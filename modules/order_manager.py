import logging
from typing import Any, Dict, Optional, List, Tuple
import time
import math

class OrderExecutionError(Exception):
    """Custom exception for order execution errors."""
    pass

class OrderManager:
    """
    Handles robust order execution, including main entry orders, stop-loss/take-profit, and retry logic.
    
    Attributes:
        exchange: ExchangeConnector instance for order placement.
        logger: Logger instance for this manager.
    """
    # Delay between main order completion and placing SL/TP orders.
    # This delay is necessary because Bybit's API exhibits eventual consistency
    # in position updates. Without this delay, SL/TP orders might be rejected
    # if placed too quickly after the main order, as the position may not be
    # fully registered in Bybit's system. The delay helps ensure the position
    # is properly established before attempting to place protective orders.
    POSITION_UPDATE_DELAY_SECONDS: float = 1.0

    def __init__(self, exchange, logger: Optional[logging.Logger] = None, max_retries: int = 3, backoff_base: float = 1.0):
        self.exchange = exchange
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.active_orders = {}
        self.MIN_NOTIONAL_USDT = 5.0  # Minimum notional value enforced by bot

    POLL_INTERVAL_SECONDS = 1
    FILL_TIMEOUT_SECONDS = 30
    OCO_POLL_INTERVAL_SECONDS = 1
    OCO_TIMEOUT_SECONDS = 60

    def _cancel_unfilled_main_and_return(self, symbol, order_id, order_responses):
        """
        Helper to cancel an unfilled main order and return early with appropriate logging and response structure.
        """
        self.logger.error(f"Main order {order_id} was not filled within timeout. Not placing SL/TP.")
        try:
            self.logger.info(f"Cancelling unfilled main order {order_id}.")
            self._retry_with_backoff(
                self.exchange.cancel_order,
                symbol=symbol,
                order_id=order_id
            )
        except Exception as cancel_exc:
            msg = str(cancel_exc)
            # Remove non-ASCII characters for Windows console compatibility
            safe_msg = msg.encode('ascii', errors='ignore').decode('ascii')
            # ErrCode 110001 = too late / order not exists
            if '110001' in safe_msg:
                self.logger.info(f"Cancel failed with 110001: order likely already filled—ignoring.")
            else:
                self.logger.error(f"Failed to cancel unfilled main order {order_id}: {safe_msg}")
        order_responses['stop_loss_order'] = None
        order_responses['take_profit_order'] = None
        return order_responses

    def place_order_with_risk(self, symbol: str, side: str, order_type: str, size: float, 
                              signal_price: Optional[float], sl_pct: Optional[float], tp_pct: Optional[float],
                              params: Optional[Dict[str, Any]] = None, 
                              reduce_only: bool = False, time_in_force: str = "GoodTillCancel") -> Dict[str, Any]:
        """
        Place a main order and, after it is filled, place associated stop-loss and take-profit orders.
        SL/TP are calculated based on actual fill price and provided percentages.
        Args:
            symbol: Trading pair symbol.
            side: 'buy' or 'sell'.
            order_type: 'market' or 'limit'.
            size: Order size.
            signal_price: Price at the time of signal generation (used as fallback or for limit orders).
            sl_pct: Stop-loss percentage (e.g., 0.01 for 1%).
            tp_pct: Take-profit percentage (e.g., 0.02 for 2%).
            params: Additional order parameters.
            reduce_only: Whether the order should be reduce-only.
            time_in_force: Time in force for the order.
        Returns:
            Dict summarizing all order responses.
        Raises:
            OrderExecutionError: If any order placement fails after retries or critical info is missing.
        """
        order_responses = {}
        filled = False
        main_order_id = None
        actual_fill_price = None
        category = 'linear'

        try:
            order_link_id = params.get('orderLinkId') if params else None

            min_order_qty, min_notional_value, qty_step = \
                self.exchange.get_min_order_amount(symbol, category=category)
            
            effective_price_for_notional_check = signal_price
            if effective_price_for_notional_check is None and order_type.lower() == 'market':
                try: # Fetch current price for notional check if signal_price is None for market order
                    ohlcv_resp = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=1)
                    ohlcv_list = ohlcv_resp.get("result", {}).get("list", [])
                    if ohlcv_list and isinstance(ohlcv_list[0], list) and len(ohlcv_list[0]) >= 5:
                        effective_price_for_notional_check = float(ohlcv_list[0][4]) # Close price of latest candle
                    else:
                        self.logger.warning("Could not fetch latest price for notional pre-check, proceeding with caution.")
                except Exception as e:
                    self.logger.warning(f"Failed to fetch latest price for notional pre-check: {e}, proceeding with caution.")

            if size is None: size = min_order_qty
            else: size = max(size, min_order_qty)

            if effective_price_for_notional_check: # Only if we have a price for check
                current_notional = size * effective_price_for_notional_check
                # Enforce Bybit's min notional value if available
                if min_notional_value and current_notional < min_notional_value:
                    raw_size = min_notional_value / effective_price_for_notional_check
                    size = math.ceil(raw_size / qty_step) * qty_step
                    size = max(size, min_order_qty)
                    self.logger.info(f"Adjusted order size to {size} to meet Bybit minNotional={min_notional_value}USDT (price={effective_price_for_notional_check}). New notional: {size * effective_price_for_notional_check:.2f}")
                
                current_notional = size * effective_price_for_notional_check # Recalculate notional with potentially adjusted size
                # Enforce bot's hard minimum notional value
                if current_notional < self.MIN_NOTIONAL_USDT:
                    raw_size = self.MIN_NOTIONAL_USDT / effective_price_for_notional_check
                    size = math.ceil(raw_size / qty_step) * qty_step
                    size = max(size, min_order_qty)
                    self.logger.info(f"Adjusted order size to {size} to meet bot minimum notional {self.MIN_NOTIONAL_USDT} USDT (price={effective_price_for_notional_check}). New notional: {size * effective_price_for_notional_check:.2f}")
            
            qty_precision = self.exchange.get_qty_precision(symbol, category)
            size = round(size, qty_precision)


            main_order_params = {
                'category': category,
                'symbol': symbol,
                'side': side.capitalize(),
                'orderType': order_type.capitalize(),
                'qty': str(size),
                'reduceOnly': reduce_only,
                'timeInForce': time_in_force,
            }
            if order_type.lower() == 'limit' and signal_price is not None:
                price_precision = self.exchange.get_price_precision(symbol, category)
                main_order_params['price'] = str(round(signal_price, price_precision))
            if order_link_id:
                main_order_params['orderLinkId'] = order_link_id
            if params: # Merge other params
                for k, v in params.items():
                    if k not in main_order_params and k not in ['sl_pct', 'tp_pct', 'signal_price']: # Avoid overriding core params
                        main_order_params[k] = v
            
            self.logger.info(f"Placing main order")
            self.logger.debug(f"Placing main order: {main_order_params}")
            main_order_response = self._retry_with_backoff(self.exchange.place_order, **main_order_params)
            self._raise_on_retcode(main_order_response, "Main order placement")
            self.log_order_status(main_order_response, "Main order")
            order_responses['main_order'] = main_order_response
            
            res_result = main_order_response.get('result', {})
            main_order_id = res_result.get('orderId') or res_result.get('order_id')

            if not main_order_id:
                raise OrderExecutionError("Main order placement did not return an order_id.")
            self.active_orders[main_order_id] = {'type': 'main', 'symbol': symbol, 'side': side, 'size': size, 'order_response': main_order_response}

            # Determine actual fill price and filled quantity
            filled_qty = 0.0

            if order_type.lower() == 'market':
                self.logger.info("Market order placed. Attempting to fetch fill details.")
                time.sleep(self.POLL_INTERVAL_SECONDS) # Small delay for order to propagate
                try:
                    order_status_resp = self._retry_with_backoff(
                        self.exchange.fetch_order,
                        symbol=symbol,
                        order_id=main_order_id,
                        category=category
                    )
                    order_data = order_status_resp.get('result', {})
                    self.logger.debug(f"Fetched market order status: {order_data}")
                    
                    # Update the main_order_response with the fetched details
                    if order_responses.get('main_order') and isinstance(order_responses['main_order'].get('result'), dict):
                        order_responses['main_order']['result'].update(order_data)
                    else:
                        # This case should ideally not happen if main_order_response was structured correctly initially
                        order_responses['main_order'] = order_status_resp 

                    if order_data.get('avgPrice') and float(order_data['avgPrice']) > 0:
                        actual_fill_price = float(order_data['avgPrice'])
                    else: # Fallback if avgPrice is zero or missing
                        actual_fill_price = signal_price # Use signal_price as a last resort
                        self.logger.warning(f"avgPrice not available or zero for market order {main_order_id}. Using signal_price {signal_price} for SL/TP calc.")

                    if order_data.get('cumExecQty') and float(order_data['cumExecQty']) > 0:
                        filled_qty = float(order_data['cumExecQty'])
                    else: # Fallback for filled quantity
                        filled_qty = size 
                        self.logger.warning(f"cumExecQty not available or zero for market order {main_order_id}. Assuming full fill size {size} for SL/TP.")
                    
                    if order_data.get('orderStatus', '').lower() in ['filled', 'partiallyfilled']:
                        filled = True
                        if main_order_id and main_order_id in self.active_orders: # Check main_order_id exists
                            self.logger.info(f"Main market order {main_order_id} is {order_data.get('orderStatus')}. Removing from active_orders.")
                            del self.active_orders[main_order_id]
                    elif order_data.get('orderStatus', '').lower() == 'new' or order_data.get('orderStatus', '').lower() == 'partiallyfilled': # Bybit might still be processing
                         self.logger.info(f"Market order {main_order_id} status is {order_data.get('orderStatus')}. Assuming fill for SL/TP placement.")
                         filled = True # Proceed with SL/TP placement optimistically
                         if main_order_id and main_order_id in self.active_orders: # Check main_order_id exists
                            self.logger.info(f"Main market order {main_order_id} assumed filled ({order_data.get('orderStatus')}). Removing from active_orders.")
                            del self.active_orders[main_order_id]
                    else:
                         self.logger.error(f"Market order {main_order_id} status is {order_data.get('orderStatus')}. Not filled.")
                         # Potentially cancel or handle as error, for now, we might not place SL/TP
                         return self._cancel_unfilled_main_and_return(symbol, main_order_id, order_responses)


                except Exception as e:
                    self.logger.error(f"Failed to fetch market order {main_order_id} details: {e}. Using signal_price and initial size for SL/TP.")
                    actual_fill_price = signal_price # Fallback
                    filled_qty = size # Fallback
                    filled = True # Optimistically assume filled to attempt SL/TP

            elif order_type.lower() == 'limit':
                self.logger.info(f"Limit order {main_order_id} placed. Waiting for fill...")
                actual_fill_price = signal_price # For limit, SL/TP are based on the limit price
                elapsed = 0
                while elapsed < self.FILL_TIMEOUT_SECONDS:
                    order_status_resp = self._retry_with_backoff(
                        self.exchange.fetch_order,
                        symbol=symbol,
                        order_id=main_order_id,
                        category=category
                    )
                    status_data = order_status_resp.get('result', {})
                    current_status = status_data.get('orderStatus', '').lower()
                    
                    if current_status == 'filled':
                        filled = True
                        filled_qty = float(status_data.get('cumExecQty', size))
                        if status_data.get('avgPrice') and float(status_data['avgPrice']) > 0 : # Update with actual avg fill price if available
                           actual_fill_price = float(status_data['avgPrice'])
                        # Update the main_order_response with the fetched details for limit orders too
                        if order_responses.get('main_order') and isinstance(order_responses['main_order'].get('result'), dict):
                            order_responses['main_order']['result'].update(status_data)
                        else:
                            order_responses['main_order'] = order_status_resp # Fallback
                        
                        if main_order_id and main_order_id in self.active_orders: # Check main_order_id exists
                            self.logger.info(f"Main limit order {main_order_id} is filled. Removing from active_orders.")
                            del self.active_orders[main_order_id]
                        break
                    if current_status == 'partiallyfilled':
                        filled = True # Partially filled is also considered for SL/TP on the filled amount
                        filled_qty = float(status_data.get('cumExecQty', 0))
                        if status_data.get('avgPrice') and float(status_data['avgPrice']) > 0 :
                           actual_fill_price = float(status_data['avgPrice'])
                        self.logger.info(f"Limit order {main_order_id} partially filled ({filled_qty}). Placing SL/TP for this amount.")
                        # Update the main_order_response with the fetched details
                        if order_responses.get('main_order') and isinstance(order_responses['main_order'].get('result'), dict):
                            order_responses['main_order']['result'].update(status_data)
                        else:
                            order_responses['main_order'] = order_status_resp # Fallback
                        
                        if main_order_id and main_order_id in self.active_orders: # Check main_order_id exists
                            self.logger.info(f"Main limit order {main_order_id} is partiallyfilled. Removing from active_orders.")
                            del self.active_orders[main_order_id]
                        break
                    if current_status in ['rejected', 'cancelled', 'expired']:
                         self.logger.error(f"Limit order {main_order_id} failed with status: {current_status}")
                         # Ensure removal from active_orders if it was added and then failed terminally
                         if main_order_id and main_order_id in self.active_orders:
                             self.logger.info(f"Main limit order {main_order_id} failed ({current_status}). Removing from active_orders.")
                             del self.active_orders[main_order_id]
                         return order_responses # Return without SL/TP

                    time.sleep(self.POLL_INTERVAL_SECONDS)
                    elapsed += self.POLL_INTERVAL_SECONDS
                
                if not filled:
                    return self._cancel_unfilled_main_and_return(symbol, main_order_id, order_responses)
            
            if not actual_fill_price or actual_fill_price <= 0: # Final check for a valid fill price
                self.logger.error(f"Could not determine a valid fill price for order {main_order_id}. Signal price: {signal_price}. Cannot proceed with SL/TP.")
                # Depending on policy, might try to cancel main_order_id if it's not confirmed closed/rejected
                raise OrderExecutionError("Failed to determine actual fill price for SL/TP calculation.")

            if filled_qty <= 0:
                self.logger.error(f"Order {main_order_id} resulted in zero filled quantity. Cannot place SL/TP.")
                raise OrderExecutionError("Main order filled quantity is zero.")

            # After main order is filled, ensure SL/TP orders are placed with retries
            if filled and actual_fill_price and filled_qty > 0:
                # Calculate SL/TP prices
                stop_loss_price_calculated = None
                take_profit_price_calculated = None
                price_precision = self.exchange.get_price_precision(symbol, category)

                if sl_pct is not None:
                    if side.lower() == "buy":
                        stop_loss_price_calculated = actual_fill_price * (1 - sl_pct)
                    elif side.lower() == "sell":
                        stop_loss_price_calculated = actual_fill_price * (1 + sl_pct)
                    if stop_loss_price_calculated is not None:
                        stop_loss_price_calculated = round(stop_loss_price_calculated, price_precision)
                
                if tp_pct is not None:
                    if side.lower() == "buy":
                        take_profit_price_calculated = actual_fill_price * (1 + tp_pct)
                    elif side.lower() == "sell":
                        take_profit_price_calculated = actual_fill_price * (1 - tp_pct)
                    if take_profit_price_calculated is not None:
                        take_profit_price_calculated = round(take_profit_price_calculated, price_precision)

                # Place stop-loss order with retries
                if sl_pct is not None and stop_loss_price_calculated is not None and stop_loss_price_calculated > 0:
                    stop_side = 'Sell' if side.lower() == 'buy' else 'Buy'
                    trigger_direction_sl = 2 if side.lower() == 'buy' else 1
                    sl_params = {
                        'category': category,
                        'symbol': symbol,
                        'side': stop_side,
                        'orderType': 'Market',
                        'qty': str(filled_qty),
                        'reduceOnly': True,
                        'triggerPrice': str(stop_loss_price_calculated),
                        'triggerBy': 'LastPrice',
                        'triggerDirection': trigger_direction_sl,
                        'positionIdx': 0,
                        'closeOnTrigger': True
                    }
                    if order_link_id: sl_params['orderLinkId'] = order_link_id + "-sl"
                    
                    self.logger.info(f"Placing stop-loss order (trigger).")
                    self.logger.debug(f"Placing stop-loss order (trigger): {sl_params}")
                    
                    # Retry SL order placement up to 3 times
                    for attempt in range(3):
                        try:
                            stop_loss_response = self._retry_with_backoff(self.exchange.place_order, **sl_params)
                            self._raise_on_retcode(stop_loss_response, "Stop-loss order placement")
                            self.log_order_status(stop_loss_response, "Stop-loss order")
                            order_responses['stop_loss_order'] = stop_loss_response
                            sl_order_id = stop_loss_response.get('result', {}).get('orderId')
                            if sl_order_id:
                                self.active_orders[sl_order_id] = {
                                    'type': 'sl',
                                    'main_order_id': main_order_id,
                                    'symbol': symbol,
                                    'side': stop_side,
                                    'size': filled_qty,
                                    'trigger_price': stop_loss_price_calculated
                                }
                            break
                        except Exception as e:
                            if attempt == 2:  # Last attempt
                                self.logger.error(f"Failed to place stop-loss order after 3 attempts: {e}")
                                raise
                            self.logger.warning(f"Attempt {attempt + 1} to place stop-loss order failed: {e}")
                            time.sleep(1)  # Wait before retry

                # Place take-profit order with retries
                if tp_pct is not None and take_profit_price_calculated is not None and take_profit_price_calculated > 0:
                    tp_side = 'Sell' if side.lower() == 'buy' else 'Buy'
                    trigger_direction_tp = 1 if side.lower() == 'buy' else 2
                    tp_params = {
                        'category': category,
                        'symbol': symbol,
                        'side': tp_side,
                        'orderType': 'Market',
                        'qty': str(filled_qty),
                        'reduceOnly': True,
                        'triggerPrice': str(take_profit_price_calculated),
                        'triggerBy': 'LastPrice',
                        'triggerDirection': trigger_direction_tp,
                        'positionIdx': 0,
                        'closeOnTrigger': True
                    }
                    if order_link_id: tp_params['orderLinkId'] = order_link_id + "-tp"

                    self.logger.info(f"Placing take-profit order (trigger).")
                    self.logger.debug(f"Placing take-profit order (trigger): {tp_params}")
                    
                    # Retry TP order placement up to 3 times
                    for attempt in range(3):
                        try:
                            take_profit_response = self._retry_with_backoff(self.exchange.place_order, **tp_params)
                            self._raise_on_retcode(take_profit_response, "Take-profit order placement")
                            self.log_order_status(take_profit_response, "Take-profit order")
                            order_responses['take_profit_order'] = take_profit_response
                            tp_order_id = take_profit_response.get('result', {}).get('orderId')
                            if tp_order_id:
                                self.active_orders[tp_order_id] = {
                                    'type': 'tp',
                                    'main_order_id': main_order_id,
                                    'symbol': symbol,
                                    'side': tp_side,
                                    'size': filled_qty,
                                    'trigger_price': take_profit_price_calculated
                                }
                            break
                        except Exception as e:
                            if attempt == 2:  # Last attempt
                                self.logger.error(f"Failed to place take-profit order after 3 attempts: {e}")
                                raise
                            self.logger.warning(f"Attempt {attempt + 1} to place take-profit order failed: {e}")
                            time.sleep(1)  # Wait before retry

                # Verify SL/TP orders are properly placed
                if (sl_pct is not None and 'stop_loss_order' not in order_responses) or \
                   (tp_pct is not None and 'take_profit_order' not in order_responses):
                    self.logger.error("Failed to place one or more protective orders. Cancelling main position.")
                    try:
                        # Cancel the main position if protective orders failed
                        self.exchange.cancel_order(symbol=symbol, order_id=main_order_id)
                        raise OrderExecutionError("Failed to place protective orders. Main position cancelled.")
                    except Exception as e:
                        self.logger.error(f"Failed to cancel main position after protective order failure: {e}")
                        raise OrderExecutionError("Failed to place protective orders and cancel main position.")

            return order_responses

        except OrderExecutionError as oee:
            self.logger.error(f"OrderExecutionError in place_order_with_risk: {oee}")
            if 'main_order' not in order_responses: order_responses['main_order'] = None
            if 'stop_loss_order' not in order_responses: order_responses['stop_loss_order'] = None
            if 'take_profit_order' not in order_responses: order_responses['take_profit_order'] = None
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in place_order_with_risk: {e}", exc_info=True)
            if 'main_order' not in order_responses: order_responses['main_order'] = None
            if 'stop_loss_order' not in order_responses: order_responses['stop_loss_order'] = None
            if 'take_profit_order' not in order_responses: order_responses['take_profit_order'] = None
            raise OrderExecutionError(f"Unexpected error: {str(e)}") from e

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """
        Retry a function with exponential backoff on failure.
        Args:
            func: Function to call.
            *args, **kwargs: Arguments for the function.
        Returns:
            Result of func(*args, **kwargs).
        Raises:
            OrderExecutionError: If all retries fail, raises the last exception wrapped.
        """
        assert callable(func), "func must be callable"
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                last_exception = exc
                wait_time = self.backoff_base * (2 ** attempt)
                self.logger.warning(f"Retry {attempt+1}/{self.max_retries} after error: {exc}. Waiting {wait_time:.2f}s.")
                time.sleep(wait_time)
        self.logger.error(f"All retries failed for {getattr(func, '__name__', str(func))}: {last_exception}")
        raise OrderExecutionError(f"All retries failed for {getattr(func, '__name__', str(func))}: {last_exception}") from last_exception

    def log_order_status(self, order_response: Dict[str, Any], order_type_context: str):
        """
        Log the status of an order.
        Args:
            order_response: The response dict from the exchange.
            order_type_context: The context of the order (e.g., "Main order", "Stop-loss order").
        """
        if not isinstance(order_response, dict):
            self.logger.error(f"Order response is not a dict: {order_response} for {order_type_context}")
            return

        result_data = order_response.get('result', {})
        # V5 API uses camelCase for fields in result
        order_id = result_data.get('orderId') or order_response.get('orderId') # Fallback to root for some cases
        status = result_data.get('orderStatus') or order_response.get('orderStatus')
        side = result_data.get('side') or order_response.get('side')
        actual_order_type = result_data.get('orderType') or order_response.get('orderType') # Actual type from response
        qty = result_data.get('qty') or order_response.get('qty')
        price = result_data.get('price') or order_response.get('price')
        avg_price = result_data.get('avgPrice')
        
        ret_code_raw = order_response.get('retCode', order_response.get('ret_code', 0))
        try:
            ret_code = int(ret_code_raw)
        except (TypeError, ValueError):
            ret_code = -1  # default to failure if non-numeric
        error_msg = order_response.get('retMsg')

        if error_msg:
            # Convert to string, remove non-ASCII chars, and replace control chars with spaces
            error_msg = str(error_msg).encode('ascii', errors='ignore').decode('ascii')
            # Replace common control characters (newlines, tabs, etc.) with spaces
            error_msg = ' '.join(error_msg.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ').split())

        log_price = avg_price if avg_price and float(avg_price) > 0 else price

        if ret_code == 0:
            self.logger.info(
                f"{order_type_context} details: id={order_id}, status={status}, side={side}, type={actual_order_type}, qty={qty}, price={log_price}"
            )
        else:
            self.logger.error(
                f"{order_type_context} failed: id={order_id}, status={status}, side={side}, type={actual_order_type}, qty={qty}, price={log_price}, msg='{error_msg}' (retCode={ret_code})"
            )
            self.logger.debug(f"Full raw order_response for failed order: {order_response}")

    def sync_active_orders_with_exchange(self, symbol: str, category: str = 'linear'):
        """
        Synchronizes the bot's list of active orders with the actual open orders on the exchange for a given symbol.
        Identifies and "adopts" any open orders on the exchange that are not tracked locally.
        Handles cancellation of counterpart SL/TP if one of them is found to be closed.
        Returns:
            List[Dict[str, Any]]: A list of order details for newly adopted orders.
        """
        self.logger.debug(f"Syncing active orders with exchange for {symbol}...")
        adopted_orders = []
        try:
            exchange_open_orders_response = self._retry_with_backoff(
                self.exchange.fetch_open_orders,
                symbol=symbol,
                params={'category': category} # Pass category via params
            )
            
            # CCXT's fetch_open_orders returns a list of order dicts directly.
            # Adapting to handle both direct list and Bybit V5-like dict response.
            open_orders_list = []
            if isinstance(exchange_open_orders_response, list):
                open_orders_list = exchange_open_orders_response
            elif isinstance(exchange_open_orders_response, dict) and \
                 exchange_open_orders_response.get('retCode') == 0 and \
                 isinstance(exchange_open_orders_response.get('result'), dict) and \
                 isinstance(exchange_open_orders_response['result'].get('list'), list):
                open_orders_list = exchange_open_orders_response['result']['list']
            else:
                # If the response is a dict but not in the expected Bybit V5 format, 
                # or some other unexpected type, log and treat as no orders.
                self.logger.warning(f"Unexpected format for open orders response for {symbol} ({category}): {exchange_open_orders_response}. Assuming no open orders.")
            
            exchange_open_orders = open_orders_list

            if not isinstance(exchange_open_orders, list): # Final safety check
                self.logger.warning(f"Exchange open orders for {symbol} was not a list: {exchange_open_orders}. Assuming no open orders.")
                exchange_open_orders = []
                
            exchange_order_ids = {o['orderId'] for o in exchange_open_orders if isinstance(o, dict) and 'orderId' in o}
            
            locally_tracked_ids_not_on_exchange = []
            for order_id, order_info in list(self.active_orders.items()): # Iterate over a copy
                if order_info.get('symbol') == symbol and order_id not in exchange_order_ids:
                    locally_tracked_ids_not_on_exchange.append(order_id)

            for order_id in locally_tracked_ids_not_on_exchange:
                if order_id not in self.active_orders: continue # Might have been processed by counterpart cancellation

                order_info = self.active_orders.pop(order_id) # Remove from tracking and get info
                order_type = order_info.get('type')
                self.logger.info(f"Order {order_id} ({order_type}) for {symbol} is tracked locally but no longer on exchange. Removed from local tracking.")

                associated_main_order_id = order_info.get('main_order_id')
                if associated_main_order_id: # This indicates it was an SL or TP order
                    if order_type == 'sl':
                        self.logger.info(f"SL order {order_id} (for main order {associated_main_order_id}) disappeared. Attempting to cancel counterpart TP.")
                        self._cancel_specific_order_type_for_main(symbol, associated_main_order_id, 'tp', category=category)
                    elif order_type == 'tp':
                        self.logger.info(f"TP order {order_id} (for main order {associated_main_order_id}) disappeared. Attempting to cancel counterpart SL.")
                        self._cancel_specific_order_type_for_main(symbol, associated_main_order_id, 'sl', category=category)
                # If a 'main' order disappeared, it likely filled or was cancelled externally.
                # SL/TP orders related to it (if any were placed) should remain managed by their own lifecycle
                # or by explicit strategy exit. No automatic cancellation of SL/TP here based on main order vanishing.

            # Adoption logic for orders on exchange but not tracked locally
            for ex_order in exchange_open_orders:
                if not isinstance(ex_order, dict): # Skip if not a dict
                    self.logger.warning(f"Found non-dict item in exchange open orders for {symbol}: {ex_order}")
                    continue
                order_id = ex_order.get('orderId')
                if not order_id:
                    self.logger.warning(f"Found an exchange order for {symbol} without an orderId: {ex_order}")
                    continue

                if order_id not in self.active_orders:
                    self.logger.info(
                        f"Order {order_id} for {symbol} (Status: {ex_order.get('orderStatus')}) is open on exchange "
                        f"but not tracked locally. Adopting it."
                    )
                    # Basic adoption. For SL/TP, main_order_id might be missing unless inferred from orderLinkId.
                    adopted_type = ex_order.get('orderType', 'unknown').lower()
                    if ex_order.get('stopOrderType') or ex_order.get('triggerPrice'): # Heuristic for SL/TP
                        # Try to infer if it's SL or TP based on common patterns or custom logic if available
                        # This part might need more sophisticated logic if you rely on main_order_id for adopted SL/TP
                        adopted_type = 'conditional' # Generic conditional
                    
                    self.active_orders[order_id] = {
                        'type': adopted_type,
                        'symbol': ex_order.get('symbol'),
                        'side': ex_order.get('side'),
                        'size': float(ex_order.get('qty', 0)),
                        'order_response': ex_order.copy(), # Store the full exchange response
                        'is_externally_synced': True,
                        'main_order_id': None # Cannot reliably determine this for adopted SL/TP without more info
                    }
                    adopted_orders.append(ex_order.copy())
                else:
                    # Order is tracked locally and on exchange. Update local status if necessary
                    local_order_info = self.active_orders[order_id]
                    exchange_status = ex_order.get('orderStatus')
                    current_local_status = local_order_info.get('order_response', {}).get('orderStatus')

                    # Log only if the status has actually changed from a previously known, different status
                    # Avoid logging if it's just the first time seeing an initial status like 'New' or 'Untriggered' if local was None/empty
                    if exchange_status and exchange_status != current_local_status:
                        if not (current_local_status is None and exchange_status in ['New', 'Untriggered', 'Active']): # Common initial states
                            self.logger.info(f"Updating status for tracked order {order_id} from '{current_local_status}' to '{exchange_status}'.")
                        
                        if isinstance(local_order_info.get('order_response'), dict):
                             local_order_info['order_response']['orderStatus'] = exchange_status
                             # Potentially update other fields like avgPrice, cumExecQty if relevant
                             if 'avgPrice' in ex_order:
                                 local_order_info['order_response']['avgPrice'] = ex_order['avgPrice']
                             if 'cumExecQty' in ex_order:
                                 local_order_info['order_response']['cumExecQty'] = ex_order['cumExecQty']

            self.logger.debug(f"Finished syncing orders for {symbol}. Adopted {len(adopted_orders)} new order(s). Active tracked: {len(self.active_orders)}")

        except Exception as e:
            self.logger.error(f"Error during sync_active_orders_with_exchange for {symbol}: {e}", exc_info=True)
        
        return adopted_orders

    def _cancel_specific_order_type_for_main(self, symbol: str, main_order_id: str, order_type_to_cancel: str, category: str = 'linear') -> None:
        """Cancels a specific type of order (sl or tp) associated with a main_order_id."""
        counterpart_order_id_to_cancel = None
        for order_id_iter, order_info_iter in list(self.active_orders.items()): # Iterate copy
            if (order_info_iter.get('symbol') == symbol and
                order_info_iter.get('main_order_id') == main_order_id and
                order_info_iter.get('type') == order_type_to_cancel):
                counterpart_order_id_to_cancel = order_id_iter
                break 
            
        if counterpart_order_id_to_cancel:
            self.logger.info(f"Attempting to cancel counterpart {order_type_to_cancel} order {counterpart_order_id_to_cancel} (for main order {main_order_id}) on {symbol}.")
            try:
                cancel_args = {'symbol': symbol, 'id': counterpart_order_id_to_cancel}
                if hasattr(self.exchange, 'unified_cancel_order_requires_category') and self.exchange.unified_cancel_order_requires_category:
                    cancel_args['params'] = {'category': category}

                self._retry_with_backoff(
                    self.exchange.cancel_order,
                    **cancel_args
                )
                self.logger.info(f"Successfully cancelled counterpart {order_type_to_cancel} order {counterpart_order_id_to_cancel}.")
                if counterpart_order_id_to_cancel in self.active_orders: 
                    del self.active_orders[counterpart_order_id_to_cancel]
            except OrderExecutionError as oee:
                # Bybit error codes for "already filled/cancelled" or "not found"
                # 10001: params error (can sometimes mean orderId invalid if it was already processed)
                # 110001: Order has been filled or cancelled
                # 110007: Order not found or status does not allow cancellation
                # CCXT might also raise specific exceptions like OrderNotFound
                msg_lower = str(oee).lower()
                if '110001' in msg_lower or '110007' in msg_lower or \
                   'order has been filled or cancelled' in msg_lower or \
                   'order does not exist' in msg_lower or 'order not found' in msg_lower:
                    self.logger.info(f"Counterpart {order_type_to_cancel} order {counterpart_order_id_to_cancel} likely already closed/cancelled. Message: {str(oee)}. Removing from tracking.")
                    if counterpart_order_id_to_cancel in self.active_orders:
                        del self.active_orders[counterpart_order_id_to_cancel]
                else:
                    self.logger.error(f"Failed to cancel counterpart {order_type_to_cancel} order {counterpart_order_id_to_cancel} after retries: {oee}")
            except Exception as e: # Catch other unexpected errors
                self.logger.error(f"Unexpected error cancelling counterpart {order_type_to_cancel} order {counterpart_order_id_to_cancel}: {e}", exc_info=True)
        else:
            self.logger.info(f"No active counterpart {order_type_to_cancel} order found for main order {main_order_id} on {symbol} to cancel.")


    def _cancel_all_sl_tp_for_main_order(self, symbol: str, main_order_id: str, category: str = 'linear') -> None:
        """Cancels all SL and TP orders associated with a given main_order_id."""
        self.logger.info(f"Cancelling all SL/TP orders linked to main order {main_order_id} for {symbol}.")
        orders_to_cancel_ids = []
        for order_id, order_info in list(self.active_orders.items()): 
            if (order_info.get('symbol') == symbol and
                order_info.get('main_order_id') == main_order_id and
                order_info.get('type') in ['sl', 'tp']):
                orders_to_cancel_ids.append(order_id)
                
        for order_id_to_cancel in orders_to_cancel_ids:
            order_info = self.active_orders.get(order_id_to_cancel) 
            if not order_info: continue

            order_type = order_info.get('type', 'unknown')
            self.logger.info(f"Attempting to cancel {order_type} order {order_id_to_cancel} (for main order {main_order_id}) on {symbol}.")
            try:
                cancel_args = {'symbol': symbol, 'id': order_id_to_cancel}
                if hasattr(self.exchange, 'unified_cancel_order_requires_category') and self.exchange.unified_cancel_order_requires_category:
                    cancel_args['params'] = {'category': category}
                
                self._retry_with_backoff(
                    self.exchange.cancel_order,
                    **cancel_args
                )
                self.logger.info(f"Successfully cancelled {order_type} order {order_id_to_cancel}.")
                if order_id_to_cancel in self.active_orders: 
                    del self.active_orders[order_id_to_cancel]
            except OrderExecutionError as oee:
                msg_lower = str(oee).lower()
                if '110001' in msg_lower or '110007' in msg_lower or \
                   'order has been filled or cancelled' in msg_lower or \
                   'order does not exist' in msg_lower or 'order not found' in msg_lower:
                    self.logger.info(f"{order_type} order {order_id_to_cancel} likely already closed/cancelled. Message: {str(oee)}. Removing from tracking.")
                    if order_id_to_cancel in self.active_orders:
                        del self.active_orders[order_id_to_cancel]
                else:
                    self.logger.error(f"Failed to cancel {order_type} order {order_id_to_cancel} after retries: {oee}")
            except Exception as e: 
                self.logger.error(f"Unexpected error cancelling {order_type} order {order_id_to_cancel}: {e}", exc_info=True)

    def execute_strategy_exit(self, symbol: str, position_to_close: Dict[str, Any], category: str = 'linear') -> Dict[str, Any]:
        """
        Executes a market order to close an existing position based on strategy signal
        and then cancels any related SL/TP orders.
        Args:
            symbol: The trading symbol.
            position_to_close: A dict containing details of the position,
                               expected to have 'side', 'size', 'main_order_id'.
        Returns:
            Dict summarizing the exit order response, or empty if error.
        Raises:
            OrderExecutionError: If the exit process fails critically.
        """
        exit_order_responses = {}
        original_side = position_to_close.get('side')
        position_size = position_to_close.get('size')
        # main_order_id_of_position is crucial for identifying which SL/TP orders to cancel.
        main_order_id_of_position = position_to_close.get('main_order_id') 

        if not all([original_side, position_size]): # main_order_id might be None if position was adopted
            msg = f"Missing side or size in position_to_close for strategy exit on {symbol}. Details: {position_to_close}"
            self.logger.error(msg)
            raise OrderExecutionError(msg)
        
        if not main_order_id_of_position:
            self.logger.warning(f"main_order_id not found in position_to_close for {symbol}. SL/TP cancellation might not be targeted if multiple positions/orders exist. Position details: {position_to_close}")
            # If main_order_id is None, _cancel_all_sl_tp_for_main_order will not find orders to cancel.
            # This is acceptable if the position was adopted and we don't have its original main_order_id.

        exit_side = 'sell' if original_side.lower() == 'buy' else 'buy'
        
        try:
            self.logger.info(f"Strategy signaled exit for {symbol}. Closing position (Original Side: {original_side}, Size: {position_size}) with a market {exit_side} order.")
            
            exit_market_order_params = {
                'category': 'linear', # Assuming linear USDT perpetual
                'symbol': symbol,
                'side': exit_side.capitalize(),
                'orderType': 'Market',
                'qty': str(position_size),
                'reduceOnly': True 
            }
            
            # Use orderLinkId for the exit order for better tracking if needed
            # exit_market_order_params['orderLinkId'] = f"exit-{main_order_id_of_position[:8]}-{int(time.time())}" if main_order_id_of_position else f"exit-adhoc-{int(time.time())}"

            exit_order_response = self._retry_with_backoff(self.exchange.place_order, **exit_market_order_params)
            self._raise_on_retcode(exit_order_response, "Strategy exit market order")
            self.log_order_status(exit_order_response, "Strategy Exit Market Order")
            exit_order_responses['exit_market_order'] = exit_order_response
            
            exit_order_id = exit_order_response.get('result', {}).get('orderId')
            # Wait for this exit market order to be confirmed filled before cancelling SL/TP.
            # This is important to prevent cancelling SL/TP if the exit market order itself fails or is rejected.
            if exit_order_id:
                self.logger.info(f"Strategy exit market order {exit_order_id} placed. Waiting for fill confirmation...")
                # Simplified fill check; a more robust one would poll get_order_status
                time.sleep(self.POSITION_UPDATE_DELAY_SECONDS) # Wait for propagation
                # Ideally, confirm fill status of exit_order_id before proceeding

            if main_order_id_of_position:
                self.logger.info(f"Position for {symbol} (related to main order {main_order_id_of_position}) assumed closed by strategy exit. Cancelling associated SL/TP orders.")
                self._cancel_all_sl_tp_for_main_order(symbol, main_order_id_of_position, category=category)
            else:
                self.logger.info(f"Position for {symbol} closed by strategy exit. No main_order_id was available, so targeted SL/TP cancellation skipped. Any active SL/TP for this symbol might need manual review or broader cleanup if not specific to this exited position.")


        except OrderExecutionError as oee: # Catch OrderExecutionError from _retry_with_backoff or _raise_on_retcode
            self.logger.error(f"OrderExecutionError during execute_strategy_exit for {symbol}: {oee}", exc_info=False) # exc_info False as it's already logged by _retry
            raise # Re-raise to be caught by bot.py
        except Exception as e: # Catch any other unexpected errors
            self.logger.error(f"Unexpected error during execute_strategy_exit for {symbol}: {e}", exc_info=True)
            raise OrderExecutionError(f"Unexpected failure in execute_strategy_exit for {symbol}: {str(e)}") from e
            
        return exit_order_responses

    def _raise_on_retcode(self, response, context):
        ret_code = response.get('retCode', response.get('ret_code', None))
        # Allow for string retCode from some exchanges, try to convert to int
        if isinstance(ret_code, str):
            try:
                ret_code = int(ret_code)
            except ValueError:
                self.logger.warning(f"Non-integer retCode '{ret_code}' received for {context}. Assuming error if not '0'.")
                if ret_code != '0': # If it's a string but not "0", treat as error.
                    error_msg = response.get('retMsg', f'Unknown error with string retCode {ret_code}')
                    self.logger.error(f"{context} failed with retCode {ret_code}: {error_msg}")
                    raise OrderExecutionError(f"{context} failed: {error_msg} (retCode {ret_code})")
                else: # String "0" is OK
                    ret_code = 0


        if ret_code is not None and ret_code != 0:
            error_msg = response.get('retMsg', 'Unknown error')
            self.logger.error(f"{context} failed with retCode {ret_code}: {error_msg}")
            self.logger.debug(f"Full error response for {context}: {response}")
            raise OrderExecutionError(f"{context} failed: {error_msg} (retCode {ret_code})") 

    def _classify_conditional_order(self, order: Dict[str, Any]) -> Optional[str]:
        """
        Classifies an order as 'sl' (Stop Loss) or 'tp' (Take Profit) based on its info.
        Returns None if not a conditional order of interest.
        """
        if not order or not isinstance(order.get('info'), dict):
            return None

        info = order['info']
        # Bybit V5 API uses stopOrderType for conditional orders
        # Common values: 'StopLoss', 'TakeProfit', 'TrailingStop'
        # We consider 'StopLoss' and 'TrailingStop' as SL types for simplicity here.
        stop_order_type = info.get('stopOrderType', '').lower()
        # trigger_direction = info.get('triggerDirection') # 1: rise, 2: fall. Not used for now but good to know.

        if stop_order_type == 'stoploss' or stop_order_type == 'trailingstop':
            return 'sl'
        elif stop_order_type == 'takeprofit':
            return 'tp'
        
        # Fallback for older API versions or different structures if needed,
        # by looking at price relative to market for stop orders
        # This is a simplified heuristic and might need adjustment based on actual API response
        # For now, we rely primarily on 'stopOrderType'
        # order_type = info.get('orderType', '').lower()
        # order_status = info.get('orderStatus', '').lower()

        # if 'stop' in order_type and order_status in ['new', 'untriggered', 'active']: 
        #     pass # Complex logic, deferring to stopOrderType

        return None

    def check_and_cancel_orphaned_conditional_orders(self, symbol: str, category: str = 'linear'):
        """
        Checks for orphaned SL/TP orders for a given symbol and cancels them.
        Orphaned means:
        1. Any SL/TP order if there's no active position for the symbol.
        2. A lone SL without a TP, or a lone TP without an SL, when a position *does* exist.
        """
        self.logger.debug(f"Checking for orphaned conditional orders for {symbol} (category: {category})...")
        open_sl_orders: List[Dict[str, Any]] = []
        open_tp_orders: List[Dict[str, Any]] = []

        try:
            # Assuming fetch_open_orders in exchange module can take 'category'
            open_orders_response = self._retry_with_backoff(
                self.exchange.fetch_open_orders,
                symbol=symbol,
                params={'category': category} # Pass category via params if underlying ccxt method needs it
            )
            # self._raise_on_retcode(open_orders_response, f"Fetching open orders for {symbol}") # CCXT usually doesn't have retcode for fetch*
            
            # CCXT's fetch_open_orders returns a list of order dicts directly
            # The structure might vary slightly if using Pybit directly; adapt as needed.
            # For CCXT, the response is the list itself.
            open_orders = open_orders_response if isinstance(open_orders_response, list) else []
            
            # If Pybit V5 response structure is {'retCode': 0, 'result': {'list': [...]}}
            if isinstance(open_orders_response, dict) and 'result' in open_orders_response and 'list' in open_orders_response['result']:
                 open_orders = open_orders_response['result']['list']


            if not open_orders:
                self.logger.debug(f"No open orders found for {symbol} in category {category}. No orphans to check.")
                return

            for order in open_orders:
                order_id = order.get('id') or order.get('orderId') # CCXT uses 'id', Bybit V5 API uses 'orderId'
                info = order.get('info', {}) # Ensure info is a dict
                
                # Re-wrapping order with 'info' if it's not already structured like CCXT unified order
                # This helps _classify_conditional_order work consistently
                ccxt_like_order = {'info': info, **order}


                classification = self._classify_conditional_order(ccxt_like_order)
                if classification == 'sl':
                    open_sl_orders.append(ccxt_like_order) # Store the ccxt_like_order
                    self.logger.debug(f"Found open SL order: {order_id} for {symbol}")
                elif classification == 'tp':
                    open_tp_orders.append(ccxt_like_order) # Store the ccxt_like_order
                    self.logger.debug(f"Found open TP order: {order_id} for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to fetch or classify open orders for {symbol}: {e}")
            return

        # Check for active position
        position_size = 0.0
        try:
            # Call fetch_positions with symbol and category (per ExchangeConnector interface)
            positions_response = self._retry_with_backoff(
                self.exchange.fetch_positions,
                symbol=symbol,
                category=category
            )

            active_position = None
            # Bybit V5 style: {'result': {'list': [...]}}
            if isinstance(positions_response, dict) and 'result' in positions_response and 'list' in positions_response['result']:
                positions_list = positions_response['result']['list']
                if positions_list:
                    for pos_data in positions_list:
                        if pos_data.get('symbol') == symbol:
                            pos_size_str = pos_data.get('size', '0')
                            if pos_size_str is not None:
                                position_size = float(pos_size_str)
                                if position_size > 0:
                                    active_position = pos_data
                                    break
            # Fallback for other formats (e.g., CCXT)
            elif isinstance(positions_response, list):
                for pos in positions_response:
                    if pos.get('symbol') == symbol:
                        pos_size_str = pos.get('contracts')
                        if pos_size_str is None:
                            pos_size_str = pos.get('info', {}).get('size', '0')
                        if pos_size_str is not None:
                            position_size = float(pos_size_str)
                            if position_size > 0:
                                active_position = pos
                                break
            self.logger.debug(f"Current position size for {symbol}: {position_size}")

        except Exception as e:
            self.logger.error(f"Failed to fetch position for {symbol}: {e}")
            return

        orders_to_cancel: List[Tuple[Dict[str, Any], str]] = [] 

        if position_size == 0:
            self.logger.info(f"No active position for {symbol}. All conditional orders are orphans.")
            for sl_order in open_sl_orders:
                orders_to_cancel.append((sl_order, "No active position"))
            for tp_order in open_tp_orders:
                orders_to_cancel.append((tp_order, "No active position"))
        else: 
            if len(open_sl_orders) == 1 and len(open_tp_orders) == 0:
                self.logger.warning(f"Found 1 SL order and 0 TP orders for active position on {symbol}. Cancelling SL.")
                orders_to_cancel.append((open_sl_orders[0], "Lone SL with active position"))
            elif len(open_tp_orders) == 1 and len(open_sl_orders) == 0:
                self.logger.warning(f"Found 1 TP order and 0 SL orders for active position on {symbol}. Cancelling TP.")
                orders_to_cancel.append((open_tp_orders[0], "Lone TP with active position"))
            elif len(open_sl_orders) == 0 and len(open_tp_orders) == 0:
                self.logger.debug(f"Position exists for {symbol}, but no SL/TP orders found. This is normal.")
            elif len(open_sl_orders) > 1 or len(open_tp_orders) > 1:
                 self.logger.warning(f"Multiple SL ({len(open_sl_orders)}) or TP ({len(open_tp_orders)}) orders found for {symbol} with an active position. Manual review might be needed. No automatic cancellation for this scenario.")
            elif len(open_sl_orders) == 1 and len(open_tp_orders) == 1:
                self.logger.debug(f"Found 1 SL and 1 TP order for active position on {symbol}. Looks good.")
            

        for order_to_cancel, reason in orders_to_cancel:
            # CCXT uses 'id'. Bybit V5 API uses 'orderId'. 'info' should contain raw details.
            order_id = order_to_cancel.get('id') or order_to_cancel.get('info', {}).get('orderId')
            order_link_id = order_to_cancel.get('info', {}).get('orderLinkId')
            
            # For Bybit conditional orders, cancellation might need order_id or conditional_order_id if using trigger orders
            # If these are TP/SL attached to a position (e.g. via place_order with slLimitPrice/tpLimitPrice),
            # they might not be separate 'conditional' orders but part of the position's metadata or separate limit orders.
            # The current logic assumes SL/TP are distinct sto/limit orders found by fetch_open_orders.

            actual_id_to_use_for_cancel = order_id 
            # Bybit V5: cancel_order uses orderId or orderLinkId.
            # If it's a conditional order not yet triggered, orderId might be the one.
            # If _classify_conditional_order relies on 'stopOrderType', these are likely Bybit V5 conditional orders.

            if not actual_id_to_use_for_cancel:
                self.logger.error(f"Cannot cancel order for {symbol}, missing order ID. Order data: {order_to_cancel.get('info')}")
                continue
            
            self.logger.info(f"Attempting to cancel orphaned conditional order ID: {actual_id_to_use_for_cancel} for {symbol}. Reason: {reason}")
            try:
                cancel_params = {'symbol': symbol}
                # For CCXT general cancel_order
                # For Bybit (using CCXT wrapper for bybit):
                #   - Regular orders: 'id' is order_id
                #   - Conditional orders (stopOrderType): 'id' is order_id.
                #     It might be that cancel_order on CCXT for Bybit handles this.
                #   - If using Pybit directly, parameters would be specific (e.g. orderId=, orderLinkId=)
                # The `exchange.py` wrapper should abstract this.
                # Assuming `self.exchange.cancel_order` takes `id` and `symbol`.
                # And for Bybit V5, it might need `category` in params.
                
                cancel_args = {'id': actual_id_to_use_for_cancel, 'symbol': symbol}
                # If exchange module expects category for Bybit V5:
                if hasattr(self.exchange, 'unified_cancel_order_requires_category') and self.exchange.unified_cancel_order_requires_category:
                    cancel_args['params'] = {'category': category}


                cancel_response = self._retry_with_backoff(
                    self.exchange.cancel_order,
                    **cancel_args
                )
                # self._raise_on_retcode(cancel_response, f"Cancelling order {actual_id_to_use_for_cancel}") # CCXT does not use retcode for cancel
                self.logger.info(f"Cancellation attempt for order {actual_id_to_use_for_cancel} response: {cancel_response}")
                # To confirm cancellation, one might re-fetch orders or check response details specific to exchange.
                # CCXT cancel_order typically returns info about the order, sometimes including its new status.

            except Exception as e:
                error_message = str(e).lower()
                # Common CCXT/Bybit error patterns for already closed/non-existent orders
                if 'order_not_exists' in error_message or \
                   'order has been filled or canceled' in error_message or \
                   'order does not exist' in error_message or \
                   'already been filled or cancelled' in error_message or \
                   'too late to cancel' in error_message or \
                   '110001' in error_message or '30034' in error_message or '10001' in error_message: # 10001: "order not found" general
                    self.logger.info(f"Order {actual_id_to_use_for_cancel} likely already closed/cancelled for {symbol}: {e}")
                else:
                    self.logger.error(f"Failed to cancel orphaned conditional order {actual_id_to_use_for_cancel} for {symbol}: {e}") 