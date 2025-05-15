import logging
from typing import Any, Dict, Optional
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
        # oco_order_ids = {} # Commented out as full OCO monitoring logic is not yet implemented here
        filled = False
        main_order_id = None
        actual_fill_price = None
        category = 'linear' # Assuming USDT Perpetual

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
                    self.logger.warning(f"Adjusted order size to {size} to meet Bybit minNotional={min_notional_value}USDT (price={effective_price_for_notional_check}). New notional: {size * effective_price_for_notional_check:.2f}")
                
                current_notional = size * effective_price_for_notional_check # Recalculate notional with potentially adjusted size
                # Enforce bot's hard minimum notional value
                if current_notional < self.MIN_NOTIONAL_USDT:
                    raw_size = self.MIN_NOTIONAL_USDT / effective_price_for_notional_check
                    size = math.ceil(raw_size / qty_step) * qty_step
                    size = max(size, min_order_qty)
                    self.logger.warning(f"Adjusted order size to {size} to meet bot minimum notional {self.MIN_NOTIONAL_USDT} USDT (price={effective_price_for_notional_check}). New notional: {size * effective_price_for_notional_check:.2f}")
            
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
            
            self.logger.info(f"Placing main order: {main_order_params}")
            main_order_response = self._retry_with_backoff(self.exchange.place_order, **main_order_params)
            self.log_order_status(main_order_response)
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
                    self.logger.info(f"Fetched market order status: {order_data}")
                    
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
                    elif order_data.get('orderStatus', '').lower() == 'new' or order_data.get('orderStatus', '').lower() == 'partiallyfilled': # Bybit might still be processing
                         self.logger.info(f"Market order {main_order_id} status is {order_data.get('orderStatus')}. Assuming fill for SL/TP placement.")
                         filled = True # Proceed with SL/TP placement optimistically
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
                        break
                    if current_status == 'partiallyfilled':
                        filled = True # Partially filled is also considered for SL/TP on the filled amount
                        filled_qty = float(status_data.get('cumExecQty', 0))
                        if status_data.get('avgPrice') and float(status_data['avgPrice']) > 0 :
                           actual_fill_price = float(status_data['avgPrice'])
                        self.logger.info(f"Limit order {main_order_id} partially filled ({filled_qty}). Placing SL/TP for this amount.")
                        break
                    if current_status in ['rejected', 'cancelled', 'expired']:
                         self.logger.error(f"Limit order {main_order_id} failed with status: {current_status}")
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

            # Calculate SL/TP prices based on actual_fill_price and percentages
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

            # Place stop-loss order if SL percentage was provided
            if sl_pct is not None and stop_loss_price_calculated is not None and stop_loss_price_calculated > 0:
                stop_side = 'Sell' if side.lower() == 'buy' else 'Buy'
                sl_params = {
                    'category': category,
                    'symbol': symbol,
                    'side': stop_side,
                    'orderType': 'Market', # Bybit SL/TP orders are often Market triggered by price
                    'qty': str(filled_qty), # Use actual filled quantity
                    'reduceOnly': True,
                    'tpslMode': 'Partial', # Assuming partial SL/TP mode for now
                    'slOrderType': 'Market', # Type of order created when SL is triggered
                    'stopLoss': str(stop_loss_price_calculated)
                }
                if order_link_id: sl_params['orderLinkId'] = order_link_id + "-sl"
                
                self.logger.info(f"Placing stop-loss order: {sl_params}")
                stop_loss_response = self._retry_with_backoff(self.exchange.place_order, **sl_params)
                self.log_order_status(stop_loss_response, "Stop-loss order")
                order_responses['stop_loss_order'] = stop_loss_response
                sl_order_id = stop_loss_response.get('result', {}).get('orderId')
                if sl_order_id: self.active_orders[sl_order_id] = {'type': 'sl', 'main_order_id': main_order_id, 'symbol': symbol}
            else:
                self.logger.info(f"Stop-loss not placed (sl_pct: {sl_pct}, calculated_price: {stop_loss_price_calculated})")
                order_responses['stop_loss_order'] = None

            # Place take-profit order if TP percentage was provided
            if tp_pct is not None and take_profit_price_calculated is not None and take_profit_price_calculated > 0:
                tp_side = 'Sell' if side.lower() == 'buy' else 'Buy'
                tp_params = {
                    'category': category,
                    'symbol': symbol,
                    'side': tp_side,
                    'orderType': 'Market', # Bybit SL/TP orders are often Market triggered by price
                    'qty': str(filled_qty), # Use actual filled quantity
                    'reduceOnly': True,
                    'tpslMode': 'Partial',
                    'tpOrderType': 'Market', # Type of order created when TP is triggered
                    'takeProfit': str(take_profit_price_calculated)
                }
                if order_link_id: tp_params['orderLinkId'] = order_link_id + "-tp"

                self.logger.info(f"Placing take-profit order: {tp_params}")
                take_profit_response = self._retry_with_backoff(self.exchange.place_order, **tp_params)
                self.log_order_status(take_profit_response, "Take-profit order")
                order_responses['take_profit_order'] = take_profit_response
                tp_order_id = take_profit_response.get('result', {}).get('orderId')
                if tp_order_id: self.active_orders[tp_order_id] = {'type': 'tp', 'main_order_id': main_order_id, 'symbol': symbol}
            else:
                self.logger.info(f"Take-profit not placed (tp_pct: {tp_pct}, calculated_price: {take_profit_price_calculated})")
                order_responses['take_profit_order'] = None
            
            # Basic OCO simulation: If one SL/TP fills, cancel the other.
            # This needs a more robust monitoring loop elsewhere if orders are not linked on exchange side.
            # For Bybit, stopLoss/takeProfit params on main order or separate SL/TP orders might be handled by exchange.
            # The current placement is separate SL and TP market trigger orders.

        except OrderExecutionError as oee: # Catch specific execution errors
            self.logger.error(f"OrderExecutionError in place_order_with_risk: {oee}")
            # Ensure main_order_response is in dict even on failure for consistent return structure
            if 'main_order' not in order_responses: order_responses['main_order'] = None
            if 'stop_loss_order' not in order_responses: order_responses['stop_loss_order'] = None
            if 'take_profit_order' not in order_responses: order_responses['take_profit_order'] = None
            # Optionally re-raise or handle by returning error state in dict
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in place_order_with_risk: {e}", exc_info=True)
            if 'main_order' not in order_responses: order_responses['main_order'] = None
            if 'stop_loss_order' not in order_responses: order_responses['stop_loss_order'] = None
            if 'take_profit_order' not in order_responses: order_responses['take_profit_order'] = None
            raise OrderExecutionError(f"Unexpected error: {str(e)}") from e
        
        return order_responses

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

    def log_order_status(self, order_response: Dict[str, Any], order_type: str):
        """
        Log the status of an order.
        Args:
            order_response: The response dict from the exchange.
            order_type: The type of the order.
        """
        if not isinstance(order_response, dict):
            self.logger.error(f"Order response is not a dict: {order_response}")
            return
        order_id = order_response.get('result', {}).get('order_id') or order_response.get('order_id')
        status = order_response.get('result', {}).get('order_status') or order_response.get('order_status')
        side = order_response.get('result', {}).get('side') or order_response.get('side')
        order_type = order_response.get('result', {}).get('order_type') or order_response.get('order_type')
        qty = order_response.get('result', {}).get('qty') or order_response.get('qty')
        price = order_response.get('result', {}).get('price') or order_response.get('price')
        error_msg = order_response.get('ret_msg')
        # Remove non-ASCII characters for Windows console compatibility
        if error_msg:
            error_msg = str(error_msg).encode('ascii', errors='ignore').decode('ascii')
        ret_code = order_response.get('retCode', order_response.get('ret_code'))

        if ret_code == 0:
            self.logger.info(
                f"{order_type} order placed: id={order_id}, status={status}, side={side}, type={order_type}, size={qty}, price={price}"
            )
        else:
            self.logger.error(
                f"{order_type} order failed: id={order_id}, status={status}, side={side}, type={order_type}, size={qty}, price={price}, error={error_msg}"
            )

    def sync_active_orders_with_exchange(self, symbol: str):
        """
        Sync self.active_orders with the actual open orders on the exchange for the given symbol.
        Removes orders from self.active_orders that are no longer open on the exchange.
        Logs discrepancies for audit and debugging.
        """
        try:
            response = self.exchange.fetch_open_orders(symbol)
            open_orders = set()
            # Bybit V5 returns open orders in result.list
            order_list = response.get('result', {}).get('list', [])
            for order in order_list:
                order_id = order.get('orderId') or order.get('order_id')
                if order_id:
                    open_orders.add(order_id)
            # Remove locally tracked orders that are no longer open
            local_order_ids = set(self.active_orders.keys())
            for order_id in local_order_ids:
                if order_id not in open_orders:
                    self.logger.info(f"Order {order_id} no longer open on exchange. Removing from active_orders.")
                    self.active_orders.pop(order_id, None)
            # Optionally, log orders open on exchange but not tracked locally
            for order_id in open_orders:
                if order_id not in self.active_orders:
                    self.logger.warning(f"Order {order_id} is open on exchange but not tracked locally.")
        except Exception as exc:
            self.logger.error(f"Failed to sync active orders with exchange: {exc}") 