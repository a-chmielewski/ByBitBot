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
            self.logger.error(f"Failed to cancel unfilled main order {order_id}: {cancel_exc}")
        order_responses['stop_loss_order'] = None
        order_responses['take_profit_order'] = None
        return order_responses

    def place_order_with_risk(self, symbol: str, side: str, order_type: str, size: float, price: Optional[float], stop_loss: float, take_profit: float, params: Optional[Dict[str, Any]] = None, reduce_only: bool = False, time_in_force: str = "GoodTillCancel", stop_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a main order and, after it is filled, place associated stop-loss and take-profit orders.
        Implements OCO logic: monitors SL/TP, cancels the other when one is filled.
        Args:
            symbol: Trading pair symbol.
            side: 'buy' or 'sell'.
            order_type: 'market' or 'limit'.
            size: Order size.
            price: Limit price (if applicable).
            stop_loss: Stop-loss price or percent (absolute price for now).
            take_profit: Take-profit price or percent (absolute price for now).
            params: Additional order parameters.
            reduce_only: Whether the order should be reduce-only.
            time_in_force: Time in force for the order.
            stop_price: The stop trigger price (for stop/TP orders).
        Returns:
            Dict summarizing all order responses.
        Raises:
            OrderExecutionError: If any order placement fails after retries.
        """
        order_responses = {}
        oco_order_ids = {}  # Local map for SL/TP order IDs in this call
        try:
            # Determine Bybit category (assume USDT Perp for now)
            category = 'linear'
            order_link_id = params.get('orderLinkId') if params else None

            # Enforce minimum order size and notional value
            min_order_qty, min_notional_value, qty_step = \
                self.exchange.get_min_order_amount(symbol, category=category)
            # assert min_order_qty is not None, f"min_order_qty could not be determined for {symbol}."
            if size is None:
                size = min_order_qty
            else:
                size = max(size, min_order_qty)
            # Determine effective price for notional check
            effective_price = price
            if order_type.lower() == 'market' or effective_price is None:
                # Fetch latest close price from OHLCV data
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe='1m', limit=1)
                    # Bybit V5 returns list of dicts or list of lists; handle both
                    if isinstance(ohlcv, list) and len(ohlcv) > 0:
                        last_candle = ohlcv[-1]
                        if isinstance(last_candle, dict):
                            effective_price = float(last_candle.get('close', 0))
                        elif isinstance(last_candle, list) and len(last_candle) >= 5:
                            effective_price = float(last_candle[4])
                        else:
                            effective_price = 0
                    else:
                        effective_price = 0
                except Exception as e:
                    self.logger.error(f"Failed to fetch latest price for notional check: {e}")
                    effective_price = 0
            if min_notional_value and effective_price:
                # compute the raw size needed to reach the min‐notional
                size = math.ceil(min_notional_value / effective_price / qty_step) * qty_step
                # never go below the exchange’s min order quantity
                size = max(size, min_order_qty)
                self.logger.warning(f"Adjusted order size to {size} to meet Bybit minNotional={min_notional_value}USDT (step={qty_step} {symbol}, price={effective_price}).")
            if size == min_order_qty:
                self.logger.warning(f"Requested order size was None or below Bybit minimum {min_order_qty} for {symbol}. Using minimum size {min_order_qty}.")

            # Place main order
            main_order_params = {
                'category': category,
                'symbol': symbol,
                'side': side.capitalize(),
                'orderType': order_type.capitalize(),
                'qty': str(size),
                'reduceOnly': reduce_only,
                'timeInForce': time_in_force,
            }
            # Only include price for limit orders
            if order_type.lower() == 'limit' and price is not None:
                main_order_params['price'] = str(price)
            if order_link_id:
                main_order_params['orderLinkId'] = order_link_id
            if params:
                for k, v in params.items():
                    if k not in main_order_params:
                        main_order_params[k] = v

            self.logger.info(
                f"Placing main order: {main_order_params}"
            )
            main_order_response = self._retry_with_backoff(
                self.exchange.place_order,
                **main_order_params
            )
            self.log_order_status(main_order_response)
            order_responses['main_order'] = main_order_response
            main_order_id = main_order_response.get('result', {}).get('order_id') or main_order_response.get('order_id')
            if main_order_id:
                self.active_orders[main_order_id] = {'type': 'main', 'symbol': symbol, 'side': side}

            # Wait for main order to fill before placing SL/TP
            order_id = main_order_id
            if not order_id:
                raise Exception("Main order did not return an order_id.")
            self.logger.info(f"Waiting for main order {order_id} to fill before placing SL/TP...")
            filled = False
            elapsed = 0
            while elapsed < self.FILL_TIMEOUT_SECONDS:
                order_status_resp = self._retry_with_backoff(
                    self.exchange.fetch_order,
                    symbol=symbol,
                    order_id=order_id
                )
                status = order_status_resp.get('result', {}).get('order_status') or order_status_resp.get('order_status')
                self.logger.debug(f"Order {order_id} status: {status}")
                if status in ('Filled', 'PartiallyFilled', 'filled', 'partially_filled'):
                    filled = True
                    break
                time.sleep(self.POLL_INTERVAL_SECONDS)
                elapsed += self.POLL_INTERVAL_SECONDS
            if not filled:
                return self._cancel_unfilled_main_and_return(symbol, order_id, order_responses)

            filled_qty = size
            partial_fill = False
            if filled:
                # Check for partial fill and adjust SL/TP size
                order_status_resp = self._retry_with_backoff(
                    self.exchange.fetch_order,
                    symbol=symbol,
                    order_id=order_id
                )
                filled_qty = (
                    order_status_resp.get('result', {}).get('cum_exec_qty')
                    or order_status_resp.get('cum_exec_qty')
                    or order_status_resp.get('result', {}).get('qty')
                    or order_status_resp.get('qty')
                    or size
                )
                try:
                    filled_qty = float(filled_qty)
                except Exception:
                    filled_qty = size
                partial_fill = (filled_qty != size)
                self.logger.info(f"Using filled quantity for SL/TP: {filled_qty}")

            # Place stop-loss order
            stop_side = 'Sell' if side.lower() == 'buy' else 'Buy'
            stop_loss_params = {
                'category': category,
                'symbol': symbol,
                'side': stop_side,
                'orderType': 'Market',
                'qty': str(filled_qty),
                'reduceOnly': True,
                'timeInForce': time_in_force,
                'stopLoss': str(stop_loss),
                'slOrderType': 'Market',
                'tpslMode': 'Partial' if partial_fill else 'Full',
            }
            if stop_price is not None:
                stop_loss_params['triggerPrice'] = str(stop_price)
            if order_link_id:
                stop_loss_params['orderLinkId'] = order_link_id + "-sl"
            self.logger.info(
                f"Placing stop-loss order: {stop_loss_params}"
            )
            stop_loss_response = self._retry_with_backoff(
                self.exchange.place_order,
                **stop_loss_params
            )
            self.log_order_status(stop_loss_response)
            order_responses['stop_loss_order'] = stop_loss_response
            stop_loss_id = stop_loss_response.get('result', {}).get('order_id') or stop_loss_response.get('order_id')
            if stop_loss_id:
                oco_order_ids['stop_loss'] = stop_loss_id

            # Place take-profit order
            take_profit_params = {
                'category': category,
                'symbol': symbol,
                'side': stop_side,
                'orderType': 'Market',
                'qty': str(filled_qty),
                'reduceOnly': True,
                'timeInForce': time_in_force,
                'takeProfit': str(take_profit),
                'tpOrderType': 'Market',
                'tpslMode': 'Partial' if partial_fill else 'Full',
            }
            if stop_price is not None:
                take_profit_params['triggerPrice'] = str(stop_price)
            if order_link_id:
                take_profit_params['orderLinkId'] = order_link_id + "-tp"
            self.logger.info(
                f"Placing take-profit order: {take_profit_params}"
            )
            take_profit_response = self._retry_with_backoff(
                self.exchange.place_order,
                **take_profit_params
            )
            self.log_order_status(take_profit_response)
            order_responses['take_profit_order'] = take_profit_response
            take_profit_id = take_profit_response.get('result', {}).get('order_id') or take_profit_response.get('order_id')
            if take_profit_id:
                oco_order_ids['take_profit'] = take_profit_id

            # Ensure stop_loss_id and take_profit_id are always defined before OCO loop
            stop_loss_id = oco_order_ids.get('stop_loss')
            take_profit_id = oco_order_ids.get('take_profit')

            # OCO logic: monitor both SL and TP, cancel the other when one is filled
            self.logger.info(f"Monitoring OCO: stop_loss_id={stop_loss_id}, take_profit_id={take_profit_id}")
            oco_elapsed = 0
            oco_filled = False
            while oco_elapsed < self.OCO_TIMEOUT_SECONDS and not oco_filled:
                # Check both orders
                sl_status = None
                tp_status = None
                if oco_order_ids.get('stop_loss'):
                    sl_resp = self._retry_with_backoff(
                        self.exchange.fetch_order,
                        symbol=symbol,
                        order_id=oco_order_ids['stop_loss']
                    )
                    sl_status = sl_resp.get('result', {}).get('order_status') or sl_resp.get('order_status')
                if oco_order_ids.get('take_profit'):
                    tp_resp = self._retry_with_backoff(
                        self.exchange.fetch_order,
                        symbol=symbol,
                        order_id=oco_order_ids['take_profit']
                    )
                    tp_status = tp_resp.get('result', {}).get('order_status') or tp_resp.get('order_status')
                self.logger.debug(f"OCO status: SL={sl_status}, TP={tp_status}")
                if sl_status in ('Filled', 'filled') and tp_status not in ('Filled', 'filled'):
                    # Cancel TP
                    self.logger.info(f"Stop-loss filled. Cancelling take-profit order {oco_order_ids.get('take_profit')}.")
                    if oco_order_ids.get('take_profit'):
                        self._retry_with_backoff(
                            self.exchange.cancel_order,
                            symbol=symbol,
                            order_id=oco_order_ids['take_profit']
                        )
                        oco_order_ids.pop('take_profit', None)
                    oco_filled = True
                elif tp_status in ('Filled', 'filled') and sl_status not in ('Filled', 'filled'):
                    # Cancel SL
                    self.logger.info(f"Take-profit filled. Cancelling stop-loss order {oco_order_ids.get('stop_loss')}.")
                    if oco_order_ids.get('stop_loss'):
                        self._retry_with_backoff(
                            self.exchange.cancel_order,
                            symbol=symbol,
                            order_id=oco_order_ids['stop_loss']
                        )
                        oco_order_ids.pop('stop_loss', None)
                    oco_filled = True
                elif sl_status in ('Filled', 'filled') and tp_status in ('Filled', 'filled'):
                    oco_filled = True
                else:
                    time.sleep(self.OCO_POLL_INTERVAL_SECONDS)
                    oco_elapsed += self.OCO_POLL_INTERVAL_SECONDS
            if not oco_filled:
                self.logger.warning(f"OCO monitoring timed out for stop_loss_id={stop_loss_id}, take_profit_id={take_profit_id}")
                # Attempt to cancel any remaining open SL/TP orders
                for leg, oid in list(oco_order_ids.items()):
                    if oid:
                        try:
                            self.logger.info(f"OCO timeout: Cancelling remaining {leg} order {oid}.")
                            self._retry_with_backoff(
                                self.exchange.cancel_order,
                                symbol=symbol,
                                order_id=oid
                            )
                        except Exception as cancel_exc:
                            self.logger.error(f"Failed to cancel {leg} order {oid} after OCO timeout: {cancel_exc}")
                oco_order_ids.clear()
            # No need to persist oco_order_ids after this call
            return order_responses
        except Exception as exc:
            self.logger.error(f"Order placement with risk failed: {exc}")
            raise OrderExecutionError(f"Order placement with risk failed: {exc}") from exc

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

    def log_order_status(self, order_response: Dict[str, Any]):
        """
        Log the status of an order.
        Args:
            order_response: The response dict from the exchange.
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
        ret_code = order_response.get('ret_code')

        if ret_code == 0:
            self.logger.info(
                f"Order placed: id={order_id}, status={status}, side={side}, type={order_type}, size={qty}, price={price}"
            )
        else:
            self.logger.error(
                f"Order failed: id={order_id}, status={status}, side={side}, type={order_type}, size={qty}, price={price}, error={error_msg}"
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