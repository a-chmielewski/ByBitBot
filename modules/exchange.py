import logging
import time
from typing import Any, Dict, Optional
from pybit.unified_trading import HTTP

class ExchangeError(Exception):
    pass

class ExchangeConnector:
    """
    Abstracts ByBit exchange connection and API calls using Pybit.
    Handles authentication, order placement, balance/position queries, and market data retrieval.
    """
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False, logger: Optional[logging.Logger] = None, max_retries: int = 3, backoff_base: float = 1.0):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._instrument_info_cache = {} # Cache for instrument details
        self._time_offset = 0  # Track time difference with server
        self.client = None
        
        # Initialize with a larger recv_window
        self._initialize_client()
        # Initial time sync after client is initialized
        self._sync_time()

    def _initialize_client(self):
        """Initialize the HTTP client with proper configuration."""
        try:
            self.client = HTTP(
                testnet=self.testnet,
                api_key=self.api_key,
                api_secret=self.api_secret,
                recv_window=30000  # Increased to 30 seconds
            )
            self.logger.info(f"Initialized ByBit client ({'testnet' if self.testnet else 'mainnet'})")
        except Exception as e:
            self.logger.error(f"Client initialization failed: {e}")
            raise ExchangeError("Client initialization failed")

    def _sync_time(self):
        """Synchronize local time with ByBit v5 server time (with full fallbacks)."""
        if not self.client:
            self.logger.error("Cannot sync time: Client not initialized")
            return

        try:
            resp = self.client.get_server_time()
            if not isinstance(resp, dict):
                raise ValueError(f"Unexpected response type: {resp!r}")

            # 1) Try top-level 'time' (ms)
            server_ms = resp.get("time")

            # 2) Fallback to result.timeNano (ns → ms)
            if not server_ms:
                result = resp.get("result", {})
                time_nano = result.get("timeNano")
                if time_nano:
                    server_ms = int(int(time_nano) / 1_000_000)

            # 3) Fallback to result.timeSecond (s → ms)
            if not server_ms:
                result = resp.get("result", {})
                time_sec = result.get("timeSecond")
                if time_sec:
                    server_ms = int(float(time_sec) * 1000)

            if not server_ms:
                self.logger.error(f"Unable to parse server time from: {resp}")
                return

            local_ms = int(time.time() * 1000)
            # Offset in seconds
            self._time_offset = (server_ms - local_ms) / 1000.0

            self.logger.info(
                f"Time synchronized with ByBit server. Offset: {self._time_offset:.3f}s"
            )
            self.logger.debug(
                f"  server_ms={server_ms}  local_ms={local_ms}"
            )

        except Exception as e:
            self.logger.error(f"Failed to sync time with ByBit server: {e}")
            # Don't raise so your bot can still run, but your next _api_call will retry

    def _get_adjusted_timestamp(self) -> int:
        """Get timestamp adjusted for server time offset."""
        return int((time.time() + self._time_offset) * 1000)

    def _check_response(self, response: dict, context: str = "API call") -> dict:
        """
        Helper to check Bybit API response for errors.
        Raises ExchangeError if ret_code or retCode is not 0.
        """
        if not isinstance(response, dict):
            self.logger.error(f"{context} failed: Response is not a dict: {response}")
            raise ExchangeError(f"{context} failed: Invalid response format.")

        # Prioritize retCode (Unified API)
        api_ret_code = response.get("retCode")
        api_ret_msg = response.get("retMsg")

        if api_ret_code is not None:  # retCode is present
            if api_ret_code != 0:
                msg_to_log = api_ret_msg or "Unified API Error (no specific message)"
                self.logger.error(f"{context} failed (retCode={api_ret_code}): {msg_to_log} | Response: {response}")
                raise ExchangeError(f"{context} failed: {msg_to_log} (retCode: {api_ret_code})")
            # If api_ret_code is 0, it's success, proceed to return response
        else:  # retCode is NOT present, try ret_code (Older API / other endpoints)
            api_ret_code = response.get("ret_code")
            api_ret_msg = response.get("ret_msg")
            if api_ret_code is not None:  # ret_code is present
                if api_ret_code != 0:
                    msg_to_log = api_ret_msg or "Legacy API Error (no specific message)"
                    self.logger.error(f"{context} failed (ret_code={api_ret_code}): {msg_to_log} | Response: {response}")
                    raise ExchangeError(f"{context} failed: {msg_to_log} (ret_code: {api_ret_code})")
                # If api_ret_code (from ret_code field) is 0, it's success, proceed to return response
            else:
                # Neither retCode nor ret_code is present.
                # This could be a successful response from an endpoint that doesn't use this convention,
                # or a malformed response. For now, log and pass through.
                self.logger.debug(f"{context}: No standard retCode/ret_code found. Assuming success or non-standard response. Response: {response}")

        return response

    def _api_call_with_backoff(self, func, *args, **kwargs):
        retries = 0
        while True:
            try:
                # Add timestamp to kwargs if not present
                if 'timestamp' not in kwargs:
                    kwargs['timestamp'] = self._get_adjusted_timestamp()
                
                response = func(*args, **kwargs)
                
                # Check for HTTP error codes in response if available
                if hasattr(response, 'status_code') and response.status_code in (429, 500):
                    raise ExchangeError(f"HTTP error {response.status_code}")
                    
                return response
                
            except Exception as e:
                err_msg = str(e)
                
                # Handle timestamp errors
                if "timestamp" in err_msg.lower() and retries < self.max_retries:
                    self.logger.warning(f"Timestamp error detected. Resyncing time with server...")
                    self._sync_time()  # Resync time
                    wait = self.backoff_base * (2 ** retries)
                    self.logger.warning(f"Retrying in {wait:.2f}s (attempt {retries+1}/{self.max_retries})...")
                    time.sleep(wait)
                    retries += 1
                    continue
                    
                # Handle rate limits and server errors
                elif ("429" in err_msg or "rate limit" in err_msg or "500" in err_msg) and retries < self.max_retries:
                    wait = self.backoff_base * (2 ** retries)
                    self.logger.warning(f"Rate limit/server error encountered. Retrying in {wait:.2f}s (attempt {retries+1}/{self.max_retries})...")
                    time.sleep(wait)
                    retries += 1
                    continue
                    
                self.logger.error(f"API call failed after {retries} retries: {e}")
                raise

    def place_order(self, **kwargs) -> Dict[str, Any]:
        '''
        Place an order on ByBit, forwarding all Bybit API parameters as-is.
        Args:
            **kwargs: All Bybit API order parameters (camelCase expected).
        Returns:
            Order response dict
        '''
        try:
            if not self.client:
                raise ExchangeError("Client not authenticated")
            response = self._api_call_with_backoff(self.client.place_order, **kwargs)
            checked = self._check_response(response, context="place_order")
            self.logger.info(f"Order placed")
            self.logger.debug(f"Order placed: {kwargs} | Response: {checked}")
            return checked
        except Exception as e:
            self.logger.error(f"Order placement failed: {e}")
            raise ExchangeError(f"Order placement failed: {str(e)}")

    def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balances from ByBit.
        Returns:
            Balance dict
        """
        try:
            if not self.client:
                raise ExchangeError("Client not authenticated")
            response = self._api_call_with_backoff(self.client.get_wallet_balance, coin="USDT")
            checked = self._check_response(response, context="fetch_balance")
            self.logger.info(f"Fetched balance: {checked}")
            return checked
        except Exception as e:
            self.logger.error(f"Fetch balance failed: {e}")
            raise ExchangeError(f"Fetch balance failed: {str(e)}")

    def fetch_positions(self, symbol: str, category: str = "linear") -> Dict[str, Any]:
        """
        Fetch open positions for a symbol from ByBit.
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            category: Bybit category (default 'linear')
        Returns:
            Positions dict
        """
        try:
            if not self.client:
                raise ExchangeError("Client not authenticated")
            norm_symbol = symbol.replace("/", "").upper()
            response = self._api_call_with_backoff(self.client.get_positions, category=category, symbol=norm_symbol)
            checked = self._check_response(response, context="fetch_positions")
            self.logger.debug(f"Fetched positions for {norm_symbol}: {checked}")
            return checked
        except Exception as e:
            self.logger.error(f"Fetch positions failed: {e}")
            raise ExchangeError(f"Fetch positions failed: {str(e)}")

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 1000) -> Any:
        """
        Fetch OHLCV market data for a symbol.
        Args:
            symbol: Trading pair
            timeframe: Data timeframe (e.g., '1m', '5m')
            limit: Number of data points
        Returns:
            OHLCV data
        """
        try:
            if not self.client:
                raise ExchangeError("Client not authenticated")
            norm_symbol = symbol.replace("/", "").upper()
            response = self._api_call_with_backoff(
                self.client.get_kline,
                category="linear",
                symbol=norm_symbol,
                interval=timeframe,
                limit=limit
            )
            checked = self._check_response(response, context=f"fetch_ohlcv {symbol} {timeframe}")
            self.logger.debug(f"Fetched OHLCV for {symbol} ({timeframe}): {checked}")
            return checked
        except Exception as e:
            self.logger.error(f"Fetch OHLCV failed: {e}")
            raise ExchangeError(f"Fetch OHLCV failed: {str(e)}")

    def cancel_order(self, symbol: str, order_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Cancel an order by order_id for a given symbol.
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            order_id: The order ID to cancel
            params: Additional parameters
        Returns:
            API response dict
        """
        try:
            if not self.client:
                raise ExchangeError("Client not authenticated")
            norm_symbol = symbol.replace("/", "").upper()
            cancel_params = {
                "symbol": norm_symbol,
                "order_id": order_id,
                "category": "linear"
            }
            if params:
                cancel_params.update(params)
            response = self._api_call_with_backoff(self.client.cancel_order, **cancel_params)
            checked = self._check_response(response, context="cancel_order")
            self.logger.info(f"Order cancelled: {cancel_params} | Response: {checked}")
            return checked
        except Exception as e:
            self.logger.error(f"Cancel order failed: {e}")
            raise ExchangeError(f"Cancel order failed: {str(e)}")

    def fetch_open_orders(self, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch all open orders for a given symbol.
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            params: Additional parameters
        Returns:
            API response dict
        """
        try:
            if not self.client:
                raise ExchangeError("Client not authenticated")
            norm_symbol = symbol.replace("/", "").upper()
            open_params = {
                "symbol": norm_symbol,
                "category": "linear"
            }
            if params:
                open_params.update(params)
            response = self._api_call_with_backoff(self.client.get_open_orders, **open_params)
            checked = self._check_response(response, context="fetch_open_orders")
            self.logger.debug(f"Fetched open orders: {open_params} | Response: {checked}")
            return checked
        except Exception as e:
            self.logger.error(f"Fetch open orders failed: {e}")
            raise ExchangeError(f"Fetch open orders failed: {str(e)}")

    def fetch_order(self, symbol: str, order_id: str, category: str = "linear", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch a single order by order_id for a given symbol and category.
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            order_id: The order ID to fetch.
            category: The category of the product (e.g., 'linear', 'spot'). Defaults to 'linear'.
            params: Optional additional parameters for the API call (e.g., {'orderLinkId': '...'}).
                    This should not include 'symbol', 'orderId', or 'category' as they are handled directly.
        Returns:
            API response dict for the found order.
        Raises:
            ExchangeError if the order is not found or an API error occurs.
        """
        if not self.client:
            raise ExchangeError("Client not authenticated")
        
        norm_symbol = symbol.replace("/", "").upper()
        
        # Base parameters for pybit call
        base_call_params = {
            "category": category,
            "symbol": norm_symbol,
            "orderId": order_id
        }
        
        # Merge additional unique params, ensuring no overlap with base_call_params keys
        if params:
            for key, value in params.items():
                if key not in base_call_params or base_call_params[key] is None:
                    base_call_params[key] = value
                elif base_call_params[key] != value:
                    raise ExchangeError(f"Parameter conflict in fetch_order: Cannot override primary param '{key}' (value: '{base_call_params[key]}') with different value '{value}' from params dictionary.")

        # First try get_order_history since it accepts orderId
        try:
            self.logger.debug(f"Attempting to fetch order {order_id} using get_order_history with params: {base_call_params}")
            response = self._api_call_with_backoff(self.client.get_order_history, **base_call_params)
            checked_response = self._check_response(response, context="fetch_order via get_order_history")
            
            order_list = checked_response.get("result", {}).get("list", [])
            if order_list:
                found_order_data = order_list[0]
                if found_order_data.get('orderId') == order_id:
                    self.logger.info(f"Order {order_id} found in order history.")
                    self.logger.debug(f"Order {order_id} found in order history. Data: {found_order_data}")
                    return {
                        "retCode": checked_response.get("retCode", 0),
                        "retMsg": checked_response.get("retMsg", "OK"),
                        "result": found_order_data,
                        "retExtInfo": checked_response.get("retExtInfo", {}),
                        "time": checked_response.get("time", time.time() * 1000)
                    }
        except Exception as e:
            self.logger.debug(f"Order {order_id} not found in history: {e}")

        # If not found in history, try get_open_orders without orderId and filter client-side
        try:
            open_params = {k: v for k, v in base_call_params.items() if k != 'orderId'}
            self.logger.debug(f"Attempting to fetch order {order_id} from open orders with params: {open_params}")
            response = self._api_call_with_backoff(self.client.get_open_orders, **open_params)
            checked_response = self._check_response(response, context="fetch_order via get_open_orders")
            
            order_list = checked_response.get("result", {}).get("list", [])
            for order in order_list:
                if order.get('orderId') == order_id:
                    self.logger.info(f"Order {order_id} found in open orders. Data: {order}")
                    return {
                        "retCode": checked_response.get("retCode", 0),
                        "retMsg": checked_response.get("retMsg", "OK"),
                        "result": order,
                        "retExtInfo": checked_response.get("retExtInfo", {}),
                        "time": checked_response.get("time", time.time() * 1000)
                    }
        except Exception as e:
            self.logger.debug(f"Order {order_id} not found in open orders: {e}")

        self.logger.error(f"Order {order_id} for symbol {symbol} (category {category}) not found after checking history and open orders.")
        raise ExchangeError(f"Order {order_id} not found after checking history and open orders.")

    def _fetch_instrument_info(self, symbol: str, category: str = 'linear'):
        norm_symbol = symbol.replace("/", "").upper()
        cache_key = f"{norm_symbol}_{category}"
        if cache_key in self._instrument_info_cache:
            return self._instrument_info_cache[cache_key]
        
        try:
            response = self._api_call_with_backoff(
                self.client.get_instruments_info,
                category=category,
                symbol=norm_symbol
            )
            checked = self._check_response(response, context=f"fetch_instrument_info for {norm_symbol}")
            instruments = checked.get('result', {}).get('list', [])
            if instruments:
                self._instrument_info_cache[cache_key] = instruments[0] # Assuming symbol is unique
                return instruments[0]
            else:
                self.logger.error(f"No instrument info found for {norm_symbol} in category {category}.")
                return None
        except Exception as e:
            self.logger.error(f"Failed to fetch instrument info for {norm_symbol}: {e}")
            return None

    def get_price_precision(self, symbol: str, category: str = 'linear') -> int:
        info = self._fetch_instrument_info(symbol, category)
        if info and info.get('priceFilter') and info['priceFilter'].get('tickSize'):
            tick_size_str = str(info['priceFilter']['tickSize'])
            if '.' in tick_size_str:
                return len(tick_size_str.split('.')[1])
            return 0 # No decimal places if integer or not found
        self.logger.warning(f"Could not determine price precision for {symbol}, defaulting to 8.")
        return 8 # Default precision

    def get_qty_precision(self, symbol: str, category: str = 'linear') -> int:
        info = self._fetch_instrument_info(symbol, category)
        if info and info.get('lotSizeFilter') and info['lotSizeFilter'].get('qtyStep'):
            qty_step_str = str(info['lotSizeFilter']['qtyStep'])
            if '.' in qty_step_str:
                return len(qty_step_str.split('.')[1])
            return 0 # No decimal places
        self.logger.warning(f"Could not determine qty precision for {symbol}, defaulting to 3.")
        return 3 # Default precision

    def get_min_order_amount(self, symbol: str, category: str = 'linear') -> tuple:
        """
        Fetch the minimum order amount and minimum notional value for a given symbol from Bybit.
        Caches the result for efficiency.
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            category: Bybit category ('linear' for perps, 'spot' for spot)
        Returns:
            Tuple: (Minimum order quantity as float, Minimum notional value as float, Quantity step as float)
        Raises:
            ExchangeError if fetch fails or info is missing
        """
        if not hasattr(self, '_min_order_cache'):
            self._min_order_cache = {}
        cache_key = (symbol, category)
        if cache_key in self._min_order_cache:
            return self._min_order_cache[cache_key]
        try:
            response = self.client.get_instruments_info(
                category=category,
                symbol=symbol
            )
            checked = self._check_response(response, context="get_instruments_info")
            instruments = checked["result"]["list"]
            assert instruments, f"No instrument info found for {symbol}"
            instrument_info = instruments[0]
            lot_filter = instrument_info["lotSizeFilter"]
            min_order_qty = float(lot_filter["minOrderQty"])
            min_notional = float(lot_filter.get("minOrderAmt", 0))
            qty_step = float(lot_filter.get("qtyStep", min_order_qty))
            self._min_order_cache[cache_key] = (min_order_qty, min_notional, qty_step)
            return min_order_qty, min_notional, qty_step
        except Exception as error:
            # Remove non-ASCII characters from error message for Windows console compatibility
            safe_error = str(error).encode('ascii', errors='ignore').decode('ascii')
            self.logger.error(f"Error fetching minimum order amount for {symbol}: {safe_error}")
            raise ExchangeError(f"Error fetching minimum order amount for {symbol}: {safe_error}") 