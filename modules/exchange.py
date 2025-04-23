import logging
import time
from typing import Any, Dict, Optional
from pybit import HTTP

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
        self.client = None
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._authenticate()

    def _authenticate(self):
        """
        Authenticate with ByBit using provided credentials via Pybit.
        """
        try:
            endpoint = "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"
            self.client = HTTP(
                endpoint,
                api_key=self.api_key,
                api_secret=self.api_secret,
                request_timeout=10
            )
            self.logger.info(f"Authenticated to ByBit ({'testnet' if self.testnet else 'mainnet'}) successfully.")
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            raise ExchangeError("Authentication failed")

    def _check_response(self, response: dict, context: str = "API call") -> dict:
        """
        Helper to check Bybit API response for errors.
        Raises ExchangeError if ret_code is not 0.
        """
        if not isinstance(response, dict):
            self.logger.error(f"{context} failed: Response is not a dict: {response}")
            raise ExchangeError(f"{context} failed: Invalid response format.")
        if response.get("ret_code", 0) != 0:
            msg = response.get("ret_msg", "Unknown error")
            self.logger.error(f"{context} failed: {msg} | Response: {response}")
            raise ExchangeError(f"{context} failed: {msg}")
        return response

    def _api_call_with_backoff(self, func, *args, **kwargs):
        retries = 0
        while True:
            try:
                response = func(*args, **kwargs)
                # Check for HTTP error codes in response if available
                if hasattr(response, 'status_code') and response.status_code in (429, 500):
                    raise ExchangeError(f"HTTP error {response.status_code}")
                return response
            except Exception as e:
                # Check for HTTP 429/500 in exception message
                err_msg = str(e)
                if ("429" in err_msg or "rate limit" in err_msg or "500" in err_msg) and retries < self.max_retries:
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
            self.logger.info(f"Order placed: {kwargs} | Response: {checked}")
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

    def fetch_positions(self) -> Dict[str, Any]:
        """
        Fetch open positions from ByBit.
        Returns:
            Positions dict
        """
        try:
            if not self.client:
                raise ExchangeError("Client not authenticated")
            response = self._api_call_with_backoff(self.client.get_positions, category="linear")
            checked = self._check_response(response, context="fetch_positions")
            self.logger.info(f"Fetched positions: {checked}")
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
            self.logger.info(f"Fetched OHLCV for {symbol} ({timeframe}): {checked}")
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
            self.logger.info(f"Fetched open orders: {open_params} | Response: {checked}")
            return checked
        except Exception as e:
            self.logger.error(f"Fetch open orders failed: {e}")
            raise ExchangeError(f"Fetch open orders failed: {str(e)}")

    def fetch_order(self, symbol: str, order_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch a single order by order_id for a given symbol.
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            order_id: The order ID to fetch
            params: Additional parameters
        Returns:
            API response dict
        """
        try:
            if not self.client:
                raise ExchangeError("Client not authenticated")
            norm_symbol = symbol.replace("/", "").upper()
            order_params = {
                "symbol": norm_symbol,
                "order_id": order_id,
                "category": "linear"
            }
            if params:
                order_params.update(params)
            response = self._api_call_with_backoff(self.client.get_order, **order_params)
            checked = self._check_response(response, context="fetch_order")
            self.logger.info(f"Fetched order: {order_params} | Response: {checked}")
            return checked
        except Exception as e:
            self.logger.error(f"Fetch order failed: {e}")
            raise ExchangeError(f"Fetch order failed: {str(e)}") 