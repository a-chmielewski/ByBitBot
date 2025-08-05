import logging
from typing import Any, Dict, Optional
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from pybit.unified_trading import WebSocket
import threading
import time
from datetime import datetime, timedelta
import random

DEFAULT_WINDOW_SIZE = 1000

class DataFetchError(Exception):
    """Custom exception for data fetching errors."""
    pass

class LiveDataFetcher:
    """
    Manages OHLCV data for strategies, maintaining a rolling window and providing real-time updates via ByBit WebSocket.
    Fetches initial data using REST and updates using WebSocket for low-latency trading.
    
    Attributes:
        exchange: ExchangeConnector instance for REST API calls.
        symbol: Trading pair symbol (e.g., 'BTCUSDT').
        timeframe: Timeframe string (e.g., '1m').
        window_size: Number of bars to keep in the rolling window.
        logger: Logger instance for this fetcher.
    """
    def _normalize_timeframe_to_bybit_interval(self, timeframe: str) -> str:
        """
        Convert CCXT-style timeframe (e.g., '1m', '1h', '1d') to ByBit interval.
        ByBit intervals: 1, 3, 5, 15, 30, 60, 120, 240, 360, 720 (minutes), D, W, M.
        """
        tf = timeframe.strip()
        # monthly (CCXT = '1M')
        if tf.endswith('M') and tf[:-1].isdigit():
            return 'M'
        # minutes (e.g. '1m', '15m')
        if tf.endswith('m') and tf[:-1].isdigit():
            return tf[:-1]
        # hours (e.g. '1h', '4h')
        if tf.endswith(('h','H')) and tf[:-1].isdigit():
            return str(int(tf[:-1]) * 60)
        # days / weeks
        if tf.lower() == '1d':
            return 'D'
        if tf.lower() == '1w':
            return 'W'
        # bare number = minutes
        if tf.isdigit():
            return tf
        self.logger.warning(f"Unsupported timeframe '{timeframe}', sending raw to API.")

        # Default to trying to convert to int, assuming it's minutes if no suffix
        try:
            return str(int(timeframe))
        except ValueError:
            self.logger.warning(f"Unsupported timeframe format '{timeframe}'. Using as is. This may cause API errors.")
            return timeframe  # Fallback, might cause issues.

    def __init__(self, exchange, symbol: str, timeframe: str, window_size: int = DEFAULT_WINDOW_SIZE, logger: Optional[logging.Logger] = None):
        assert exchange is not None, "Exchange connector must be provided."
        assert isinstance(symbol, str) and symbol, "Symbol must be a non-empty string."
        assert isinstance(timeframe, str) and timeframe, "Timeframe must be a non-empty string."
        assert isinstance(window_size, int) and window_size > 0, "Window size must be a positive integer."
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.exchange = exchange
        self.symbol = symbol.upper().replace("/", "")
        self.timeframe_orig = timeframe  # Keep original for logging/display if needed
        self.bybit_interval = self._normalize_timeframe_to_bybit_interval(timeframe)
        self.window_size = window_size
        self.data = pd.DataFrame()
        self.ws_client = None
        self.ws_running = False
        self._ws_thread = None
        self._ws_lock = threading.Lock()
        
        # Enhanced WebSocket connection management
        self._connection_state = "disconnected"  # disconnected, connecting, connected, error
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._base_reconnect_delay = 1.0  # seconds
        self._max_reconnect_delay = 60.0  # seconds
        self._last_successful_connection = None
        self._last_data_received = None
        self._connection_timeout = 30.0  # seconds to wait for first data after connection
        self._data_timeout = 120.0  # seconds without data before considering connection stale

    def _ohlcv_to_df(self, ohlcv_raw: list) -> pd.DataFrame:
        """
        Convert ByBit OHLCV list (newest first) to a DataFrame (oldest first, correct columns).
        Args:
            ohlcv_raw: List of OHLCV lists from ByBit API.
        Returns:
            DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        if not ohlcv_raw:
            return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Reverse if list is newest-first; sorting will ensure correct order in all cases
        ohlcv_raw = ohlcv_raw[::-1]
        df = pd.DataFrame(ohlcv_raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # Ensure data is ordered from oldest to newest
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df

    def fetch_initial_data(self) -> pd.DataFrame:
        """
        Fetch initial OHLCV data to fill the rolling window.
        Returns:
            DataFrame with OHLCV data.
        Raises:
            DataFetchError: If data fetch fails or no data is returned.
        """
        try:
            response = self.exchange.fetch_ohlcv(self.symbol, self.bybit_interval, self.window_size)
            ohlcv_raw = response.get('result', {}).get('list', [])
            if not ohlcv_raw:
                raise DataFetchError("No OHLCV data returned from exchange.")
            df = self._ohlcv_to_df(ohlcv_raw)
            if len(df) > self.window_size:
                df = df.iloc[-self.window_size:]
            self.data = df.reset_index(drop=True)
            self.logger.info(f"Fetched initial OHLCV data: {len(self.data)} rows for {self.symbol} {self.timeframe_orig}")
            return self.data.copy()
        except Exception as exc:
            self.logger.error(f"Initial data fetch failed: {exc}")
            raise DataFetchError(f"Initial data fetch failed: {exc}")

    def update_data(self) -> pd.DataFrame:
        """
        Fetch the latest OHLCV bar and update the rolling window.
        Returns:
            Updated DataFrame with OHLCV data.
        Raises:
            DataFetchError: If data update fails.
        """
        try:
            response = self.exchange.fetch_ohlcv(self.symbol, self.bybit_interval, limit=2)
            ohlcv_raw = response.get('result', {}).get('list', [])
            if not ohlcv_raw:
                raise DataFetchError("No OHLCV data returned from exchange.")
            df_new = self._ohlcv_to_df(ohlcv_raw)
            with self._ws_lock:
                # Compare newest entries based on timestamp
                if self.data.empty or df_new['timestamp'].iloc[-1] > self.data['timestamp'].iloc[-1]:
                    self.data = pd.concat([self.data, df_new.iloc[[-1]]], ignore_index=True)
                elif df_new['timestamp'].iloc[-1] == self.data['timestamp'].iloc[-1]:
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        self.data.at[self.data.index[-1], col] = df_new.iloc[-1][col]
                self.remove_old_data()
            self.logger.debug(f"Updated OHLCV data: {len(self.data)} rows for {self.symbol} {self.timeframe_orig}")
            return self.data.copy()
        except Exception as exc:
            self.logger.error(f"Data update failed: {exc}")
            raise DataFetchError(f"Data update failed: {exc}")

    def get_data(self) -> pd.DataFrame:
        """
        Get the current rolling window of OHLCV data, always trimmed to window_size.
        Returns:
            DataFrame with OHLCV data.
        """
        with self._ws_lock:
            if not self.data.empty and len(self.data) > self.window_size:
                return self.data.iloc[-self.window_size:].copy().reset_index(drop=True)
            return self.data.copy().reset_index(drop=True)

    def remove_old_data(self):
        """
        Remove data outside the rolling window (keep only the most recent window_size rows).
        """
        try:
            if not self.data.empty and len(self.data) > self.window_size:
                old_len = len(self.data)
                self.data = self.data.iloc[-self.window_size:].reset_index(drop=True)
                self.logger.debug(f"Removed old data: trimmed from {old_len} to {len(self.data)} rows.")
        except Exception as exc:
            self.logger.error(f"Remove old data failed: {exc}")
            raise

    def _calculate_reconnect_delay(self) -> float:
        """
        Calculate exponential backoff delay with jitter for reconnection attempts.
        
        Returns:
            Delay in seconds before next reconnection attempt
        """
        # Exponential backoff: base_delay * (2 ^ attempt_number)
        delay = self._base_reconnect_delay * (2 ** min(self._reconnect_attempts, 6))  # Cap at 2^6 = 64x multiplier
        delay = min(delay, self._max_reconnect_delay)
        
        # Add jitter (±25% random variation) to prevent thundering herd
        jitter = delay * 0.25 * (2 * random.random() - 1)  # -25% to +25%
        delay += jitter
        
        return max(0.1, delay)  # Minimum 100ms delay

    def _is_connection_stale(self) -> bool:
        """
        Check if the WebSocket connection is stale (no data received for too long).
        
        Returns:
            True if connection should be considered stale and needs reconnection
        """
        if not self._last_data_received:
            # No data received yet - check if we've waited too long since connection
            if self._last_successful_connection:
                time_since_connection = (datetime.now() - self._last_successful_connection).total_seconds()
                return time_since_connection > self._connection_timeout
            return False
        
        # Check if too much time has passed since last data
        time_since_data = (datetime.now() - self._last_data_received).total_seconds()
        return time_since_data > self._data_timeout

    def _reset_connection_state(self):
        """Reset connection state tracking for a fresh start."""
        self._connection_state = "disconnected"
        self._reconnect_attempts = 0
        self._last_successful_connection = None
        self._last_data_received = None

    def start_websocket(self):
        """
        Start the WebSocket client with enhanced error recovery and connection management.
        Implements exponential backoff, connection health monitoring, and automatic recovery.
        Raises:
            Exception: If WebSocket fails to start initially.
        """
        try:
            if self.ws_running:
                self.logger.info("WebSocket already running.")
                return
                
            self.ws_running = True
            self._reset_connection_state()
            
            def ws_run():
                while self.ws_running:
                    try:
                        # Check if we've exceeded maximum reconnection attempts
                        if self._reconnect_attempts >= self._max_reconnect_attempts:
                            self.logger.error(f"WebSocket connection failed after {self._max_reconnect_attempts} attempts. "
                                           f"Stopping reconnection attempts for {self.symbol} {self.timeframe_orig}")
                            self._connection_state = "error"
                            break
                        
                        # Calculate and apply reconnection delay if this is a retry
                        if self._reconnect_attempts > 0:
                            delay = self._calculate_reconnect_delay()
                            self.logger.info(f"WebSocket reconnection attempt {self._reconnect_attempts + 1}/{self._max_reconnect_attempts} "
                                           f"for {self.symbol} {self.timeframe_orig} in {delay:.1f} seconds...")
                            time.sleep(delay)
                        
                        # Attempt connection
                        self._connection_state = "connecting"
                        self._reconnect_attempts += 1
                        
                        self.ws_client = WebSocket(
                            testnet=self.exchange.testnet,
                            channel_type="linear"
                        )
                        
                        self.ws_client.kline_stream(
                            symbol=self.symbol,
                            interval=self.bybit_interval,
                            callback=self.on_kline_message
                        )
                        
                        self._connection_state = "connected"
                        self._last_successful_connection = datetime.now()
                        
                        # Reset reconnect attempts on successful connection
                        if self._reconnect_attempts > 1:
                            self.logger.info(f"✅ WebSocket reconnected successfully for {self.symbol} {self.timeframe_orig} "
                                           f"after {self._reconnect_attempts} attempts")
                        else:
                            self.logger.info(f"✅ WebSocket connected for {self.symbol} {self.timeframe_orig}")
                        
                        self._reconnect_attempts = 0
                        
                        # Main connection monitoring loop
                        while self.ws_running and self._connection_state == "connected":
                            time.sleep(1)  # Check every second
                            
                            # Check for stale connection
                            if self._is_connection_stale():
                                self.logger.warning(f"WebSocket connection stale for {self.symbol} {self.timeframe_orig} "
                                                  f"(no data for {self._data_timeout}s). Triggering reconnection...")
                                self._connection_state = "error"
                                break
                        
                        # If we exit the monitoring loop but ws_running is still True, it means we detected an issue
                        if self.ws_running and self._connection_state == "error":
                            raise Exception("Connection became stale or unhealthy")
                            
                    except Exception as exc:
                        self._connection_state = "error"
                        
                        # Classify error types
                        error_type = "unknown"
                        is_recoverable = True
                        
                        if "authentication" in str(exc).lower() or "unauthorized" in str(exc).lower():
                            error_type = "authentication"
                            is_recoverable = False
                        elif "invalid symbol" in str(exc).lower() or "not found" in str(exc).lower():
                            error_type = "invalid_symbol"
                            is_recoverable = False
                        elif "network" in str(exc).lower() or "connection" in str(exc).lower() or "timeout" in str(exc).lower():
                            error_type = "network"
                            is_recoverable = True
                        elif "rate limit" in str(exc).lower() or "too many" in str(exc).lower():
                            error_type = "rate_limit"
                            is_recoverable = True
                        
                        if not is_recoverable:
                            self.logger.error(f"WebSocket unrecoverable error ({error_type}) for {self.symbol} {self.timeframe_orig}: {exc}")
                            break
                        else:
                            self.logger.warning(f"WebSocket recoverable error ({error_type}) for {self.symbol} {self.timeframe_orig}: {exc}")
                        
                        # Don't retry if we're stopping
                        if not self.ws_running:
                            break
                            
                    finally:
                        # Clean up current connection
                        if self.ws_client:
                            try:
                                self.ws_client.exit()
                            except Exception:
                                pass  # Ignore cleanup errors
                            self.ws_client = None
                
                # Final state update
                if self._connection_state != "error":
                    self._connection_state = "disconnected"
                    
                self.logger.info(f"WebSocket thread ended for {self.symbol} {self.timeframe_orig}")
            
            self._ws_thread = threading.Thread(target=ws_run, daemon=True)
            self._ws_thread.start()
            
        except Exception as exc:
            self.logger.error(f"Failed to start WebSocket: {exc}")
            self._connection_state = "error"
            raise

    def stop_websocket(self):
        """
        Stop the WebSocket client and unsubscribe from k-line updates.
        Ensures the thread's run loop checks ws_running frequently for graceful shutdown.
        Raises:
            Exception: If WebSocket fails to stop.
        """
        try:
            self.ws_running = False
            if self.ws_client:
                self.ws_client.exit()
                self.ws_client = None
            if self._ws_thread and self._ws_thread.is_alive():
                self._ws_thread.join(timeout=2)
            self.logger.info(f"WebSocket stopped for {self.symbol} {self.timeframe_orig}")
        except Exception as exc:
            self.logger.error(f"Failed to stop WebSocket: {exc}")
            raise

    def on_kline_message(self, message: Dict[str, Any]):
        """
        Handle incoming k-line message and update the rolling window.
        Updates connection health tracking.
        Args:
            message: The k-line message dict from Pybit WebSocket.
        """
        try:
            # Update connection health tracking
            self._last_data_received = datetime.now()
            
            kline = message.get('data')
            data_list = message.get('data') or []
            if not isinstance(data_list, list) or not data_list:
                return
            kline = data_list[-1]
            row = {
                'timestamp': pd.to_datetime(kline['start'], unit='ms'),
                'open': float(kline['open']),
                'high': float(kline['high']),
                'low': float(kline['low']),
                'close': float(kline['close']),
                'volume': float(kline['volume'])
            }
            with self._ws_lock:
                if self.data.empty or row['timestamp'] > self.data['timestamp'].iloc[-1]:
                    self.data = pd.concat([self.data, pd.DataFrame([row])], ignore_index=True)
                elif row['timestamp'] == self.data['timestamp'].iloc[-1]:
                    self.data.iloc[-1] = list(row.values())
                self.remove_old_data()
            self.logger.debug(f"WebSocket kline update: {row['timestamp']} {self.symbol} {self.timeframe_orig}")
        except Exception as exc:
            self.logger.error(f"Failed to process kline message for {self.symbol} {self.timeframe_orig}: {exc}")
            # Don't update _last_data_received on processing errors

    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current WebSocket connection status and health information.
        
        Returns:
            Dictionary containing connection state, health metrics, and diagnostics
        """
        now = datetime.now()
        
        # Calculate time since last data
        time_since_data = None
        if self._last_data_received:
            time_since_data = (now - self._last_data_received).total_seconds()
        
        # Calculate time since connection
        time_since_connection = None
        if self._last_successful_connection:
            time_since_connection = (now - self._last_successful_connection).total_seconds()
        
        # Determine overall health
        is_healthy = (
            self.ws_running and 
            self._connection_state == "connected" and
            not self._is_connection_stale()
        )
        
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe_orig,
            'ws_running': self.ws_running,
            'connection_state': self._connection_state,
            'is_healthy': is_healthy,
            'reconnect_attempts': self._reconnect_attempts,
            'max_reconnect_attempts': self._max_reconnect_attempts,
            'time_since_last_data': time_since_data,
            'time_since_connection': time_since_connection,
            'is_connection_stale': self._is_connection_stale(),
            'data_rows': len(self.data) if not self.data.empty else 0,
            'connection_timeout': self._connection_timeout,
            'data_timeout': self._data_timeout
        }
