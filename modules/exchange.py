import logging
import time
from typing import Any, Dict, Optional, List
from pybit.unified_trading import HTTP
from requests.exceptions import ReadTimeout, RequestException, ConnectionError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from datetime import datetime, timedelta
import random
from collections import deque

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
        
        # Enhanced API health monitoring and circuit breaker
        self._api_health_window = 300  # 5 minutes
        self._api_calls_history = deque(maxlen=1000)  # Track last 1000 API calls
        self._error_rates = {
            'connection_errors': deque(maxlen=100),
            'rate_limit_errors': deque(maxlen=100),
            'server_errors': deque(maxlen=100),
            'timeout_errors': deque(maxlen=100)
        }
        
        # Circuit breaker state
        self._circuit_state = "closed"  # closed, half_open, open
        self._circuit_failure_count = 0
        self._circuit_failure_threshold = 5  # failures before opening circuit
        self._circuit_recovery_timeout = 60  # seconds to wait before half_open
        self._circuit_open_time = None
        self._circuit_success_threshold = 3  # successes needed to close from half_open
        self._circuit_success_count = 0
        
        # Rate limiting and health tracking
        self._last_api_call_time = 0
        self._min_api_call_interval = 0.05  # 50ms minimum between calls
        self._rate_limit_detected_until = None  # Timestamp until when to slow down
        
        # Initialize with a larger recv_window
        self._initialize_client()
        # Initial time sync after client is initialized
        self._sync_time()

    def _initialize_client(self):
        """Initialize the HTTP client with proper configuration."""
        try:
            # Use a much larger recv_window to handle timestamp sync issues
            # ByBit allows up to 60000ms (60 seconds)
            self.client = HTTP(
                testnet=self.testnet,
                api_key=self.api_key,
                api_secret=self.api_secret,
                recv_window=60000,  # Maximum allowed: 60 seconds
                timeout=60  # Increase client timeout as well
            )
            self.logger.info(f"Initialized ByBit client with 60s recv_window ({'testnet' if self.testnet else 'mainnet'})")
        except Exception as e:
            self.logger.error(f"Client initialization failed: {e}")
            raise ExchangeError("Client initialization failed")

    def _sync_time(self):
        """Synchronize local time with ByBit v5 server time (with full fallbacks)."""
        if not self.client:
            self.logger.error("Cannot sync time: Client not initialized")
            return

        try:
            # Perform multiple time sync attempts for better accuracy
            offsets = []
            for attempt in range(3):
                start_local = time.time()
                resp = self.client.get_server_time()
                end_local = time.time()
                
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
                    continue

                # Account for network latency by using the midpoint of the request
                midpoint_local = (start_local + end_local) / 2
                local_ms = int(midpoint_local * 1000)
                
                # Calculate offset and add to list
                offset = (server_ms - local_ms) / 1000.0
                offsets.append(offset)
                
                # Small delay between attempts
                if attempt < 2:
                    time.sleep(0.1)
            
            if offsets:
                # Use median offset for better accuracy
                offsets.sort()
                if len(offsets) % 2 == 0:
                    self._time_offset = (offsets[len(offsets)//2-1] + offsets[len(offsets)//2]) / 2
                else:
                    self._time_offset = offsets[len(offsets)//2]
                
                self.logger.info(
                    f"Time synchronized with ByBit server. Offset: {self._time_offset:.3f}s (median of {len(offsets)} attempts)"
                )
                self.logger.debug(f"  All offsets: {[f'{o:.3f}s' for o in offsets]}")
            else:
                self.logger.error("Failed to get any valid time sync responses")

        except Exception as e:
            self.logger.error(f"Failed to sync time with ByBit server: {e}")
            # Don't raise so your bot can still run, but your next _api_call will retry

    def _get_adjusted_timestamp(self) -> int:
        """Get timestamp adjusted for server time offset with conservative approach."""
        current_time = time.time()
        
        # Apply offset with additional conservative adjustment for network latency
        # Subtract a small buffer to ensure we're not ahead of server time
        conservative_offset = self._time_offset - 0.5  # Subtract 500ms buffer
        adjusted_time = current_time + conservative_offset
        timestamp_ms = int(adjusted_time * 1000)
        
        # Safeguard: ensure timestamp is reasonable (not in far future or past)
        current_ms = int(current_time * 1000)
        max_offset_ms = 300000  # 5 minutes in milliseconds
        
        if abs(timestamp_ms - current_ms) > max_offset_ms:
            self.logger.warning(f"Adjusted timestamp {timestamp_ms} seems unreasonable. Using current time with small buffer.")
            self.logger.debug(f"  current_ms={current_ms}, offset={self._time_offset}, adjusted={timestamp_ms}")
            # Use current time minus small buffer as fallback
            timestamp_ms = current_ms - 1000  # 1 second behind current time
            
        self.logger.debug(f"Generated timestamp: {timestamp_ms} (offset: {self._time_offset:.3f}s, conservative_offset: {conservative_offset:.3f}s)")
        return timestamp_ms

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

    def _record_api_call(self, method_name: str, success: bool, error_type: str = None, response_time: float = None):
        """Record API call result for health monitoring and circuit breaker logic."""
        now = datetime.now()
        
        call_record = {
            'timestamp': now,
            'method': method_name,
            'success': success,
            'error_type': error_type,
            'response_time': response_time
        }
        
        self._api_calls_history.append(call_record)
        
        # Record specific error types
        if not success and error_type:
            if error_type in self._error_rates:
                self._error_rates[error_type].append(now)
        
        # Update circuit breaker state
        if success:
            self._circuit_success_count += 1
            if self._circuit_state == "half_open" and self._circuit_success_count >= self._circuit_success_threshold:
                self._circuit_state = "closed"
                self._circuit_failure_count = 0
                self._circuit_success_count = 0
                self.logger.info("Circuit breaker closed - API calls restored to normal")
        else:
            self._circuit_failure_count += 1
            self._circuit_success_count = 0
            
            if self._circuit_state == "closed" and self._circuit_failure_count >= self._circuit_failure_threshold:
                self._circuit_state = "open"
                self._circuit_open_time = now
                self.logger.error(f"Circuit breaker opened - API calls will be limited for {self._circuit_recovery_timeout}s")

    def _check_circuit_breaker(self, method_name: str) -> bool:
        """Check if circuit breaker allows the API call. Returns True if call should proceed."""
        now = datetime.now()
        
        if self._circuit_state == "closed":
            return True
        elif self._circuit_state == "open":
            # Check if recovery timeout has elapsed
            if self._circuit_open_time and (now - self._circuit_open_time).total_seconds() >= self._circuit_recovery_timeout:
                self._circuit_state = "half_open"
                self._circuit_success_count = 0
                self.logger.info("Circuit breaker entering half-open state - testing API calls")
                return True
            else:
                # Circuit is open, reject non-critical calls
                critical_methods = {'get_positions', 'cancel_order', 'get_wallet_balance'}
                if method_name not in critical_methods:
                    self.logger.warning(f"Circuit breaker open - rejecting non-critical API call: {method_name}")
                    return False
                else:
                    self.logger.warning(f"Circuit breaker open - allowing critical API call: {method_name}")
                    return True
        elif self._circuit_state == "half_open":
            # Allow calls but monitor closely
            return True
        
        return True

    def _calculate_backoff_with_jitter(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter to prevent thundering herd."""
        # Base exponential backoff
        delay = self.backoff_base * (2 ** min(attempt, 6))  # Cap at 2^6 = 64x multiplier
        
        # Add jitter (±25% random variation)
        jitter = delay * 0.25 * (2 * random.random() - 1)  # -25% to +25%
        delay += jitter
        
        # Additional delay if rate limiting detected
        if self._rate_limit_detected_until and datetime.now() < self._rate_limit_detected_until:
            delay += 2.0  # Add 2 seconds if we're in rate limit cooldown
        
        return max(0.1, delay)  # Minimum 100ms delay

    def _apply_rate_limiting(self):
        """Apply rate limiting between API calls."""
        now = time.time()
        time_since_last_call = now - self._last_api_call_time
        
        if time_since_last_call < self._min_api_call_interval:
            sleep_time = self._min_api_call_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self._last_api_call_time = time.time()

    def get_api_health_status(self) -> Dict[str, Any]:
        """Get current API health status and circuit breaker state."""
        now = datetime.now()
        cutoff_time = now - timedelta(seconds=self._api_health_window)
        
        # Calculate recent error rates
        recent_calls = [call for call in self._api_calls_history if call['timestamp'] > cutoff_time]
        total_recent_calls = len(recent_calls)
        failed_recent_calls = len([call for call in recent_calls if not call['success']])
        
        error_rates = {}
        for error_type, error_deque in self._error_rates.items():
            recent_errors = len([ts for ts in error_deque if ts > cutoff_time])
            error_rates[error_type] = recent_errors
        
        # Calculate average response time
        response_times = [call.get('response_time') for call in recent_calls if call.get('response_time')]
        avg_response_time = sum(response_times) / len(response_times) if response_times else None
        
        return {
            'circuit_state': self._circuit_state,
            'circuit_failure_count': self._circuit_failure_count,
            'total_api_calls': len(self._api_calls_history),
            'recent_calls': total_recent_calls,
            'recent_failures': failed_recent_calls,
            'success_rate': (total_recent_calls - failed_recent_calls) / total_recent_calls if total_recent_calls > 0 else 1.0,
            'error_rates': error_rates,
            'average_response_time': avg_response_time,
            'rate_limit_active': self._rate_limit_detected_until is not None and datetime.now() < self._rate_limit_detected_until,
            'time_offset': self._time_offset
        }

    def reset_circuit_breaker(self, reason: str = "Manual reset"):
        """
        Manually reset the circuit breaker to closed state.
        
        Args:
            reason: Reason for the reset (for logging)
        """
        old_state = self._circuit_state
        self._circuit_state = "closed"
        self._circuit_failure_count = 0
        self._circuit_success_count = 0
        self._circuit_open_time = None
        
        self.logger.info(f"Circuit breaker manually reset from '{old_state}' to 'closed'. Reason: {reason}")

    def force_time_resync(self):
        """Force a time resynchronization with the ByBit server."""
        self.logger.info("Forcing time resynchronization with ByBit server...")
        old_offset = self._time_offset
        self._sync_time()
        new_offset = self._time_offset
        
        if abs(new_offset - old_offset) > 1.0:  # More than 1 second difference
            self.logger.warning(f"Significant time offset change detected: {old_offset:.3f}s -> {new_offset:.3f}s")
        else:
            self.logger.info(f"Time resynchronization completed: offset = {new_offset:.3f}s")

    def get_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive diagnostics including API health, circuit breaker state, 
        connection status, and recent error patterns.
        """
        api_health = self.get_api_health_status()
        now = datetime.now()
        
        # Analyze recent errors for patterns
        recent_errors = {}
        for error_type, error_times in self._error_rates.items():
            recent_errors[error_type] = len([ts for ts in error_times if (now - ts).total_seconds() < 300])
        
        # Calculate uptime
        if hasattr(self, '_start_time'):
            uptime_seconds = (now - self._start_time).total_seconds()
        else:
            uptime_seconds = None
        
        return {
            'api_health': api_health,
            'client_info': {
                'testnet': self.testnet,
                'recv_window': 60000,
                'uptime_seconds': uptime_seconds
            },
            'error_analysis': {
                'recent_errors_5min': recent_errors,
                'circuit_breaker_trips': getattr(self, '_circuit_trips_history', []),
                'total_api_calls': len(self._api_calls_history)
            },
            'performance_metrics': {
                'avg_response_time': api_health.get('average_response_time'),
                'success_rate': api_health.get('success_rate'),
                'rate_limit_incidents': len([ts for ts in self._error_rates.get('rate_limit_errors', []) 
                                           if (now - ts).total_seconds() < 3600])  # Last hour
            },
            'recommendations': self._generate_health_recommendations(api_health, recent_errors)
        }

    def _generate_health_recommendations(self, api_health: Dict, recent_errors: Dict) -> List[str]:
        """Generate health recommendations based on current status."""
        recommendations = []
        
        if api_health.get('success_rate', 1.0) < 0.8:
            recommendations.append("Low API success rate - consider reducing request frequency")
        
        if api_health.get('circuit_state') == 'open':
            recommendations.append("Circuit breaker is open - check network connectivity and ByBit API status")
        
        if recent_errors.get('rate_limit_errors', 0) > 3:
            recommendations.append("Frequent rate limiting detected - implement longer delays between requests")
        
        if recent_errors.get('timeout_errors', 0) > 5:
            recommendations.append("High timeout rate - check network stability and consider increasing timeout values")
        
        if api_health.get('average_response_time', 0) > 2.0:
            recommendations.append("High API response times - monitor ByBit server status")
        
        if not recommendations:
            recommendations.append("API health is good - no issues detected")
        
        return recommendations

    def _api_call_with_backoff(self, func, *args, **kwargs):
        method_name = getattr(func, '__name__', str(func))
        start_time = time.time()
        retries = 0
        
        # Check circuit breaker
        if not self._check_circuit_breaker(method_name):
            raise ExchangeError(f"Circuit breaker open - API call {method_name} rejected")
        
        while True:
            try:
                # Apply rate limiting
                self._apply_rate_limiting()
                
                # Only add timestamp to methods that actually need it
                # Some ByBit methods (like set_leverage) don't accept timestamp parameters
                timestamp_required_methods = {
                    'place_order', 'cancel_order', 'cancel_all_orders', 'amend_order',
                    'get_open_orders', 'get_order_history', 'get_trade_history'
                }
                
                if method_name in timestamp_required_methods and 'timestamp' not in kwargs:
                    kwargs['timestamp'] = self._get_adjusted_timestamp()
                
                call_start = time.time()
                response = func(*args, **kwargs)
                response_time = time.time() - call_start
                
                # Check for HTTP error codes in response if available
                if hasattr(response, 'status_code') and response.status_code in (429, 500, 502, 503, 504):
                    self.logger.warning(f"'{method_name}' returned HTTP error {response.status_code}. Raising ExchangeError to trigger retry.")
                    raise ExchangeError(f"HTTP error {response.status_code} from response object")
                
                # Record successful API call
                self._record_api_call(method_name, True, response_time=response_time)
                
                # Log successful retry if this wasn't the first attempt
                if retries > 0:
                    total_time = time.time() - start_time
                    self.logger.info(f"✅ API call '{method_name}' succeeded after {retries} retries in {total_time:.2f}s")
                
                return response
                
            except (ReadTimeout, URLLib3ReadTimeoutError) as e:
                self._record_api_call(method_name, False, 'timeout_errors')
                
                if retries < self.max_retries:
                    wait = self._calculate_backoff_with_jitter(retries)
                    self.logger.warning(
                        f"Read timeout for '{method_name}'. Attempt {retries + 1}/{self.max_retries}. "
                        f"Retrying in {wait:.2f}s. Error: {str(e)}"
                    )
                    time.sleep(wait)
                    retries += 1
                    continue
                
                total_time = time.time() - start_time
                self.logger.error(f"API call '{method_name}' failed after {self.max_retries} retries due to ReadTimeout in {total_time:.2f}s: {str(e)}")
                raise ExchangeError(f"API call '{method_name}' failed after {self.max_retries} retries (ReadTimeout.)") from e

            except RequestException as e: # Catches ConnectionError, HTTPError (if pybit raises them), etc.
                err_msg_lower = str(e).lower()
                
                # Classify error type for monitoring
                error_type = 'connection_errors' if isinstance(e, ConnectionError) else 'server_errors'
                if "429" in err_msg_lower or "rate limit" in err_msg_lower or "too many visits" in err_msg_lower:
                    error_type = 'rate_limit_errors'
                    # Set rate limit cooldown period
                    self._rate_limit_detected_until = datetime.now() + timedelta(seconds=30)
                
                self._record_api_call(method_name, False, error_type)
                
                # Determine if this specific RequestException is retryable
                is_retryable_http_error = (
                    "429" in err_msg_lower or "rate limit" in err_msg_lower or "too many visits" in err_msg_lower or
                    "500" in err_msg_lower or "502" in err_msg_lower or "503" in err_msg_lower or "504" in err_msg_lower or
                    "server error" in err_msg_lower
                )
                is_connection_error = isinstance(e, ConnectionError)

                if (is_retryable_http_error or is_connection_error) and retries < self.max_retries:
                    wait = self._calculate_backoff_with_jitter(retries)
                    log_event_type = "Retryable HTTP/Server Error" if is_retryable_http_error else "Connection Error"
                    self.logger.warning(
                        f"{log_event_type} for '{method_name}'. Attempt {retries + 1}/{self.max_retries}. "
                        f"Retrying in {wait:.2f}s. Error: {str(e)}"
                    )
                    time.sleep(wait)
                    retries += 1
                    continue
                
                total_time = time.time() - start_time
                self.logger.error(f"API call '{method_name}' failed due to non-retried RequestException in {total_time:.2f}s: {str(e)}")
                raise ExchangeError(f"API call '{method_name}' failed (RequestException.)") from e

            except Exception as e: # General fallback, including handling self-raised ExchangeError from status_code check
                err_msg_lower = str(e).lower()
                
                # Handle specific non-retryable errors for cancel_order
                if func.__name__ == 'cancel_order':
                    # Check for specific ByBit error codes that shouldn't be retried
                    non_retryable_cancel_errors = [
                        "110001",  # order not exists or too late to cancel
                        "110003",  # order does not exist
                        "110004",  # order status does not allow cancellation
                        "110005",  # order has been cancelled
                        "110006",  # order has been filled
                        "110007",  # order has been partially filled
                    ]
                    
                    # Check if this is a non-retryable cancel error
                    if any(error_code in str(e) for error_code in non_retryable_cancel_errors):
                        # Extract the specific error code for logging
                        found_error_code = next((code for code in non_retryable_cancel_errors if code in str(e)), "unknown")
                        self.logger.info(f"Cancel order completed - order was already processed (Error {found_error_code}): {str(e)}")
                        # Return a success-like response for these expected failures
                        return {
                            "retCode": 0,
                            "retMsg": f"Order already processed - Error {found_error_code}",
                            "result": {"status": "already_processed", "error_code": found_error_code},
                            "retExtInfo": {},
                            "time": int(time.time() * 1000)
                        }
                
                # Handle timestamp errors
                if "timestamp" in err_msg_lower and retries < self.max_retries:
                    self._record_api_call(method_name, False, 'server_errors')  # Timestamp is usually a server sync issue
                    self.logger.warning(f"Timestamp error detected for '{method_name}'. Resyncing time with server...")
                    self._sync_time()  # Resync time
                    wait = self._calculate_backoff_with_jitter(retries)
                    self.logger.warning(
                        f"Retrying '{method_name}' after timestamp resync. Attempt {retries + 1}/{self.max_retries}. "
                        f"Retrying in {wait:.2f}s..."
                    )
                    time.sleep(wait)
                    retries += 1
                    continue
                
                # Classify error type for monitoring
                error_type = 'server_errors'  # Default classification
                if "429" in err_msg_lower or "rate limit" in err_msg_lower or "too many visits" in err_msg_lower:
                    error_type = 'rate_limit_errors'
                    # Set rate limit cooldown period
                    self._rate_limit_detected_until = datetime.now() + timedelta(seconds=30)
                elif "connection" in err_msg_lower or "network" in err_msg_lower:
                    error_type = 'connection_errors'
                elif "timeout" in err_msg_lower:
                    error_type = 'timeout_errors'
                
                self._record_api_call(method_name, False, error_type)
                    
                # Handle rate limits/server errors if they came as a generic Exception or our self-raised ExchangeError
                is_retryable_keyword_error = (
                    "429" in err_msg_lower or "rate limit" in err_msg_lower or "too many visits" in err_msg_lower or
                    "500" in err_msg_lower or "502" in err_msg_lower or "503" in err_msg_lower or "504" in err_msg_lower or
                    "server error" in err_msg_lower or "http error" in err_msg_lower # Catches our self-raised one
                )

                if is_retryable_keyword_error and retries < self.max_retries:
                    wait = self._calculate_backoff_with_jitter(retries)
                    self.logger.warning(
                        f"Generic error with retryable keywords for '{method_name}'. Attempt {retries + 1}/{self.max_retries}. "
                        f"Retrying in {wait:.2f}s. Error: {str(e)}"
                    )
                    time.sleep(wait)
                    retries += 1
                    continue
                
                total_time = time.time() - start_time
                self.logger.error(f"API call '{method_name}' failed after {retries} retries with unhandled exception in {total_time:.2f}s: {str(e)}")
                if isinstance(e, ExchangeError):
                    raise # Re-raise if it's already an ExchangeError we classified
                else:
                    raise ExchangeError(f"API call '{func.__name__}' failed after {retries} retries (General Unhandled Exception.)") from e

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
            self.logger.info("Order placed")
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

    def fetch_all_positions(self, category: str = "linear") -> Dict[str, Any]:
        """
        Fetch all open positions across all symbols for USDT pairs.
        Args:
            category: Bybit category (default 'linear')
        Returns:
            Positions dict
        """
        try:
            if not self.client:
                raise ExchangeError("Client not authenticated")
            response = self._api_call_with_backoff(self.client.get_positions, category=category, settleCoin="USDT")
            checked = self._check_response(response, context="fetch_all_positions")
            self.logger.debug(f"Fetched all positions: Found {len(checked.get('list', []))} positions")
            return checked
        except Exception as e:
            self.logger.error(f"Fetch all positions failed: {e}")
            raise ExchangeError(f"Fetch all positions failed: {str(e)}")

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

    def fetch_all_open_orders(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch all open orders across all symbols for USDT pairs.
        Args:
            params: Additional parameters
        Returns:
            API response dict
        """
        try:
            if not self.client:
                raise ExchangeError("Client not authenticated")
            open_params = {
                "category": "linear",
                "settleCoin": "USDT"  # Filter for USDT-based contracts to avoid API error
            }
            if params:
                open_params.update(params)
            response = self._api_call_with_backoff(self.client.get_open_orders, **open_params)
            checked = self._check_response(response, context="fetch_all_open_orders")
            self.logger.debug(f"Fetched all open orders: {open_params} | Found {len(checked.get('list', []))} orders")
            return checked
        except Exception as e:
            self.logger.error(f"Fetch all open orders failed: {e}")
            raise ExchangeError(f"Fetch all open orders failed: {str(e)}")

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
                decimal_part = tick_size_str.split('.')[1].rstrip('0')
                return len(decimal_part)
            return 0  # No decimal places if integer or not found
        self.logger.warning(f"Could not determine price precision for {symbol}, defaulting to 8.")
        return 8 # Default precision

    def get_qty_precision(self, symbol: str, category: str = 'linear') -> int:
        info = self._fetch_instrument_info(symbol, category)
        if info and info.get('lotSizeFilter') and info['lotSizeFilter'].get('qtyStep'):
            qty_step_str = str(info['lotSizeFilter']['qtyStep'])
            if '.' in qty_step_str:
                decimal_part = qty_step_str.split('.')[1].rstrip('0')
                return len(decimal_part)
            return 0  # No decimal places
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

    def fetch_all_tickers(self, category: str = 'linear') -> Dict[str, Any]:
        """
        Fetch all ticker information including 24h volume for the specified category.
        
        Args:
            category: The category of instruments ('linear' for USDT perpetuals)
            
        Returns:
            Dict containing ticker information for all symbols
        """
        try:
            self.logger.debug(f"Fetching all tickers for category: {category}")
            response = self._api_call_with_backoff(
                self.client.get_tickers,
                category=category
            )
            
            checked_response = self._check_response(response, f"fetch_all_tickers({category})")
            self.logger.debug(f"Successfully fetched tickers. Response keys: {list(checked_response.keys())}")
            
            return checked_response
            
        except Exception as e:
            self.logger.error(f"Error fetching all tickers for category {category}: {e}")
            raise ExchangeError(f"Failed to fetch tickers: {str(e)}") from e

    def get_top_volume_symbols(self, category: str = 'linear', count: int = 10, min_volume_usdt: float = 1000000) -> List[str]:
        """
        Get top symbols by 24h volume from ByBit.
        
        Args:
            category: The category of instruments ('linear' for USDT perpetuals)
            count: Number of top symbols to return
            min_volume_usdt: Minimum 24h volume in USDT to include
            
        Returns:
            List of symbol names sorted by 24h volume (highest first)
        """
        try:
            # Fetch all tickers
            tickers_response = self.fetch_all_tickers(category)
            tickers_list = tickers_response.get('result', {}).get('list', [])
            
            # Filter and sort by volume
            valid_symbols = []
            for ticker in tickers_list:
                symbol = ticker.get('symbol', '')
                volume_24h = float(ticker.get('turnover24h', 0))
                
                # Filter by minimum volume and ensure it's USDT pair
                if volume_24h >= min_volume_usdt and symbol.endswith('USDT'):
                    valid_symbols.append((symbol, volume_24h))
            
            # Sort by volume (descending) and return top symbols
            valid_symbols.sort(key=lambda x: x[1], reverse=True)
            top_symbols = [symbol for symbol, volume in valid_symbols[:count]]
            
            self.logger.info(f"Found {len(valid_symbols)} symbols with volume >= ${min_volume_usdt:,.0f}")
            self.logger.debug(f"Top {count} symbols by volume: {top_symbols}")
            
            return top_symbols
            
        except Exception as e:
            self.logger.error(f"Error getting top volume symbols: {e}")
            raise ExchangeError(f"Failed to get top volume symbols: {str(e)}") from e

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current market price for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            
        Returns:
            Current price as float, or None if unable to fetch
        """
        try:
            norm_symbol = symbol.replace("/", "").upper()
            
            # Use get_tickers to get current price
            response = self._api_call_with_backoff(
                self.client.get_tickers,
                category='linear',
                symbol=norm_symbol
            )
            
            checked_response = self._check_response(response, f"get_current_price({norm_symbol})")
            tickers_list = checked_response.get('result', {}).get('list', [])
            
            if tickers_list:
                ticker = tickers_list[0]
                last_price = ticker.get('lastPrice')
                if last_price:
                    price = float(last_price)
                    self.logger.debug(f"Current price for {norm_symbol}: {price}")
                    return price
                    
            self.logger.warning(f"No price data found for {norm_symbol}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def set_leverage(self, symbol: str, leverage: int, category: str = 'linear') -> Dict[str, Any]:
        """
        Set leverage for a trading symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            leverage: Leverage value (1-50 for most ByBit instruments)
            category: The category of instruments ('linear' for USDT perpetuals)
            
        Returns:
            Dict containing the response from ByBit API
            
        Raises:
            ExchangeError: If leverage setting fails
        """
        try:
            self.logger.info(f"Setting leverage to {leverage}x for {symbol} ({category})")
            
            # Rely on Pybit's internal timestamp handling to avoid 10002 errors
            response = self.client.set_leverage(
                category=category,
                symbol=symbol,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            
            # Handle specific case where leverage is already set (error code 110043)
            api_ret_code = response.get("retCode") or response.get("ret_code")
            if api_ret_code == 110043:
                self.logger.info(f"Leverage for {symbol} is already set to {leverage}x (no change needed)")
                # Return a successful response structure
                return {
                    "retCode": 0,
                    "retMsg": "Leverage already set to requested value",
                    "result": {},
                    "time": int(time.time() * 1000)
                }
            
            # For all other responses, use normal error checking
            checked_response = self._check_response(response, f"set_leverage({symbol}, {leverage}x)")
            self.logger.info(f"Successfully set leverage to {leverage}x for {symbol}")
            
            return checked_response
            
        except Exception as e:
            # Catch any other exceptions including pybit errors
            error_msg = str(e).lower()
            if "110043" in error_msg or "leverage not modified" in error_msg:
                self.logger.info(f"Leverage for {symbol} is already set to {leverage}x (no change needed)")
                # Return a successful response structure
                return {
                    "retCode": 0,
                    "retMsg": "Leverage already set to requested value",
                    "result": {},
                    "time": int(time.time() * 1000)
                }
            elif "timestamp" in error_msg or "recv_window" in error_msg or "10002" in error_msg or "bad request" in error_msg:
                # Timestamp/sync errors - implement comprehensive fallback strategy
                self.logger.warning(f"Leverage setting failed due to timestamp/sync issues for {symbol}.")
                self.logger.warning(f"Error details: {str(e)}")
                
                # Try alternative approach: resync time and retry once
                self.logger.info("Attempting time resync and single retry...")
                try:
                    # Force time resync
                    self._sync_time()
                    time.sleep(1)  # Brief pause after sync
                    
                    # Retry once letting Pybit supply its own timestamp
                    retry_response = self.client.set_leverage(
                        category=category,
                        symbol=symbol,
                        buyLeverage=str(leverage),
                        sellLeverage=str(leverage)
                    )
                    
                    # If retry succeeds, check and return
                    checked_response = self._check_response(retry_response, f"set_leverage_retry({symbol}, {leverage}x)")
                    self.logger.info(f"✅ Successfully set leverage to {leverage}x for {symbol} after retry")
                    return checked_response
                    
                except Exception as retry_err:
                    self.logger.warning(f"Retry also failed: {retry_err}")

                # Final explicit attempt: use server time minus a small buffer (<1s)
                try:
                    self.logger.info("Attempting final leverage set using server time reference…")

                    server_time_resp = self.client.get_server_time()
                    # Extract server time in ms (covers different response shapes)
                    server_ms = server_time_resp.get("time") or \
                               int(int(server_time_resp.get("result", {}).get("timeNano", 0)) / 1_000_000) or \
                               int(float(server_time_resp.get("result", {}).get("timeSecond", 0)) * 1000)

                    if not server_ms:
                        raise ValueError(f"Unexpected server_time response: {server_time_resp}")

                    safe_timestamp = server_ms - 500  # 0.5 s behind server, always valid

                    final_response = self.client.set_leverage(
                        category=category,
                        symbol=symbol,
                        buyLeverage=str(leverage),
                        sellLeverage=str(leverage),
                        timestamp=safe_timestamp
                    )

                    checked_final = self._check_response(final_response, f"set_leverage_final({symbol}, {leverage}x)")
                    self.logger.info(f"✅ Successfully set leverage to {leverage}x for {symbol} with server-time reference")
                    return checked_final

                except Exception as final_err:
                    self.logger.warning(f"Final server-time attempt failed: {final_err}")
                
                # Final fallback - check current leverage and continue
                self.logger.warning(f"Leverage setting completely failed for {symbol}. Bot will continue with existing leverage.")
                try:
                    positions = self.fetch_positions(symbol, category)
                    current_leverage_info = positions.get('result', {}).get('list', [])
                    if current_leverage_info:
                        current_leverage = current_leverage_info[0].get('leverage', 'unknown')
                        self.logger.info(f"📊 Current leverage for {symbol}: {current_leverage}x (timestamp sync issues prevented changes)")
                    else:
                        self.logger.info(f"📊 No existing position found for {symbol}. Default leverage will be used.")
                        current_leverage = 'default'
                    
                    return {
                        "retCode": 0,
                        "retMsg": f"Leverage setting bypassed due to persistent timestamp issues. Trading will continue.",
                        "result": {"current_leverage": current_leverage, "bypass_reason": "timestamp_sync_failure"},
                        "time": int(time.time() * 1000)
                    }
                except Exception as pos_err:
                    self.logger.debug(f"Could not fetch current position info: {pos_err}")
                    return {
                        "retCode": 0,
                        "retMsg": f"Leverage setting bypassed due to timestamp issues. Bot will continue with default settings.",
                        "result": {"bypass_reason": "timestamp_sync_failure"},
                        "time": int(time.time() * 1000)
                    }
            else:
                self.logger.error(f"Error setting leverage to {leverage}x for {symbol}: {e}")
                raise ExchangeError(f"Failed to set leverage for {symbol}: {str(e)}") from e 