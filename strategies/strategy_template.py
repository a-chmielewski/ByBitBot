import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
from datetime import datetime, timezone

class StrategyTemplate(ABC):
    """
    Abstract base class for all trading strategies.
    All strategies must inherit from this class and implement required methods.

    NOTE: This template only supports single-symbol operation. The fields order_pending and active_order_id are not symbol-aware and will cause conflicts if used for multiple symbols simultaneously. For multi-symbol support, refactor these fields to be keyed by symbol.
    """
    def __init__(self, data: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Args:
            data: Market data (passed in by the bot, e.g., OHLCV DataFrame or dict)
            config: Strategy-specific and global config parameters (dict)
            logger: Logger instance for strategy-specific logging
        Usage:
            - self.position should be updated when an order is filled (see on_order_update).
            - self.position should be cleared/updated on trade exit (see on_trade_update).
        """
        self.data = data.copy() # Ensure the strategy works on a copy of the initial data
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.position: Dict[str, Optional[Dict[str, Any]]] = {}  # Track current position state per symbol, explicitly Optional
        self.order_pending: Dict[str, bool] = {} # Tracks if an order is awaiting confirmation per symbol
        self.active_order_id: Dict[str, Optional[str]] = {} # Stores the ID of the pending/active main order per symbol
        
        # Renamed last_state to current_operational_state for clarity and per-symbol tracking
        self.current_operational_state: Dict[str, Optional[str]] = {} 
        
        self.on_init()
        self.init_indicators()

    def on_init(self) -> None:
        """
        Optional: Pre-start setup (e.g., warm up indicators on extra historical data).
        Override in your strategy if needed.
        """
        pass

    @abstractmethod
    def init_indicators(self) -> None:
        """
        Initialize indicators required by the strategy.
        """
        pass

    def _base_check_entry(self, symbol: str) -> bool:
        """
        Base check to see if a new entry can be considered for a specific symbol.
        Prevents new entries if a position is already open or an order is pending for that symbol.
        """
        current_pos = self.position.get(symbol)
        current_pending = self.order_pending.get(symbol, False)
        # self.logger.debug(f"{self.__class__.__name__} _base_check_entry for {symbol}: current_position={current_pos}, current_order_pending={current_pending}")

        if current_pos is not None:
            # self.logger.debug(f"{self.__class__.__name__} _base_check_entry for {symbol}: Returning False (position already open). Details: {current_pos}")
            return False
        if current_pending:
            # self.logger.debug(f"{self.__class__.__name__} _base_check_entry for {symbol}: Returning False (order pending). Active Order ID: {self.active_order_id.get(symbol)}")
            return False
        # self.logger.debug(f"{self.__class__.__name__} _base_check_entry for {symbol}: Returning True (ok to check entry conditions)")
        return True

    def check_entry(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Public method to check entry conditions for a specific symbol.
        Wraps _check_entry_conditions with checks for existing positions or pending orders.
        Args:
            symbol: The trading symbol (e.g., 'BTC/USDT')
        Returns:
            dict | None: Order details if entry signal, else None.
        """
        if not self._base_check_entry(symbol):
            return None
        
        entry_signal = self._check_entry_conditions(symbol)
        
        if entry_signal:
            # Make a shallow copy to avoid mutating the original dict
            entry_signal = entry_signal.copy()
            # Ensure essential keys are present, and add risk parameters
            if not all(k in entry_signal for k in ['side']): # size is optional now, price is not required for market orders
                self.logger.error(f"Strategy {self.__class__.__name__} _check_entry_conditions missing 'side'. Signal: {entry_signal}")
                return None

            risk_params = self.get_risk_parameters() # expects sl_pct, tp_pct
            entry_signal.update(risk_params) # Add sl_pct and tp_pct to the signal

            # Default size from config if not provided by strategy
            if 'size' not in entry_signal or entry_signal['size'] is None:
                 entry_signal['size'] = self.config.get('default', {}).get('default_order_size', None) # Example path, adjust as per your config
                 if entry_signal['size'] is None:
                     self.logger.info(f"Strategy {self.__class__.__name__} did not specify 'size' and no default_order_size in config. OrderManager will use exchange minimum.")
            
            # Ensure sl_pct and tp_pct are present, even if None
            if 'sl_pct' not in entry_signal: entry_signal['sl_pct'] = None
            if 'tp_pct' not in entry_signal: entry_signal['tp_pct'] = None
            
            # Mark order as pending before returning the signal
            self.order_pending[symbol] = True
            # self.active_order_id[symbol] = "PENDING_PLACEMENT" # Placeholder until actual ID is known
            self.logger.info(f"{self.__class__.__name__} for {symbol}: Entry signal generated. Order pending placement.")
            
        return entry_signal

    @abstractmethod
    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Protected method for strategy-specific entry logic for a given symbol.
        Called by check_entry if no position is open and no order is pending for that symbol.
        Args:
            symbol: The trading symbol (e.g., 'BTC/USDT')
        Returns:
            dict | None: Order details if entry signal, else None.
        """
        pass

    @abstractmethod
    def check_exit(self, symbol: str) -> Optional[Dict[str, Any]]: # Updated to take symbol, return signal
        """
        Check exit conditions for an open position for a specific symbol.
        Args:
            symbol: The trading symbol (e.g., 'BTCUSDT')
        Returns:
            Optional[Dict[str, Any]]: Exit signal details if conditions met, else None.
                                      Example: {'type': 'market', 'reason': 'RSI overbought'}
        """
        pass

    @abstractmethod
    def get_risk_parameters(self) -> Dict[str, Any]:
        """
        Return risk parameters (stop-loss as sl_pct, take-profit as tp_pct, etc.) for the strategy.
        These should be percentages (e.g., 0.01 for 1%).
        Returns:
            dict: {"sl_pct": float_or_None, "tp_pct": float_or_None, ...}
        """
        pass

    def on_order_update(self, order_responses: Dict[str, Any], symbol: str) -> None:
        """
        Handle order status updates from OrderManager for a specific symbol.
        This method now receives the full response dictionary from OrderManager.
        Args:
            order_responses: Dict from OrderManager, contains 'main_order', 'stop_loss_order', 'take_profit_order'.
            symbol: The trading symbol (e.g., 'BTC/USDT')
        """
        self.logger.info(f"{self.__class__.__name__} on_order_update for {symbol}.")
        self.logger.debug(f"{self.__class__.__name__} on_order_update for {symbol}. Received order_responses: {order_responses}")
        
        main_order_response = order_responses.get('main_order')
        sl_order_response = order_responses.get('stop_loss_order')
        tp_order_response = order_responses.get('take_profit_order')

        # Logic to handle position opening from main order
        if main_order_response:
            order_result = main_order_response.get('result', {})
            order_id = order_result.get('orderId')
            order_status = order_result.get('orderStatus', '').lower()

            if not order_id:
                self.logger.error(f"{self.__class__.__name__}: Main order response missing orderId for {symbol}. Response: {main_order_response}")
                if self.order_pending.get(symbol, False): # If an order was pending
                    self.logger.warning(f"{self.__class__.__name__}: Order response for a pending order lacks ID for {symbol}. Resetting pending state and looking for entry.")
                    self.order_pending[symbol] = False
                    self.active_order_id[symbol] = None
                    self.log_state_change(symbol, "awaiting_entry", f"{self.__class__.__name__} for {symbol}: Main order failed or ID missing. Looking for new entry conditions...")
                return

            self.logger.info(f"{self.__class__.__name__}: Main order update for {order_id} ({symbol}): Status - '{order_status}'")
            self.active_order_id[symbol] = order_id # Track the main order ID

            if order_status in ['filled', 'partiallyfilled']:
                self.logger.info(f"{self.__class__.__name__}: Main order {order_id} ({symbol}) is {order_status}. Updating position.")
                
                sl_order_id = sl_order_response.get('result', {}).get('orderId') if sl_order_response else None
                tp_order_id = tp_order_response.get('result', {}).get('orderId') if tp_order_response else None

                self.position[symbol] = {
                    'main_order_id': order_id,
                    'symbol': symbol,
                    'side': order_result.get('side', 'unknown').lower(),
                    'size': float(order_result.get('cumExecQty', order_result.get('qty', 0))),
                    'entry_price': float(order_result.get('avgPrice', 0)),
                    'status': 'open',
                    'timestamp': order_result.get('updatedTime', datetime.now(timezone.utc).isoformat()),
                    'sl_order_id': sl_order_id,
                    'tp_order_id': tp_order_id,
                    'raw_entry_response': order_result
                }
                self.order_pending[symbol] = False 
                # active_order_id remains set to main_order_id while position is open
                self.log_state_change(symbol, "monitoring_trade", f"{self.__class__.__name__} for {symbol}: Trade active (main order {order_id}). Monitoring for exit or SL/TP.")
            
            elif order_status in ['new', 'active', 'untriggered', 'pendingcancel']: # Still active or pending placement/cancellation
                 self.logger.info(f"{self.__class__.__name__}: Main order {order_id} ({symbol}) is {order_status}. Still pending or active.")
                 self.order_pending[symbol] = True # Keep it true if 'new' or 'active' but not yet filled
                 # Potentially a different state like "awaiting_fill" could be used here
                 # For now, "monitoring_trade" can cover this phase too if an order is placed.
                 if self.current_operational_state.get(symbol) != "monitoring_trade":
                     self.log_state_change(symbol, "awaiting_fill", f"{self.__class__.__name__} for {symbol}: Main order {order_id} placed. Waiting for fill/SL/TP confirmation.")


            elif order_status in ['rejected', 'cancelled', 'expired']:
                self.logger.info(f"{self.__class__.__name__}: Main order {order_id} ({symbol}) is {order_status}. Resetting state.")
                self.clear_position(symbol) # This will trigger "awaiting_entry" log
                self.order_pending[symbol] = False
                self.active_order_id[symbol] = None
                # log_state_change is handled by clear_position

        # Logic to handle position closing from SL/TP orders
        # This assumes SL/TP fill means the position is closed.
        for order_type_key, order_resp in [('stop_loss_order', sl_order_response), ('take_profit_order', tp_order_response)]:
            if order_resp:
                order_result = order_resp.get('result', {})
                order_id = order_result.get('orderId')
                order_status = order_result.get('orderStatus', '').lower()
                
                if not order_id: continue # Skip if no ID

                self.logger.info(f"{self.__class__.__name__}: {order_type_key} update for {order_id} ({symbol}): Status - '{order_status}'")

                if order_status in ['filled', 'triggered']: # 'triggered' for market SL/TP
                    self.logger.info(f"{self.__class__.__name__}: {order_type_key} {order_id} ({symbol}) is {order_status}. Position considered closed.")
                    self.clear_position(symbol) # This will trigger "awaiting_entry" log
                    # order_pending and active_order_id are reset in clear_position indirectly or directly.
                elif order_status in ['cancelled', 'rejected', 'expired'] and self.position.get(symbol):
                    self.logger.warning(f"{self.__class__.__name__}: {order_type_key} {order_id} ({symbol}) is {order_status} but position for {symbol} is still considered open. Checking counterpart.")
                    # If one SL/TP is cancelled/rejected, the other might still be active or the position might need explicit closing.
                    # The orphan check in OrderManager should handle lone SL/TPs.
                    # Here, we just log. If both SL/TP are gone and position is open, it's a potential issue.
                    current_pos = self.position.get(symbol)
                    if current_pos:
                        active_sl = self.active_order_id.get(f"{symbol}_sl")
                        active_tp = self.active_order_id.get(f"{symbol}_tp")
                        if not active_sl and not active_tp: # Pseudo-check, real check is via OrderManager
                             self.logger.warning(f"{self.__class__.__name__}: {order_type_key} {order_id} failed and no other SL/TP seems active for open position on {symbol}. Manual check or exit signal might be needed.")
        
        # Handling an exit order that is part of order_responses (e.g. from execute_strategy_exit)
        exit_order_response = order_responses.get('exit_order') # Assuming key 'exit_order'
        if exit_order_response:
            order_result = exit_order_response.get('result', {})
            order_id = order_result.get('orderId')
            order_status = order_result.get('orderStatus', '').lower()
            if order_id:
                self.logger.info(f"{self.__class__.__name__}: Exit order update for {order_id} ({symbol}): Status - '{order_status}'")
                if order_status == 'filled':
                    self.logger.info(f"{self.__class__.__name__}: Exit order {order_id} ({symbol}) is filled. Position closed.")
                    self.clear_position(symbol) # This will trigger "awaiting_entry" log


    def on_trade_update(self, trade: Dict[str, Any], symbol: str) -> None:
        """
        Handle trade updates (e.g., position closed due to SL/TP or manual exit) for a specific symbol.
        This method is typically called by the bot after a trade is recorded by PerformanceTracker.
        Args:
            trade: Trade update details (dict). Expected to have 'exit': True if position closed.
            symbol: The trading symbol (e.g., 'BTC/USDT')
        """
        self.logger.debug(f"{self.__class__.__name__} on_trade_update for {symbol}. Trade: {trade}")
        if trade.get('exit'): # This is a bit redundant if on_order_update handles all closures
            self.logger.info(f"{self.__class__.__name__}: Position for {symbol} confirmed closed by trade_update. Ensuring state is awaiting_entry.")
            self.clear_position(symbol) # Ensures state is set to awaiting_entry

    def on_error(self, exception: Exception) -> None:
        """
        Optional: Handle errors raised during strategy logic.
        Args:
            exception: The exception instance.
        Default: logs a warning.
        """
        self.logger.warning(f"Strategy {self.__class__.__name__} encountered an error: {exception}")

    def log_state_change(self, symbol: str, new_state: str, message: str) -> None:
        """
        Logs a message if the strategy's operational state for a symbol changes.
        Args:
            symbol: The trading symbol (e.g., 'BTCUSDT').
            new_state: The new operational state (e.g., 'awaiting_entry', 'monitoring_trade').
            message: The message to log.
        """
        current_symbol_state = self.current_operational_state.get(symbol)
        if current_symbol_state != new_state:
            self.current_operational_state[symbol] = new_state
            self.logger.info(message)
        else:
            self.logger.debug(f"State for {symbol} is already '{new_state}'. Suppressing duplicate log: {message}")

    def clear_position(self, symbol: str) -> None:
        """
        Clears the active position for the given symbol and resets related state.
        Triggers a state change log to 'awaiting_entry'.
        """
        if self.position.get(symbol) is not None: # Only log if a position was actually cleared
            self.logger.info(f"{self.__class__.__name__} for {symbol}: Clearing active position details.")
        
        self.position[symbol] = None
        self.order_pending[symbol] = False # No order should be pending if we are clearing position
        self.active_order_id[symbol] = None # No active main order ID associated with a non-existent position

        # This is the key log for "looking for new entry" after a trade closes or fails.
        self.log_state_change(symbol, "awaiting_entry", f"{self.__class__.__name__} for {symbol}: Position closed/cleared. Looking for new entry conditions...")


    def _log_untracked_order(self, order_id: str, order_status: str, symbol: str) -> None:
        """Helper to log updates for orders not actively tracked by the strategy's primary pending/active state."""
        self.logger.info(f"{self.__class__.__name__}: Update for an externally managed or non-primary order {order_id} ({symbol}): Status - '{order_status}'. This strategy instance might not be directly managing it.")

    def on_externally_synced_order(self, order_details: Dict[str, Any], symbol: str) -> None:
        """
        Handles order details that were synced from the exchange but not initially placed by this strategy instance's current lifecycle.
        (e.g. orders found on exchange at startup, or SL/TP orders whose main order was handled by a previous bot session).
        The strategy might choose to adopt them or just log them.
        Args:
            order_details (Dict[str, Any]): The full order details from the exchange.
            symbol (str): The trading symbol.
        """
        order_id = order_details.get('orderId')
        order_status = order_details.get('orderStatus', '').lower()
        stop_order_type = order_details.get('stopOrderType', '').lower() # For Bybit V5
        
        self.logger.info(f"{self.__class__.__name__} for {symbol}: Received externally synced order {order_id}, status: {order_status}, type: {order_details.get('orderType')}, stopType: {stop_order_type}.")

        # Basic adoption logic: if it's an open conditional order (SL/TP) and we have a position,
        # try to link it. This is complex and needs careful handling of main_order_id.
        # For now, this method primarily serves as an informational hook.
        # More sophisticated adoption would require matching orderLinkIds or other heuristics.

        current_pos = self.position.get(symbol)
        if current_pos:
            if stop_order_type == 'stoploss' and not current_pos.get('sl_order_id'):
                # Potentially adopt if matches position side etc.
                self.logger.info(f"Potentially adopting synced SL order {order_id} for existing position on {symbol}.")
                # self.position[symbol]['sl_order_id'] = order_id 
                # self.active_order_id[f"{symbol}_sl"] = order_id # Example tracking
            elif stop_order_type == 'takeprofit' and not current_pos.get('tp_order_id'):
                self.logger.info(f"Potentially adopting synced TP order {order_id} for existing position on {symbol}.")
                # self.position[symbol]['tp_order_id'] = order_id
                # self.active_order_id[f"{symbol}_tp"] = order_id # Example tracking
        else: # No current position
            if order_status in ['new', 'active', 'untriggered'] and not stop_order_type : # A regular limit/market order
                # This could be an old entry order. If bot policy is to take over,
                # one might set self.order_pending = True and self.active_order_id.
                # However, this is risky without knowing the full context of that order.
                self.logger.warning(f"{self.__class__.__name__} for {symbol}: Synced active non-conditional order {order_id} but no local position. Manual review may be needed.")
            elif stop_order_type and order_status in ['new', 'active', 'untriggered']:
                 self.log_state_change(symbol, "awaiting_entry", f"{self.__class__.__name__} for {symbol}: Synced active conditional order {order_id}. No current position. Looking for new entry conditions...") 