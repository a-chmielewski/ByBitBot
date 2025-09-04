import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Import risk utilities for advanced risk management
try:
    from modules.risk_utilities import (
        compute_atr, atr_stop_levels, progressive_take_profit_levels,
        update_trailing_stop, position_size_vol_normalized, kelly_fraction_capped,
        volatility_regime, RiskUtilitiesError
    )
    RISK_UTILITIES_AVAILABLE = True
except ImportError:
    RISK_UTILITIES_AVAILABLE = False
    
# Import market analyzer for regime-based adjustments
try:
    from modules.market_analyzer import MarketAnalyzer, MarketAnalysisError
    MARKET_ANALYZER_AVAILABLE = True
except ImportError:
    MARKET_ANALYZER_AVAILABLE = False

class StrategyTemplate(ABC):
    """
    Abstract base class for all trading strategies.
    All strategies must inherit from this class and implement required methods.

    NOTE: This template only supports single-symbol operation. The fields order_pending and active_order_id are not symbol-aware and will cause conflicts if used for multiple symbols simultaneously. For multi-symbol support, refactor these fields to be keyed by symbol.
    """
    
    # Market type tags - strategies should override this to indicate market conditions they work best in
    MARKET_TYPE_TAGS: List[str] = []  # e.g., ['TRENDING', 'HIGH_VOLATILITY'], ['RANGING'], ['TRANSITIONAL'], etc.
    
    # Strategy visibility - strategies can set this to False to hide from selection menu (e.g., dev/example strategies)
    SHOW_IN_SELECTION: bool = True
    
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
        
        # Enhanced Risk Management Configuration
        self._init_risk_management_config()
        
        # Initialize trailing stop tracking
        self.trailing_stops: Dict[str, Optional[float]] = {}  # symbol -> trailing stop price
        
        # Initialize market analyzer if available
        self.market_analyzer: Optional[Any] = None
        if MARKET_ANALYZER_AVAILABLE and hasattr(config, 'get'):
            try:
                # Note: Market analyzer requires exchange instance, will be set by bot if available
                pass
            except Exception as e:
                self.logger.warning(f"Could not initialize market analyzer: {e}")
        
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

    def update_indicators_for_new_row(self) -> None:
        """
        Optional: Update indicators efficiently for the latest row only.
        This method can be overridden by strategies to provide incremental updates
        instead of recalculating all indicators from scratch.
        Default implementation falls back to full init_indicators().
        """
        self.init_indicators()

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

    def check_entry(self, symbol: str, directional_bias: str = 'NEUTRAL', bias_strength: str = 'WEAK') -> Optional[Dict[str, Any]]:
        """
        Public method to check entry conditions for a specific symbol.
        Wraps _check_entry_conditions with checks for existing positions or pending orders.
        Args:
            symbol: The trading symbol (e.g., 'BTC/USDT')
            directional_bias: Higher timeframe directional bias ('BULLISH', 'BEARISH', 'NEUTRAL', etc.)
            bias_strength: Strength of the bias ('STRONG', 'MODERATE', 'WEAK')
        Returns:
            dict | None: Order details if entry signal, else None.
        """
        if not self._base_check_entry(symbol):
            return None
        
        entry_signal = self._check_entry_conditions(symbol)
        
        if entry_signal:
            # Multi-timeframe bias alignment filter
            if not self._check_bias_alignment(entry_signal, directional_bias, bias_strength):
                self.logger.info(f"Entry signal for {symbol} filtered out due to bias misalignment: signal={entry_signal.get('side')} vs bias={directional_bias}")
                return None
            
            # Make a shallow copy to avoid mutating the original dict
            entry_signal = entry_signal.copy()
            # Ensure essential keys are present, and add risk parameters
            if not all(k in entry_signal for k in ['side']): # size is optional now, price is not required for market orders
                self.logger.error(f"Strategy {self.__class__.__name__} _check_entry_conditions missing 'side'. Signal: {entry_signal}")
                return None

            # Get entry price for enhanced risk calculations
            entry_price = entry_signal.get('price') or self._get_current_price(symbol)
            account_equity = self._get_account_equity()
            
            # Apply liquidity and spread filters before generating risk parameters
            market_data = self._get_market_data(symbol)
            if market_data:
                if not self.check_liquidity_filter(symbol, market_data):
                    self.logger.info(f"Entry signal for {symbol} filtered out due to liquidity constraints")
                    return None
                    
                if not self.check_spread_slippage_guard(symbol, market_data):
                    self.logger.info(f"Entry signal for {symbol} filtered out due to spread/slippage constraints")
                    return None

            # Get enhanced risk parameters with full context
            # Handle backward compatibility for legacy strategies
            try:
                risk_params = self.get_risk_parameters(
                    symbol=symbol,
                    entry_price=entry_price,
                    side=entry_signal['side'],
                    account_equity=account_equity
                )
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    # Legacy strategy - call without parameters
                    self.logger.debug(f"Using legacy get_risk_parameters() for {self.__class__.__name__}")
                    risk_params = self.get_risk_parameters()
                else:
                    raise
            entry_signal.update(risk_params)

            # Use calculated position size if available
            if 'size' not in entry_signal or entry_signal['size'] is None:
                calculated_size = risk_params.get('position_size')
                if calculated_size:
                    entry_signal['size'] = calculated_size
                else:
                    # Fallback to config default
                    entry_signal['size'] = self.config.get('default', {}).get('default_order_size', None)
                    if entry_signal['size'] is None:
                        self.logger.info(f"Strategy {self.__class__.__name__} did not specify 'size' and no calculated size available. OrderManager will use exchange minimum.")
            
            # Ensure backward compatibility with sl_pct and tp_pct
            if 'sl_pct' not in entry_signal: entry_signal['sl_pct'] = None
            if 'tp_pct' not in entry_signal: entry_signal['tp_pct'] = None
            
            # Mark order as pending before returning the signal
            self.order_pending[symbol] = True
            # self.active_order_id[symbol] = "PENDING_PLACEMENT" # Placeholder until actual ID is known
            self.logger.info(f"{self.__class__.__name__} for {symbol}: Enhanced entry signal generated (regime: {risk_params.get('volatility_regime')}, sizing: {risk_params.get('position_sizing_mode')}). Order pending placement.")
            
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

    def get_risk_parameters(self, symbol: str = 'BTCUSDT', entry_price: Optional[float] = None, 
                           side: Optional[str] = None, account_equity: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """
        Return risk parameters using enhanced risk management hooks.
        
        This method now uses the configurable risk management system and calls RiskUtilities.
        Individual strategies can override this to provide custom risk logic, but should
        generally prefer configuring the hooks instead.
        
        Args:
            symbol: Trading symbol (for regime-based adjustments)
            entry_price: Proposed entry price (for ATR-based calculations)
            side: Position side ('long' or 'short')
            account_equity: Current account equity (for position sizing)
            
        Returns:
            dict: Enhanced risk parameters with calculated values
        """
        try:
            # Get volatility regime for regime-based adjustments
            volatility_regime = self.get_current_volatility_regime(symbol)
            
            risk_params = {
                'sl_pct': None,
                'tp_pct': None,
                'position_size': None,
                'trailing_stop_enabled': self.trailing_stop_enabled,
                'volatility_regime': volatility_regime
            }
            
            # Calculate stop loss
            if entry_price and side:
                sl_price = self.calculate_stop_loss_price(symbol, entry_price, side)
                if sl_price and entry_price > 0:
                    if side.lower() == 'long':
                        risk_params['sl_pct'] = (entry_price - sl_price) / entry_price
                    else:  # short
                        risk_params['sl_pct'] = (sl_price - entry_price) / entry_price
            else:
                # Fallback to configured percentage
                risk_params['sl_pct'] = self.stop_loss_fixed_pct
            
            # Calculate take profit  
            if entry_price and side:
                tp_prices = self.calculate_take_profit_prices(symbol, entry_price, side)
                if tp_prices:
                    # Use first take profit level for sl_pct/tp_pct compatibility
                    tp_price = tp_prices[0]
                    if side.lower() == 'long':
                        risk_params['tp_pct'] = (tp_price - entry_price) / entry_price
                    else:  # short
                        risk_params['tp_pct'] = (entry_price - tp_price) / entry_price
                    
                    # Include all TP levels for advanced order management
                    risk_params['tp_prices'] = tp_prices
            else:
                # Fallback to configured percentage
                risk_params['tp_pct'] = self.take_profit_fixed_pct
            
            # Calculate position size if we have all required parameters
            if entry_price and account_equity:
                position_size = self.calculate_position_size(
                    symbol=symbol,
                    entry_price=entry_price,
                    account_equity=account_equity,
                    current_volatility_regime=volatility_regime
                )
                risk_params['position_size'] = position_size
            
            # Add configuration metadata for OrderManager
            risk_params.update({
                'stop_loss_mode': self.stop_loss_mode,
                'take_profit_mode': self.take_profit_mode,
                'position_sizing_mode': self.position_sizing_mode,
                'leverage_multiplier': self.leverage_by_regime.get(volatility_regime or 'normal', 1.0)
            })
            
            return risk_params
            
        except Exception as e:
            self.logger.error(f"Error calculating risk parameters for {symbol}: {e}")
            # Return safe defaults
            return {
                'sl_pct': self.stop_loss_fixed_pct,
                'tp_pct': self.take_profit_fixed_pct,
                'position_size': None,
                'trailing_stop_enabled': False,
                'volatility_regime': None
            }

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

    def clear_position(self, symbol: str, reason: str = "Position closed/cleared") -> None:
        """
        Clears the active position for the given symbol and resets related state.
        Triggers a state change log to 'awaiting_entry'.
        
        Args:
            symbol: The trading symbol
            reason: Reason for clearing position (for logging)
        """
        if self.position.get(symbol) is not None: # Only log if a position was actually cleared
            self.logger.info(f"{self.__class__.__name__} for {symbol}: Clearing active position details. Reason: {reason}")
        
        self.position[symbol] = None
        self.order_pending[symbol] = False # No order should be pending if we are clearing position
        self.active_order_id[symbol] = None # No active main order ID associated with a non-existent position

        # This is the key log for "looking for new entry" after a trade closes or fails.
        self.log_state_change(symbol, "awaiting_entry", f"{self.__class__.__name__} for {symbol}: {reason}. Looking for new entry conditions...")


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

    # ==================== ENHANCED RISK MANAGEMENT HOOKS ====================
    
    def _init_risk_management_config(self) -> None:
        """
        Initialize enhanced risk management configuration with sensible defaults.
        Strategies can override these settings via config or by overriding this method.
        """
        strategy_config = self.config.get('strategy_configs', {}).get(self.__class__.__name__, {})
        risk_config = strategy_config.get('risk_management', {})
        
        # Stop loss configuration
        self.stop_loss_mode = risk_config.get('stop_loss_mode', 'fixed_pct')  # 'fixed_pct' | 'atr_mult'
        self.stop_loss_fixed_pct = risk_config.get('stop_loss_fixed_pct', 0.02)  # 2% default
        self.stop_loss_atr_multiplier = risk_config.get('stop_loss_atr_multiplier', 2.0)  # 2x ATR default
        
        # Take profit configuration
        self.take_profit_mode = risk_config.get('take_profit_mode', 'fixed_pct')  # 'fixed_pct' | 'progressive_levels'
        self.take_profit_fixed_pct = risk_config.get('take_profit_fixed_pct', 0.02)  # 2% default for quicker profits
        self.take_profit_progressive_levels = risk_config.get('take_profit_progressive_levels', [0.015, 0.035, 0.08])  # 1.5%, 3.5%, 8% - quicker first target
        
        # Trailing stop configuration
        self.trailing_stop_enabled = risk_config.get('trailing_stop_enabled', False)
        self.trailing_stop_mode = risk_config.get('trailing_stop_mode', 'price_pct')  # 'price_pct' | 'atr_mult'
        self.trailing_stop_offset_pct = risk_config.get('trailing_stop_offset_pct', 0.025)  # 2.5% trailing offset - wider
        self.trailing_stop_atr_multiplier = risk_config.get('trailing_stop_atr_multiplier', 2.5)  # 2.5x ATR trailing offset - wider
        
        # Position sizing configuration
        self.position_sizing_mode = risk_config.get('position_sizing_mode', 'fixed_notional')  # 'fixed_notional' | 'vol_normalized' | 'kelly_capped'
        self.position_fixed_notional = risk_config.get('position_fixed_notional', 1000.0)  # $1000 default
        self.position_risk_per_trade = risk_config.get('position_risk_per_trade', 0.01)  # 1% account risk per trade
        self.position_kelly_cap = risk_config.get('position_kelly_cap', 0.1)  # 10% max Kelly allocation
        
        # Leverage by volatility regime
        self.leverage_by_regime = risk_config.get('leverage_by_regime', {
            'low': 1.2,    # 20% higher leverage in low vol
            'normal': 1.0, # Base leverage in normal vol
            'high': 0.8    # 20% lower leverage in high vol
        })
        
        # Liquidity and spread filters
        self.min_liquidity_filter = risk_config.get('min_liquidity_filter', {
            'enabled': True,
            'min_bid_ask_volume': 10000,  # Min $10k on each side
            'max_spread_bps': 10  # Max 10 bps spread
        })
        
        self.spread_slippage_guard = risk_config.get('spread_slippage_guard', {
            'enabled': True,
            'max_spread_pct': 0.001,  # Max 0.1% spread
            'max_slippage_pct': 0.002  # Max 0.2% expected slippage
        })
        
        # ATR period for calculations
        self.atr_period = risk_config.get('atr_period', 14)

    def calculate_position_size(self, symbol: str, entry_price: float, account_equity: float, 
                               current_volatility_regime: Optional[str] = None) -> float:
        """
        Calculate position size based on configured position sizing mode.
        
        Args:
            symbol: Trading symbol
            entry_price: Proposed entry price
            account_equity: Current account equity
            current_volatility_regime: Current volatility regime ('low', 'normal', 'high')
            
        Returns:
            Position size in base currency units
        """
        try:
            base_size = 0.0
            
            if self.position_sizing_mode == 'fixed_notional':
                base_size = self.position_fixed_notional / entry_price
                
            elif self.position_sizing_mode == 'vol_normalized' and RISK_UTILITIES_AVAILABLE:
                # Get ATR for volatility normalization
                current_atr = self._get_current_atr(symbol)
                if current_atr and current_atr > 0:
                    tick_value = entry_price * 0.0001  # Assume 0.01% tick value
                    base_size = position_size_vol_normalized(
                        account_equity=account_equity,
                        risk_per_trade=self.position_risk_per_trade,
                        atr=current_atr,
                        tick_value=tick_value
                    )
                else:
                    # Fallback to fixed notional if ATR unavailable
                    self.logger.warning(f"ATR unavailable for {symbol}, falling back to fixed notional sizing")
                    base_size = self.position_fixed_notional / entry_price
                    
            elif self.position_sizing_mode == 'kelly_capped' and RISK_UTILITIES_AVAILABLE:
                # Get historical win rate and average return for Kelly calculation
                win_prob, edge = self._estimate_strategy_edge(symbol)
                if win_prob and edge:
                    kelly_fraction = kelly_fraction_capped(
                        edge=edge,
                        win_prob=win_prob,
                        cap=self.position_kelly_cap
                    )
                    base_size = (account_equity * kelly_fraction) / entry_price
                else:
                    # Fallback to risk-based sizing
                    risk_amount = account_equity * self.position_risk_per_trade
                    base_size = risk_amount / entry_price
            
            else:
                # Default fallback
                base_size = self.position_fixed_notional / entry_price
            
            # Apply volatility regime adjustment
            if current_volatility_regime and current_volatility_regime in self.leverage_by_regime:
                regime_multiplier = self.leverage_by_regime[current_volatility_regime]
                base_size *= regime_multiplier
                self.logger.debug(f"Applied {current_volatility_regime} vol regime multiplier {regime_multiplier} to position size")
            
            return max(base_size, 0.001)  # Minimum position size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            # Safe fallback
            return self.position_fixed_notional / entry_price

    def calculate_stop_loss_price(self, symbol: str, entry_price: float, side: str) -> Optional[float]:
        """
        Calculate stop loss price based on configured stop loss mode.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            side: Position side ('long' or 'short')
            
        Returns:
            Stop loss price or None if no stop loss
        """
        try:
            if self.stop_loss_mode == 'fixed_pct':
                if side.lower() == 'long':
                    return entry_price * (1 - self.stop_loss_fixed_pct)
                else:  # short
                    return entry_price * (1 + self.stop_loss_fixed_pct)
                    
            elif self.stop_loss_mode == 'atr_mult' and RISK_UTILITIES_AVAILABLE:
                current_atr = self._get_current_atr(symbol)
                if current_atr and current_atr > 0:
                    stop_levels = atr_stop_levels(
                        entry_price=entry_price,
                        side=side.lower(),
                        atr=current_atr,
                        atr_mult_sl=self.stop_loss_atr_multiplier,
                        atr_mult_tp=self.stop_loss_atr_multiplier  # Use same multiplier for TP calculation, we only use SL result
                    )
                    return stop_levels.get('stop_loss')
                else:
                    # Fallback to fixed percentage
                    self.logger.warning(f"ATR unavailable for {symbol}, using fixed percentage stop loss")
                    if side.lower() == 'long':
                        return entry_price * (1 - self.stop_loss_fixed_pct)
                    else:
                        return entry_price * (1 + self.stop_loss_fixed_pct)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss for {symbol}: {e}")
            return None

    def calculate_take_profit_prices(self, symbol: str, entry_price: float, side: str) -> List[float]:
        """
        Calculate take profit price(s) based on configured take profit mode.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            side: Position side ('long' or 'short')
            
        Returns:
            List of take profit prices (may be empty)
        """
        try:
            if self.take_profit_mode == 'fixed_pct':
                if side.lower() == 'long':
                    tp_price = entry_price * (1 + self.take_profit_fixed_pct)
                else:  # short
                    tp_price = entry_price * (1 - self.take_profit_fixed_pct)
                return [tp_price]
                
            elif self.take_profit_mode == 'progressive_levels' and RISK_UTILITIES_AVAILABLE:
                return progressive_take_profit_levels(
                    entry_price=entry_price,
                    side=side.lower(),
                    levels=self.take_profit_progressive_levels
                )
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit levels for {symbol}: {e}")
            return []

    def update_trailing_stop_if_enabled(self, symbol: str, current_price: float, side: str) -> Optional[float]:
        """
        Update trailing stop if enabled and return new trailing stop price.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            side: Position side ('long' or 'short')
            
        Returns:
            Updated trailing stop price or None if not enabled/applicable
        """
        if not self.trailing_stop_enabled or not RISK_UTILITIES_AVAILABLE:
            return None
            
        try:
            current_trail_price = self.trailing_stops.get(symbol)
            
            if self.trailing_stop_mode == 'price_pct':
                new_trail_price = update_trailing_stop(
                    side=side.lower(),
                    current_price=current_price,
                    trail_price=current_trail_price,
                    trail_offset=self.trailing_stop_offset_pct
                )
            elif self.trailing_stop_mode == 'atr_mult':
                current_atr = self._get_current_atr(symbol)
                if current_atr and current_atr > 0:
                    # Convert ATR multiplier to price percentage
                    atr_offset_pct = (current_atr * self.trailing_stop_atr_multiplier) / current_price
                    new_trail_price = update_trailing_stop(
                        side=side.lower(),
                        current_price=current_price,
                        trail_price=current_trail_price,
                        trail_offset=atr_offset_pct
                    )
                else:
                    # Fallback to percentage-based trailing
                    new_trail_price = update_trailing_stop(
                        side=side.lower(),
                        current_price=current_price,
                        trail_price=current_trail_price,
                        trail_offset=self.trailing_stop_offset_pct
                    )
            else:
                return None
            
            # Update stored trailing stop
            self.trailing_stops[symbol] = new_trail_price
            return new_trail_price
            
        except Exception as e:
            self.logger.error(f"Error updating trailing stop for {symbol}: {e}")
            return None

    def check_liquidity_filter(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Check if current market conditions pass liquidity filters.
        
        Args:
            symbol: Trading symbol
            market_data: Current market data (bid, ask, volume, etc.)
            
        Returns:
            True if liquidity conditions are acceptable
        """
        if not self.min_liquidity_filter['enabled']:
            return True
            
        try:
            bid_price = market_data.get('bid', 0)
            ask_price = market_data.get('ask', 0)
            bid_size = market_data.get('bid_size', 0)
            ask_size = market_data.get('ask_size', 0)
            
            if not all([bid_price, ask_price, bid_size, ask_size]):
                self.logger.warning(f"Incomplete market data for liquidity check on {symbol}")
                return False
            
            # Check minimum liquidity on each side
            min_volume = self.min_liquidity_filter['min_bid_ask_volume']
            bid_value = bid_price * bid_size
            ask_value = ask_price * ask_size
            
            if bid_value < min_volume or ask_value < min_volume:
                self.logger.debug(f"Liquidity filter failed for {symbol}: bid_value={bid_value}, ask_value={ask_value}, min_required={min_volume}")
                return False
            
            # Check spread
            spread_bps = ((ask_price - bid_price) / bid_price) * 10000
            max_spread_bps = self.min_liquidity_filter['max_spread_bps']
            
            if spread_bps > max_spread_bps:
                self.logger.debug(f"Spread filter failed for {symbol}: spread={spread_bps:.1f}bps, max_allowed={max_spread_bps}bps")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking liquidity filter for {symbol}: {e}")
            return False

    def check_spread_slippage_guard(self, symbol: str, market_data: Dict[str, Any]) -> bool:
        """
        Check if current spread and expected slippage are within acceptable limits.
        
        Args:
            symbol: Trading symbol  
            market_data: Current market data
            
        Returns:
            True if spread/slippage conditions are acceptable
        """
        if not self.spread_slippage_guard['enabled']:
            return True
            
        try:
            bid_price = market_data.get('bid', 0)
            ask_price = market_data.get('ask', 0)
            
            if not all([bid_price, ask_price]):
                self.logger.warning(f"Incomplete market data for spread check on {symbol}")
                return False
            
            # Check spread percentage
            mid_price = (bid_price + ask_price) / 2
            spread_pct = (ask_price - bid_price) / mid_price
            
            if spread_pct > self.spread_slippage_guard['max_spread_pct']:
                self.logger.debug(f"Spread guard failed for {symbol}: spread={spread_pct:.4f}, max_allowed={self.spread_slippage_guard['max_spread_pct']:.4f}")
                return False
            
            # Simple slippage estimation (half the spread)
            estimated_slippage_pct = spread_pct / 2
            
            if estimated_slippage_pct > self.spread_slippage_guard['max_slippage_pct']:
                self.logger.debug(f"Slippage guard failed for {symbol}: estimated_slippage={estimated_slippage_pct:.4f}, max_allowed={self.spread_slippage_guard['max_slippage_pct']:.4f}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking spread/slippage guard for {symbol}: {e}")
            return False

    def get_current_volatility_regime(self, symbol: str) -> Optional[str]:
        """
        Get current volatility regime for the symbol if market analyzer is available.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Volatility regime ('low', 'normal', 'high') or None if unavailable
        """
        if not MARKET_ANALYZER_AVAILABLE or not self.market_analyzer:
            return None
            
        try:
            return self.market_analyzer.get_vol_regime(symbol, '1m')
        except Exception as e:
            self.logger.warning(f"Could not get volatility regime for {symbol}: {e}")
            return None

    def _get_current_atr(self, symbol: str) -> Optional[float]:
        """
        Get current ATR value for the symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current ATR value or None if unavailable
        """
        try:
            if hasattr(self.data, 'iloc') and len(self.data) >= self.atr_period:
                # Calculate ATR from current data
                if RISK_UTILITIES_AVAILABLE:
                    atr_series = compute_atr(self.data, period=self.atr_period)
                    if not atr_series.empty:
                        return float(atr_series.iloc[-1])
                
                # Fallback ATR calculation
                high_low = self.data['high'] - self.data['low']
                high_close = abs(self.data['high'] - self.data['close'].shift(1))
                low_close = abs(self.data['low'] - self.data['close'].shift(1))
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(window=self.atr_period, min_periods=self.atr_period).mean()
                
                if not pd.isna(atr.iloc[-1]):
                    return float(atr.iloc[-1])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR for {symbol}: {e}")
            return None

    def _estimate_strategy_edge(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimate strategy edge for Kelly criterion calculation.
        
        This is a placeholder that should be overridden by strategies with access
        to historical performance data.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Tuple of (win_probability, edge_ratio) or (None, None) if unavailable
        """
        # Default conservative estimates - strategies should override this
        # with actual historical performance data
        return (0.55, 1.5)  # 55% win rate, 1.5 reward/risk ratio

    def set_market_analyzer(self, market_analyzer: Any) -> None:
        """
        Set market analyzer instance for regime-based adjustments.
        
        Args:
            market_analyzer: MarketAnalyzer instance
        """
        self.market_analyzer = market_analyzer

    # ==================== UTILITY METHODS FOR ENHANCED FUNCTIONALITY ====================
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for the symbol.
        
        This is a placeholder that should be overridden by strategies or set by
        the bot with actual market data access.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current price or None if unavailable
        """
        try:
            if hasattr(self.data, 'iloc') and len(self.data) > 0:
                return float(self.data['close'].iloc[-1])
            return None
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def _get_account_equity(self) -> Optional[float]:
        """
        Get current account equity.
        
        This is a placeholder that should be set by the bot with actual account data.
        Individual strategies should not need to override this.
        
        Returns:
            Account equity or None if unavailable
        """
        # This should be provided by the bot's context
        # For now, return a placeholder that strategies can use for testing
        return getattr(self, '_account_equity', None)

    def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get current market data (bid, ask, volume, etc.) for liquidity checks.
        
        This is a placeholder that should be set by the bot with actual market data.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Market data dictionary or None if unavailable
        """
        # This should be provided by the bot's context
        # For now, return None which will cause liquidity filters to be skipped
        return getattr(self, '_market_data', {}).get(symbol)

    def set_account_equity(self, equity: float) -> None:
        """
        Set current account equity (called by bot).
        
        Args:
            equity: Current account equity
        """
        self._account_equity = equity

    def set_market_data(self, symbol: str, market_data: Dict[str, Any]) -> None:
        """
        Set current market data for a symbol (called by bot).
        
        Args:
            symbol: Trading symbol
            market_data: Market data dictionary
        """
        if not hasattr(self, '_market_data'):
            self._market_data = {}
        self._market_data[symbol] = market_data

    def _check_bias_alignment(self, entry_signal: Dict[str, Any], directional_bias: str, bias_strength: str) -> bool:
        """
        Check if entry signal aligns with higher timeframe directional bias.
        
        Args:
            entry_signal: Entry signal dictionary containing 'side'
            directional_bias: Higher timeframe bias ('BULLISH', 'BEARISH', 'NEUTRAL', etc.)
            bias_strength: Strength of bias ('STRONG', 'MODERATE', 'WEAK')
            
        Returns:
            True if signal aligns with bias or bias is weak/neutral
        """
        signal_side = entry_signal.get('side', '').lower()
        
        # Always allow if bias is neutral or weak
        if directional_bias == 'NEUTRAL' or bias_strength == 'WEAK':
            return True
            
        # For moderate/strong bias, enforce alignment
        if bias_strength in ['MODERATE', 'STRONG']:
            if directional_bias in ['BULLISH', 'BULLISH_BIASED']:
                return signal_side == 'buy'
            elif directional_bias in ['BEARISH', 'BEARISH_BIASED']:
                return signal_side == 'sell'
                
        return True

    def update_trailing_stops(self, symbol: str, current_price: float) -> None:
        """
        Update trailing stops for active positions.
        
        This should be called periodically by the bot for positions with trailing stops enabled.
        
        Args:
            symbol: Trading symbol  
            current_price: Current market price
        """
        if not self.trailing_stop_enabled:
            return
            
        position = self.position.get(symbol)
        if not position or position.get('status') != 'open':
            return
            
        side = position.get('side', 'long')
        new_trail_price = self.update_trailing_stop_if_enabled(symbol, current_price, side)
        
        if new_trail_price:
            self.logger.debug(f"Updated trailing stop for {symbol} {side} position: {new_trail_price}")
            # The actual trailing stop order update should be handled by OrderManager
            # This method just calculates and stores the new trailing price

    def get_enhanced_exit_signal(self, symbol: str, current_price: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Enhanced exit signal that includes trailing stop checks.
        
        This extends the standard check_exit with trailing stop logic.
        Strategies can call this instead of check_exit directly, or it can be called by the bot.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price (if not provided, will try to get from data)
            
        Returns:
            Exit signal dictionary or None
        """
        # First check standard exit conditions
        exit_signal = self.check_exit(symbol)
        if exit_signal:
            return exit_signal
            
        # Check trailing stop if enabled
        if not self.trailing_stop_enabled:
            return None
            
        position = self.position.get(symbol)
        if not position or position.get('status') != 'open':
            return None
            
        if current_price is None:
            current_price = self._get_current_price(symbol)
            if current_price is None:
                return None
        
        # Update trailing stop and check if hit
        side = position.get('side', 'long')
        trailing_stop_price = self.trailing_stops.get(symbol)
        
        if trailing_stop_price:
            should_exit = False
            if side.lower() == 'long' and current_price <= trailing_stop_price:
                should_exit = True
            elif side.lower() == 'short' and current_price >= trailing_stop_price:
                should_exit = True
                
            if should_exit:
                return {
                    'type': 'market',
                    'reason': 'trailing_stop_hit',
                    'trailing_stop_price': trailing_stop_price,
                    'current_price': current_price
                }
        
        return None 