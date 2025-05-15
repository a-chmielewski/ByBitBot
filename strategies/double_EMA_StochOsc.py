import pandas as pd
import pandas_ta as ta
from typing import Any, Dict, Optional
import logging

from .strategy_template import StrategyTemplate

class StrategyDoubleEMAStochOsc(StrategyTemplate):
    """
    Strategy using Double EMA crossover and Stochastic Oscillator for entry/exit signals.
    """

    def __init__(self, data: pd.DataFrame, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the strategy with data, configuration, and a logger.

        Args:
            data: DataFrame containing OHLCV market data.
            config: Dictionary with strategy-specific and global parameters.
            logger: Optional logger instance for strategy-specific logging.
        """
        # Default strategy-specific parameters
        self.strategy_config = {
            "ema_slow_period": config.get("ema_slow_period", 150),
            "ema_fast_period": config.get("ema_fast_period", 50),
            "stoch_k_period": config.get("stoch_k_period", 5),
            "stoch_d_period": config.get("stoch_d_period", 3),
            "stoch_slowing_period": config.get("stoch_slowing_period", 3),
            "stop_loss_pct": config.get("stop_loss_pct", 0.01),
            "take_profit_pct": config.get("take_profit_pct", 0.02),
            "stoch_overbought": config.get("stoch_overbought", 80),
            "stoch_oversold": config.get("stoch_oversold", 20),
            "time_stop_bars": config.get("time_stop_bars", 30),
            "ema_proximity_pct": config.get("ema_proximity_pct", 0.003)
        }
        super().__init__(data, self.strategy_config, logger)
        
        self.entry_bar_index: Optional[int] = None

    def on_init(self) -> None:
        self.logger.info(f"{self.__class__.__name__} on_init called.")
        if len(self.data) < max(self.strategy_config["ema_slow_period"], self.strategy_config["ema_fast_period"]):
            self.logger.warning("Initial data may be too short for reliable EMA calculation.")

    def init_indicators(self) -> None:
        """
        Initialize indicators required by the strategy.
        This method is called by the parent StrategyTemplate's __init__.
        """
        if self.data is None or self.data.empty:
            self.logger.error("Data is not available for indicator initialization.")
            # Potentially raise an error or set a flag to prevent trading
            return

        try:
            self.logger.debug(f"Initializing indicators with data columns: {self.data.columns.tolist()} and length: {len(self.data)}")
            # EMA Fast and Slow
            self.data['ema_fast'] = self.data.ta.ema(close=self.data['close'], length=self.strategy_config["ema_fast_period"])
            self.data['ema_slow'] = self.data.ta.ema(close=self.data['close'], length=self.strategy_config["ema_slow_period"])

            # Stochastic Oscillator
            stoch_df = self.data.ta.stoch(
                high='high', low='low', close='close',
                k=self.strategy_config["stoch_k_period"],
                d=self.strategy_config["stoch_d_period"],
                smooth_k=self.strategy_config["stoch_slowing_period"]
            )
            
            # Dynamically find column names for stochastic K and D lines
            k_col_name = next((col for col in stoch_df.columns if col.lower().startswith('stochk')), None)
            d_col_name = next((col for col in stoch_df.columns if col.lower().startswith('stochd')), None)

            if k_col_name and d_col_name:
                self.data['stoch_k'] = stoch_df[k_col_name]
                self.data['stoch_d'] = stoch_df[d_col_name]
                self.logger.debug(f"Stochastic K ({k_col_name}) and D ({d_col_name}) lines added to data.")
            else:
                self.data['stoch_k'] = pd.Series(index=self.data.index, dtype='float64')
                self.data['stoch_d'] = pd.Series(index=self.data.index, dtype='float64')

            self.logger.info(f"{self.__class__.__name__} indicators initialized. Data length now: {len(self.data)}")
            if self.data[['ema_fast', 'ema_slow', 'stoch_k', 'stoch_d']].isnull().all().all():
                 self.logger.warning("All indicator columns are NaN after initialization. Check data and periods.")
            elif self.data[['ema_fast', 'ema_slow', 'stoch_k', 'stoch_d']].isnull().any().any():
                 self.logger.info("Some NaN values present in indicator columns, typically at the start of the series.")

        except AttributeError as ae:
            if 'ta' in str(ae).lower():
                 self.logger.error(f"Pandas_TA extension 'ta' not available on DataFrame. Ensure pandas_ta is correctly installed and imported. Error: {ae}", exc_info=True)
            else:
                 self.logger.error(f"AttributeError during indicator initialization: {ae}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Error initializing indicators: {e}", exc_info=True)
            raise

    def update_indicators_for_new_row(self) -> None:
        """
        Efficiently update indicators only for the latest row in self.data.
        Should be called after a new row is appended to the rolling window.
        """
        if self.data is None or self.data.empty:
            return
        idx = self.data.index[-1]
        # EMA Fast and Slow (using pandas_ta for consistency)
        fast_period = self.strategy_config["ema_fast_period"]
        slow_period = self.strategy_config["ema_slow_period"]
        # Only recalculate for the window needed for EMA
        self.data['ema_fast'] = self.data.ta.ema(length=fast_period)
        self.data['ema_slow'] = self.data.ta.ema(length=slow_period)
        # Stochastic Oscillator (calculate for the window needed)
        stoch_k_period = self.strategy_config["stoch_k_period"]
        stoch_d_period = self.strategy_config["stoch_d_period"]
        stoch_slowing_period = self.strategy_config["stoch_slowing_period"]
        stoch_df = self.data.ta.stoch(
            high='high', low='low', close='close',
            k=stoch_k_period, d=stoch_d_period, smooth_k=stoch_slowing_period
        )
        k_col_name = next((col for col in stoch_df.columns if col.lower().startswith('stochk')), None)
        d_col_name = next((col for col in stoch_df.columns if col.lower().startswith('stochd')), None)
        if k_col_name and d_col_name:
            self.data.loc[:, 'stoch_k'] = stoch_df[k_col_name]
            self.data.loc[:, 'stoch_d'] = stoch_df[d_col_name]
        else:
            self.data.loc[:, 'stoch_k'] = pd.Series(index=self.data.index, dtype='float64')
            self.data.loc[:, 'stoch_d'] = pd.Series(index=self.data.index, dtype='float64')

    def _get_current_values(self) -> Optional[Dict[str, Any]]:
        """ Helper to get the latest (and previous for some fields) values from the data. """
        if self.data is None or self.data.empty:
            self.logger.debug("Data is None or empty in _get_current_values.")
            return None
        
        # Ensure enough data for current and previous values
        if len(self.data) < 2:
            self.logger.debug(f"Not enough data for current and previous values. Have {len(self.data)} rows.")
            # Allow proceeding if only 1 row, but prev_ values will be same as current
            # return None # Stricter: requires at least 2 rows

        latest = self.data.iloc[-1]
        # Use previous if available, otherwise use latest (for the very first bar after warmup)
        previous = self.data.iloc[-2] if len(self.data) >= 2 else latest

        required_cols = ['close', 'ema_fast', 'ema_slow', 'stoch_k', 'stoch_d']
        # Check if all required columns exist in the DataFrame
        if not all(col in self.data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            self.logger.warning(f"Missing one or more required columns in data for _get_current_values: {missing_cols}. Available: {self.data.columns.tolist()}")
            return None
        
        # Check for NaN values in critical indicators for the latest bar
        if latest[required_cols].isnull().any():
            self.logger.debug(f"NaN values detected in critical indicators for the latest bar. Data: {latest[required_cols].to_dict()}")
            return None
        
        # Previous values might have NaNs if it's at the very beginning of the series after indicator calculation
        # This is often acceptable for crossover conditions.
        # if previous[required_cols].isnull().any().any() and len(self.data) >=2:
        #     self.logger.debug(f"NaN values detected in critical indicators for the previous bar. Data: {previous[required_cols].to_dict()}")
            # Depending on strategy, might return None or proceed with caution

        return {
            "current_price": latest['close'],
            "ema_fast": latest['ema_fast'],
            "ema_slow": latest['ema_slow'],
            "stoch_k": latest['stoch_k'],
            "stoch_d": latest['stoch_d'],
            "prev_stoch_k": previous['stoch_k'], # Can be NaN if len(self.data) < k_period or similar
            "prev_stoch_d": previous['stoch_d'], # Can be NaN
            "current_bar_datetime": latest.name # Assuming datetime index for self.data
        }

    def check_entry(self) -> Optional[Dict[str, Any]]:
        """
        Check entry conditions.
        Returns:
            dict | None: Order details if entry signal, else None.
                         Example: {"side": "buy", "size": 0.001, "type": "market",
                                   "stop_loss_price": 12300.0, "take_profit_price": 12400.0}
        """
        if self.position:  # Already in a position, no new entry
            self.log_state_change('in_position', f"{type(self).__name__}: In position, monitoring for exit.")
            return None

        vals = self._get_current_values()
        if not vals:
            self.log_state_change('waiting_for_entry', f"{type(self).__name__}: Waiting for entry opportunity (insufficient data).")
            self.logger.info(f"Waiting for entry: insufficient data or indicators. Latest close: {self.data['close'].iloc[-1] if not self.data.empty else 'N/A'}")
            return None

        current_price = vals["current_price"]
        ema_fast = vals["ema_fast"]
        ema_slow = vals["ema_slow"]
        stoch_k = vals["stoch_k"]
        stoch_d = vals["stoch_d"]
        prev_stoch_k = vals["prev_stoch_k"]
        prev_stoch_d = vals["prev_stoch_d"]

        # Handle potential NaN in prev_stoch values for crossover detection
        # If previous stochastic values are NaN (e.g. at the start of data series),
        # a crossover cannot be reliably determined.
        if pd.isna(prev_stoch_k) or pd.isna(prev_stoch_d):
            self.logger.info(f"Waiting for entry: prev_stoch_k or prev_stoch_d is NaN. prev_stoch_k={prev_stoch_k}, prev_stoch_d={prev_stoch_d}")
            return None 

        # --- Entry Conditions ---
        # Long entry conditions based on original logic:
        # ema_uptrend: EMA_fast > EMA_slow
        # price_at_ema: Price is near fast EMA (within proximity %)
        # stoch_oversold: Current Stochastic K is below oversold level
        # stoch_crossing_up: Stochastic K crossed above Stochastic D (K[-1] < D[-1] and K[0] > D[0])

        ema_uptrend = ema_fast > ema_slow
        price_near_fast_ema = abs(current_price - ema_fast) < (self.strategy_config["ema_proximity_pct"] * current_price)
        
        stoch_k_is_oversold = stoch_k < self.strategy_config["stoch_oversold"]
        stoch_crossing_up = (prev_stoch_k < prev_stoch_d) and (stoch_k > stoch_d)
        
        long_condition = (
            ema_uptrend and
            price_near_fast_ema and
            stoch_k_is_oversold and # Original: self.stoch.percK[-1] < self.p.stoch_oversold (prev K) and self.stoch.percK[0] > self.stoch.percD[0] (current K > current D)
                                  # The original also had stoch K[0] > D[0] which is part of crossing_up
            stoch_crossing_up
        )

        # Short entry conditions based on original logic:
        # ema_downtrend: EMA_fast < EMA_slow
        # price_at_ema: Price is near fast EMA
        # stoch_overbought: Current Stochastic K is above overbought level
        # stoch_crossing_down: Stochastic K crossed below Stochastic D (K[-1] > D[-1] and K[0] < D[0])

        ema_downtrend = ema_fast < ema_slow
        # price_near_fast_ema is the same condition for proximity
        
        stoch_k_is_overbought = stoch_k > self.strategy_config["stoch_overbought"]
        stoch_crossing_down = (prev_stoch_k > prev_stoch_d) and (stoch_k < stoch_d)

        short_condition = (
            ema_downtrend and
            price_near_fast_ema and
            stoch_k_is_overbought and # Original: self.stoch.percK[-1] > self.p.stoch_overbought (prev K) and self.stoch.percK[0] < self.stoch.percD[0] (current K < current D)
            stoch_crossing_down
        )

        # Always log the current state and entry condition components
        self.logger.debug(
            f"Waiting for entry: price={current_price}, ema_fast={ema_fast}, ema_slow={ema_slow}, "
            f"stoch_k={stoch_k}, stoch_d={stoch_d}, "
            f"ema_uptrend={ema_uptrend}, price_near_fast_ema={price_near_fast_ema}, "
            f"stoch_k_is_oversold={stoch_k_is_oversold}, stoch_crossing_up={stoch_crossing_up}, "
            f"ema_downtrend={ema_downtrend}, stoch_k_is_overbought={stoch_k_is_overbought}, stoch_crossing_down={stoch_crossing_down}"
        )

        order_details = None
        if long_condition:
            self.log_state_change('entry_signal', f"{type(self).__name__}: Long entry conditions met, placing order.")
            self.logger.info(f"Long entry signal detected at price {current_price}. EMAFast={ema_fast:.2f}, EMASlow={ema_slow:.2f}, StochK={stoch_k:.2f}, StochD={stoch_d:.2f}")
            order_details = {
                "side": "buy",
                "type": "market", 
                # Do not set a default order size here; OrderManager will enforce Bybit minimum
                "size": self.strategy_config.get("order_size"),
                "price": current_price
            }

        elif short_condition:
            self.log_state_change('entry_signal', f"{type(self).__name__}: Short entry conditions met, placing order.")
            self.logger.info(f"Short entry signal detected at price {current_price}. EMAFast={ema_fast:.2f}, EMASlow={ema_slow:.2f}, StochK={stoch_k:.2f}, StochD={stoch_d:.2f}")
            order_details = {
                "side": "sell",
                "type": "market",
                # Do not set a default order size here; OrderManager will enforce Bybit minimum
                "size": self.strategy_config.get("order_size"),
                "price": current_price
            }
        else:
            self.log_state_change('waiting_for_entry', f"{type(self).__name__}: Waiting for entry opportunity.")
        if order_details:
            risk_params = self.get_risk_parameters(current_price=current_price, side=order_details["side"])
            # Map risk_params keys to 'stop_loss' and 'take_profit' for bot.py compatibility
            if "stop_loss_price" in risk_params:
                order_details["stop_loss"] = risk_params["stop_loss_price"]
            if "take_profit_price" in risk_params:
                order_details["take_profit"] = risk_params["take_profit_price"]
            self.logger.info(f"Prepared order: {order_details}")

        return order_details

    def check_exit(self, position: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check exit conditions for an open position.
        Args:
            position: Current open position details (passed by the bot, should match self.position).
                      Example: {"side": "buy", "entry_price": 12345.6, "size": 1.0, 
                                "order_id": "...", "entry_timestamp": "..."}
        Returns:
            dict | None: Exit order details if exit signal, else None.
                         Example: {"reason": "time_stop", "size": 1.0, "type": "market", "side": "sell"}
        """
        if not self.position: # No active position to exit
            # This check might be redundant if bot only calls check_exit when a position exists,
            # but good for robustness.
            self.log_state_change('waiting_for_entry', f"{type(self).__name__}: Waiting for entry opportunity.")
            return None
        
        # Ensure the passed 'position' arg matches the strategy's internal state if needed for consistency checks
        # For now, we rely on self.position and self.entry_bar_index which are set by on_order_update
        if not self.entry_bar_index:
            self.logger.warning("check_exit: self.entry_bar_index is not set. Cannot evaluate time-based exit.")
            # Decide if to proceed with other exit checks or return. For now, let's allow other checks.

        vals = self._get_current_values()
        if not vals:
            self.logger.debug("check_exit: Not enough data or NaN values from _get_current_values.")
            return None

        current_price = vals["current_price"]
        ema_fast = vals["ema_fast"]
        # current_bar_datetime = vals["current_bar_datetime"] # If using datetime for bar counting
        # current_bar_df_index = self.data.index.get_loc(current_bar_datetime) # If self.data.index is DatetimeIndex

        exit_signal = False
        reason = None

        # 1. Time-based stop
        if self.entry_bar_index is not None:
            # Assuming self.entry_bar_index is the integer index of the entry bar in the self.data DataFrame
            # And self.data is continuously updated, so len(self.data)-1 is the current bar index.
            current_df_index = len(self.data) - 1 
            num_bars_held = current_df_index - self.entry_bar_index
            
            if num_bars_held >= self.strategy_config["time_stop_bars"]:
                exit_signal = True
                reason = "time_stop"
                self.logger.info(f"Exit (time_stop): Position held for {num_bars_held} bars (limit: {self.strategy_config['time_stop_bars']}). Current price: {current_price}")
        else:
            self.logger.debug("Time-based stop skipped as entry_bar_index is not set.")

        # 2. Price crosses EMA against the trade direction (if not already triggered by time stop)
        if not exit_signal:
            if self.position['side'] == 'buy' and current_price < ema_fast:
                exit_signal = True
                reason = "price_cross_ema_long_exit"
                self.logger.info(f"Exit (price_cross_ema_long_exit): Price {current_price:.2f} < EMAFast {ema_fast:.2f}")
            elif self.position['side'] == 'sell' and current_price > ema_fast:
                exit_signal = True
                reason = "price_cross_ema_short_exit"
                self.logger.info(f"Exit (price_cross_ema_short_exit): Price {current_price:.2f} > EMAFast {ema_fast:.2f}")
        
        if exit_signal:
            self.log_state_change('exit_signal', f"{type(self).__name__}: Exit conditions met ({reason}), closing position.")
            # Determine side for the closing order
            close_side = "sell" if self.position['side'] == 'buy' else "buy"
            return {
                "reason": reason,
                "size": self.position.get("size"),  # Exit the full size of the current position
                "type": "market",
                "side": close_side
                # Add order_id of the position to close if OrderManager needs it: "position_order_id": self.position.get("order_id")
            }
            
        else:
            self.log_state_change('in_position', f"{type(self).__name__}: In position, monitoring for exit.")
            return None

    def get_risk_parameters(self, current_price: float, side: str) -> Dict[str, Any]:
        """
        Return risk parameters (stop-loss price, take-profit price) for the strategy.
        These are determined at the point of entry based on percentages.
        Args:
            current_price: The price at which the entry is being considered.
            side: "buy" or "sell", indicating the direction of the trade.
        Returns:
            dict: {"stop_loss_price": float, "take_profit_price": float}
                  Returns an empty dict if side is invalid.
        """
        sl_pct = self.strategy_config["stop_loss_pct"]
        tp_pct = self.strategy_config["take_profit_pct"]
        
        # The original strategy's backtest code used swing_high/low for SL in some cases.
        # This refactored version uses percentage-based SL/TP from the entry price
        # as per the simplified parameters derived from the original params.
        # If swing-based SL/TP is desired, self.data would need to be analyzed here
        # to find recent swing points, or they'd need to be tracked continuously.

        stop_loss_price = None
        take_profit_price = None

        if side == "buy":
            stop_loss_price = current_price * (1 - sl_pct)
            take_profit_price = current_price * (1 + tp_pct)
        elif side == "sell":
            stop_loss_price = current_price * (1 + sl_pct)
            take_profit_price = current_price * (1 - tp_pct)
        else:
            self.logger.error(f"Invalid side '{side}' provided for get_risk_parameters. Cannot calculate SL/TP.")
            return {} # Return empty if side is unrecognized

        # Rounding to a reasonable number of decimal places (e.g., 8 for crypto)
        # The exact precision might depend on the trading pair.
        # Consider making precision configurable or deriving from exchange info.
        price_precision = self.strategy_config.get("price_precision", 8)

        return {
            "stop_loss_price": round(stop_loss_price, price_precision),
            "take_profit_price": round(take_profit_price, price_precision)
        }

    def on_order_update(self, order: Dict[str, Any]) -> None:
        self.logger.debug(f"on_order_update received: {order}")
        # Basic implementation, will be detailed later
        order_status = order.get('status', order.get('main_order', {}).get('result', {}).get('order_status'))
        if order_status in ('Filled', 'filled', 'FILLED'):
            if not self.position:
                self.position = {
                    "side": order.get("side"),
                    "entry_price": order.get("price", order.get("average")),
                    "size": order.get("filled", order.get("amount")),
                    "entry_timestamp": order.get("timestamp", pd.Timestamp.now(tz='UTC').isoformat()),
                    "order_id": order.get("id")
                }
                if self.data is not None and not self.data.empty:
                    self.entry_bar_index = len(self.data) - 1
                    self.logger.info(f"Position opened: {self.position}. Entry bar index: {self.entry_bar_index}")
                else:
                    self.logger.warning("Could not set entry_bar_index as data is empty/None.")
        elif order_status in ('Canceled', 'canceled', 'Rejected', 'rejected', 'Expired', 'expired'):
            self.logger.info(f"Order {order.get('id')} ended without fill: {order_status}")


    def on_trade_update(self, trade: Dict[str, Any]) -> None:
        self.logger.debug(f"on_trade_update received: {trade}")
        if trade.get('status') == 'closed' or trade.get('exit_order') or trade.get('exit'):
            self.logger.info(f"Trade closed. PnL: {trade.get('pnl')}, Reason: {trade.get('exit_reason', 'N/A')}. Resetting position.")
            self.position = None
            self.entry_bar_index = None

    def on_error(self, exception: Exception) -> None:
        self.logger.error(f"Strategy {self.__class__.__name__} encountered an error: {exception}", exc_info=True)

