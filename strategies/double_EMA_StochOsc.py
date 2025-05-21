import pandas as pd
import pandas_ta as ta
from typing import Any, Dict, Optional
import logging

from .strategy_template import StrategyTemplate

class StrategyDoubleEMAStochOsc(StrategyTemplate):
    """
    Strategy using Double EMA crossover and Stochastic Oscillator for entry/exit signals.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 data: pd.DataFrame,
                 logger: logging.Logger):
        super().__init__(config=config, data=data, logger=logger)

    def on_init(self) -> None:
        super().on_init() # Call base on_init
        self.logger.info(f"{self.__class__.__name__} on_init called.")
        
        # Access parameters from self.config (which is the config dict passed to __init__)
        # It's assumed these parameters are now directly in the config object
        # or nested under a strategy-specific key in the main config.json
        # For consistency with StrategyExample, let's assume they are fetched like this:
        strategy_specific_params = self.config.get('strategy_configs', {}).get(self.__class__.__name__, {})

        self.ema_slow_period = strategy_specific_params.get("ema_slow_period", 150)
        self.ema_fast_period = strategy_specific_params.get("ema_fast_period", 50)
        self.stoch_k_period = strategy_specific_params.get("stoch_k_period", 5)
        self.stoch_d_period = strategy_specific_params.get("stoch_d_period", 3)
        self.stoch_slowing_period = strategy_specific_params.get("stoch_slowing_period", 3)
        # Risk parameters (sl_pct, tp_pct) are handled by get_risk_parameters
        self.stoch_overbought = strategy_specific_params.get("stoch_overbought", 80)
        self.stoch_oversold = strategy_specific_params.get("stoch_oversold", 20)
        self.time_stop_bars = strategy_specific_params.get("time_stop_bars", 30) # Used in check_exit
        self.ema_proximity_pct = strategy_specific_params.get("ema_proximity_pct", 0.003)

        # Cache risk parameters
        self.sl_pct = strategy_specific_params.get('sl_pct',
                    strategy_specific_params.get('stop_loss_pct', 0.01))
        self.tp_pct = strategy_specific_params.get('tp_pct',
                    strategy_specific_params.get('take_profit_pct', 0.02))
        if 'sl_pct' not in strategy_specific_params and 'stop_loss_pct' not in strategy_specific_params:
            self.logger.info(f"{self.__class__.__name__}: No stop loss config found for 'sl_pct' or 'stop_loss_pct'. Using default value: {self.sl_pct}")
        if 'tp_pct' not in strategy_specific_params and 'take_profit_pct' not in strategy_specific_params:
            self.logger.info(f"{self.__class__.__name__}: No take profit config found for 'tp_pct' or 'take_profit_pct'. Using default value: {self.tp_pct}")

        self.logger.info(f"{self.__class__.__name__} parameters: EMA Fast={self.ema_fast_period}, EMA Slow={self.ema_slow_period}, Stoch K={self.stoch_k_period}, D={self.stoch_d_period}, Slowing={self.stoch_slowing_period}")

        if len(self.data) < max(self.ema_slow_period, self.ema_fast_period):
            self.logger.warning("Initial data may be too short for reliable EMA calculation.")
        
        self.entry_bar_index: Optional[int] = None # This seems specific to time-based exits

    def init_indicators(self) -> None:
        """
        Initialize indicators required by the strategy.
        """
        if self.data is None or self.data.empty:
            self.logger.error(f"'{self.__class__.__name__}': Data is not available for indicator initialization.")
            # Create empty columns if data is a DataFrame, to prevent key errors later
            if isinstance(self.data, pd.DataFrame):
                for col_name in ['ema_fast', 'ema_slow', 'stoch_k', 'stoch_d']:
                    self.data[col_name] = pd.Series(dtype='float64')
            return

        if 'close' not in self.data.columns:
            self.logger.error(f"'{self.__class__.__name__}': 'close' column missing from data. Cannot init indicators.")
            # Create empty columns to prevent key errors later
            for col_name in ['ema_fast', 'ema_slow', 'stoch_k', 'stoch_d']:
                self.data[col_name] = pd.Series(dtype='float64')
            return
            
        try:
            self.logger.debug(f"Initializing indicators with data columns: {self.data.columns.tolist()} and length: {len(self.data)}")
            # EMA Fast and Slow
            self.data['ema_fast'] = self.data.ta.ema(close=self.data['close'], length=self.ema_fast_period)
            self.data['ema_slow'] = self.data.ta.ema(close=self.data['close'], length=self.ema_slow_period)

            # Stochastic Oscillator
            stoch_df = self.data.ta.stoch(
                high='high', low='low', close='close',
                k=self.stoch_k_period,
                d=self.stoch_d_period,
                smooth_k=self.stoch_slowing_period,
            )
            
            k_col_name = next((col for col in stoch_df.columns if col.lower().startswith('stochk')), None)
            d_col_name = next((col for col in stoch_df.columns if col.lower().startswith('stochd')), None)

            if k_col_name and d_col_name:
                self.data['stoch_k'] = stoch_df[k_col_name]
                self.data['stoch_d'] = stoch_df[d_col_name]
                self.logger.debug(f"Stochastic K ({k_col_name}) and D ({d_col_name}) lines added to data.")
            else:
                self.logger.error("Could not find Stochastic K or D columns in pandas_ta output.")
                # Create empty series that will contain NaN values
                self.data['stoch_k'] = pd.Series(index=self.data.index, dtype='float64')
                self.data['stoch_d'] = pd.Series(index=self.data.index, dtype='float64')

            self.logger.debug(f"{self.__class__.__name__} indicators initialized. Data length now: {len(self.data)}")
            # Log information about the warm-up period
            nan_counts = self.data[['ema_fast', 'ema_slow', 'stoch_k', 'stoch_d']].isnull().sum()
            self.logger.debug(f"Indicator warm-up periods (NaN counts): {nan_counts.to_dict()}")
            # Determine the maximum warm-up period needed
            max_warmup = max(nan_counts)
            self.logger.debug(f"Maximum warm-up period needed: {max_warmup} bars")

        except AttributeError as ae:
            if 'ta' in str(ae).lower():
                 self.logger.error(f"Pandas_TA extension 'ta' not available on DataFrame. Error: {ae}", exc_info=True)
            else:
                 self.logger.error(f"AttributeError during indicator initialization: {ae}", exc_info=True)
            # Avoid raising here if data columns were created, allow bot to proceed but strategy won't signal.
        except Exception as e:
            self.logger.error(f"Error initializing indicators: {e}", exc_info=True)
            # Avoid raising here.

    def update_indicators_for_new_row(self) -> None:
        """
        Efficiently update indicators only for the latest row in self.data.
        Called by bot.py after a new row is appended to the rolling window.
        EMAs are calculated incrementally.
        Stochastic Oscillator is calculated using pandas_ta on a tail of the data.
        """
        if self.data is None or self.data.empty or 'close' not in self.data.columns:
            self.logger.debug(f"'{self.__class__.__name__}': Data not ready for update_indicators_for_new_row (data is None, empty, or 'close' column missing).")
            return

        if len(self.data) < 2:
            self.logger.debug(f"'{self.__class__.__name__}': Not enough data for incremental update (need at least 2 rows). Current rows: {len(self.data)}. Falling back to full init_indicators.")
            # Fallback to full init for safety if data is too short, though ideally this path isn't hit often
            # if DataFetcher ensures sufficient initial data and then appends one by one.
            self.init_indicators() 
            return

        try:
            # --- Incremental EMA Calculation ---
            latest_idx = self.data.index[-1]
            prev_idx = self.data.index[-2]

            current_close = self.data.loc[latest_idx, 'close']
            
            # EMA Fast
            prev_ema_fast = self.data.loc[prev_idx, 'ema_fast']
            if pd.isna(prev_ema_fast):
                # Use SMA of first ema_fast_period closes as seed if enough data, else fallback to current close
                if len(self.data) >= self.ema_fast_period:
                    sma_seed = self.data['close'].iloc[-self.ema_fast_period:].mean()
                    self.logger.debug(f"Previous fast EMA is NaN, initializing with SMA({self.ema_fast_period}) seed: {sma_seed}")
                    prev_ema_fast = sma_seed
                else:
                    self.logger.debug(f"Previous fast EMA is NaN, not enough data for SMA seed, using current close price: {current_close}")
                    prev_ema_fast = current_close
            alpha_fast = 2 / (self.ema_fast_period + 1)
            new_ema_fast = (current_close * alpha_fast) + (prev_ema_fast * (1 - alpha_fast))
            self.data.loc[latest_idx, 'ema_fast'] = new_ema_fast

            # EMA Slow
            prev_ema_slow = self.data.loc[prev_idx, 'ema_slow']
            if pd.isna(prev_ema_slow):
                # Use SMA of first ema_slow_period closes as seed if enough data, else fallback to current close
                if len(self.data) >= self.ema_slow_period:
                    sma_seed = self.data['close'].iloc[-self.ema_slow_period:].mean()
                    self.logger.debug(f"Previous slow EMA is NaN, initializing with SMA({self.ema_slow_period}) seed: {sma_seed}")
                    prev_ema_slow = sma_seed
                else:
                    self.logger.debug(f"Previous slow EMA is NaN, not enough data for SMA seed, using current close price: {current_close}")
                    prev_ema_slow = current_close
            alpha_slow = 2 / (self.ema_slow_period + 1)
            new_ema_slow = (current_close * alpha_slow) + (prev_ema_slow * (1 - alpha_slow))
            self.data.loc[latest_idx, 'ema_slow'] = new_ema_slow
            
            # --- Stochastic Oscillator Calculation (on data tail) ---
            # Throttle recalculation to every 3 bars for performance
            if not hasattr(self, '_last_stoch_calc_bar'):
                self._last_stoch_calc_bar = -1
                self._last_stoch_values = {'k': 0.0, 'd': 0.0}
            stoch_recalc_interval = 3
            min_stoch_data_len = self.stoch_k_period + self.stoch_slowing_period + self.stoch_d_period + 5 
            current_bar = len(self.data) - 1
            if len(self.data) >= min_stoch_data_len and (current_bar % stoch_recalc_interval == 0 or self._last_stoch_calc_bar == -1):
                stoch_input_df = self.data.tail(min_stoch_data_len).copy()
                stoch_df_tail = stoch_input_df.ta.stoch(
                    high='high', low='low', close='close',
                    k=self.stoch_k_period,
                    d=self.stoch_d_period,
                    smooth_k=self.stoch_slowing_period,
                )
                if stoch_df_tail is not None and not stoch_df_tail.empty:
                    k_col_name_tail = next((col for col in stoch_df_tail.columns if col.lower().startswith('stochk')), None)
                    d_col_name_tail = next((col for col in stoch_df_tail.columns if col.lower().startswith('stochd')), None)
                    if k_col_name_tail and d_col_name_tail:
                        k_val = stoch_df_tail[k_col_name_tail].iloc[-1]
                        d_val = stoch_df_tail[d_col_name_tail].iloc[-1]
                        self.data.loc[latest_idx, 'stoch_k'] = k_val
                        self.data.loc[latest_idx, 'stoch_d'] = d_val
                        self._last_stoch_values = {'k': k_val, 'd': d_val}
                        self._last_stoch_calc_bar = current_bar
                    else:
                        self.logger.warning(f"'{self.__class__.__name__}': Could not find Stochastic K or D columns in pandas_ta output on tail data. Setting to 0 for latest row.")
                        self.data.loc[latest_idx, 'stoch_k'] = 0.0
                        self.data.loc[latest_idx, 'stoch_d'] = 0.0
                        self._last_stoch_values = {'k': 0.0, 'd': 0.0}
                        self._last_stoch_calc_bar = current_bar
                else:
                    self.logger.warning(f"'{self.__class__.__name__}': pandas_ta.stoch on tail data returned None or empty. Setting to 0 for latest row.")
                    self.data.loc[latest_idx, 'stoch_k'] = 0.0
                    self.data.loc[latest_idx, 'stoch_d'] = 0.0
                    self._last_stoch_values = {'k': 0.0, 'd': 0.0}
                    self._last_stoch_calc_bar = current_bar
            elif len(self.data) >= min_stoch_data_len:
                # Use cached values
                self.data.loc[latest_idx, 'stoch_k'] = self._last_stoch_values['k']
                self.data.loc[latest_idx, 'stoch_d'] = self._last_stoch_values['d']
            else:
                self.logger.debug(f"'{self.__class__.__name__}': Not enough data ({len(self.data)} rows) for Stoch tail calculation (need {min_stoch_data_len}). Setting Stoch K/D to 0 for latest row.")
                self.data.loc[latest_idx, 'stoch_k'] = 0.0
                self.data.loc[latest_idx, 'stoch_d'] = 0.0

        except KeyError as ke:
            self.logger.error(f"'{self.__class__.__name__}': KeyError in update_indicators_for_new_row: {ke}. This might happen if 'ema_fast', 'ema_slow' or 'close' are missing from previous rows, or if self.data was modified unexpectedly.", exc_info=True)
            # Fallback to full init if something is wrong with assumptions for incremental
            self.init_indicators()
        except Exception as e:
            self.logger.error(f"'{self.__class__.__name__}': Error in update_indicators_for_new_row: {e}", exc_info=True)
            # Fallback for other errors
            self.init_indicators()

    def _get_current_values(self) -> Optional[Dict[str, Any]]:
        """ Helper to get the latest (and previous for some fields) values from the data. """
        if self.data is None or self.data.empty:
            self.logger.debug("Data is None or empty in _get_current_values.")
            return None
        
        if len(self.data) < 2: # Need at least 2 for previous values
            self.logger.debug(f"Not enough data for current and previous values. Have {len(self.data)} rows.")
            return None

        latest = self.data.iloc[-1]
        # Use previous if available, otherwise use latest (for the very first bar after warmup)
        previous = self.data.iloc[-2] if len(self.data) >= 2 else latest # This should be fine now

        required_cols = ['close', 'ema_fast', 'ema_slow', 'stoch_k', 'stoch_d']
        if not all(col in self.data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            self.logger.warning(f"Missing one or more required columns: {missing_cols}. Available: {self.data.columns.tolist()}")
            return None
        
        # Check for NaN values in critical indicators for the latest bar
        # fillna(0) was used, so NaNs here mean the column wasn't even created properly or all data was bad.
        if latest[required_cols].isnull().any():
            self.logger.info(f"NaN values detected in latest bar indicators: {latest[required_cols].to_dict()}")
            # This might indicate an issue if fillna(0) was expected to handle it.
            # However, if source 'close'/'high'/'low' is NaN, output can be NaN.
            return None 
        
        # Check previous for NaNs for crossover logic
        prev_stoch_k_val = previous['stoch_k'] if 'stoch_k' in previous else float('nan')
        prev_stoch_d_val = previous['stoch_d'] if 'stoch_d' in previous else float('nan')

        return {
            "current_price": latest['close'],
            "ema_fast": latest['ema_fast'],
            "ema_slow": latest['ema_slow'],
            "stoch_k": latest['stoch_k'],
            "stoch_d": latest['stoch_d'],
            "prev_stoch_k": prev_stoch_k_val,
            "prev_stoch_d": prev_stoch_d_val,
            "current_bar_datetime": latest.name 
        }

    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Check entry conditions for a given symbol.
        Returns:
            dict | None: Order details if entry signal, else None.
        """
        vals = self._get_current_values()
        if not vals:
            self.logger.debug(f"{type(self).__name__}: No entry conditions met for {symbol} at {vals.get('current_bar_datetime')}.")
            return None

        current_price = vals["current_price"]
        ema_fast = vals["ema_fast"]
        ema_slow = vals["ema_slow"]
        stoch_k = vals["stoch_k"]
        stoch_d = vals["stoch_d"]
        prev_stoch_k = vals["prev_stoch_k"]
        prev_stoch_d = vals["prev_stoch_d"]

        # Check if we have all required values (not in warm-up period)
        required_values = [current_price, ema_fast, ema_slow, stoch_k, stoch_d, prev_stoch_k, prev_stoch_d]
        if any(pd.isna(val) for val in required_values):
            self.logger.info("One or more indicators still in warm-up period or missing data, skipping entry check.")
            return None

        # --- Entry Conditions ---
        ema_uptrend = ema_fast > ema_slow
        price_near_fast_ema = abs(current_price - ema_fast) < (self.ema_proximity_pct * current_price)
        stoch_k_is_oversold = stoch_k < self.stoch_oversold
        stoch_crossing_up = (prev_stoch_k < prev_stoch_d) and (stoch_k > stoch_d)
        
        long_condition = (
            ema_uptrend and
            price_near_fast_ema and
            stoch_k_is_oversold and
            stoch_crossing_up
        )

        ema_downtrend = ema_fast < ema_slow
        stoch_k_is_overbought = stoch_k > self.stoch_overbought
        stoch_crossing_down = (prev_stoch_k > prev_stoch_d) and (stoch_k < stoch_d)

        short_condition = (
            ema_downtrend and
            price_near_fast_ema and
            stoch_k_is_overbought and
            stoch_crossing_down
        )

        order_side = None
        if long_condition:
            order_side = "buy"
        elif short_condition:
            order_side = "sell"

        if order_side:
            self.log_state_change(symbol, 'entry_conditions_met', f"{type(self).__name__} for {symbol}: {order_side.upper()} entry conditions met.")
            
            strat_order_size = self.config.get('strategy_configs', {}).get(self.__class__.__name__, {}).get('order_size')
            
            order_details = {
                "side": order_side,
                "price": current_price, 
                "size": strat_order_size 
            }
            return order_details
        
        # If neither long_condition nor short_condition was met
        self.logger.debug(f"{type(self).__name__}: No entry conditions met for {symbol} at {vals.get('current_bar_datetime')} (Price: {vals.get('current_price')}).")
        return None

    def check_exit(self, symbol: str = None, **kwargs) -> bool:
        """
        Check exit conditions for an open position for a specific symbol.
        Args:
            symbol: The trading symbol (e.g., 'BTCUSDT')
        Returns:
            bool: True if exit signal, else False.
        """
        if symbol is None:
            raise ValueError(f"{self.__class__.__name__}.check_exit() requires a 'symbol' argument.")

        position_data = self.position.get(symbol)
        if not position_data:
            self.log_state_change(symbol, 'no_position_for_exit_check', f"{type(self).__name__}: No position to check for exit.")
            return False

        position_side = position_data.get('side', '').lower()
        if not position_side:
            self.logger.error(f"{self.__class__.__name__}: Cannot determine position side from position_data. Details: {position_data}. Expected 'side' key.")
            return False

        vals = self._get_current_values()
        if not vals:
            self.logger.debug("check_exit: Not enough data or NaN values from _get_current_values.")
            return False

        current_price = vals["current_price"]
        ema_fast = vals["ema_fast"]

        if pd.isna(current_price) or pd.isna(ema_fast):
            self.logger.debug("check_exit: Current price or EMA fast is NaN.")
            return False

        exit_signal = False
        reason = None

        # 1. Time-based stop
        if self.entry_bar_index is not None:
            current_df_index = len(self.data) - 1 
            num_bars_held = current_df_index - self.entry_bar_index
            if num_bars_held >= self.time_stop_bars:
                exit_signal = True
                reason = "time_stop"
                self.logger.info(f"Exit (time_stop): Position held for {num_bars_held} bars (limit: {self.time_stop_bars}). Current price: {current_price:.2f}")
        else:
            self.logger.debug("Time-based stop skipped as entry_bar_index is None.")

        # 2. Price crosses EMA against the trade direction (if not already triggered by time stop)
        if not exit_signal:
            if position_side == 'buy' and current_price < ema_fast:
                exit_signal = True
                reason = "price_cross_ema_long_exit"
                self.logger.info(f"Exit (price_cross_ema_long_exit): Price {current_price:.2f} < EMAFast {ema_fast:.2f}")
            elif position_side == 'sell' and current_price > ema_fast:
                exit_signal = True
                reason = "price_cross_ema_short_exit"
                self.logger.info(f"Exit (price_cross_ema_short_exit): Price {current_price:.2f} > EMAFast {ema_fast:.2f}")
        
        if exit_signal:
            self.log_state_change(symbol, 'exit_signal', f"{type(self).__name__}: Exit conditions met ({reason}), closing position.")
            return True
        return False

    def get_risk_parameters(self) -> Dict[str, Any]:
        """
        Return risk parameters as percentages for the strategy.
        Returns:
            dict: {"sl_pct": float_or_None, "tp_pct": float_or_None}
        """
        return {
            "sl_pct": self.sl_pct,
            "tp_pct": self.tp_pct
        }

    # Override on_order_update to set entry_bar_index
    def on_order_update(self, order_responses: Dict[str, Any], symbol: str) -> None:
        super().on_order_update(order_responses, symbol) # Call base class method first
        
        # After base class processing, if a position was opened, set entry_bar_index
        if self.position and not self.order_pending: # Position is set and order is no longer pending (i.e., filled)
            # Check if entry_bar_index was already set for this position (e.g. from a partial fill then full fill)
            # For simplicity, we set/reset it if position is confirmed.
            if self.data is not None and not self.data.empty:
                self.entry_bar_index = len(self.data) - 1
                self.logger.info(f"{self.__class__.__name__}: Position confirmed. Entry bar index set to: {self.entry_bar_index}")
            else:
                self.logger.warning(f"{self.__class__.__name__}: Could not set entry_bar_index as data is empty/None after position confirmation.")
        elif not self.position and not self.order_pending and self.active_order_id is None:
            # This case implies an order failed or was cancelled, and no position was taken.
            # Base class on_order_update handles resetting self.order_pending and self.active_order_id.
            # Ensure entry_bar_index is also reset if it was somehow set for a pending order that failed.
            if self.entry_bar_index is not None:
                self.logger.info(f"{self.__class__.__name__}: Order did not result in a position. Resetting entry_bar_index.")
                self.entry_bar_index = None

    # Override on_trade_update to reset entry_bar_index
    def on_trade_update(self, trade: Dict[str, Any], symbol: str) -> None:
        super().on_trade_update(trade, symbol) # Call base class method first
        
        # After base class processing (which clears self.position), clear entry_bar_index
        if trade.get('exit'): # If it was an exit
            if self.entry_bar_index is not None:
                self.logger.info(f"{self.__class__.__name__}: Trade closed. Resetting entry_bar_index.")
                self.entry_bar_index = None

    def on_error(self, exception: Exception) -> None:
        self.logger.error(f"Strategy {self.__class__.__name__} encountered an error: {exception}", exc_info=True)

