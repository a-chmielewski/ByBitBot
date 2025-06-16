import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Any, Dict, Optional, List
import logging

from .strategy_template import StrategyTemplate

class StrategyBreakoutAndRetest(StrategyTemplate):
    """Breakout and Retest Strategy - Captures transitional trend changes.

    **STRATEGY MATRIX ROLE**: Active strategy for trending markets with mixed 1-minute conditions
    **MATRIX USAGE**: 
    - TRENDING(5m) + RANGING(1m) → 1-minute execution
    - TRENDING(5m) + TRANSITIONAL(1m) → 1-minute execution
    **EXECUTION TIMEFRAME**: 1-minute for precision entry timing

    Strategy waits for:
    1. Clear breakout of support/resistance with volume confirmation
    2. Retest of the broken level
    3. Reversal confirmation at the retest level
    4. Entry on bounce from retest with tight stop loss

    Market Type:
        Transitional trend change – e.g. a market that has been trending on 5-minute timeframe
        but showing ranging or transitional behavior on 1-minute timeframe. This strategy captures
        the continuation of the 5-minute trend by entering on precise 1-minute retest levels.
        Perfect for trending markets that need precision timing due to short-term consolidation.

    Indicators & Parameters:
        - Support/Resistance levels (drawn from recent highs/lows)
        - Volume to confirm breakout strength on the initial break
        - Trend indicator (50-period EMA or Supertrend on 5-min) for bigger trend bias
        - Momentum oscillator (e.g. RSI) for entry timing

    Entry Conditions:
        1. Initial breakout:
           - Clear breakout of a key level (e.g. price breaks above $250)
           - Large candle with high volume
           - Wait for retest rather than chasing immediately

        2. Retest entry:
           - Price retraces to breakout level (old resistance → new support)
           - Look for bullish reversal signs (double bottom, dojis, engulfing candle)
           - Enter on bounce with decent uptick
           - For bearish breakout/retest, invert the logic

        3. Indicator filter:
           - RSI should stay above 40 in uptrend pullback
           - Monitor ADX for momentum strength

        4. Volume:
           - Declining volume on pullback
           - Volume pickup on bounce
           - High volume on deep pullback suggests breakout failure

    Exit Conditions:
        1. Trend continuation:
           - Ride until clear reversal pattern
           - Target next major resistance
           - Use measured move concept (prior range height)

        2. Partial exits:
           - Take partial at +0.5% or structural level
           - Trail stop on remainder
           - Consider breakeven stop after initial profit

    Stop Loss / Take Profit:
        Stop Loss:
            - Place just below retested level (e.g. $248-249 for $250 level)
            - Tight enough to avoid noise but catch failures
            - Invert for short positions

        Take Profit:
            - Target next resistance level
            - Scale out at 1R (e.g. $254 for $251 entry with $248 stop)
            - Trail remainder with breakeven stop

    Notes:
        - High success rate due to confirmation wait
        - Good reward-to-risk ratio
        - Not all breakouts retest
        - Suitable for high leverage due to tight stops
        - Avoid trading during major news
        - Wait for clear entry signals
        - Focus on clear levels and patient confirmation
    """
    
    # Market type tags indicating this strategy works for trending markets with mixed 1-min conditions
    MARKET_TYPE_TAGS: List[str] = ['TRENDING', 'RANGING', 'TRANSITIONAL']
    
    # Strategy Matrix integration
    SHOW_IN_SELECTION: bool = True  # Available for manual selection and automatic matrix selection

    def __init__(self,
                 data: pd.DataFrame,
                 config: Dict[str, Any],
                 logger: logging.Logger):
        super().__init__(data=data, config=config, logger=logger)

    def on_init(self) -> None:
        super().on_init()
        self.logger.info(f"{self.__class__.__name__} on_init called.")
        
        # Get strategy-specific parameters from config
        strategy_specific_params = self.config.get('strategy_configs', {}).get(self.__class__.__name__, {})

        # Support/Resistance parameters
        self.sr_lookback_period = strategy_specific_params.get("sr_lookback_period", 30)
        self.sr_min_touches = strategy_specific_params.get("sr_min_touches", 3)
        self.sr_tolerance_pct = strategy_specific_params.get("sr_tolerance_pct", 0.002)  # 0.2%
        
        # Breakout parameters
        self.breakout_min_pct = strategy_specific_params.get("breakout_min_pct", 0.003)  # 0.3%
        self.volume_breakout_multiplier = strategy_specific_params.get("volume_breakout_multiplier", 1.5)
        self.volume_avg_period = strategy_specific_params.get("volume_avg_period", 20)
        
        # Trend filter parameters
        self.ema_trend_period = strategy_specific_params.get("ema_trend_period", 50)
        self.use_trend_filter = strategy_specific_params.get("use_trend_filter", True)
        
        # Momentum filter parameters
        self.rsi_period = strategy_specific_params.get("rsi_period", 14)
        self.rsi_pullback_min = strategy_specific_params.get("rsi_pullback_min", 40)
        self.rsi_pullback_max = strategy_specific_params.get("rsi_pullback_max", 60)
        
        # Retest parameters
        self.retest_timeout_bars = strategy_specific_params.get("retest_timeout_bars", 15)
        self.retest_tolerance_pct = strategy_specific_params.get("retest_tolerance_pct", 0.002)  # 0.2%
        self.reversal_confirmation_bars = strategy_specific_params.get("reversal_confirmation_bars", 2)
        
        # Pattern recognition parameters
        self.engulfing_min_ratio = strategy_specific_params.get("engulfing_min_ratio", 1.2)
        self.hammer_ratio = strategy_specific_params.get("hammer_ratio", 2.0)
        
        # Risk parameters
        self.stop_loss_buffer_pct = strategy_specific_params.get("stop_loss_buffer_pct", 0.001)  # 0.1%
        self.first_target_pct = strategy_specific_params.get("first_target_pct", 0.005)  # 0.5%
        self.measured_move_multiplier = strategy_specific_params.get("measured_move_multiplier", 1.0)
        
        # Volume decline threshold during pullback
        self.min_breakout_volume_decline = strategy_specific_params.get("min_breakout_volume_decline", 0.7)

        # Cache risk parameters for get_risk_parameters()
        self.sl_pct = strategy_specific_params.get('sl_pct', 0.01)  # 1% default
        self.tp_pct = strategy_specific_params.get('tp_pct', 0.02)  # 2% default

        # State tracking variables
        self.support_levels = []
        self.resistance_levels = []
        self.last_sr_update = 0
        
        # Breakout tracking
        self.breakout_detected = False
        self.breakout_direction = None
        self.breakout_level = 0
        self.breakout_price = 0
        self.breakout_bar = None
        self.breakout_volume = 0
        self.range_height = 0
        
        # Retest tracking
        self.waiting_for_retest = False
        self.retest_level = 0
        self.retest_confirmed = False
        self.reversal_start_bar = None

        self.logger.info(f"{self.__class__.__name__} parameters: SR_lookback={self.sr_lookback_period}, "
                        f"Volume_mult={self.volume_breakout_multiplier}, EMA_trend={self.ema_trend_period}")

    def init_indicators(self) -> None:
        """Initialize indicators required by the strategy."""
        if self.data is None or self.data.empty:
            self.logger.error(f"'{self.__class__.__name__}': Data is not available for indicator initialization.")
            return

        required_cols = ['close', 'high', 'low', 'volume']
        if not all(col in self.data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            self.logger.error(f"'{self.__class__.__name__}': Missing required columns: {missing_cols}")
            return
            
        try:
            self.logger.debug(f"Starting indicator initialization with data shape: {self.data.shape}")
            self.logger.debug(f"Data columns before initialization: {list(self.data.columns)}")
            self.logger.debug(f"Data index: {self.data.index.tolist()[:5]}... (showing first 5)")
            
            # Volume SMA for breakout confirmation
            self.data['volume_sma'] = self.data['volume'].rolling(window=self.volume_avg_period, min_periods=1).mean()
            self.logger.debug(f"Volume SMA created successfully with {self.data['volume_sma'].notna().sum()} valid values")
            self.logger.debug(f"Data columns after volume_sma: {list(self.data.columns)}")
            
            # EMA for trend bias (if enabled)
            if self.use_trend_filter:
                try:
                    self.logger.debug(f"Attempting EMA calculation with period {self.ema_trend_period}")
                    self.data['ema_trend'] = self.data.ta.ema(close=self.data['close'], length=self.ema_trend_period)
                    if 'ema_trend' not in self.data.columns or self.data['ema_trend'].isna().all():
                        self.logger.warning(f"EMA calculation failed, creating manual EMA for trend bias")
                        # Fallback to manual EMA calculation
                        self.data['ema_trend'] = self.data['close'].ewm(span=self.ema_trend_period, adjust=False).mean()
                    self.logger.debug(f"EMA trend created successfully with {self.data['ema_trend'].notna().sum()} valid values")
                except Exception as ema_error:
                    self.logger.error(f"Error calculating EMA trend: {ema_error}")
                    # Fallback to manual EMA calculation
                    self.data['ema_trend'] = self.data['close'].ewm(span=self.ema_trend_period, adjust=False).mean()
                    self.logger.info(f"Using fallback EMA calculation")
            
            # RSI for momentum filter
            try:
                self.logger.debug(f"Attempting RSI calculation with period {self.rsi_period}")
                self.data['rsi'] = self.data.ta.rsi(close=self.data['close'], length=self.rsi_period)
                if 'rsi' not in self.data.columns or self.data['rsi'].isna().all():
                    self.logger.warning(f"RSI calculation failed, using default values")
                    self.data['rsi'] = 50.0  # Neutral default
                self.logger.debug(f"RSI created successfully with {self.data['rsi'].notna().sum()} valid values")
            except Exception as rsi_error:
                self.logger.error(f"Error calculating RSI: {rsi_error}")
                self.data['rsi'] = 50.0  # Neutral default
            
            # Initialize support/resistance tracking columns
            self.logger.debug("Creating support/resistance tracking columns")
            self.data['current_support'] = pd.Series(dtype='float64', index=self.data.index)
            self.data['current_resistance'] = pd.Series(dtype='float64', index=self.data.index)
            self.logger.debug(f"Support/resistance columns created")
            
            # Update initial support/resistance levels
            self.logger.debug("Updating initial support/resistance levels")
            self._update_support_resistance_levels()

            # Verify all required columns exist
            expected_cols = ['volume_sma', 'rsi', 'current_support', 'current_resistance']
            if self.use_trend_filter:
                expected_cols.append('ema_trend')
            
            self.logger.debug(f"Data columns after full initialization: {list(self.data.columns)}")
            self.logger.debug(f"Expected columns: {expected_cols}")
            
            missing_after_init = [col for col in expected_cols if col not in self.data.columns]
            if missing_after_init:
                self.logger.error(f"Indicator initialization incomplete, missing columns: {missing_after_init}")
                # Let's also check if columns exist but are all NaN
                for col in expected_cols:
                    if col in self.data.columns:
                        nan_count = self.data[col].isna().sum()
                        total_count = len(self.data[col])
                        self.logger.debug(f"Column '{col}': {nan_count}/{total_count} are NaN")
            else:
                self.logger.debug(f"{self.__class__.__name__} indicators initialized successfully.")
                # Log sample values from the last row
                if len(self.data) > 0:
                    last_row = self.data.iloc[-1]
                    sample_values = {col: last_row.get(col, 'N/A') for col in expected_cols}
                    self.logger.debug(f"Sample indicator values from last row: {sample_values}")
            
        except Exception as e:
            self.logger.error(f"Error initializing indicators: {e}", exc_info=True)

    def update_indicators_for_new_row(self) -> None:
        """Update indicators for the latest row efficiently."""
        if self.data is None or self.data.empty or len(self.data) < 2:
            self.logger.debug(f"'{self.__class__.__name__}': Not enough data for incremental update.")
            return

        try:
            # Check if required columns exist, if not, fall back to full initialization
            required_indicator_cols = ['volume_sma', 'rsi', 'current_support', 'current_resistance']
            if self.use_trend_filter:
                required_indicator_cols.append('ema_trend')
            
            self.logger.debug(f"'{self.__class__.__name__}': Checking for required columns: {required_indicator_cols}")
            self.logger.debug(f"'{self.__class__.__name__}': Current data columns: {list(self.data.columns)}")
            
            missing_cols = [col for col in required_indicator_cols if col not in self.data.columns]
            if missing_cols:
                self.logger.warning(f"'{self.__class__.__name__}': Missing indicator columns {missing_cols}, falling back to full initialization.")
                self.logger.debug(f"'{self.__class__.__name__}': Data shape before init fallback: {self.data.shape}")
                self.init_indicators()
                self.logger.debug(f"'{self.__class__.__name__}': Data shape after init fallback: {self.data.shape}")
                self.logger.debug(f"'{self.__class__.__name__}': Data columns after init fallback: {list(self.data.columns)}")
                return
            
            self.logger.debug(f"'{self.__class__.__name__}': All required columns present, proceeding with incremental update")
            latest_idx = self.data.index[-1]
            
            # Update volume SMA incrementally
            if len(self.data) >= self.volume_avg_period:
                volume_window = self.data['volume'].iloc[-self.volume_avg_period:].mean()
                self.data.loc[latest_idx, 'volume_sma'] = volume_window
            else:
                self.data.loc[latest_idx, 'volume_sma'] = self.data['volume'].iloc[-len(self.data):].mean()
            
            # Update EMA trend incrementally
            if self.use_trend_filter:
                prev_idx = self.data.index[-2]
                current_close = self.data.loc[latest_idx, 'close']
                prev_ema = self.data.loc[prev_idx, 'ema_trend']
                
                if pd.isna(prev_ema):
                    if len(self.data) >= self.ema_trend_period:
                        prev_ema = self.data['close'].iloc[-self.ema_trend_period:].mean()
                    else:
                        prev_ema = current_close
                        
                alpha = 2 / (self.ema_trend_period + 1)
                new_ema = (current_close * alpha) + (prev_ema * (1 - alpha))
                self.data.loc[latest_idx, 'ema_trend'] = new_ema
            
            # Update RSI (recalculate on tail for accuracy)
            min_rsi_data = self.rsi_period + 5
            if len(self.data) >= min_rsi_data:
                rsi_tail = self.data.tail(min_rsi_data).ta.rsi(close='close', length=self.rsi_period)
                if rsi_tail is not None and not rsi_tail.empty:
                    self.data.loc[latest_idx, 'rsi'] = rsi_tail.iloc[-1]
                else:
                    self.data.loc[latest_idx, 'rsi'] = 50.0  # neutral default
            else:
                self.data.loc[latest_idx, 'rsi'] = 50.0
            
            # Update support/resistance levels periodically
            current_bar = len(self.data) - 1
            if current_bar - self.last_sr_update >= 10:  # Update every 10 bars
                self._update_support_resistance_levels()
                
        except Exception as e:
            self.logger.error(f"'{self.__class__.__name__}': Error in update_indicators_for_new_row: {e}", exc_info=True)

    def _update_support_resistance_levels(self) -> None:
        """Update support and resistance levels based on recent price action."""
        if len(self.data) < self.sr_lookback_period:
            return
            
        self.last_sr_update = len(self.data) - 1
        
        try:
            # Get recent highs and lows for analysis
            recent_data = self.data.tail(self.sr_lookback_period)
            recent_highs = recent_data['high'].tolist()
            recent_lows = recent_data['low'].tolist()
            
            # Find resistance levels (cluster of highs)
            self.resistance_levels = self._find_levels(recent_highs, 'resistance')
            
            # Find support levels (cluster of lows)
            self.support_levels = self._find_levels(recent_lows, 'support')
            
            # Update data columns with current primary levels
            latest_idx = self.data.index[-1]
            if self.support_levels:
                self.data.loc[latest_idx, 'current_support'] = self.support_levels[0]['level']
            if self.resistance_levels:
                self.data.loc[latest_idx, 'current_resistance'] = self.resistance_levels[0]['level']
                
        except Exception as e:
            self.logger.error(f"Error updating support/resistance levels: {e}")

    def _find_levels(self, prices: List[float], level_type: str) -> List[Dict[str, Any]]:
        """Find support or resistance levels from price clusters."""
        levels = []
        price_clusters = {}
        
        for price in prices:
            found_cluster = False
            for cluster_price in price_clusters:
                if abs(price - cluster_price) / cluster_price < self.sr_tolerance_pct:
                    price_clusters[cluster_price] += 1
                    found_cluster = True
                    break
            if not found_cluster:
                price_clusters[price] = 1
        
        # Filter levels with minimum touches
        for price, touches in price_clusters.items():
            if touches >= self.sr_min_touches:
                levels.append({'level': price, 'touches': touches, 'strength': touches})
        
        # Sort by strength (most touches first)
        levels.sort(key=lambda x: x['strength'], reverse=True)
        
        return levels[:5]  # Keep top 5 levels

    def _get_current_values(self) -> Optional[Dict[str, Any]]:
        """Helper to get current market values for analysis."""
        if self.data is None or self.data.empty or len(self.data) < 2:
            return None

        latest = self.data.iloc[-1]
        previous = self.data.iloc[-2]

        required_cols = ['close', 'high', 'low', 'volume']
        if not all(col in self.data.columns for col in required_cols):
            return None

        return {
            "current_price": latest['close'],
            "current_high": latest['high'],
            "current_low": latest['low'],
            "current_volume": latest['volume'],
            "volume_sma": latest.get('volume_sma', 0),
            "ema_trend": latest.get('ema_trend'),
            "rsi": latest.get('rsi', 50),
            "prev_close": previous['close'],
            "current_bar_datetime": latest.name
        }

    def _detect_breakout(self, vals: Dict[str, Any]) -> tuple:
        """Detect breakout of key support/resistance levels."""
        if not (self.support_levels or self.resistance_levels):
            return False, None, None
            
        current_price = vals["current_price"]
        current_high = vals["current_high"]
        current_low = vals["current_low"]
        current_volume = vals["current_volume"]
        avg_volume = vals["volume_sma"]
        
        # Check volume confirmation
        if avg_volume > 0 and current_volume < avg_volume * self.volume_breakout_multiplier:
            return False, None, None
        
        # Check resistance breakouts (bullish)
        for resistance in self.resistance_levels:
            level = resistance['level']
            if (current_high > level * (1 + self.breakout_min_pct) and 
                current_price > level):
                
                range_height = self._calculate_range_height(level, 'resistance')
                return True, 'long', {'level': level, 'range_height': range_height}
        
        # Check support breakdowns (bearish)
        for support in self.support_levels:
            level = support['level']
            if (current_low < level * (1 - self.breakout_min_pct) and 
                current_price < level):
                
                range_height = self._calculate_range_height(level, 'support')
                return True, 'short', {'level': level, 'range_height': range_height}
        
        return False, None, None

    def _calculate_range_height(self, broken_level: float, level_type: str) -> float:
        """Calculate the height of the range for measured move targets."""
        if level_type == 'resistance':
            # Find nearest support below
            nearest_support = 0
            for support in self.support_levels:
                if support['level'] < broken_level:
                    nearest_support = max(nearest_support, support['level'])
            return broken_level - nearest_support if nearest_support > 0 else broken_level * 0.02
        else:
            # Find nearest resistance above
            nearest_resistance = float('inf')
            for resistance in self.resistance_levels:
                if resistance['level'] > broken_level:
                    nearest_resistance = min(nearest_resistance, resistance['level'])
            return nearest_resistance - broken_level if nearest_resistance < float('inf') else broken_level * 0.02

    def _check_trend_bias(self, direction: str, vals: Dict[str, Any]) -> bool:
        """Check if breakout aligns with trend bias."""
        if not self.use_trend_filter or vals["ema_trend"] is None:
            return True
            
        current_price = vals["current_price"]
        ema_value = vals["ema_trend"]
        
        if pd.isna(ema_value):
            return True  # Allow if EMA not ready yet
        
        if direction == 'long':
            return current_price > ema_value
        else:
            return current_price < ema_value

    def _detect_retest(self, vals: Dict[str, Any]) -> bool:
        """Detect retest of the broken level."""
        if not self.waiting_for_retest:
            return False
            
        current_price = vals["current_price"]
        current_high = vals["current_high"]
        current_low = vals["current_low"]
        current_volume = vals["current_volume"]
        
        # Check if price has returned to retest level
        if self.breakout_direction == 'long':
            # For bullish breakout, retest from above
            retest_occurring = (current_low <= self.retest_level * (1 + self.retest_tolerance_pct) and
                              current_price >= self.retest_level * (1 - self.retest_tolerance_pct))
        else:
            # For bearish breakout, retest from below
            retest_occurring = (current_high >= self.retest_level * (1 - self.retest_tolerance_pct) and
                              current_price <= self.retest_level * (1 + self.retest_tolerance_pct))
        
        if retest_occurring:
            # Check volume characteristics during pullback
            volume_ok = current_volume < self.breakout_volume * self.min_breakout_volume_decline
            
            # Check RSI during pullback
            rsi_ok = self._check_rsi_pullback(vals)
            
            return volume_ok and rsi_ok
        
        return False

    def _check_rsi_pullback(self, vals: Dict[str, Any]) -> bool:
        """Check RSI behavior during pullback."""
        current_rsi = vals["rsi"]
        
        if pd.isna(current_rsi):
            return True  # Allow if RSI not ready
            
        if self.breakout_direction == 'long':
            return current_rsi > self.rsi_pullback_min
        else:
            return current_rsi < self.rsi_pullback_max

    def _detect_reversal_confirmation(self, vals: Dict[str, Any]) -> bool:
        """Detect reversal patterns at retest level."""
        if not self.retest_confirmed:
            return False
            
        if len(self.data) < 2:
            return False
        
        # Check for engulfing pattern
        if self._is_engulfing_pattern():
            return True
            
        # Check for hammer/doji pattern
        if self._is_hammer_or_doji():
            return True
            
        # Check for multi-bar confirmation
        if self._is_multi_bar_reversal(vals):
            return True
        
        return False

    def _is_engulfing_pattern(self) -> bool:
        """Check for engulfing candlestick pattern."""
        if len(self.data) < 2:
            return False
            
        latest = self.data.iloc[-1]
        previous = self.data.iloc[-2]
        
        prev_open = previous.get('open', previous['close'])
        prev_close = previous['close']
        curr_open = latest.get('open', latest['close'])
        curr_close = latest['close']
        
        prev_body = abs(prev_close - prev_open)
        curr_body = abs(curr_close - curr_open)
        
        if curr_body < prev_body * self.engulfing_min_ratio:
            return False
        
        if self.breakout_direction == 'long':
            # Bullish engulfing
            return (prev_close < prev_open and curr_close > curr_open and
                    curr_open < prev_close and curr_close > prev_open)
        else:
            # Bearish engulfing
            return (prev_close > prev_open and curr_close < curr_open and
                    curr_open > prev_close and curr_close < prev_open)

    def _is_hammer_or_doji(self) -> bool:
        """Check for hammer or doji patterns."""
        if len(self.data) < 1:
            return False
            
        latest = self.data.iloc[-1]
        curr_open = latest.get('open', latest['close'])
        curr_close = latest['close']
        curr_high = latest['high']
        curr_low = latest['low']
        
        body_size = abs(curr_close - curr_open)
        total_range = curr_high - curr_low
        
        if total_range == 0:
            return False
            
        body_ratio = body_size / total_range
        
        if self.breakout_direction == 'long':
            # Hammer: small body, long lower wick
            lower_wick = min(curr_open, curr_close) - curr_low
            upper_wick = curr_high - max(curr_open, curr_close)
            return (body_ratio < 0.3 and lower_wick > body_size * self.hammer_ratio and
                    upper_wick < body_size)
        else:
            # Inverted hammer: small body, long upper wick
            lower_wick = min(curr_open, curr_close) - curr_low
            upper_wick = curr_high - max(curr_open, curr_close)
            return (body_ratio < 0.3 and upper_wick > body_size * self.hammer_ratio and
                    lower_wick < body_size)

    def _is_multi_bar_reversal(self, vals: Dict[str, Any]) -> bool:
        """Check for multiple bar reversal confirmation."""
        if self.reversal_start_bar is None:
            return False
            
        bars_since_reversal = len(self.data) - self.reversal_start_bar
        
        if bars_since_reversal >= self.reversal_confirmation_bars:
            current_price = vals["current_price"]
            
            if self.breakout_direction == 'long':
                return current_price > self.retest_level
            else:
                return current_price < self.retest_level
        
        return False

    def _reset_breakout_state(self) -> None:
        """Reset breakout and retest state variables."""
        self.breakout_detected = False
        self.breakout_direction = None
        self.waiting_for_retest = False
        self.retest_confirmed = False
        self.reversal_start_bar = None

    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check entry conditions for the breakout and retest strategy."""
        vals = self._get_current_values()
        if not vals:
            return None

        current_bar = len(self.data) - 1
        
        if not self.waiting_for_retest:
            # Step 1: Look for breakouts
            breakout_occurred, direction, breakout_info = self._detect_breakout(vals)
            
            if breakout_occurred and self._check_trend_bias(direction, vals):
                self.breakout_detected = True
                self.breakout_direction = direction
                self.breakout_level = breakout_info['level']
                self.breakout_price = vals["current_price"]
                self.breakout_bar = current_bar
                self.breakout_volume = vals["current_volume"]
                self.range_height = breakout_info['range_height']
                
                # Set up retest waiting
                self.waiting_for_retest = True
                self.retest_level = self.breakout_level
                
                self.logger.info(f"{self.__class__.__name__}: Breakout detected - {direction} at {self.breakout_level}")
                
        else:
            # Step 2: Check for retest
            if self._detect_retest(vals):
                self.retest_confirmed = True
                self.reversal_start_bar = current_bar
                self.logger.info(f"{self.__class__.__name__}: Retest confirmed at level {self.retest_level}")
                
            # Step 3: Look for reversal confirmation
            if self.retest_confirmed and self._detect_reversal_confirmation(vals):
                # Enter position on confirmed reversal
                self.logger.info(f"{self.__class__.__name__}: Reversal confirmation detected - preparing {self.breakout_direction} entry")
                
                order_side = "buy" if self.breakout_direction == 'long' else "sell"
                
                strat_order_size = self.config.get('strategy_configs', {}).get(self.__class__.__name__, {}).get('order_size')
                
                order_details = {
                    "side": order_side,
                    "price": vals["current_price"],
                    "size": strat_order_size
                }
                
                # Reset state after entry signal
                self._reset_breakout_state()
                
                return order_details
            
            # Timeout retest waiting
            if (self.breakout_bar is not None and 
                current_bar - self.breakout_bar > self.retest_timeout_bars):
                self.logger.info(f"{self.__class__.__name__}: Retest timeout - resetting state")
                self._reset_breakout_state()
        
        return None

    def check_exit(self, symbol: str) -> bool:
        """Check for strategy-specific exit conditions."""
        # This strategy primarily relies on SL/TP for exits
        # Could add advanced exit logic here if needed
        return False

    def get_risk_parameters(self) -> Dict[str, Any]:
        """Return risk parameters for the strategy."""
        return {
            "sl_pct": self.sl_pct,
            "tp_pct": self.tp_pct
        }

    def on_error(self, exception: Exception) -> None:
        """Handle strategy errors."""
        self.logger.error(f"Strategy {self.__class__.__name__} encountered an error: {exception}", exc_info=True) 