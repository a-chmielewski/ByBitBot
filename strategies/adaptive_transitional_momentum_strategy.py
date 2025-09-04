"""
Adaptive Transitional Momentum Breakout Strategy

Strategy #14 in the Strategy Matrix

Market Conditions: Best fit for TRANSITIONAL markets
Description: Adaptive strategy for transitional market phases with momentum detection
"""

import logging
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

from .strategy_template import StrategyTemplate

class StrategyAdaptiveTransitionalMomentum(StrategyTemplate):
    """
    Adaptive Transitional Momentum Breakout Strategy
    
    A sophisticated strategy designed for transitional market conditions where the market
    is shifting from one regime to another (low vol to high vol, trend to reversal, etc.).
    The strategy adapts its behavior based on volatility regime changes and works during
    in-between phases that other strategies might not cover.
    
    Market Type:
    -----------
    - Transitional market conditions
    - Volatility regime changes (low to high, high to low)
    - Trend to reversal transitions
    - Range to trend breakouts
    - Momentum ignition phases
    - Market character shifts
    
    Strategy Logic:
    --------------
    1. Adaptive Momentum Detection:
       - Rate of Change (ROC) with adaptive lookback period
       - Period adjusts based on volatility regime (10-30 bars)
       - EMA smoothing to filter noise
       - Faster response in high volatility, slower in low volatility
    
    2. Volatility Regime Detection:
       - ATR expansion/contraction analysis
       - Bollinger Band width changes
       - Multi-bar volatility trend detection
       - Regime transition identification
    
    3. Trend Context Analysis:
       - Moving average crossovers (20/50)
       - DMI crossovers and ADX transitions
       - Trend strength and direction changes
       - Multi-timeframe alignment
    
    4. Pattern Recognition:
       - Bullish/bearish engulfing patterns
       - Wide-range breakout bars
       - Volatility squeeze releases
       - Momentum ignition candles
    
    Entry Conditions:
    ----------------
    1. Volatility-Driven Entries:
       - ATR expansion (1.5x threshold over 3 bars)
       - Bollinger Band width expansion
       - Momentum oscillator slope confirmation
       - Volume surge validation
    
    2. Momentum Reversal Entries:
       - Trend indicator flips (MA/DMI crossovers)
       - Strong momentum oscillator signals
       - Engulfing pattern confirmation
       - Multi-timeframe alignment
    
    3. Breakout Bar Triggers:
       - Wide-range bars (2x average range)
       - Engulfing patterns with volume
       - Volatility expansion confirmation
       - Adaptive momentum alignment
    
    4. Multi-Signal Confirmation:
       - At least 2 supporting signals required
       - Volatility regime validation
       - Pattern and momentum alignment
       - Risk/reward assessment
    
    Exit Conditions:
    ---------------
    1. Fixed Time Exits:
       - Maximum hold period (5 bars default)
       - Quick profit capture approach
       - Avoid post-spike chop
       - Configurable hold duration
    
    2. Trailing Stops:
       - ATR-based trailing (2x ATR)
       - Trend emergence detection
       - Profit protection mechanism
       - Dynamic stop adjustment
    
    3. Momentum Fade Exits:
       - Momentum oscillator reversal
       - Slope flattening detection
       - Peak identification
       - Early exit on momentum loss
    
    4. Emergency Exits:
       - Failed breakout detection
       - Volatility collapse
       - Price reversal to pre-breakout levels
       - False signal protection
    
    Risk Management:
    --------------
    - Reduced position sizing (50% of normal)
    - Tight initial stops (1.5x ATR)
    - Quick breakeven moves
    - Two-stage stop system
    - Volatility-based risk adjustment
    """
    
    MARKET_TYPE_TAGS: List[str] = ['TRANSITIONAL']
    SHOW_IN_SELECTION: bool = True
    
    def __init__(self, data: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(data, config, logger)
        
        # Adaptive momentum state
        self.current_volatility_regime = "unknown"  # "low", "high", "transitioning"
        self.previous_regime = "unknown"
        self.adaptive_momentum_period = self.config.get('momentum_base_period', 20)
        self.regime_change_bar = None
        
        # Trade management
        self.entry_price = None
        self.entry_bar = None
        self.stop_price = None
        self.target_price = None
        self.trailing_stop = None
        
        # Historical data for calculations
        self.atr_history = []
        self.bb_width_history = []
        self.momentum_history = []
        self.candle_ranges = []
        
        # Pattern detection
        self.last_engulfing_bar = None
        self.last_wide_range_bar = None
        self.last_trade_bar = -self.config.get('cooldown_bars', 5)
        
        self.logger.info("Adaptive Transitional Momentum strategy initialized")
    
    def init_indicators(self) -> None:
        """Initialize adaptive momentum and volatility indicators"""
        try:
            # Import pandas_ta with fallback
            try:
                import pandas_ta as ta
                self.has_pandas_ta = True
            except ImportError:
                self.logger.warning("pandas_ta not available, using manual calculations")
                self.has_pandas_ta = False
            
            # Get strategy parameters
            atr_period = self.config.get('atr_period', 14)
            bb_period = self.config.get('bb_period', 20)
            bb_std = self.config.get('bb_std', 2.0)
            fast_ma_period = self.config.get('fast_ma_period', 20)
            slow_ma_period = self.config.get('slow_ma_period', 50)
            adx_period = self.config.get('adx_period', 14)
            dmi_period = self.config.get('dmi_period', 14)
            volume_period = self.config.get('volume_avg_period', 20)
            
            # ATR for volatility measurement
            if self.has_pandas_ta:
                self.data['atr'] = ta.atr(self.data['high'], self.data['low'], self.data['close'], length=atr_period)
            else:
                self.data['atr'] = self._calculate_atr_manual(atr_period)
            
            # Bollinger Bands for volatility detection
            if self.has_pandas_ta:
                bb_data = ta.bbands(self.data['close'], length=bb_period, std=bb_std)
                self.data['bb_upper'] = bb_data[f'BBU_{bb_period}_{bb_std}']
                self.data['bb_middle'] = bb_data[f'BBM_{bb_period}_{bb_std}']
                self.data['bb_lower'] = bb_data[f'BBL_{bb_period}_{bb_std}']
            else:
                self.data['bb_middle'] = self.data['close'].rolling(window=bb_period).mean()
                bb_std_dev = self.data['close'].rolling(window=bb_period).std()
                self.data['bb_upper'] = self.data['bb_middle'] + (bb_std_dev * bb_std)
                self.data['bb_lower'] = self.data['bb_middle'] - (bb_std_dev * bb_std)
            
            # Moving averages for trend context
            self.data['fast_ma'] = self.data['close'].ewm(span=fast_ma_period).mean()
            self.data['slow_ma'] = self.data['close'].ewm(span=slow_ma_period).mean()
            
            # ADX and DMI for trend analysis
            if self.has_pandas_ta:
                adx_data = ta.adx(self.data['high'], self.data['low'], self.data['close'], length=adx_period)
                self.data['adx'] = adx_data[f'ADX_{adx_period}']
                self.data['dmi_plus'] = adx_data[f'DMP_{adx_period}']
                self.data['dmi_minus'] = adx_data[f'DMN_{adx_period}']
            else:
                self.data['adx'] = self._calculate_adx_manual(adx_period)
                self.data['dmi_plus'], self.data['dmi_minus'] = self._calculate_dmi_manual(dmi_period)
            
            # Volume analysis
            self.data['volume_sma'] = self.data['volume'].rolling(window=volume_period).mean()
            
            self.logger.debug("All adaptive momentum indicators initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing indicators: {str(e)}")
            raise
    
    def _calculate_atr_manual(self, period: int) -> pd.Series:
        """Manual ATR calculation as fallback"""
        try:
            high_low = self.data['high'] - self.data['low']
            high_close = np.abs(self.data['high'] - self.data['close'].shift())
            low_close = np.abs(self.data['low'] - self.data['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            
            return atr
        except Exception:
            return pd.Series([0.01] * len(self.data), index=self.data.index)
    
    def _calculate_adx_manual(self, period: int) -> pd.Series:
        """Manual ADX calculation as fallback"""
        try:
            # Simplified ADX calculation
            high_diff = self.data['high'].diff()
            low_diff = self.data['low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            plus_dm_series = pd.Series(plus_dm, index=self.data.index)
            minus_dm_series = pd.Series(minus_dm, index=self.data.index)
            
            tr = self._calculate_atr_manual(1) * len(self.data)  # True range
            
            plus_di = 100 * (plus_dm_series.rolling(window=period).mean() / tr.rolling(window=period).mean())
            minus_di = 100 * (minus_dm_series.rolling(window=period).mean() / tr.rolling(window=period).mean())
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return adx
        except Exception:
            return pd.Series([25.0] * len(self.data), index=self.data.index)
    
    def _calculate_dmi_manual(self, period: int) -> Tuple[pd.Series, pd.Series]:
        """Manual DMI calculation as fallback"""
        try:
            high_diff = self.data['high'].diff()
            low_diff = self.data['low'].diff()
            
            plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            plus_dm_series = pd.Series(plus_dm, index=self.data.index)
            minus_dm_series = pd.Series(minus_dm, index=self.data.index)
            
            tr = self._calculate_atr_manual(1) * len(self.data)  # True range
            
            plus_di = 100 * (plus_dm_series.rolling(window=period).mean() / tr.rolling(window=period).mean())
            minus_di = 100 * (minus_dm_series.rolling(window=period).mean() / tr.rolling(window=period).mean())
            
            return plus_di, minus_di
        except Exception:
            return (pd.Series([25.0] * len(self.data), index=self.data.index),
                    pd.Series([25.0] * len(self.data), index=self.data.index))
    
    def detect_volatility_regime(self, idx: int) -> str:
        """Detect current volatility regime and transitions"""
        try:
            vol_regime_lookback = self.config.get('vol_regime_lookback', 30)
            if idx < vol_regime_lookback:
                return "unknown"
            
            # Update ATR history
            current_atr = self.data['atr'].iloc[idx]
            if pd.isna(current_atr):
                return self.current_volatility_regime
            
            self.atr_history.append(current_atr)
            if len(self.atr_history) > vol_regime_lookback:
                self.atr_history.pop(0)
            
            # Calculate ATR statistics
            if len(self.atr_history) >= vol_regime_lookback:
                atr_median = np.median(self.atr_history)
                atr_recent = np.mean(self.atr_history[-5:])  # Last 5 bars average
                
                # Check for volatility expansion
                vol_expansion_threshold = self.config.get('vol_expansion_threshold', 1.5)
                vol_expansion_ratio = atr_recent / atr_median if atr_median > 0 else 1.0
                
                # Bollinger Band width analysis
                bb_upper = self.data['bb_upper'].iloc[idx]
                bb_lower = self.data['bb_lower'].iloc[idx]
                bb_middle = self.data['bb_middle'].iloc[idx]
                
                if not pd.isna(bb_upper) and not pd.isna(bb_lower) and bb_middle > 0:
                    bb_width = (bb_upper - bb_lower) / bb_middle
                    self.bb_width_history.append(bb_width)
                    if len(self.bb_width_history) > 20:
                        self.bb_width_history.pop(0)
                    
                    # Determine regime with hysteresis
                    bb_squeeze_threshold = self.config.get('bb_squeeze_threshold', 0.02)
                    bb_expansion_threshold = self.config.get('bb_expansion_threshold', 0.04)
                    regime_hysteresis = self.config.get('regime_hysteresis', 0.1)
                    
                    if vol_expansion_ratio > vol_expansion_threshold:
                        if bb_width > bb_expansion_threshold:
                            new_regime = "high"
                        else:
                            new_regime = "transitioning"
                    elif vol_expansion_ratio < (1.0 + regime_hysteresis):
                        if bb_width < bb_squeeze_threshold:
                            new_regime = "low"
                        else:
                            new_regime = "transitioning"
                    else:
                        new_regime = "transitioning"
                    
                    # Update regime with hysteresis to prevent flip-flopping
                    if new_regime != self.current_volatility_regime:
                        if self.current_volatility_regime == "unknown":
                            self.current_volatility_regime = new_regime
                        else:
                            # Apply hysteresis
                            if new_regime == "high" and vol_expansion_ratio > (vol_expansion_threshold + regime_hysteresis):
                                self.previous_regime = self.current_volatility_regime
                                self.current_volatility_regime = new_regime
                                self.regime_change_bar = idx
                            elif new_regime == "low" and vol_expansion_ratio < (1.0 - regime_hysteresis):
                                self.previous_regime = self.current_volatility_regime
                                self.current_volatility_regime = new_regime
                                self.regime_change_bar = idx
                            elif new_regime == "transitioning":
                                self.previous_regime = self.current_volatility_regime
                                self.current_volatility_regime = new_regime
                                self.regime_change_bar = idx
            
            return self.current_volatility_regime
            
        except Exception as e:
            self.logger.error(f"Error in detect_volatility_regime: {str(e)}")
            return self.current_volatility_regime
    
    def calculate_adaptive_momentum(self, idx: int) -> float:
        """Calculate momentum with adaptive period based on volatility regime"""
        try:
            # Adjust momentum period based on volatility regime
            momentum_min_period = self.config.get('momentum_min_period', 10)
            momentum_max_period = self.config.get('momentum_max_period', 30)
            momentum_base_period = self.config.get('momentum_base_period', 20)
            
            if self.current_volatility_regime == "high":
                self.adaptive_momentum_period = momentum_min_period
            elif self.current_volatility_regime == "low":
                self.adaptive_momentum_period = momentum_max_period
            else:  # transitioning
                self.adaptive_momentum_period = momentum_base_period
            
            # Calculate Rate of Change (ROC) momentum
            if idx >= self.adaptive_momentum_period:
                current_price = self.data['close'].iloc[idx]
                past_price = self.data['close'].iloc[idx - self.adaptive_momentum_period]
                if past_price != 0:
                    momentum = (current_price - past_price) / past_price * 100
                else:
                    momentum = 0
            else:
                momentum = 0
            
            # Apply EMA smoothing
            momentum_ema_period = self.config.get('momentum_ema_period', 7)
            self.momentum_history.append(momentum)
            if len(self.momentum_history) > momentum_ema_period:
                self.momentum_history.pop(0)
            
            if len(self.momentum_history) >= momentum_ema_period:
                # Simple EMA calculation
                alpha = 2.0 / (momentum_ema_period + 1)
                smoothed_momentum = self.momentum_history[0]
                for i in range(1, len(self.momentum_history)):
                    smoothed_momentum = alpha * self.momentum_history[i] + (1 - alpha) * smoothed_momentum
                return smoothed_momentum
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error in calculate_adaptive_momentum: {str(e)}")
            return 0.0
    
    def detect_trend_context(self, idx: int) -> Dict[str, str]:
        """Detect trend context using MA crossover and DMI"""
        try:
            if idx < 1:
                return {"ma_trend": "unknown", "dmi_signal": "unknown", "adx_state": "unknown"}
            
            fast_ma_curr = self.data['fast_ma'].iloc[idx]
            slow_ma_curr = self.data['slow_ma'].iloc[idx]
            fast_ma_prev = self.data['fast_ma'].iloc[idx-1]
            slow_ma_prev = self.data['slow_ma'].iloc[idx-1]
            
            # MA crossover analysis
            ma_trend = "neutral"
            if not pd.isna(fast_ma_curr) and not pd.isna(slow_ma_curr):
                if fast_ma_curr > slow_ma_curr:
                    if fast_ma_prev <= slow_ma_prev:
                        ma_trend = "bullish_crossover"
                    else:
                        ma_trend = "bullish"
                elif fast_ma_curr < slow_ma_curr:
                    if fast_ma_prev >= slow_ma_prev:
                        ma_trend = "bearish_crossover"
                    else:
                        ma_trend = "bearish"
            
            # DMI analysis
            dmi_signal = "neutral"
            dmi_plus_curr = self.data['dmi_plus'].iloc[idx]
            dmi_minus_curr = self.data['dmi_minus'].iloc[idx]
            dmi_plus_prev = self.data['dmi_plus'].iloc[idx-1]
            dmi_minus_prev = self.data['dmi_minus'].iloc[idx-1]
            
            if not pd.isna(dmi_plus_curr) and not pd.isna(dmi_minus_curr):
                dmi_crossover_threshold = self.config.get('dmi_crossover_threshold', 5)
                di_diff = abs(dmi_plus_curr - dmi_minus_curr)
                
                if di_diff > dmi_crossover_threshold:
                    if dmi_plus_curr > dmi_minus_curr:
                        # Check for recent crossover
                        if dmi_plus_prev <= dmi_minus_prev:
                            dmi_signal = "bullish_crossover"
                        else:
                            dmi_signal = "bullish"
                    else:
                        if dmi_plus_prev >= dmi_minus_prev:
                            dmi_signal = "bearish_crossover"
                        else:
                            dmi_signal = "bearish"
            
            # ADX trend strength
            adx_state = "ranging"
            adx_curr = self.data['adx'].iloc[idx]
            if not pd.isna(adx_curr):
                adx_low_threshold = self.config.get('adx_low_threshold', 25)
                adx_rising_threshold = self.config.get('adx_rising_threshold', 30)
                
                if adx_curr > adx_rising_threshold:
                    adx_state = "trending"
                elif adx_curr < adx_low_threshold:
                    adx_state = "ranging"
                else:
                    adx_state = "transitioning"
            
            return {
                "ma_trend": ma_trend,
                "dmi_signal": dmi_signal,
                "adx_state": adx_state
            }
            
        except Exception as e:
            self.logger.error(f"Error in detect_trend_context: {str(e)}")
            return {"ma_trend": "unknown", "dmi_signal": "unknown", "adx_state": "unknown"}
    
    def detect_candlestick_patterns(self, idx: int) -> Dict[str, Any]:
        """Detect engulfing patterns and wide-range bars"""
        try:
            if idx < 1:
                return {"engulfing": None, "wide_range": False}
            
            # Current and previous candle data
            curr_open = self.data['open'].iloc[idx]
            curr_close = self.data['close'].iloc[idx]
            curr_high = self.data['high'].iloc[idx]
            curr_low = self.data['low'].iloc[idx]
            curr_range = curr_high - curr_low
            
            prev_open = self.data['open'].iloc[idx-1]
            prev_close = self.data['close'].iloc[idx-1]
            prev_high = self.data['high'].iloc[idx-1]
            prev_low = self.data['low'].iloc[idx-1]
            prev_range = prev_high - prev_low
            
            # Engulfing pattern detection
            engulfing = None
            engulfing_min_ratio = self.config.get('engulfing_min_ratio', 1.2)
            
            if curr_range > prev_range * engulfing_min_ratio:
                # Bullish engulfing
                if (curr_close > curr_open and prev_close < prev_open and 
                    curr_close > prev_open and curr_open < prev_close):
                    engulfing = "bullish"
                    self.last_engulfing_bar = idx
                # Bearish engulfing
                elif (curr_close < curr_open and prev_close > prev_open and 
                      curr_close < prev_open and curr_open > prev_close):
                    engulfing = "bearish"
                    self.last_engulfing_bar = idx
            
            # Wide-range bar detection
            self.candle_ranges.append(curr_range)
            range_lookback = self.config.get('range_lookback', 10)
            if len(self.candle_ranges) > range_lookback:
                self.candle_ranges.pop(0)
            
            wide_range = False
            if len(self.candle_ranges) >= range_lookback:
                wide_range_multiplier = self.config.get('wide_range_multiplier', 2.0)
                avg_range = sum(self.candle_ranges[:-1]) / (len(self.candle_ranges) - 1)  # Exclude current
                if curr_range > avg_range * wide_range_multiplier:
                    wide_range = True
                    self.last_wide_range_bar = idx
            
            return {"engulfing": engulfing, "wide_range": wide_range}
            
        except Exception as e:
            self.logger.error(f"Error in detect_candlestick_patterns: {str(e)}")
            return {"engulfing": None, "wide_range": False}
    
    def check_entry_conditions(self, idx: int) -> Tuple[bool, Optional[str]]:
        """Main entry logic for transitional momentum breakouts"""
        try:
            # Detect current volatility regime
            vol_regime = self.detect_volatility_regime(idx)
            
            # Only trade during regime transitions or when regime just changed
            if vol_regime not in ["transitioning", "high"]:
                return False, None
            
            # Check if we're in a recent regime change
            if self.regime_change_bar is None:
                return False, None
            
            bars_since_change = idx - self.regime_change_bar
            if bars_since_change > 10:  # Only trade within 10 bars of regime change
                return False, None
            
            # Calculate adaptive momentum
            momentum = self.calculate_adaptive_momentum(idx)
            
            # Get trend context
            trend_context = self.detect_trend_context(idx)
            
            # Detect candlestick patterns
            patterns = self.detect_candlestick_patterns(idx)
            
            # Entry conditions based on momentum direction
            direction = None
            momentum_threshold = self.config.get('momentum_threshold', 0.5)
            
            # Bullish entry conditions
            if momentum > momentum_threshold:
                # Check for supporting signals
                bullish_signals = 0
                
                # Volatility expansion with upward momentum
                if vol_regime == "high" or (vol_regime == "transitioning" and self.previous_regime == "low"):
                    bullish_signals += 1
                
                # Trend context support
                if trend_context["ma_trend"] in ["bullish", "bullish_crossover"]:
                    bullish_signals += 1
                if trend_context["dmi_signal"] in ["bullish", "bullish_crossover"]:
                    bullish_signals += 1
                
                # Pattern support
                if patterns["engulfing"] == "bullish":
                    bullish_signals += 2  # Strong signal
                if patterns["wide_range"] and self.data['close'].iloc[idx] > self.data['open'].iloc[idx]:
                    bullish_signals += 1
                
                # Volume confirmation
                volume_curr = self.data['volume'].iloc[idx]
                volume_avg = self.data['volume_sma'].iloc[idx]
                if not pd.isna(volume_avg) and volume_avg > 0:
                    volume_surge_multiplier = self.config.get('volume_surge_multiplier', 2.0)  # Enhanced from 1.5 to 2.0
                    if volume_curr > volume_avg * volume_surge_multiplier:
                        bullish_signals += 1
                
                # Require at least 2 supporting signals
                if bullish_signals >= 2:
                    direction = "long"
            
            # Bearish entry conditions
            elif momentum < -momentum_threshold:
                # Check for supporting signals
                bearish_signals = 0
                
                # Volatility expansion with downward momentum
                if vol_regime == "high" or (vol_regime == "transitioning" and self.previous_regime == "low"):
                    bearish_signals += 1
                
                # Trend context support
                if trend_context["ma_trend"] in ["bearish", "bearish_crossover"]:
                    bearish_signals += 1
                if trend_context["dmi_signal"] in ["bearish", "bearish_crossover"]:
                    bearish_signals += 1
                
                # Pattern support
                if patterns["engulfing"] == "bearish":
                    bearish_signals += 2  # Strong signal
                if patterns["wide_range"] and self.data['close'].iloc[idx] < self.data['open'].iloc[idx]:
                    bearish_signals += 1
                
                # Volume confirmation
                volume_curr = self.data['volume'].iloc[idx]
                volume_avg = self.data['volume_sma'].iloc[idx]
                if not pd.isna(volume_avg) and volume_avg > 0:
                    volume_surge_multiplier = self.config.get('volume_surge_multiplier', 2.0)  # Enhanced from 1.5 to 2.0
                    if volume_curr > volume_avg * volume_surge_multiplier:
                        bearish_signals += 1
                
                # Require at least 2 supporting signals
                if bearish_signals >= 2:
                    direction = "short"
            
            return direction is not None, direction
            
        except Exception as e:
            self.logger.error(f"Error in check_entry_conditions: {str(e)}")
            return False, None
    
    def calculate_stops_and_targets(self, idx: int, direction: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            current_price = self.data['close'].iloc[idx]
            atr_value = self.data['atr'].iloc[idx]
            
            if pd.isna(atr_value) or atr_value == 0:
                atr_value = current_price * 0.01
            
            # Stop loss based on ATR
            stop_atr_multiplier = self.config.get('stop_atr_multiplier', 1.5)
            stop_distance = atr_value * stop_atr_multiplier
            
            risk_reward_ratio = self.config.get('risk_reward_ratio', 2.0)
            
            if direction == "long":
                stop_price = current_price - stop_distance
                target_price = current_price + (stop_distance * risk_reward_ratio)
            else:  # short
                stop_price = current_price + stop_distance
                target_price = current_price - (stop_distance * risk_reward_ratio)
            
            return stop_price, target_price
            
        except Exception as e:
            self.logger.error(f"Error in calculate_stops_and_targets: {str(e)}")
            current_price = self.data['close'].iloc[idx]
            return current_price * 0.98, current_price * 1.02  # Fallback values
    
    def check_exit_conditions(self, idx: int) -> Tuple[bool, Optional[str]]:
        """Check various exit conditions for transitional trades"""
        try:
            if not hasattr(self, 'entry_bar') or self.entry_bar is None:
                return False, None
            
            current_price = self.data['close'].iloc[idx]
            
            # Fixed bar exit (if enabled)
            use_fixed_exit = self.config.get('use_fixed_exit', False)
            max_hold_bars = self.config.get('max_hold_bars', 5)
            
            if use_fixed_exit:
                if idx - self.entry_bar >= max_hold_bars:
                    return True, "fixed_time_exit"
            
            # Stop loss hit
            if self.stop_price:
                if self.entry_side == 'long' and current_price <= self.stop_price:
                    return True, "stop_loss"
                elif self.entry_side == 'short' and current_price >= self.stop_price:
                    return True, "stop_loss"
            
            # Target hit
            if self.target_price:
                if self.entry_side == 'long' and current_price >= self.target_price:
                    return True, "target_hit"
                elif self.entry_side == 'short' and current_price <= self.target_price:
                    return True, "target_hit"
            
            # Momentum fade exit
            momentum = self.calculate_adaptive_momentum(idx)
            momentum_fade_threshold = self.config.get('momentum_fade_threshold', 0.2)
            
            if self.entry_side == 'long' and momentum < momentum_fade_threshold:
                return True, "momentum_fade"
            elif self.entry_side == 'short' and momentum > -momentum_fade_threshold:
                return True, "momentum_fade"
            
            # Emergency exit on regime reversal
            emergency_exit_bars = self.config.get('emergency_exit_bars', 2)
            if idx - self.entry_bar >= emergency_exit_bars:
                vol_regime = self.detect_volatility_regime(idx)
                if vol_regime == "low" and self.current_volatility_regime != "low":
                    return True, "emergency_regime_reversal"
            
            # Trailing stop
            if self.trailing_stop:
                if self.entry_side == 'long' and current_price <= self.trailing_stop:
                    return True, "trailing_stop"
                elif self.entry_side == 'short' and current_price >= self.trailing_stop:
                    return True, "trailing_stop"
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error in check_exit_conditions: {str(e)}")
            return False, None
    
    def update_trailing_stop(self, idx: int) -> None:
        """Update trailing stop based on ATR"""
        try:
            if not hasattr(self, 'entry_price') or self.entry_price is None:
                return
            
            current_price = self.data['close'].iloc[idx]
            atr_value = self.data['atr'].iloc[idx]
            
            if pd.isna(atr_value) or atr_value == 0:
                return
            
            trailing_atr_multiplier = self.config.get('trailing_atr_multiplier', 2.0)
            trail_distance = atr_value * trailing_atr_multiplier
            
            # Only activate trailing stop after some profit
            if self.entry_side == 'long':  # Long position
                profit = current_price - self.entry_price
                if profit > atr_value:  # At least 1 ATR profit
                    new_stop = current_price - trail_distance
                    if self.trailing_stop is None or new_stop > self.trailing_stop:
                        self.trailing_stop = new_stop
            elif self.entry_side == 'short':  # Short position
                profit = self.entry_price - current_price
                if profit > atr_value:  # At least 1 ATR profit
                    new_stop = current_price + trail_distance
                    if self.trailing_stop is None or new_stop < self.trailing_stop:
                        self.trailing_stop = new_stop
                        
        except Exception as e:
            self.logger.error(f"Error in update_trailing_stop: {str(e)}")
    
    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for adaptive transitional momentum entry opportunities"""
        try:
            idx = len(self.data) - 1
            if idx < 0:
                return None
            
            # Check cooldown period
            cooldown_bars = self.config.get('cooldown_bars', 5)
            if idx - self.last_trade_bar < cooldown_bars:
                return None
            
            # Need minimum bars for all indicators
            min_bars = max(
                self.config.get('vol_regime_lookback', 30),
                self.config.get('momentum_max_period', 30),
                self.config.get('slow_ma_period', 50)
            )
            if idx < min_bars:
                return None
            
            # Check for new entries
            should_enter, direction = self.check_entry_conditions(idx)
            
            if should_enter and direction:
                current_price = self.data['close'].iloc[idx]
                
                # Calculate stops and targets
                stop_price, target_price = self.calculate_stops_and_targets(idx, direction)
                
                # Initialize trade state
                self.entry_price = current_price
                self.entry_bar = idx
                self.entry_side = direction
                self.stop_price = stop_price
                self.target_price = target_price
                self.trailing_stop = None
                
                momentum = self.calculate_adaptive_momentum(idx)
                
                self.logger.info(f"Adaptive transitional {direction} entry at {current_price:.4f}, momentum: {momentum:.3f}, regime: {self.current_volatility_regime}")
                
                return {
                    'side': 'buy' if direction == 'long' else 'sell',
                    'price': current_price,
                    'confidence': 0.75,
                    'reason': f'adaptive_transitional_{direction}_momentum_{momentum:.2f}_regime_{self.current_volatility_regime}'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _check_entry_conditions: {str(e)}")
            return None
    
    def check_exit(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check various exit conditions for adaptive transitional trades"""
        try:
            idx = len(self.data) - 1
            if idx < 0:
                return None
            
            current_price = self.data['close'].iloc[idx]
            
            # Update trailing stop
            self.update_trailing_stop(idx)
            
            # Check exit conditions
            should_exit, exit_reason = self.check_exit_conditions(idx)
            
            if should_exit:
                self.logger.info(f"Adaptive transitional exit: {exit_reason}")
                return {
                    'action': 'exit',
                    'price': current_price,
                    'reason': exit_reason
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in check_exit: {str(e)}")
            return None
    
    def on_trade_closed(self, symbol: str, trade_result: Dict[str, Any]) -> None:
        """Handle trade closure cleanup"""
        try:
            self.last_trade_bar = len(self.data) - 1
            
            # Reset trade state
            self.entry_price = None
            self.entry_bar = None
            self.entry_side = None
            self.stop_price = None
            self.target_price = None
            self.trailing_stop = None
            
            # Log trade result
            exit_reason = trade_result.get('reason', 'unknown')
            pnl = trade_result.get('pnl', 0)
            
            self.logger.info(f"Adaptive transitional trade closed - {exit_reason}, PnL: {pnl}")
            
        except Exception as e:
            self.logger.error(f"Error in on_trade_closed: {str(e)}")
    
    def get_risk_parameters(self) -> Dict[str, Any]:
        """Get adaptive risk management parameters"""
        try:
            # Reduced position sizing for transitional trades (higher uncertainty)
            position_size_factor = self.config.get('position_size_factor', 0.5)
            max_position_pct = self.config.get('max_position_pct', 2.0) * position_size_factor
            
            # ATR-based risk parameters
            if hasattr(self, 'entry_price') and self.entry_price and len(self.data) > 0:
                current_price = self.data['close'].iloc[-1]
                atr_value = self.data['atr'].iloc[-1]
                
                if not pd.isna(atr_value) and atr_value > 0:
                    stop_atr_multiplier = self.config.get('stop_atr_multiplier', 1.5)
                    sl_pct = (atr_value * stop_atr_multiplier) / current_price
                    
                    risk_reward_ratio = self.config.get('risk_reward_ratio', 2.0)
                    tp_pct = sl_pct * risk_reward_ratio
                    
                    return {
                        "sl_pct": sl_pct,
                        "tp_pct": tp_pct,
                        "max_position_pct": max_position_pct,
                        "risk_reward_ratio": risk_reward_ratio
                    }
            
            # Fallback to config defaults
            return {
                "sl_pct": self.config.get('sl_pct', 0.05),  # Widened from 0.025
                "tp_pct": self.config.get('tp_pct', 0.03),  # Quicker from 0.05
                "max_position_pct": max_position_pct,
                "risk_reward_ratio": self.config.get('risk_reward_ratio', 2.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_risk_parameters: {str(e)}")
            return {
                "sl_pct": 0.05,  # Widened from 0.025
                "tp_pct": 0.03,  # Quicker from 0.05
                "max_position_pct": 1.0,
                "risk_reward_ratio": 2.0
            }