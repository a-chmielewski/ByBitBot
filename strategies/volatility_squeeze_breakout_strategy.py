"""
Volatility Squeeze Breakout Anticipation Strategy

Strategy #8 in the Strategy Matrix

Market Conditions: Good for LOW_VOLATILITY and TRANSITIONAL markets
Description: Detects volatility squeezes and anticipates breakout direction
"""

import logging
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

from .strategy_template import StrategyTemplate

class StrategyVolatilitySqueezeBreakout(StrategyTemplate):
    """
    Volatility Squeeze Breakout Anticipation Strategy
    
    A transitional strategy designed to catch breakouts from low-volatility squeeze
    conditions. The strategy identifies volatility squeezes using Bollinger Bands,
    ADX, and volume, then positions for breakouts in either direction using OCO-style
    logic to catch the initial explosive move when volatility returns.
    
    Market Type:
    -----------
    - Transition from low-volatility to high-volatility
    - Markets currently quiet but likely to break out soon
    - Pre-breakout positioning during squeeze conditions
    - Consolidation periods before major moves
    
    Strategy Logic:
    --------------
    1. Squeeze Detection:
       - Bollinger Band squeeze (width at multi-period low)
       - ADX very low (under 30) indicating no trend
       - Volume declining during consolidation
       - Minimum squeeze duration for validity
    
    2. Range Identification:
       - Clear support and resistance levels
       - Tight consolidation boundaries
       - Range validation with minimum touches
       - Range size requirements for tradability
    
    3. Breakout Preparation:
       - OCO-style breakout orders (both directions)
       - Buy stop above resistance
       - Sell stop below support
       - Market decides direction, strategy follows
    
    4. Confirmation and Filtering:
       - Volume spike confirmation (optional)
       - Candle close confirmation
       - False breakout protection
       - Quick exit on failed breakouts
    
    Entry Conditions:
    ----------------
    1. Squeeze Prerequisites:
       - Bollinger Band squeeze detected
       - ADX below threshold (low trend strength)
       - Volume below average (quiet conditions)
       - Minimum squeeze duration met
    
    2. Range Validation:
       - Clear support/resistance identified
       - Minimum range touches confirmed
       - Range size within tradable limits
       - Stable consolidation pattern
    
    3. Breakout Triggers:
       - Price breaks above resistance (long)
       - Price breaks below support (short)
       - Optional volume/candle confirmation
       - OCO logic cancels opposite direction
    
    4. Confirmation Filters:
       - Volume spike (1.5x average)
       - Candle close position (60% of range)
       - Immediate follow-through required
       - False breakout detection
    
    Exit Conditions:
    ---------------
    1. Quick Profit Taking:
       - Range height projection (0.8x)
       - ATR-based targets (1.5x ATR)
       - Partial exits (40% on first target)
       - Momentum-based trailing
    
    2. Failed Breakout Exits:
       - Quick reversal back into range
       - Time-based exit (5 bars max)
       - Tight stops just inside range
       - False breakout protection
    
    3. Trend Development:
       - Trail stops for sustained moves
       - Partial profit management
       - Range projection targets
       - Maximum hold time limits
    
    4. Squeeze Reset:
       - New squeeze detection after breakout
       - State reset for next opportunity
       - Cooldown periods between trades
    
    Risk Management:
    --------------
    - Tight initial stops inside range
    - Conservative position sizing (90% normal)
    - Quick exits on false breakouts
    - Range-based profit targets
    - Time stops for failed moves
    """
    
    MARKET_TYPE_TAGS: List[str] = ['LOW_VOLATILITY', 'TRANSITIONAL']
    SHOW_IN_SELECTION: bool = True
    
    def __init__(self, data: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(data, config, logger)
        
        # Squeeze detection state
        self.squeeze_detected = False
        self.squeeze_start_bar = None
        self.squeeze_bars_count = 0
        
        # Range detection variables
        self.range_support = 0
        self.range_resistance = 0
        self.range_middle = 0
        self.range_height = 0
        self.range_confirmed = False
        
        # Breakout management
        self.pending_breakout_long = False
        self.pending_breakout_short = False
        self.breakout_long_price = 0
        self.breakout_short_price = 0
        self.breakout_direction = None
        self.breakout_confirmed = False
        self.first_target_hit = False
        
        # Trade management
        self.entry_bar = None
        self.entry_side = None
        self.entry_price = None
        self.initial_atr = 0
        self.last_trade_bar = -self.config.get('cooldown_bars', 5)
        
        self.logger.info("Volatility Squeeze Breakout strategy initialized")
    
    def init_indicators(self) -> None:
        """Initialize squeeze detection indicators"""
        try:
            # Import pandas_ta with fallback
            try:
                import pandas_ta as ta
                self.has_pandas_ta = True
            except ImportError:
                self.logger.warning("pandas_ta not available, using manual calculations")
                self.has_pandas_ta = False
            
            # Get strategy parameters
            bb_period = self.config.get('bb_period', 20)
            bb_std = self.config.get('bb_std', 2.0)
            adx_period = self.config.get('adx_period', 14)
            atr_period = self.config.get('atr_period', 14)
            volume_avg_period = self.config.get('volume_avg_period', 20)
            
            # Bollinger Bands for squeeze detection
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
            
            # ADX for trend strength
            if self.has_pandas_ta:
                adx_data = ta.adx(self.data['high'], self.data['low'], self.data['close'], length=adx_period)
                self.data['adx'] = adx_data[f'ADX_{adx_period}']
            else:
                self.data['adx'] = self._calculate_adx_manual(adx_period)
            
            # ATR for volatility measurement
            if self.has_pandas_ta:
                self.data['atr'] = ta.atr(self.data['high'], self.data['low'], self.data['close'], length=atr_period)
            else:
                self.data['atr'] = self._calculate_atr_manual(atr_period)
            
            # Volume analysis
            self.data['volume_sma'] = self.data['volume'].rolling(window=volume_avg_period).mean()
            
            self.logger.debug("All squeeze detection indicators initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing indicators: {str(e)}")
            raise
    
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
    
    def is_bollinger_squeeze(self, idx: int) -> bool:
        """Detect Bollinger Band squeeze conditions"""
        try:
            if idx < 1:
                return False
            
            current_price = self.data['close'].iloc[idx]
            bb_upper = self.data['bb_upper'].iloc[idx]
            bb_lower = self.data['bb_lower'].iloc[idx]
            adx_value = self.data['adx'].iloc[idx]
            current_volume = self.data['volume'].iloc[idx]
            avg_volume = self.data['volume_sma'].iloc[idx]
            
            if pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(adx_value) or pd.isna(avg_volume):
                return False
            
            # Calculate Bollinger Band width as percentage
            bb_width_pct = (bb_upper - bb_lower) / current_price if current_price > 0 else 0
            
            # Get configuration thresholds
            bb_squeeze_threshold = self.config.get('bb_squeeze_threshold', 0.02)
            adx_low_threshold = self.config.get('adx_low_threshold', 30)
            volume_low_threshold = self.config.get('volume_low_threshold', 0.85)
            enable_simple_squeeze = self.config.get('enable_simple_squeeze', True)
            
            if enable_simple_squeeze:
                # Simpler squeeze detection - only require BB squeeze
                return bb_width_pct < bb_squeeze_threshold
            else:
                # Original complex squeeze detection
                bb_squeeze = bb_width_pct < bb_squeeze_threshold
                adx_low = adx_value < adx_low_threshold
                volume_low = current_volume < (avg_volume * volume_low_threshold)
                
                # All criteria must be met for squeeze
                return bb_squeeze and adx_low and volume_low
                
        except Exception as e:
            self.logger.error(f"Error in is_bollinger_squeeze: {str(e)}")
            return False
    
    def is_simple_range_detected(self, idx: int) -> bool:
        """Simplified range detection for less restrictive trading"""
        try:
            if idx < 10:  # Need minimum data
                return False
            
            # Look at recent highs and lows
            lookback = min(15, idx + 1)
            start_idx = max(0, idx - lookback + 1)
            end_idx = idx + 1
            
            recent_highs = self.data['high'].iloc[start_idx:end_idx].tolist()
            recent_lows = self.data['low'].iloc[start_idx:end_idx].tolist()
            
            # Simple range: highest high and lowest low
            resistance = max(recent_highs)
            support = min(recent_lows)
            
            # Validate range makes sense
            min_range_size_pct = self.config.get('min_range_size_pct', 0.003)
            if resistance <= support or (resistance - support) / support < min_range_size_pct:
                return False
            
            self.range_resistance = resistance
            self.range_support = support
            self.range_middle = (resistance + support) / 2
            self.range_height = resistance - support
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in is_simple_range_detected: {str(e)}")
            return False
    
    def detect_support_resistance_range(self, idx: int) -> bool:
        """Detect support and resistance levels during squeeze period"""
        try:
            range_detection_bars = self.config.get('range_detection_bars', 20)
            if idx < range_detection_bars:
                return False
            
            # Look back at recent price action during squeeze period
            lookback_bars = min(range_detection_bars, self.squeeze_bars_count)
            if lookback_bars < 10:  # Need minimum data
                return False
            
            start_idx = max(0, idx - lookback_bars + 1)
            end_idx = idx + 1
            
            highs = self.data['high'].iloc[start_idx:end_idx].tolist()
            lows = self.data['low'].iloc[start_idx:end_idx].tolist()
            
            # Find resistance level (cluster of highs)
            range_touch_tolerance = self.config.get('range_touch_tolerance', 0.003)
            min_range_touches = self.config.get('min_range_touches', 2)
            
            high_clusters = {}
            for high in highs:
                found_cluster = False
                for cluster_level in high_clusters:
                    if abs(high - cluster_level) / cluster_level < range_touch_tolerance:
                        high_clusters[cluster_level] += 1
                        found_cluster = True
                        break
                if not found_cluster:
                    high_clusters[high] = 1
            
            # Find support level (cluster of lows)
            low_clusters = {}
            for low in lows:
                found_cluster = False
                for cluster_level in low_clusters:
                    if abs(low - cluster_level) / cluster_level < range_touch_tolerance:
                        low_clusters[cluster_level] += 1
                        found_cluster = True
                        break
                if not found_cluster:
                    low_clusters[low] = 1
            
            # Get most significant levels
            resistance_candidates = [(level, count) for level, count in high_clusters.items() 
                                   if count >= min_range_touches]
            support_candidates = [(level, count) for level, count in low_clusters.items() 
                                if count >= min_range_touches]
            
            if not resistance_candidates or not support_candidates:
                return False
            
            # Choose levels with most touches
            resistance = max(resistance_candidates, key=lambda x: x[1])[0]
            support = min(support_candidates, key=lambda x: x[1])[0]
            
            # Validate range makes sense
            min_range_size_pct = self.config.get('min_range_size_pct', 0.003)
            if resistance <= support or (resistance - support) / support < min_range_size_pct:
                return False
            
            self.range_resistance = resistance
            self.range_support = support
            self.range_middle = (resistance + support) / 2
            self.range_height = resistance - support
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in detect_support_resistance_range: {str(e)}")
            return False
    
    def setup_breakout_levels(self) -> bool:
        """Calculate breakout trigger levels"""
        try:
            if not self.range_confirmed:
                return False
            
            # Set breakout levels with buffer
            breakout_buffer_pct = self.config.get('breakout_buffer_pct', 0.003)
            self.breakout_long_price = self.range_resistance * (1 + breakout_buffer_pct)
            self.breakout_short_price = self.range_support * (1 - breakout_buffer_pct)
            
            self.pending_breakout_long = True
            self.pending_breakout_short = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in setup_breakout_levels: {str(e)}")
            return False
    
    def check_breakout_trigger(self, idx: int) -> Tuple[bool, Optional[str]]:
        """Check if price has triggered a breakout"""
        try:
            if not (self.pending_breakout_long or self.pending_breakout_short):
                return False, None
            
            current_high = self.data['high'].iloc[idx]
            current_low = self.data['low'].iloc[idx]
            
            # Check long breakout
            if self.pending_breakout_long and current_high >= self.breakout_long_price:
                return True, 'long'
            
            # Check short breakout
            if self.pending_breakout_short and current_low <= self.breakout_short_price:
                return True, 'short'
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error in check_breakout_trigger: {str(e)}")
            return False, None
    
    def confirm_breakout(self, idx: int, direction: str) -> bool:
        """Confirm breakout with volume and candle analysis"""
        try:
            confirmation_required = self.config.get('confirmation_required', False)
            if not confirmation_required:
                return True
            
            current_volume = self.data['volume'].iloc[idx]
            avg_volume = self.data['volume_sma'].iloc[idx]
            current_open = self.data['open'].iloc[idx]
            current_close = self.data['close'].iloc[idx]
            current_high = self.data['high'].iloc[idx]
            current_low = self.data['low'].iloc[idx]
            
            if pd.isna(avg_volume) or avg_volume == 0:
                return True  # Skip volume check if no data
            
            # Volume confirmation
            volume_breakout_multiplier = self.config.get('volume_breakout_multiplier', 1.5)
            volume_spike = current_volume >= (avg_volume * volume_breakout_multiplier)
            
            # Candle confirmation
            candle_range = current_high - current_low
            if candle_range == 0:
                return False
            
            candle_close_pct = self.config.get('candle_close_pct', 0.6)
            
            if direction == 'long':
                # For long breakout, candle should close in upper part of its range
                close_position = (current_close - current_low) / candle_range
                candle_confirmation = (close_position >= candle_close_pct and 
                                     current_close > current_open)
            else:
                # For short breakout, candle should close in lower part of its range
                close_position = (current_high - current_close) / candle_range
                candle_confirmation = (close_position >= candle_close_pct and 
                                     current_close < current_open)
            
            return volume_spike and candle_confirmation
            
        except Exception as e:
            self.logger.error(f"Error in confirm_breakout: {str(e)}")
            return True  # Default to confirmed on error
    
    def calculate_targets_and_stops(self, direction: str, entry_price: float) -> Tuple[float, float, float]:
        """Calculate profit targets and stop loss levels"""
        try:
            # Stop loss just inside the range
            initial_stop_buffer_pct = self.config.get('initial_stop_buffer_pct', 0.002)
            
            if direction == 'long':
                stop_price = self.range_resistance * (1 - initial_stop_buffer_pct)
            else:
                stop_price = self.range_support * (1 + initial_stop_buffer_pct)
            
            # Profit targets
            quick_profit_pct = self.config.get('quick_profit_pct', 0.004)
            quick_target = entry_price * (1 + quick_profit_pct) if direction == 'long' else entry_price * (1 - quick_profit_pct)
            
            # Range projection target
            range_projection_multiplier = self.config.get('range_projection_multiplier', 0.8)
            range_target = entry_price + (self.range_height * range_projection_multiplier) if direction == 'long' else entry_price - (self.range_height * range_projection_multiplier)
            
            # ATR target
            atr_target_multiplier = self.config.get('atr_target_multiplier', 1.5)
            atr_target = entry_price + (self.initial_atr * atr_target_multiplier) if direction == 'long' else entry_price - (self.initial_atr * atr_target_multiplier)
            
            # Use the closer of range or ATR target as main target
            main_target = min(range_target, atr_target) if direction == 'long' else max(range_target, atr_target)
            
            return stop_price, quick_target, main_target
            
        except Exception as e:
            self.logger.error(f"Error in calculate_targets_and_stops: {str(e)}")
            return entry_price * 0.98, entry_price * 1.02, entry_price * 1.04  # Fallback values
    
    def check_exit_conditions(self, idx: int) -> Tuple[bool, Optional[str]]:
        """Check for breakout position exit conditions"""
        try:
            if not hasattr(self, 'entry_bar') or self.entry_bar is None:
                return False, None
            
            current_price = self.data['close'].iloc[idx]
            bars_held = idx - self.entry_bar
            
            # Time-based exit
            max_hold_bars = self.config.get('max_hold_bars', 50)
            if bars_held >= max_hold_bars:
                return True, "time_exit"
            
            # Check for fake breakout (quick reversal back into range)
            fake_breakout_exit_bars = self.config.get('fake_breakout_exit_bars', 5)
            if (bars_held <= fake_breakout_exit_bars and 
                self.range_support < current_price < self.range_resistance):
                return True, "fake_breakout"
            
            # Calculate current profit/loss
            if self.entry_price:
                quick_profit_pct = self.config.get('quick_profit_pct', 0.004)
                trail_stop_pct = self.config.get('trail_stop_pct', 0.004)
                
                if self.entry_side == 'long':
                    profit_pct = (current_price - self.entry_price) / self.entry_price
                    
                    # Quick profit target
                    if profit_pct >= quick_profit_pct and not self.first_target_hit:
                        self.first_target_hit = True
                        return True, "quick_profit_long"
                    
                    # Range projection or ATR target
                    range_projection_multiplier = self.config.get('range_projection_multiplier', 0.8)
                    atr_target_multiplier = self.config.get('atr_target_multiplier', 1.5)
                    
                    range_profit = (self.range_height * range_projection_multiplier) / self.entry_price
                    atr_profit = (self.initial_atr * atr_target_multiplier) / self.entry_price
                    target_profit = min(range_profit, atr_profit)
                    
                    if profit_pct >= target_profit:
                        return True, "main_target_long"
                    
                    # Trailing stop after first target
                    if self.first_target_hit and profit_pct > 0:
                        trail_level = self.entry_price * (1 + profit_pct - trail_stop_pct)
                        if current_price < trail_level:
                            return True, "trail_stop_long"
                
                elif self.entry_side == 'short':
                    profit_pct = (self.entry_price - current_price) / self.entry_price
                    
                    # Quick profit target
                    if profit_pct >= quick_profit_pct and not self.first_target_hit:
                        self.first_target_hit = True
                        return True, "quick_profit_short"
                    
                    # Range projection or ATR target
                    range_projection_multiplier = self.config.get('range_projection_multiplier', 0.8)
                    atr_target_multiplier = self.config.get('atr_target_multiplier', 1.5)
                    
                    range_profit = (self.range_height * range_projection_multiplier) / self.entry_price
                    atr_profit = (self.initial_atr * atr_target_multiplier) / self.entry_price
                    target_profit = min(range_profit, atr_profit)
                    
                    if profit_pct >= target_profit:
                        return True, "main_target_short"
                    
                    # Trailing stop after first target
                    if self.first_target_hit and profit_pct > 0:
                        trail_level = self.entry_price * (1 - profit_pct + trail_stop_pct)
                        if current_price > trail_level:
                            return True, "trail_stop_short"
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error in check_exit_conditions: {str(e)}")
            return False, None
    
    def reset_squeeze_state(self) -> None:
        """Reset all squeeze and breakout state variables"""
        self.squeeze_detected = False
        self.squeeze_start_bar = None
        self.squeeze_bars_count = 0
        self.range_confirmed = False
        self.pending_breakout_long = False
        self.pending_breakout_short = False
        self.breakout_direction = None
        self.breakout_confirmed = False
    
    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for squeeze breakout entry opportunities"""
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
                self.config.get('bb_period', 20),
                self.config.get('adx_period', 14),
                self.config.get('atr_period', 14),
                self.config.get('volume_avg_period', 20),
                self.config.get('range_detection_bars', 20)
            )
            if idx < min_bars:
                return None
            
            current_price = self.data['close'].iloc[idx]
            if pd.isna(current_price):
                return None
            
            # Step 1: Detect Bollinger Band squeeze
            if self.is_bollinger_squeeze(idx):
                if not self.squeeze_detected:
                    self.squeeze_detected = True
                    self.squeeze_start_bar = idx
                    self.squeeze_bars_count = 1
                else:
                    self.squeeze_bars_count += 1
                
                # Check if squeeze has lasted long enough and not too long
                bb_squeeze_bars = self.config.get('bb_squeeze_bars', 5)
                max_squeeze_bars = self.config.get('max_squeeze_bars', 150)
                
                if (self.squeeze_bars_count >= bb_squeeze_bars and 
                    self.squeeze_bars_count <= max_squeeze_bars):
                    
                    # Step 2: Detect support/resistance range
                    if not self.range_confirmed:
                        enable_simple_squeeze = self.config.get('enable_simple_squeeze', True)
                        
                        # Try simple range detection first, fall back to complex if needed
                        if enable_simple_squeeze:
                            range_detected = self.is_simple_range_detected(idx)
                        else:
                            range_detected = self.detect_support_resistance_range(idx)
                        
                        if range_detected:
                            self.range_confirmed = True
                            self.setup_breakout_levels()
                    
                    # Step 3: Check for breakout triggers
                    if self.range_confirmed:
                        breakout_triggered, direction = self.check_breakout_trigger(idx)
                        
                        if breakout_triggered:
                            # Step 4: Confirm breakout
                            if self.confirm_breakout(idx, direction):
                                # Initialize trade state
                                self.initial_atr = self.data['atr'].iloc[idx] if not pd.isna(self.data['atr'].iloc[idx]) else current_price * 0.01
                                
                                # Cancel the opposite direction (OCO behavior)
                                if direction == 'long':
                                    self.pending_breakout_short = False
                                    self.entry_side = 'long'
                                    self.entry_bar = idx
                                    self.entry_price = current_price
                                    self.breakout_direction = 'long'
                                    self.first_target_hit = False
                                    
                                    self.logger.info(f"Squeeze breakout long entry above {self.range_resistance:.4f}")
                                    
                                    return {
                                        'side': 'buy',
                                        'price': current_price,
                                        'confidence': 0.8,
                                        'reason': f'squeeze_breakout_long_above_{self.range_resistance:.4f}'
                                    }
                                
                                elif direction == 'short':
                                    self.pending_breakout_long = False
                                    self.entry_side = 'short'
                                    self.entry_bar = idx
                                    self.entry_price = current_price
                                    self.breakout_direction = 'short'
                                    self.first_target_hit = False
                                    
                                    self.logger.info(f"Squeeze breakout short entry below {self.range_support:.4f}")
                                    
                                    return {
                                        'side': 'sell',
                                        'price': current_price,
                                        'confidence': 0.8,
                                        'reason': f'squeeze_breakout_short_below_{self.range_support:.4f}'
                                    }
                                
                                # Reset squeeze state after breakout
                                self.reset_squeeze_state()
                            else:
                                # Breakout not confirmed, reset trigger for this direction
                                if direction == 'long':
                                    self.pending_breakout_long = False
                                else:
                                    self.pending_breakout_short = False
            else:
                # No longer in squeeze, reset state
                if self.squeeze_detected:
                    self.reset_squeeze_state()
                
                # Alternative entry: Look for simple range breakouts even without formal squeeze
                enable_simple_squeeze = self.config.get('enable_simple_squeeze', True)
                if enable_simple_squeeze and idx % 5 == 0:  # Check every 5 bars to avoid overtrading
                    if self.is_simple_range_detected(idx):
                        current_high = self.data['high'].iloc[idx]
                        current_low = self.data['low'].iloc[idx]
                        
                        # Setup simple breakout levels
                        breakout_buffer_pct = self.config.get('breakout_buffer_pct', 0.003)
                        breakout_long_price = self.range_resistance * (1 + breakout_buffer_pct)
                        breakout_short_price = self.range_support * (1 - breakout_buffer_pct)
                        
                        # Check for immediate breakout
                        if current_high >= breakout_long_price:
                            # Long breakout
                            self.initial_atr = self.data['atr'].iloc[idx] if not pd.isna(self.data['atr'].iloc[idx]) else current_price * 0.01
                            self.entry_side = 'long'
                            self.entry_bar = idx
                            self.entry_price = current_price
                            self.breakout_direction = 'long'
                            self.first_target_hit = False
                            
                            self.logger.info(f"Simple range breakout long entry above {self.range_resistance:.4f}")
                            
                            return {
                                'side': 'buy',
                                'price': current_price,
                                'confidence': 0.75,
                                'reason': f'simple_range_breakout_long_above_{self.range_resistance:.4f}'
                            }
                        
                        elif current_low <= breakout_short_price:
                            # Short breakout
                            self.initial_atr = self.data['atr'].iloc[idx] if not pd.isna(self.data['atr'].iloc[idx]) else current_price * 0.01
                            self.entry_side = 'short'
                            self.entry_bar = idx
                            self.entry_price = current_price
                            self.breakout_direction = 'short'
                            self.first_target_hit = False
                            
                            self.logger.info(f"Simple range breakout short entry below {self.range_support:.4f}")
                            
                            return {
                                'side': 'sell',
                                'price': current_price,
                                'confidence': 0.75,
                                'reason': f'simple_range_breakout_short_below_{self.range_support:.4f}'
                            }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _check_entry_conditions: {str(e)}")
            return None
    
    def check_exit(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check various exit conditions for squeeze breakout trades"""
        try:
            idx = len(self.data) - 1
            if idx < 0:
                return None
            
            current_price = self.data['close'].iloc[idx]
            
            # Check exit conditions
            should_exit, exit_reason = self.check_exit_conditions(idx)
            
            if should_exit:
                if exit_reason in ["quick_profit_long", "quick_profit_short"]:
                    # Take partial profits
                    partial_exit_pct = self.config.get('partial_exit_pct', 0.4)
                    self.logger.info(f"Squeeze breakout quick profit hit - taking {partial_exit_pct*100}% profits")
                    
                    return {
                        'action': 'partial_exit',
                        'price': current_price,
                        'partial_pct': partial_exit_pct,
                        'reason': exit_reason
                    }
                else:
                    # Exit completely
                    self.logger.info(f"Squeeze breakout exit: {exit_reason}")
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
            self.entry_bar = None
            self.entry_side = None
            self.entry_price = None
            self.breakout_direction = None
            self.first_target_hit = False
            
            # Log trade result
            exit_reason = trade_result.get('reason', 'unknown')
            pnl = trade_result.get('pnl', 0)
            
            self.logger.info(f"Squeeze breakout trade closed - {exit_reason}, PnL: {pnl}")
            
        except Exception as e:
            self.logger.error(f"Error in on_trade_closed: {str(e)}")
    
    def get_risk_parameters(self) -> Dict[str, Any]:
        """Get squeeze-based risk management parameters"""
        try:
            # Calculate range-based risk parameters
            if (hasattr(self, 'range_height') and self.range_height > 0 and
                hasattr(self, 'entry_price') and self.entry_price and
                hasattr(self, 'entry_side') and self.entry_side):
                
                current_price = self.data['close'].iloc[-1]
                initial_stop_buffer_pct = self.config.get('initial_stop_buffer_pct', 0.004)  # Widened from 0.002
                
                if self.entry_side == 'long':
                    # Stop just inside range resistance
                    stop_price = self.range_resistance * (1 - initial_stop_buffer_pct)
                    sl_pct = abs(current_price - stop_price) / current_price
                elif self.entry_side == 'short':
                    # Stop just inside range support
                    stop_price = self.range_support * (1 + initial_stop_buffer_pct)
                    sl_pct = abs(stop_price - current_price) / current_price
                else:
                    sl_pct = initial_stop_buffer_pct
                
                # Calculate target based on range projection
                range_projection_multiplier = self.config.get('range_projection_multiplier', 0.8)
                tp_pct = (self.range_height * range_projection_multiplier) / current_price
                
                # Conservative position sizing for breakout trades
                position_size_reduction = self.config.get('position_size_reduction', 0.9)
                max_position_pct = self.config.get('max_position_pct', 2.0) * position_size_reduction
                
                return {
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "max_position_pct": max_position_pct,
                    "risk_reward_ratio": tp_pct / sl_pct if sl_pct > 0 else 2.0
                }
            
            # Fallback to config defaults
            return {
                "sl_pct": self.config.get('sl_pct', 0.03),  # Widened from 0.015
                "tp_pct": self.config.get('tp_pct', 0.02),  # Quicker from 0.03
                "max_position_pct": self.config.get('max_position_pct', 2.0) * self.config.get('position_size_reduction', 0.9),
                "risk_reward_ratio": 2.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_risk_parameters: {str(e)}")
            return {
                "sl_pct": 0.03,  # Widened from 0.015
                "tp_pct": 0.02,  # Quicker from 0.03
                "max_position_pct": 1.8,
                "risk_reward_ratio": 2.0
            }