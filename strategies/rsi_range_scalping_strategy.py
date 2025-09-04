"""
RSI Range Scalping with Candlestick Confirmation Strategy

Strategy #4 in the Strategy Matrix

Market Conditions: Best fit for RANGING markets
Description: Uses RSI with candlestick patterns for range-bound scalping with tight stops
"""

import logging
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

from .strategy_template import StrategyTemplate

class StrategyRSIRangeScalping(StrategyTemplate):
    """
    RSI Range Scalping with Candlestick Confirmation Strategy
    
    A scalping strategy designed for ranging markets that combines RSI overbought/oversold
    signals with candlestick reversal patterns at key support and resistance levels.
    
    Market Type:
    -----------
    - Ranging markets (sideways movement)
    - Moderate volatility choppy markets
    - Horizontal channels with frequent reversals
    - No clear long-term directional trend
    
    Strategy Logic:
    --------------
    1. Range Detection:
       - Identify support and resistance levels
       - Use recent price highs/lows clustering
       - SMA reference for range middle
       - Validate minimum range width
    
    2. RSI Timing:
       - RSI ≤ 30 for oversold (long entries)
       - RSI ≥ 70 for overbought (short entries)
       - Avoid middle zone (RSI 40-60)
       - Quick exits on RSI normalization
    
    3. Candlestick Confirmation:
       - Bullish engulfing at support
       - Bearish engulfing at resistance
       - Hammer/pin bars at extremes
       - Shooting star at resistance
    
    4. Risk Management:
       - Tight stops outside S/R levels
       - Quick profit taking on RSI normalization
       - High win rate with small R:R ratios
       - Breakout protection with consecutive stop limits
    
    Entry Conditions:
    ----------------
    1. Long Entries (Near Support):
       - Price near established support level
       - RSI ≤ 30 (oversold condition)
       - Bullish candlestick pattern confirmation
       - Optional: RSI divergence for extra confirmation
    
    2. Short Entries (Near Resistance):
       - Price near established resistance level
       - RSI ≥ 70 (overbought condition)
       - Bearish candlestick pattern confirmation
       - Optional: RSI divergence for extra confirmation
    
    3. Avoidance Rules:
       - No trades in range middle zone
       - No trades during volume breakouts
       - Disable after consecutive stops
    
    Exit Conditions:
    ---------------
    1. RSI Normalization:
       - Long: Exit when RSI reaches 40-60
       - Short: Exit when RSI reaches 40-60
       - Quick profit taking approach
    
    2. Range Middle Approach:
       - Exit when approaching range midpoint
       - Avoid holding through full range
    
    3. Time-Based Exits:
       - Maximum 20 bars for scalping
       - Quick in-and-out approach
    
    4. Stop Loss Protection:
       - Tight stops just outside S/R levels
       - Typically 0.2% or less
    
    Risk Management:
    --------------
    - Very tight stops (0.2% typical)
    - 1.2:1 reward/risk ratio
    - High win rate strategy
    - Consecutive stop protection
    - Volume breakout detection
    """
    
    MARKET_TYPE_TAGS: List[str] = ['RANGING']
    SHOW_IN_SELECTION: bool = True
    
    def __init__(self, data: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(data, config, logger)
        
        # Range detection state - Enhanced for more selective entries
        self.range_active = False
        self.support_level = 0
        self.resistance_level = 0
        self.range_middle = 0
        self.range_width = 0
        self.last_range_update = 0
        self.min_range_stability_bars = self.config.get('min_range_stability_bars', 8)  # Minimum bars for range stability
        
        # Range trading performance tracking
        self.consecutive_stops = 0
        self.range_trading_enabled = True
        self.last_stop_bar = None
        
        # Candlestick pattern detection
        self.pattern_detected = False
        self.pattern_type = None
        self.pattern_bar = None
        
        # Trade management
        self.entry_bar = None
        self.entry_side = None
        self.entry_level = None
        self.last_trade_bar = -self.config.get('cooldown_bars', 3)
        
        self.logger.info("RSI Range Scalping strategy initialized")
    
    def init_indicators(self) -> None:
        """Initialize RSI and reference indicators"""
        try:
            # Import pandas_ta with fallback
            try:
                import pandas_ta as ta
                self.has_pandas_ta = True
            except ImportError:
                self.logger.warning("pandas_ta not available, using manual calculations")
                self.has_pandas_ta = False
            
            # Get strategy parameters
            rsi_period = self.config.get('rsi_period', 14)
            sma_reference = self.config.get('sma_reference', 100)
            volume_period = self.config.get('volume_period', 20)
            
            # RSI indicator
            if self.has_pandas_ta:
                self.data['rsi'] = ta.rsi(self.data['close'], length=rsi_period)
            else:
                self.data['rsi'] = self._calculate_rsi_manual(rsi_period)
            
            # Reference SMA for range middle
            self.data['sma_reference'] = self.data['close'].rolling(window=sma_reference).mean()
            
            # Volume analysis
            self.data['volume_sma'] = self.data['volume'].rolling(window=volume_period).mean()
            
            self.logger.debug("All indicators initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing indicators: {str(e)}")
            raise
    
    def _calculate_rsi_manual(self, period: int) -> pd.Series:
        """Manual RSI calculation as fallback"""
        try:
            delta = self.data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception:
            return pd.Series([50.0] * len(self.data), index=self.data.index)
    
    def detect_range_levels(self, idx: int) -> bool:
        """Detect support and resistance levels from recent price action"""
        try:
            range_lookback = self.config.get('range_lookback', 50)
            if idx < range_lookback:
                return False
            
            # Get recent highs and lows
            lookback_period = min(range_lookback, idx + 1)
            start_idx = max(0, idx - lookback_period + 1)
            end_idx = idx + 1
            
            highs = self.data['high'].iloc[start_idx:end_idx].tolist()
            lows = self.data['low'].iloc[start_idx:end_idx].tolist()
            
            # Find significant levels by clustering highs and lows
            highs.sort(reverse=True)
            lows.sort()
            
            # Take top quartile of highs and bottom quartile of lows
            top_quartile = max(1, int(len(highs) * 0.25))
            bottom_quartile = max(1, int(len(lows) * 0.25))
            
            # Calculate potential resistance (average of top highs)
            potential_resistance = sum(highs[:top_quartile]) / top_quartile
            
            # Calculate potential support (average of bottom lows)
            potential_support = sum(lows[:bottom_quartile]) / bottom_quartile
            
            # Check if we have a valid range
            range_width = (potential_resistance - potential_support) / potential_support
            min_range_width = self.config.get('min_range_width', 0.008)
            
            if range_width >= min_range_width:
                self.support_level = potential_support
                self.resistance_level = potential_resistance
                self.range_middle = (self.support_level + self.resistance_level) / 2
                self.range_width = range_width
                self.range_active = True
                self.last_range_update = idx
                
                self.logger.debug(f"Range detected: Support {self.support_level:.4f}, Resistance {self.resistance_level:.4f}, Width {range_width:.3f}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in detect_range_levels: {str(e)}")
            return False
    
    def is_near_support(self, price: float) -> bool:
        """Check if price is near support level"""
        if not self.range_active:
            return False
        
        tolerance_pct = self.config.get('support_resistance_tolerance', 0.002)
        tolerance = self.support_level * tolerance_pct
        return self.support_level - tolerance <= price <= self.support_level + tolerance
    
    def is_near_resistance(self, price: float) -> bool:
        """Check if price is near resistance level"""
        if not self.range_active:
            return False
        
        tolerance_pct = self.config.get('support_resistance_tolerance', 0.002)
        tolerance = self.resistance_level * tolerance_pct
        return self.resistance_level - tolerance <= price <= self.resistance_level + tolerance
    
    def is_in_range_middle(self, price: float) -> bool:
        """Check if price is in the middle zone (avoid trading here)"""
        if not self.range_active:
            return True
        
        range_size = self.resistance_level - self.support_level
        middle_buffer_pct = self.config.get('range_middle_buffer', 0.3)
        middle_buffer = range_size * middle_buffer_pct
        
        middle_low = self.range_middle - middle_buffer
        middle_high = self.range_middle + middle_buffer
        
        return middle_low <= price <= middle_high
    
    def detect_candlestick_patterns(self, idx: int) -> Tuple[bool, Optional[str]]:
        """Detect bullish and bearish candlestick patterns"""
        try:
            if idx < 1:
                return False, None
            
            # Current and previous candle data
            curr_open = self.data['open'].iloc[idx]
            curr_high = self.data['high'].iloc[idx]
            curr_low = self.data['low'].iloc[idx]
            curr_close = self.data['close'].iloc[idx]
            
            prev_open = self.data['open'].iloc[idx-1]
            prev_high = self.data['high'].iloc[idx-1]
            prev_low = self.data['low'].iloc[idx-1]
            prev_close = self.data['close'].iloc[idx-1]
            
            current_price = curr_close
            
            # Bullish patterns (for support bounce)
            if self.is_near_support(current_price):
                # Bullish engulfing
                if (prev_close < prev_open and  # Previous red candle
                    curr_close > curr_open and  # Current green candle
                    curr_open < prev_close and  # Current opens below prev close
                    curr_close > prev_open):    # Current closes above prev open
                    return True, 'bullish_engulfing'
                
                # Hammer/Pin bar at support
                body_size = abs(curr_close - curr_open)
                lower_shadow = curr_open - curr_low if curr_close > curr_open else curr_close - curr_low
                upper_shadow = curr_high - curr_close if curr_close > curr_open else curr_high - curr_open
                
                if (lower_shadow > 2 * body_size and  # Long lower shadow
                    upper_shadow < body_size * 0.5):  # Small upper shadow
                    return True, 'hammer'
            
            # Bearish patterns (for resistance rejection)
            elif self.is_near_resistance(current_price):
                # Bearish engulfing
                if (prev_close > prev_open and  # Previous green candle
                    curr_close < curr_open and  # Current red candle
                    curr_open > prev_close and  # Current opens above prev close
                    curr_close < prev_open):    # Current closes below prev open
                    return True, 'bearish_engulfing'
                
                # Shooting star at resistance
                body_size = abs(curr_close - curr_open)
                upper_shadow = curr_high - curr_close if curr_close < curr_open else curr_high - curr_open
                lower_shadow = curr_open - curr_low if curr_close < curr_open else curr_close - curr_low
                
                if (upper_shadow > 2 * body_size and  # Long upper shadow
                    lower_shadow < body_size * 0.5):  # Small lower shadow
                    return True, 'shooting_star'
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error in detect_candlestick_patterns: {str(e)}")
            return False, None
    
    def check_rsi_divergence(self, idx: int, direction: str) -> bool:
        """Check for RSI divergence at support/resistance"""
        try:
            if idx < 10:
                return False
            
            current_price = self.data['close'].iloc[idx]
            current_rsi = self.data['rsi'].iloc[idx]
            
            if pd.isna(current_rsi):
                return False
            
            # Look back for previous touch of same level
            for i in range(2, min(20, idx)):
                past_idx = idx - i
                past_price = self.data['close'].iloc[past_idx]
                past_rsi = self.data['rsi'].iloc[past_idx]
                
                if pd.isna(past_rsi):
                    continue
                
                if direction == 'bullish' and self.is_near_support(past_price):
                    # Bullish divergence: lower low in price, higher low in RSI
                    if current_price < past_price and current_rsi > past_rsi:
                        return True
                elif direction == 'bearish' and self.is_near_resistance(past_price):
                    # Bearish divergence: higher high in price, lower high in RSI
                    if current_price > past_price and current_rsi < past_rsi:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in check_rsi_divergence: {str(e)}")
            return False
    
    def check_volume_breakout_signal(self, idx: int) -> bool:
        """Check for volume surge that might indicate breakout"""
        try:
            volume_period = self.config.get('volume_period', 20)
            if idx < volume_period:
                return False
            
            current_volume = self.data['volume'].iloc[idx]
            avg_volume = self.data['volume_sma'].iloc[idx]
            
            if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume == 0:
                return False
            
            volume_threshold = self.config.get('volume_surge_threshold', 1.5)
            return current_volume > avg_volume * volume_threshold
            
        except Exception:
            return False
    
    def check_range_exit_conditions(self, idx: int) -> bool:
        """Check for range scalping exit conditions"""
        try:
            if not hasattr(self, 'entry_bar') or self.entry_bar is None:
                return False
            
            current_price = self.data['close'].iloc[idx]
            current_rsi = self.data['rsi'].iloc[idx]
            bars_held = idx - self.entry_bar
            
            if pd.isna(current_rsi):
                return False
            
            # Time stop
            time_stop_bars = self.config.get('time_stop_bars', 20)
            if bars_held >= time_stop_bars:
                return True
            
            rsi_neutral_low = self.config.get('rsi_neutral_low', 40)
            rsi_neutral_high = self.config.get('rsi_neutral_high', 60)
            
            if self.entry_side == 'long':
                # Exit when RSI reaches neutral zone
                if current_rsi >= rsi_neutral_low:
                    return True
                # Exit when approaching range middle
                if current_price >= self.range_middle * 0.95:
                    return True
            
            elif self.entry_side == 'short':
                # Exit when RSI reaches neutral zone
                if current_rsi <= rsi_neutral_high:
                    return True
                # Exit when approaching range middle
                if current_price <= self.range_middle * 1.05:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in check_range_exit_conditions: {str(e)}")
            return False
    
    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for RSI range scalping entry opportunities"""
        try:
            idx = len(self.data) - 1
            if idx < 0:
                return None
            
            # Check cooldown period
            cooldown_bars = self.config.get('cooldown_bars', 3)
            if idx - self.last_trade_bar < cooldown_bars:
                return None
            
            # Need minimum bars for all indicators
            min_bars = max(
                self.config.get('rsi_period', 14),
                self.config.get('sma_reference', 100),
                self.config.get('range_lookback', 50)
            )
            if idx < min_bars:
                return None
            
            current_price = self.data['close'].iloc[idx]
            current_rsi = self.data['rsi'].iloc[idx]
            
            if pd.isna(current_price) or pd.isna(current_rsi):
                return None
            
            # Update range levels
            if not self.range_active or idx - self.last_range_update > 20:
                self.detect_range_levels(idx)
            
            # Check if range trading is disabled due to consecutive stops
            if not self.range_trading_enabled:
                # Re-enable after some bars have passed
                if self.last_stop_bar and idx - self.last_stop_bar > 50:
                    self.range_trading_enabled = True
                    self.consecutive_stops = 0
                return None
            
            # Check for volume breakout signal
            if self.check_volume_breakout_signal(idx):
                self.logger.debug("Volume breakout detected, avoiding range trades")
                return None
            
            # Enhanced range stability check
            range_stable = (self.last_range_update is not None and 
                           idx - self.last_range_update >= self.min_range_stability_bars)
            
            # Only trade if we have an active, stable range and not in middle zone
            if not self.range_active or not range_stable or self.is_in_range_middle(current_price):
                return None
            
            # Check for entry conditions
            pattern_detected, pattern_type = self.detect_candlestick_patterns(idx)
            
            rsi_oversold = self.config.get('rsi_oversold', 30)
            rsi_overbought = self.config.get('rsi_overbought', 70)
            
            # Long entry conditions (near support)
            if (self.is_near_support(current_price) and 
                current_rsi <= rsi_oversold and
                pattern_detected and pattern_type in ['bullish_engulfing', 'hammer']):
                
                # Additional confirmation with divergence (optional)
                divergence_ok = self.check_rsi_divergence(idx, 'bullish') or True
                
                if divergence_ok:
                    self.entry_side = 'long'
                    self.entry_level = self.support_level
                    self.entry_bar = idx
                    
                    self.logger.info(f"RSI range scalping long entry - RSI: {current_rsi:.1f}, Pattern: {pattern_type}")
                    
                    return {
                        'side': 'buy',
                        'price': current_price,
                        'confidence': 0.8,
                        'reason': 'rsi_oversold_bounce'
                    }
            
            # Short entry conditions (near resistance)
            elif (self.is_near_resistance(current_price) and 
                  current_rsi >= rsi_overbought and
                  pattern_detected and pattern_type in ['bearish_engulfing', 'shooting_star']):
                
                # Additional confirmation with divergence (optional)
                divergence_ok = self.check_rsi_divergence(idx, 'bearish') or True
                
                if divergence_ok:
                    self.entry_side = 'short'
                    self.entry_level = self.resistance_level
                    self.entry_bar = idx
                    
                    self.logger.info(f"RSI range scalping short entry - RSI: {current_rsi:.1f}, Pattern: {pattern_type}")
                    
                    return {
                        'side': 'sell',
                        'price': current_price,
                        'confidence': 0.8,
                        'reason': 'rsi_overbought_rejection'
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _check_entry_conditions: {str(e)}")
            return None
    
    def check_exit(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check various exit conditions for range scalping trades"""
        try:
            idx = len(self.data) - 1
            if idx < 0:
                return None
            
            current_price = self.data['close'].iloc[idx]
            
            # Check for early exit conditions
            should_exit = self.check_range_exit_conditions(idx)
            
            if should_exit:
                self.logger.info("Range scalping exit conditions met")
                return {
                    'action': 'exit',
                    'price': current_price,
                    'reason': 'range_scalp_exit'
                }
            
            # Check for volume breakout (emergency exit)
            if self.check_volume_breakout_signal(idx):
                self.logger.info("Volume breakout detected, emergency exit")
                return {
                    'action': 'exit',
                    'price': current_price,
                    'reason': 'volume_breakout'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in check_exit: {str(e)}")
            return None
    
    def on_trade_closed(self, symbol: str, trade_result: Dict[str, Any]) -> None:
        """Handle trade closure cleanup and consecutive stop tracking"""
        try:
            self.last_trade_bar = len(self.data) - 1
            
            # Reset trade state
            self.entry_bar = None
            self.entry_side = None
            self.entry_level = None
            self.entry_price = None
            
            # Track consecutive stops
            exit_reason = trade_result.get('reason', 'unknown')
            pnl = trade_result.get('pnl', 0)
            
            if pnl < 0:  # Loss
                self.consecutive_stops += 1
                self.last_stop_bar = len(self.data) - 1
                
                consecutive_limit = self.config.get('consecutive_stops_limit', 2)
                # Disable range trading after consecutive stops
                if self.consecutive_stops >= consecutive_limit:
                    self.range_trading_enabled = False
                    self.logger.warning(f"Range trading disabled after {self.consecutive_stops} consecutive stops")
            else:
                # Reset consecutive stops on winning trade
                self.consecutive_stops = 0
            
            self.logger.info(f"Trade closed - {exit_reason}, PnL: {pnl}")
            
        except Exception as e:
            self.logger.error(f"Error in on_trade_closed: {str(e)}")
    
    def get_risk_parameters(self) -> Dict[str, Any]:
        """Get range scalping risk management parameters"""
        try:
            # Calculate tight stops based on support/resistance levels
            if (self.range_active and 
                hasattr(self, 'entry_side') and 
                self.entry_side and 
                hasattr(self, 'entry_level') and 
                self.entry_level):
                
                current_price = self.data['close'].iloc[-1]
                stop_loss_pct = self.config.get('stop_loss_pct', 0.004)  # Widened from 0.002
                take_profit_ratio = self.config.get('take_profit_ratio', 1.0)  # Quicker 1:1 from 1.2:1
                
                if self.entry_side == 'long':
                    # Stop just below support
                    stop_price = self.entry_level * (1 - stop_loss_pct)
                    stop_distance = current_price - stop_price
                    target_price = current_price + (stop_distance * take_profit_ratio)
                    
                    sl_pct = abs(current_price - stop_price) / current_price
                    tp_pct = abs(target_price - current_price) / current_price
                    
                elif self.entry_side == 'short':
                    # Stop just above resistance
                    stop_price = self.entry_level * (1 + stop_loss_pct)
                    stop_distance = stop_price - current_price
                    target_price = current_price - (stop_distance * take_profit_ratio)
                    
                    sl_pct = abs(stop_price - current_price) / current_price
                    tp_pct = abs(current_price - target_price) / current_price
                
                return {
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "max_position_pct": self.config.get('max_position_pct', 2.0),
                    "risk_reward_ratio": tp_pct / sl_pct if sl_pct > 0 else take_profit_ratio
                }
            
            # Fallback to config defaults
            return {
                "sl_pct": self.config.get('stop_loss_pct', 0.004),  # Widened from 0.002
                "tp_pct": self.config.get('stop_loss_pct', 0.002) * self.config.get('take_profit_ratio', 1.0),  # Quicker 1:1
                "max_position_pct": self.config.get('max_position_pct', 2.0),
                "risk_reward_ratio": self.config.get('take_profit_ratio', 1.2)
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_risk_parameters: {str(e)}")
            return {
                "sl_pct": 0.004,  # Widened from 0.002
                "tp_pct": 0.004,  # Quicker 1:1 ratio
                "max_position_pct": 2.0,
                "risk_reward_ratio": 1.2
            } 