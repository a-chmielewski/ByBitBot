"""
Volatility Reversal Scalping Strategy

Strategy #6 in the Strategy Matrix

Market Conditions: Good for HIGH_VOLATILITY markets and reversal conditions
Description: Identifies volatility spikes and trades reversals after extreme moves
"""

import logging
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

from .strategy_template import StrategyTemplate

class StrategyVolatilityReversalScalping(StrategyTemplate):
    """
    Volatility Reversal Scalping Strategy (Fade the Extremes)
    
    An advanced contrarian strategy designed for high-volatility conditions that produces
    over-extended price spikes. The strategy identifies emotionally driven surges and
    positions for short-term reversals once exhaustion signals appear.
    
    Market Type:
    -----------
    - High-volatility conditions with extreme price spikes
    - Emotionally driven moves (news, liquidations)
    - Markets showing clear exhaustion signals
    - Liquid markets with mean reversion tendencies
    
    Strategy Logic:
    --------------
    1. Spike Detection:
       - Bollinger Band extreme breaches (2.5σ+ from mean)
       - ATR-based extreme candle identification (2.5x+ normal range)
       - Volume climax detection (2x+ average volume)
       - RSI extreme readings (80+/20- levels)
       - Scoring system for spike confirmation
    
    2. Reversal Confirmation:
       - Candlestick reversal patterns (shooting star, hammer, engulfing)
       - Failure to make new highs/lows after spike
       - RSI divergence at extremes
       - Volume exhaustion signals
    
    3. Entry Methods:
       - Immediate entry on very high confidence scores (5+)
       - Traditional spike + reversal confirmation
       - Moderate overbought/oversold with volume
       - Alternative BB breach with RSI + volume
    
    4. Quick Exit Management:
       - Fibonacci retracement targets (38.2%, 50%)
       - Bollinger Band middle or EMA mean reversion
       - Time-based exits (2-8 bars maximum)
       - Partial profit taking at first targets
    
    Entry Conditions:
    ----------------
    1. Extreme Spike Detection:
       - Price beyond 2.5σ Bollinger Bands
       - Candle range > 2.5x ATR
       - Volume > 2x average
       - RSI > 80 (up-spike) or < 20 (down-spike)
       - Minimum spike size (0.5%+ body)
    
    2. Reversal Triggers:
       - Shooting star after up-spike
       - Hammer after down-spike
       - Bearish/bullish engulfing patterns
       - Failed breakout patterns
       - RSI divergence confirmation
    
    3. Entry Scoring (0-12 points):
       - BB position: 2 pts extreme, 1 pt regular breach
       - RSI levels: 2 pts extreme, 1 pt moderate
       - Volume climax: 2 pts high, 1 pt elevated
       - ATR candle size: 2 pts extreme, 1 pt large
       - Body size: 1 pt if meets minimum
       - Candle direction: 1 pt confirmation
    
    Exit Conditions:
    ---------------
    1. Mean Reversion Targets:
       - Bollinger Band middle (20 EMA)
       - Fibonacci retracements (38.2%, 50%)
       - Previous support/resistance levels
    
    2. Quick Scalp Exits:
       - 0.2% profit on first target
       - Partial exits at key levels
       - Time stops (8 bars maximum)
    
    3. Emergency Exits:
       - Momentum continuation against position
       - Volume breakout in original direction
       - RSI re-extreme in original direction
    
    Risk Management:
    --------------
    - Very tight stops (0.2% beyond spike extreme)
    - Reduced position sizing (70% of normal)
    - Maximum 4 consecutive losses before disable
    - Quick time exits (2-8 bars)
    - Partial profit taking approach
    """
    
    MARKET_TYPE_TAGS: List[str] = ['HIGH_VOLATILITY', 'RANGING']
    SHOW_IN_SELECTION: bool = True
    
    def __init__(self, data: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(data, config, logger)
        
        # Spike detection state
        self.extreme_spike_detected = False
        self.spike_direction = None  # 'up' or 'down'
        self.spike_high = 0
        self.spike_low = 0
        self.spike_bar = None
        self.spike_size = 0
        self.spike_volume = 0
        
        # Reversal pattern tracking
        self.reversal_pattern_detected = False
        self.pattern_type = None
        self.pattern_confirmation_bar = None
        
        # Risk management
        self.consecutive_losses = 0
        self.trading_enabled = True
        
        # Trade management
        self.entry_bar = None
        self.entry_side = None
        self.entry_price = None
        self.first_target_hit = False
        self.last_trade_bar = -self.config.get('cooldown_bars', 2)
        
        self.logger.info("Volatility Reversal Scalping strategy initialized")
    
    def init_indicators(self) -> None:
        """Initialize volatility and reversal indicators"""
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
            atr_period = self.config.get('atr_period', 14)
            rsi_period = self.config.get('rsi_period', 7)
            ema_period = self.config.get('ema_period', 20)
            volume_period = self.config.get('volume_avg_period', 20)
            
            # Bollinger Bands
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
            
            # ATR indicator
            if self.has_pandas_ta:
                self.data['atr'] = ta.atr(self.data['high'], self.data['low'], self.data['close'], length=atr_period)
            else:
                self.data['atr'] = self._calculate_atr_manual(atr_period)
            
            # RSI indicator (fast for extremes)
            if self.has_pandas_ta:
                self.data['rsi'] = ta.rsi(self.data['close'], length=rsi_period)
            else:
                self.data['rsi'] = self._calculate_rsi_manual(rsi_period)
            
            # EMA for mean reversion target
            if self.has_pandas_ta:
                self.data['ema'] = ta.ema(self.data['close'], length=ema_period)
            else:
                self.data['ema'] = self.data['close'].ewm(span=ema_period).mean()
            
            # Volume analysis
            self.data['volume_sma'] = self.data['volume'].rolling(window=volume_period).mean()
            
            self.logger.info("All volatility indicators initialized successfully")
            
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
    
    def evaluate_spike_conditions(self, idx: int) -> Tuple[int, Optional[str], Optional[Dict[str, Any]]]:
        """Evaluate spike conditions using a scoring system"""
        try:
            if idx < 1:
                return 0, None, None
            
            current_price = self.data['close'].iloc[idx]
            current_high = self.data['high'].iloc[idx]
            current_low = self.data['low'].iloc[idx]
            current_open = self.data['open'].iloc[idx]
            current_volume = self.data['volume'].iloc[idx]
            
            bb_upper = self.data['bb_upper'].iloc[idx]
            bb_lower = self.data['bb_lower'].iloc[idx]
            bb_middle = self.data['bb_middle'].iloc[idx]
            atr_value = self.data['atr'].iloc[idx]
            rsi_value = self.data['rsi'].iloc[idx]
            avg_volume = self.data['volume_sma'].iloc[idx]
            
            if pd.isna(bb_upper) or pd.isna(atr_value) or pd.isna(rsi_value) or pd.isna(avg_volume):
                return 0, None, None
            
            # Calculate extreme Bollinger Band levels
            bb_range = bb_upper - bb_lower
            bb_extreme_std = self.config.get('bb_extreme_std', 2.5)
            bb_extreme_upper = bb_middle + (bb_range * bb_extreme_std / 2)
            bb_extreme_lower = bb_middle - (bb_range * bb_extreme_std / 2)
            
            # Calculate candle metrics
            candle_range = current_high - current_low
            candle_body = abs(current_price - current_open)
            candle_body_pct = candle_body / current_price if current_price > 0 else 0
            
            # Get thresholds
            atr_extreme_multiplier = self.config.get('atr_extreme_multiplier', 2.5)
            volume_climax_multiplier = self.config.get('volume_climax_multiplier', 2.0)
            rsi_overbought_extreme = self.config.get('rsi_overbought_extreme', 80)
            rsi_oversold_extreme = self.config.get('rsi_oversold_extreme', 20)
            rsi_overbought_moderate = self.config.get('rsi_overbought_moderate', 75)
            rsi_oversold_moderate = self.config.get('rsi_oversold_moderate', 25)
            min_spike_size_pct = self.config.get('min_spike_size_pct', 0.005)
            
            # Initialize scores for up and down spikes
            up_score = 0
            down_score = 0
            
            # UP-SPIKE SCORING
            # 1. Bollinger Band position (2 points for extreme, 1 for regular breach)
            if current_price > bb_extreme_upper:
                up_score += 2
            elif current_price > bb_upper:
                up_score += 1
            
            # 2. RSI levels (2 points for extreme, 1 for moderate)
            if rsi_value >= rsi_overbought_extreme:
                up_score += 2
            elif rsi_value >= rsi_overbought_moderate:
                up_score += 1
            
            # 3. Volume climax (2 points for high volume, 1 for elevated)
            if current_volume > avg_volume * volume_climax_multiplier:
                up_score += 2
            elif current_volume > avg_volume * (volume_climax_multiplier * 0.7):
                up_score += 1
            
            # 4. Candle size relative to ATR (2 points for extreme, 1 for large)
            if candle_range > atr_value * atr_extreme_multiplier:
                up_score += 2
            elif candle_range > atr_value * (atr_extreme_multiplier * 0.7):
                up_score += 1
            
            # 5. Minimum body size (1 point if met)
            if candle_body_pct > min_spike_size_pct:
                up_score += 1
            
            # 6. Bullish candle confirmation (1 point for green candle on up-spike)
            if current_price > current_open:
                up_score += 1
            
            # DOWN-SPIKE SCORING
            # 1. Bollinger Band position
            if current_price < bb_extreme_lower:
                down_score += 2
            elif current_price < bb_lower:
                down_score += 1
            
            # 2. RSI levels
            if rsi_value <= rsi_oversold_extreme:
                down_score += 2
            elif rsi_value <= rsi_oversold_moderate:
                down_score += 1
            
            # 3. Volume climax
            if current_volume > avg_volume * volume_climax_multiplier:
                down_score += 2
            elif current_volume > avg_volume * (volume_climax_multiplier * 0.7):
                down_score += 1
            
            # 4. Candle size relative to ATR
            if candle_range > atr_value * atr_extreme_multiplier:
                down_score += 2
            elif candle_range > atr_value * (atr_extreme_multiplier * 0.7):
                down_score += 1
            
            # 5. Minimum body size
            if candle_body_pct > min_spike_size_pct:
                down_score += 1
            
            # 6. Bearish candle confirmation (1 point for red candle on down-spike)
            if current_price < current_open:
                down_score += 1
            
            # Determine direction and return best score
            min_score_threshold = self.config.get('min_score_threshold', 3)
            
            if up_score >= down_score and up_score >= min_score_threshold:
                spike_size = (current_high - bb_middle) / bb_middle if bb_middle > 0 else 0
                return up_score, 'up', {
                    'direction': 'up', 
                    'size': spike_size, 
                    'extreme_price': current_high,
                    'score': up_score
                }
            elif down_score > up_score and down_score >= min_score_threshold:
                spike_size = (bb_middle - current_low) / bb_middle if bb_middle > 0 else 0
                return down_score, 'down', {
                    'direction': 'down', 
                    'size': spike_size, 
                    'extreme_price': current_low,
                    'score': down_score
                }
            
            return 0, None, None
            
        except Exception as e:
            self.logger.error(f"Error in evaluate_spike_conditions: {str(e)}")
            return 0, None, None
    
    def detect_reversal_pattern(self, idx: int, spike_direction: str) -> Tuple[bool, Optional[str]]:
        """Detect reversal candlestick patterns after extreme spike"""
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
            
            # After up-spike, look for bearish reversal patterns
            if spike_direction == 'up':
                # Shooting star (long upper shadow, small body)
                body_size = abs(curr_close - curr_open)
                upper_shadow = curr_high - max(curr_open, curr_close)
                lower_shadow = min(curr_open, curr_close) - curr_low
                
                if (upper_shadow > 2 * body_size and  # Long upper shadow
                    lower_shadow < body_size * 0.5 and  # Small lower shadow
                    curr_high < prev_high):  # Failed to make new high
                    return True, 'shooting_star'
                
                # Bearish engulfing
                if (prev_close > prev_open and  # Previous green candle
                    curr_close < curr_open and  # Current red candle
                    curr_open > prev_close and  # Opens above prev close
                    curr_close < prev_open and  # Closes below prev open
                    curr_high <= prev_high):  # Failed to exceed previous high
                    return True, 'bearish_engulfing'
                
                # Simple failure to make new high with red candle
                if (curr_close < curr_open and  # Red candle
                    curr_high <= prev_high and  # No new high
                    curr_close < prev_close):  # Closes below previous close
                    return True, 'failed_breakout_bearish'
            
            # After down-spike, look for bullish reversal patterns
            elif spike_direction == 'down':
                # Hammer (long lower shadow, small body)
                body_size = abs(curr_close - curr_open)
                lower_shadow = min(curr_open, curr_close) - curr_low
                upper_shadow = curr_high - max(curr_open, curr_close)
                
                if (lower_shadow > 2 * body_size and  # Long lower shadow
                    upper_shadow < body_size * 0.5 and  # Small upper shadow
                    curr_low > prev_low):  # Failed to make new low
                    return True, 'hammer'
                
                # Bullish engulfing
                if (prev_close < prev_open and  # Previous red candle
                    curr_close > curr_open and  # Current green candle
                    curr_open < prev_close and  # Opens below prev close
                    curr_close > prev_open and  # Closes above prev open
                    curr_low >= prev_low):  # Failed to go below previous low
                    return True, 'bullish_engulfing'
                
                # Simple failure to make new low with green candle
                if (curr_close > curr_open and  # Green candle
                    curr_low >= prev_low and  # No new low
                    curr_close > prev_close):  # Closes above previous close
                    return True, 'failed_breakout_bullish'
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error in detect_reversal_pattern: {str(e)}")
            return False, None
    
    def check_rsi_divergence(self, idx: int, spike_direction: str) -> bool:
        """Check for RSI divergence at extremes"""
        try:
            if idx < 5:
                return False
            
            current_price = self.data['close'].iloc[idx]
            current_rsi = self.data['rsi'].iloc[idx]
            
            if pd.isna(current_rsi):
                return False
            
            rsi_overbought_extreme = self.config.get('rsi_overbought_extreme', 80)
            rsi_oversold_extreme = self.config.get('rsi_oversold_extreme', 20)
            
            # Look back for previous extreme
            for i in range(2, min(10, idx)):
                past_idx = idx - i
                past_price = self.data['close'].iloc[past_idx]
                past_rsi = self.data['rsi'].iloc[past_idx]
                
                if pd.isna(past_rsi):
                    continue
                
                if spike_direction == 'up':
                    # Bearish divergence: higher high in price, lower high in RSI
                    if (current_price > past_price and 
                        current_rsi < past_rsi and 
                        past_rsi >= rsi_overbought_extreme):
                        return True
                elif spike_direction == 'down':
                    # Bullish divergence: lower low in price, higher low in RSI
                    if (current_price < past_price and 
                        current_rsi > past_rsi and 
                        past_rsi <= rsi_oversold_extreme):
                        return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in check_rsi_divergence: {str(e)}")
            return False
    
    def calculate_retracement_targets(self, idx: int, spike_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
        """Calculate Fibonacci retracement targets"""
        try:
            bb_middle = self.data['bb_middle'].iloc[idx]
            spike_price = spike_data['extreme_price']
            
            if pd.isna(bb_middle):
                return None, None
            
            fib_382 = self.config.get('fibonacci_retracement_1', 0.382)
            fib_50 = self.config.get('fibonacci_retracement_2', 0.5)
            
            if spike_data['direction'] == 'up':
                # For up-spike fade, calculate downward retracement
                spike_size = spike_price - bb_middle
                target_382 = spike_price - (spike_size * fib_382)
                target_50 = spike_price - (spike_size * fib_50)
                return target_382, target_50
            else:
                # For down-spike fade, calculate upward retracement
                spike_size = bb_middle - spike_price
                target_382 = spike_price + (spike_size * fib_382)
                target_50 = spike_price + (spike_size * fib_50)
                return target_382, target_50
                
        except Exception as e:
            self.logger.error(f"Error in calculate_retracement_targets: {str(e)}")
            return None, None
    
    def check_exit_conditions(self, idx: int) -> Tuple[bool, Optional[str]]:
        """Check for reversal scalp exit conditions"""
        try:
            if not hasattr(self, 'entry_bar') or self.entry_bar is None:
                return False, None
            
            current_price = self.data['close'].iloc[idx]
            bb_middle = self.data['bb_middle'].iloc[idx]
            bb_upper = self.data['bb_upper'].iloc[idx]
            bb_lower = self.data['bb_lower'].iloc[idx]
            ema_level = self.data['ema'].iloc[idx]
            rsi_value = self.data['rsi'].iloc[idx]
            
            bars_held = idx - self.entry_bar
            
            if pd.isna(current_price) or pd.isna(rsi_value):
                return False, None
            
            # Time-based exit
            max_hold_bars = self.config.get('max_hold_bars', 8)
            if bars_held >= max_hold_bars:
                return True, "time_exit"
            
            # Emergency stop-loss check
            if self.entry_side == 'long':
                # Emergency stop if we break significantly below entry
                if bars_held >= 2 and current_price < self.entry_price * 0.995:  # 0.5% stop
                    return True, "emergency_stop_long"
            elif self.entry_side == 'short':
                # Emergency stop if we break significantly above entry
                if bars_held >= 2 and current_price > self.entry_price * 1.005:  # 0.5% stop
                    return True, "emergency_stop_short"
            
            if self.entry_side == 'long':  # Long position (fading down-spike)
                # Target 1: Quick scalp target
                if current_price >= self.entry_price * 1.002 and not self.first_target_hit:  # 0.2% profit
                    self.first_target_hit = True
                    return True, "quick_scalp_long"
                
                # Target 2: BB middle or EMA
                target_1 = max(ema_level, bb_middle) if not pd.isna(ema_level) and not pd.isna(bb_middle) else current_price * 1.005
                if current_price >= target_1 and bars_held >= 1:
                    return True, "first_target_long"
                
                # Target 3: Further retracement
                if self.first_target_hit and current_price >= target_1 * 1.005:
                    return True, "second_target_long"
                
                # RSI reversal signal
                if rsi_value >= 70 and bars_held >= 1:  # RSI getting overbought again
                    return True, "rsi_reversal_long"
            
            elif self.entry_side == 'short':  # Short position (fading up-spike)
                # Target 1: Quick scalp target
                if current_price <= self.entry_price * 0.998 and not self.first_target_hit:  # 0.2% profit
                    self.first_target_hit = True
                    return True, "quick_scalp_short"
                
                # Target 2: BB middle or EMA
                target_1 = min(ema_level, bb_middle) if not pd.isna(ema_level) and not pd.isna(bb_middle) else current_price * 0.995
                if current_price <= target_1 and bars_held >= 1:
                    return True, "first_target_short"
                
                # Target 3: Further retracement
                if self.first_target_hit and current_price <= target_1 * 0.995:
                    return True, "second_target_short"
                
                # RSI reversal signal
                if rsi_value <= 30 and bars_held >= 1:  # RSI getting oversold again
                    return True, "rsi_reversal_short"
            
            # Momentum reversal against us (price continues in original spike direction)
            if bars_held >= 2:
                if self.entry_side == 'long' and current_price < bb_lower:  # Long but price going lower
                    return True, "momentum_reversal_long"
                elif self.entry_side == 'short' and current_price > bb_upper:  # Short but price going higher
                    return True, "momentum_reversal_short"
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error in check_exit_conditions: {str(e)}")
            return False, None
    
    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for volatility reversal scalping entry opportunities"""
        try:
            idx = len(self.data) - 1
            if idx < 0:
                return None
            
            # Check cooldown period
            cooldown_bars = self.config.get('cooldown_bars', 2)
            if idx - self.last_trade_bar < cooldown_bars:
                return None
            
            # Check if trading is disabled due to consecutive losses
            if not self.trading_enabled:
                return None
            
            # Need minimum bars for all indicators
            min_bars = max(
                self.config.get('bb_period', 20),
                self.config.get('atr_period', 14),
                self.config.get('rsi_period', 7),
                self.config.get('ema_period', 20)
            )
            if idx < min_bars:
                return None
            
            current_price = self.data['close'].iloc[idx]
            if pd.isna(current_price):
                return None
            
            # Method 1: Immediate entry on very high score (strong signals)
            score, direction, spike_data = self.evaluate_spike_conditions(idx)
            immediate_entry_score = self.config.get('immediate_entry_score', 5)
            
            if score >= immediate_entry_score and spike_data:
                self.entry_side = 'short' if spike_data['direction'] == 'up' else 'long'
                self.entry_bar = idx
                self.entry_price = current_price
                self.first_target_hit = False
                
                self.logger.info(f"Volatility reversal immediate entry - Score: {score}, Direction: {spike_data['direction']}")
                
                return {
                    'action': self.entry_side,
                    'price': current_price,
                    'confidence': 0.9,
                    'reason': f'volatility_reversal_immediate_{spike_data["direction"]}_spike'
                }
            
            # Method 2: Traditional spike detection and reversal confirmation
            min_score_threshold = self.config.get('min_score_threshold', 3)
            if score >= min_score_threshold and spike_data:
                self.extreme_spike_detected = True
                self.spike_direction = spike_data['direction']
                self.spike_size = spike_data['size']
                
                if spike_data['direction'] == 'up':
                    self.spike_high = spike_data['extreme_price']
                else:
                    self.spike_low = spike_data['extreme_price']
                
                self.spike_bar = idx
                return None  # Wait for next bar to check reversal
            
            # Method 3: Check for reversal pattern after detected spike
            reversal_confirmation_bars = self.config.get('reversal_confirmation_bars', 3)
            if (self.extreme_spike_detected and 
                self.spike_bar is not None and 
                idx - self.spike_bar <= reversal_confirmation_bars):
                
                pattern_detected, pattern_type = self.detect_reversal_pattern(idx, self.spike_direction)
                divergence_confirmed = self.check_rsi_divergence(idx, self.spike_direction)
                
                # Enter if we have pattern OR divergence (more flexible)
                if pattern_detected or divergence_confirmed:
                    self.entry_side = 'short' if self.spike_direction == 'up' else 'long'
                    self.entry_bar = idx
                    self.entry_price = current_price
                    self.first_target_hit = False
                    
                    # Reset spike detection
                    self.extreme_spike_detected = False
                    self.spike_direction = None
                    
                    confirmation_type = pattern_type if pattern_detected else 'divergence'
                    self.logger.info(f"Volatility reversal pattern entry - Pattern: {confirmation_type}")
                    
                    return {
                        'action': self.entry_side,
                        'price': current_price,
                        'confidence': 0.8,
                        'reason': f'volatility_reversal_pattern_{confirmation_type}'
                    }
            
            # Method 4: Alternative entry on moderate BB breach with volume + RSI
            elif not self.extreme_spike_detected:
                bb_upper = self.data['bb_upper'].iloc[idx]
                bb_lower = self.data['bb_lower'].iloc[idx]
                rsi_value = self.data['rsi'].iloc[idx]
                current_volume = self.data['volume'].iloc[idx]
                avg_volume = self.data['volume_sma'].iloc[idx]
                
                if pd.isna(bb_upper) or pd.isna(rsi_value) or pd.isna(avg_volume):
                    return None
                
                rsi_overbought_moderate = self.config.get('rsi_overbought_moderate', 75)
                rsi_oversold_moderate = self.config.get('rsi_oversold_moderate', 25)
                
                # Moderate overbought condition
                if (current_price > bb_upper and 
                    rsi_value >= rsi_overbought_moderate and
                    current_volume > avg_volume * 1.5):  # 1.5x volume instead of 2x
                    
                    self.entry_side = 'short'
                    self.entry_bar = idx
                    self.entry_price = current_price
                    self.first_target_hit = False
                    
                    self.logger.info(f"Volatility reversal moderate overbought entry - RSI: {rsi_value:.1f}")
                    
                    return {
                        'action': 'short',
                        'price': current_price,
                        'confidence': 0.7,
                        'reason': 'volatility_reversal_moderate_overbought'
                    }
                
                # Moderate oversold condition
                elif (current_price < bb_lower and 
                      rsi_value <= rsi_oversold_moderate and
                      current_volume > avg_volume * 1.5):
                    
                    self.entry_side = 'long'
                    self.entry_bar = idx
                    self.entry_price = current_price
                    self.first_target_hit = False
                    
                    self.logger.info(f"Volatility reversal moderate oversold entry - RSI: {rsi_value:.1f}")
                    
                    return {
                        'action': 'long',
                        'price': current_price,
                        'confidence': 0.7,
                        'reason': 'volatility_reversal_moderate_oversold'
                    }
            
            # Reset spike detection if too much time has passed
            elif (self.extreme_spike_detected and 
                  self.spike_bar is not None and 
                  idx - self.spike_bar > reversal_confirmation_bars):
                self.extreme_spike_detected = False
                self.spike_direction = None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _check_entry_conditions: {str(e)}")
            return None
    
    def check_exit(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check various exit conditions for volatility reversal trades"""
        try:
            idx = len(self.data) - 1
            if idx < 0:
                return None
            
            current_price = self.data['close'].iloc[idx]
            
            # Check exit conditions
            should_exit, exit_reason = self.check_exit_conditions(idx)
            
            if should_exit:
                self.logger.info(f"Volatility reversal exit: {exit_reason}")
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
        """Handle trade closure cleanup and consecutive loss tracking"""
        try:
            self.last_trade_bar = len(self.data) - 1
            
            # Reset trade state
            self.entry_bar = None
            self.entry_side = None
            self.entry_price = None
            self.first_target_hit = False
            
            # Track consecutive losses for risk management
            exit_reason = trade_result.get('reason', 'unknown')
            pnl = trade_result.get('pnl', 0)
            
            if pnl < 0:  # Loss
                self.consecutive_losses += 1
                max_consecutive_losses = self.config.get('max_consecutive_losses', 4)
                
                if self.consecutive_losses >= max_consecutive_losses:
                    self.trading_enabled = False
                    self.logger.warning(f"Trading disabled after {self.consecutive_losses} consecutive losses")
            else:
                # Reset consecutive losses on winning trade
                self.consecutive_losses = 0
            
            self.logger.info(f"Trade closed - {exit_reason}, PnL: {pnl}")
            
        except Exception as e:
            self.logger.error(f"Error in on_trade_closed: {str(e)}")
    
    def get_risk_parameters(self) -> Dict[str, Any]:
        """Get volatility reversal risk management parameters"""
        try:
            # Calculate dynamic stops based on spike data
            if (hasattr(self, 'entry_side') and 
                self.entry_side and 
                hasattr(self, 'entry_price') and 
                self.entry_price):
                
                current_price = self.data['close'].iloc[-1]
                stop_buffer_pct = self.config.get('stop_buffer_pct', 0.002)
                
                # Very tight stops for contrarian trades
                if self.entry_side == 'long':
                    # Stop below recent low or entry with buffer
                    sl_pct = stop_buffer_pct + 0.001  # Extra buffer for volatility
                    tp_pct = sl_pct * 1.5  # 1.5:1 reward/risk for quick scalps
                elif self.entry_side == 'short':
                    # Stop above recent high or entry with buffer
                    sl_pct = stop_buffer_pct + 0.001
                    tp_pct = sl_pct * 1.5
                else:
                    sl_pct = stop_buffer_pct
                    tp_pct = stop_buffer_pct * 1.5
                
                # Reduced position sizing for contrarian trades
                position_size_reduction = self.config.get('position_size_reduction', 0.7)
                max_position_pct = self.config.get('max_position_pct', 2.0) * position_size_reduction
                
                return {
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "max_position_pct": max_position_pct,
                    "risk_reward_ratio": tp_pct / sl_pct if sl_pct > 0 else 1.5
                }
            
            # Fallback to config defaults
            return {
                "sl_pct": self.config.get('stop_buffer_pct', 0.002) + 0.001,
                "tp_pct": (self.config.get('stop_buffer_pct', 0.002) + 0.001) * 1.5,
                "max_position_pct": self.config.get('max_position_pct', 2.0) * self.config.get('position_size_reduction', 0.7),
                "risk_reward_ratio": 1.5
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_risk_parameters: {str(e)}")
            return {
                "sl_pct": 0.003,
                "tp_pct": 0.0045,
                "max_position_pct": 1.4,
                "risk_reward_ratio": 1.5
            } 