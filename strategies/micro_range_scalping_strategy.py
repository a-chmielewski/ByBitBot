"""
Micro Range Scalping Strategy

Strategy #7 in the Strategy Matrix

Market Conditions: Best fit for LOW_VOLATILITY and RANGING markets
Description: Tight range scalping for low volatility, sideways market conditions
"""

import logging
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

from .strategy_template import StrategyTemplate

class StrategyMicroRangeScalping(StrategyTemplate):
    """
    Micro Range Scalping Strategy (Low-Vol Mean Reversion)
    
    A specialized strategy designed for ultra-low volatility environments where price
    action is slow and tight. The strategy scalps tiny swings within micro-ranges,
    essentially a finer version of range trading with very small profit targets.
    
    Market Type:
    -----------
    - Low-volatility environments with slow, tight price action
    - Narrow trading bands (typically 0.2% or less)
    - Declining volume and compressed volatility
    - Range-bound markets with unusually small ranges
    - Often precedes major breakouts ("calm before the storm")
    
    Strategy Logic:
    --------------
    1. Low Volatility Detection:
       - ATR below threshold (0.2% of price)
       - Bollinger Band squeeze (width < 0.5%)
       - Volume declining below 80% of average
       - Multiple confirmation bars required
    
    2. Micro Range Identification:
       - Very tight support/resistance levels (max 0.3% range)
       - Minimum 2 touches of each level for confirmation
       - Range tolerance of 0.1% for level identification
       - Continuous range validation
    
    3. Oscillator-Based Entries:
       - Fast RSI (7-period) for responsiveness
       - Stochastic oscillator for momentum
       - Oversold/overbought in micro-range context
       - Quick reversal signals at range extremes
    
    4. Micro Profit Management:
       - Very tight stops (0.05% beyond range)
       - Small profit targets (0.15% typical)
       - Quick exits at range middle or oscillator reversal
       - Break-even management after small profits
    
    Entry Conditions:
    ----------------
    1. Environment Prerequisites:
       - Low volatility confirmed (ATR, BB squeeze, volume)
       - Micro-range detected and validated
       - Minimum range stability period
    
    2. Long Entries (Near Micro-Support):
       - Price near identified support level
       - Fast RSI < 30 or Stochastic < 20
       - Oscillator oversold conditions
       - Any uptick confirmation from support
    
    3. Short Entries (Near Micro-Resistance):
       - Price near identified resistance level
       - Fast RSI > 70 or Stochastic > 80
       - Oscillator overbought conditions
       - Rejection signals at resistance
    
    4. Range Validation:
       - At least 2 touches of support/resistance
       - Range width between 0.1% - 0.3%
       - Stable range for minimum period
       - No recent breakout attempts
    
    Exit Conditions:
    ---------------
    1. Quick Profit Taking:
       - Target range middle or small increment (0.15%)
       - Oscillator reversal signals
       - Any sign of momentum fade
       - Time-based exits (10 bars maximum)
    
    2. Range Break Protection:
       - Immediate exit on range breakout
       - Stop loss just outside range boundaries
       - Break-even moves after small profits
    
    3. Volatility Change:
       - Exit all positions if volatility increases
       - Monitor for breakout conditions
       - Protect against "calm before storm" scenarios
    
    Risk Management:
    --------------
    - Ultra-tight stops (0.05% beyond range)
    - Very small profit targets (0.15% typical)
    - Reduced position sizing (70% of normal)
    - High win rate, small R:R approach
    - Quick time exits and break-even management
    """
    
    MARKET_TYPE_TAGS: List[str] = ['LOW_VOLATILITY', 'RANGING']
    SHOW_IN_SELECTION: bool = True
    
    def __init__(self, data: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(data, config, logger)
        
        # Range detection state
        self.micro_range_detected = False
        self.range_support = 0
        self.range_resistance = 0
        self.range_middle = 0
        self.range_width = 0
        self.range_touches_support = 0
        self.range_touches_resistance = 0
        self.last_range_update = 0
        
        # Low volatility state
        self.low_vol_confirmed = False
        self.low_vol_bars_count = 0
        
        # Trade management
        self.entry_bar = None
        self.entry_side = None
        self.entry_price = None
        self.break_even_moved = False
        self.last_trade_bar = -self.config.get('cooldown_bars', 2)
        
        self.logger.info("Micro Range Scalping strategy initialized")
    
    def init_indicators(self) -> None:
        """Initialize range detection and support/resistance indicators"""
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
            rsi_period = self.config.get('rsi_period', 7)
            stoch_k_period = self.config.get('stoch_k_period', 14)
            stoch_d_period = self.config.get('stoch_d_period', 3)
            volume_period = self.config.get('volume_avg_period', 20)
            
            # ATR for volatility measurement
            if self.has_pandas_ta:
                self.data['atr'] = ta.atr(self.data['high'], self.data['low'], self.data['close'], length=atr_period)
            else:
                self.data['atr'] = self._calculate_atr_manual(atr_period)
            
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
            
            # Fast RSI for micro-range responsiveness
            if self.has_pandas_ta:
                self.data['rsi'] = ta.rsi(self.data['close'], length=rsi_period)
            else:
                self.data['rsi'] = self._calculate_rsi_manual(rsi_period)
            
            # Stochastic oscillator
            if self.has_pandas_ta:
                stoch_data = ta.stoch(self.data['high'], self.data['low'], self.data['close'], 
                                    k=stoch_k_period, d=stoch_d_period)
                self.data['stoch_k'] = stoch_data[f'STOCHk_{stoch_k_period}_{stoch_d_period}_{stoch_d_period}']
                self.data['stoch_d'] = stoch_data[f'STOCHd_{stoch_k_period}_{stoch_d_period}_{stoch_d_period}']
            else:
                self.data['stoch_k'], self.data['stoch_d'] = self._calculate_stoch_manual(stoch_k_period, stoch_d_period)
            
            # Volume analysis
            self.data['volume_sma'] = self.data['volume'].rolling(window=volume_period).mean()
            
            self.logger.debug("All micro-range indicators initialized successfully")
            
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
    
    def _calculate_stoch_manual(self, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """Manual Stochastic calculation as fallback"""
        try:
            lowest_low = self.data['low'].rolling(window=k_period).min()
            highest_high = self.data['high'].rolling(window=k_period).max()
            
            stoch_k = 100 * (self.data['close'] - lowest_low) / (highest_high - lowest_low)
            stoch_d = stoch_k.rolling(window=d_period).mean()
            
            return stoch_k, stoch_d
        except Exception:
            return (pd.Series([50.0] * len(self.data), index=self.data.index),
                    pd.Series([50.0] * len(self.data), index=self.data.index))
    
    def is_low_volatility_environment(self, idx: int) -> bool:
        """Detect low-volatility conditions using multiple criteria"""
        try:
            if idx < 1:
                return False
            
            current_price = self.data['close'].iloc[idx]
            atr_value = self.data['atr'].iloc[idx]
            bb_upper = self.data['bb_upper'].iloc[idx]
            bb_lower = self.data['bb_lower'].iloc[idx]
            current_volume = self.data['volume'].iloc[idx]
            avg_volume = self.data['volume_sma'].iloc[idx]
            
            if pd.isna(atr_value) or pd.isna(bb_upper) or pd.isna(avg_volume):
                return False
            
            # Calculate ATR as percentage of price
            atr_pct = atr_value / current_price if current_price > 0 else 0
            
            # Calculate Bollinger Band width as percentage
            bb_width_pct = (bb_upper - bb_lower) / current_price if current_price > 0 else 0
            
            # Get thresholds
            atr_low_threshold = self.config.get('atr_low_threshold', 0.002)
            bb_squeeze_threshold = self.config.get('bb_squeeze_threshold', 0.005)
            volume_decline_threshold = self.config.get('volume_decline_threshold', 0.8)
            
            # Check low volatility criteria
            atr_low = atr_pct < atr_low_threshold
            bb_squeeze = bb_width_pct < bb_squeeze_threshold
            volume_declining = current_volume < (avg_volume * volume_decline_threshold)
            
            # All criteria must be met for low-vol confirmation
            return atr_low and bb_squeeze and volume_declining
            
        except Exception as e:
            self.logger.error(f"Error in is_low_volatility_environment: {str(e)}")
            return False
    
    def detect_micro_range(self, idx: int) -> bool:
        """Detect micro-range with tight support and resistance levels"""
        try:
            range_detection_bars = self.config.get('range_detection_bars', 20)
            if idx < range_detection_bars:
                return False
            
            # Look back at recent price action
            start_idx = max(0, idx - range_detection_bars + 1)
            end_idx = idx + 1
            
            lookback_highs = self.data['high'].iloc[start_idx:end_idx].tolist()
            lookback_lows = self.data['low'].iloc[start_idx:end_idx].tolist()
            
            # Find potential resistance (recent highs cluster)
            range_tolerance_pct = self.config.get('range_tolerance_pct', 0.001)
            min_range_touches = self.config.get('min_range_touches', 2)
            
            high_levels = []
            for i, high in enumerate(lookback_highs):
                touches = 0
                for other_high in lookback_highs:
                    if abs(other_high - high) / high < range_tolerance_pct:
                        touches += 1
                if touches >= min_range_touches:
                    high_levels.append(high)
            
            # Find potential support (recent lows cluster)
            low_levels = []
            for i, low in enumerate(lookback_lows):
                touches = 0
                for other_low in lookback_lows:
                    if abs(other_low - low) / low < range_tolerance_pct:
                        touches += 1
                if touches >= min_range_touches:
                    low_levels.append(low)
            
            if not high_levels or not low_levels:
                return False
            
            # Use the most common levels
            resistance = max(set(high_levels), key=high_levels.count)
            support = min(set(low_levels), key=low_levels.count)
            
            # Check if it's a micro-range (very tight)
            range_width_pct = (resistance - support) / support if support > 0 else 0
            micro_range_max_pct = self.config.get('micro_range_max_pct', 0.003)
            
            if range_width_pct <= micro_range_max_pct and resistance > support:
                self.range_resistance = resistance
                self.range_support = support
                self.range_middle = (resistance + support) / 2
                self.range_width = resistance - support
                self.last_range_update = idx
                
                # Count actual touches in recent bars
                self.range_touches_support = sum(1 for low in lookback_lows 
                                               if abs(low - support) / support < range_tolerance_pct)
                self.range_touches_resistance = sum(1 for high in lookback_highs 
                                                  if abs(high - resistance) / resistance < range_tolerance_pct)
                
                self.logger.debug(f"Micro-range detected: Support {support:.4f}, Resistance {resistance:.4f}, Width {range_width_pct:.3f}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in detect_micro_range: {str(e)}")
            return False
    
    def is_near_support(self, price: float) -> bool:
        """Check if price is near support level"""
        if not self.micro_range_detected:
            return False
        
        range_tolerance_pct = self.config.get('range_tolerance_pct', 0.001)
        return abs(price - self.range_support) / self.range_support < range_tolerance_pct
    
    def is_near_resistance(self, price: float) -> bool:
        """Check if price is near resistance level"""
        if not self.micro_range_detected:
            return False
        
        range_tolerance_pct = self.config.get('range_tolerance_pct', 0.001)
        return abs(price - self.range_resistance) / self.range_resistance < range_tolerance_pct
    
    def check_oscillator_oversold(self, idx: int) -> bool:
        """Check if oscillators indicate oversold conditions"""
        try:
            rsi_value = self.data['rsi'].iloc[idx]
            stoch_k = self.data['stoch_k'].iloc[idx]
            
            if pd.isna(rsi_value) or pd.isna(stoch_k):
                return False
            
            rsi_oversold = self.config.get('rsi_oversold', 30)
            stoch_oversold = self.config.get('stoch_oversold', 20)
            
            rsi_oversold_condition = rsi_value < rsi_oversold
            stoch_oversold_condition = stoch_k < stoch_oversold
            
            return rsi_oversold_condition or stoch_oversold_condition
            
        except Exception:
            return False
    
    def check_oscillator_overbought(self, idx: int) -> bool:
        """Check if oscillators indicate overbought conditions"""
        try:
            rsi_value = self.data['rsi'].iloc[idx]
            stoch_k = self.data['stoch_k'].iloc[idx]
            
            if pd.isna(rsi_value) or pd.isna(stoch_k):
                return False
            
            rsi_overbought = self.config.get('rsi_overbought', 70)
            stoch_overbought = self.config.get('stoch_overbought', 80)
            
            rsi_overbought_condition = rsi_value > rsi_overbought
            stoch_overbought_condition = stoch_k > stoch_overbought
            
            return rsi_overbought_condition or stoch_overbought_condition
            
        except Exception:
            return False
    
    def check_range_breakout(self, idx: int) -> Tuple[bool, Optional[str]]:
        """Check if the micro-range has been broken"""
        try:
            if not self.micro_range_detected:
                return False, None
            
            current_price = self.data['close'].iloc[idx]
            range_tolerance_pct = self.config.get('range_tolerance_pct', 0.001)
            
            if current_price > self.range_resistance * (1 + range_tolerance_pct):
                return True, 'upside'
            elif current_price < self.range_support * (1 - range_tolerance_pct):
                return True, 'downside'
            
            return False, None
            
        except Exception:
            return False, None
    
    def check_exit_conditions(self, idx: int) -> Tuple[bool, Optional[str]]:
        """Check for micro-scalp exit conditions"""
        try:
            if not hasattr(self, 'entry_bar') or self.entry_bar is None:
                return False, None
            
            current_price = self.data['close'].iloc[idx]
            bars_held = idx - self.entry_bar
            
            # Time-based exit
            max_hold_bars = self.config.get('max_hold_bars', 10)
            if bars_held >= max_hold_bars:
                return True, "time_exit"
            
            # Check for range breakout (exit immediately)
            breakout, direction = self.check_range_breakout(idx)
            if breakout:
                return True, f"range_breakout_{direction}"
            
            # Profit target exits
            if self.entry_price:
                take_profit_pct = self.config.get('take_profit_pct', 0.001)  # Quicker from 0.0015
                break_even_buffer_pct = self.config.get('break_even_buffer_pct', 0.0003)
                
                if self.entry_side == 'long':
                    # Take profit target or resistance approached
                    if (current_price >= self.entry_price * (1 + take_profit_pct) or
                        self.is_near_resistance(current_price) or
                        self.check_oscillator_overbought(idx)):
                        return True, "profit_target_long"
                    
                    # Move to break-even
                    if (not self.break_even_moved and 
                        current_price >= self.entry_price * (1 + break_even_buffer_pct)):
                        return True, "break_even_long"
                
                elif self.entry_side == 'short':
                    # Take profit target or support approached
                    if (current_price <= self.entry_price * (1 - take_profit_pct) or
                        self.is_near_support(current_price) or
                        self.check_oscillator_oversold(idx)):
                        return True, "profit_target_short"
                    
                    # Move to break-even
                    if (not self.break_even_moved and 
                        current_price <= self.entry_price * (1 - break_even_buffer_pct)):
                        return True, "break_even_short"
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error in check_exit_conditions: {str(e)}")
            return False, None
    
    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for micro range bounce entry opportunities"""
        try:
            idx = len(self.data) - 1
            if idx < 0:
                return None
            
            # Check cooldown period
            cooldown_bars = self.config.get('cooldown_bars', 2)
            if idx - self.last_trade_bar < cooldown_bars:
                return None
            
            # Need minimum bars for all indicators
            min_bars = max(
                self.config.get('atr_period', 14),
                self.config.get('bb_period', 20),
                self.config.get('rsi_period', 7),
                self.config.get('stoch_k_period', 14),
                self.config.get('volume_avg_period', 20),
                self.config.get('range_detection_bars', 20)
            )
            if idx < min_bars:
                return None
            
            current_price = self.data['close'].iloc[idx]
            if pd.isna(current_price):
                return None
            
            # Step 1: Detect low volatility environment
            if self.is_low_volatility_environment(idx):
                self.low_vol_confirmed = True
                self.low_vol_bars_count += 1
            else:
                self.low_vol_confirmed = False
                self.low_vol_bars_count = 0
                self.micro_range_detected = False
                return None
            
            # Step 2: Detect micro-range only in low-vol environment
            min_range_bars = self.config.get('min_range_bars', 6)
            if self.low_vol_confirmed and self.low_vol_bars_count >= min_range_bars:
                if self.detect_micro_range(idx):
                    self.micro_range_detected = True
            
            # Reset range if it's been too long since last update
            range_detection_bars = self.config.get('range_detection_bars', 20)
            if (self.micro_range_detected and 
                idx - self.last_range_update > range_detection_bars):
                self.micro_range_detected = False
            
            # Only trade if we have confirmed micro-range in low-vol environment
            if not (self.low_vol_confirmed and self.micro_range_detected):
                return None
            
            # Entry Condition 1: Long near micro-support
            if (self.is_near_support(current_price) and 
                self.check_oscillator_oversold(idx)):
                
                self.entry_side = 'long'
                self.entry_bar = idx
                self.entry_price = current_price
                self.break_even_moved = False
                
                self.logger.info(f"Micro range scalping long entry at support {self.range_support:.4f}")
                
                # Get risk parameters for this entry
                risk_params = self.get_risk_parameters()
                
                return {
                    'side': 'buy',
                    'price': current_price,
                    'confidence': 0.8,
                    'reason': 'micro_range_support_bounce',
                    'sl_pct': risk_params.get('sl_pct', 0.0005),
                    'tp_pct': risk_params.get('tp_pct', 0.0015),
                    'max_position_pct': risk_params.get('max_position_pct', 1.4),
                    'risk_reward_ratio': risk_params.get('risk_reward_ratio', 3.0),
                    'size': None  # Will be calculated by bot's risk manager
                }
            
            # Entry Condition 2: Short near micro-resistance
            elif (self.is_near_resistance(current_price) and 
                  self.check_oscillator_overbought(idx)):
                
                self.entry_side = 'short'
                self.entry_bar = idx
                self.entry_price = current_price
                self.break_even_moved = False
                
                self.logger.info(f"Micro range scalping short entry at resistance {self.range_resistance:.4f}")
                
                # Get risk parameters for this entry
                risk_params = self.get_risk_parameters()
                
                return {
                    'side': 'sell',
                    'price': current_price,
                    'confidence': 0.8,
                    'reason': 'micro_range_resistance_rejection',
                    'sl_pct': risk_params.get('sl_pct', 0.0005),
                    'tp_pct': risk_params.get('tp_pct', 0.0015),
                    'max_position_pct': risk_params.get('max_position_pct', 1.4),
                    'risk_reward_ratio': risk_params.get('risk_reward_ratio', 3.0),
                    'size': None  # Will be calculated by bot's risk manager
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _check_entry_conditions: {str(e)}")
            return None
    
    def check_exit(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check various exit conditions for micro-range scalping trades"""
        try:
            idx = len(self.data) - 1
            if idx < 0:
                return None
            
            current_price = self.data['close'].iloc[idx]
            
            # Check exit conditions
            should_exit, exit_reason = self.check_exit_conditions(idx)
            
            if should_exit:
                if exit_reason in ["break_even_long", "break_even_short"]:
                    # Move stop to break-even instead of closing
                    self.break_even_moved = True
                    self.logger.info(f"Micro range scalping break-even move: {exit_reason}")
                    return None  # Don't exit, just track break-even move
                else:
                    # Close position
                    self.logger.info(f"Micro range scalping exit: {exit_reason}")
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
            self.break_even_moved = False
            
            # Log trade result
            exit_reason = trade_result.get('reason', 'unknown')
            pnl = trade_result.get('pnl', 0)
            
            self.logger.info(f"Micro range trade closed - {exit_reason}, PnL: {pnl}")
            
        except Exception as e:
            self.logger.error(f"Error in on_trade_closed: {str(e)}")
    
    def get_risk_parameters(self) -> Dict[str, Any]:
        """Get micro-scalping risk management parameters"""
        try:
            # Calculate ultra-tight stops based on micro-range
            if (self.micro_range_detected and 
                hasattr(self, 'entry_side') and 
                self.entry_side and 
                hasattr(self, 'entry_price') and 
                self.entry_price):
                
                current_price = self.data['close'].iloc[-1]
                stop_loss_buffer_pct = self.config.get('stop_loss_buffer_pct', 0.001)  # Widened from 0.0005
                take_profit_pct = self.config.get('take_profit_pct', 0.001)  # Quicker from 0.0015
                
                if self.entry_side == 'long':
                    # Stop just below support
                    stop_price = self.range_support * (1 - stop_loss_buffer_pct)
                    sl_pct = abs(current_price - stop_price) / current_price
                    tp_pct = take_profit_pct
                elif self.entry_side == 'short':
                    # Stop just above resistance
                    stop_price = self.range_resistance * (1 + stop_loss_buffer_pct)
                    sl_pct = abs(stop_price - current_price) / current_price
                    tp_pct = take_profit_pct
                else:
                    sl_pct = stop_loss_buffer_pct
                    tp_pct = take_profit_pct
                
                # Reduced position sizing for micro-scalping
                position_size_reduction = self.config.get('position_size_reduction', 0.7)
                max_position_pct = self.config.get('max_position_pct', 2.0) * position_size_reduction
                
                return {
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "max_position_pct": max_position_pct,
                    "risk_reward_ratio": tp_pct / sl_pct if sl_pct > 0 else 3.0
                }
            
            # Fallback to config defaults
            return {
                "sl_pct": self.config.get('stop_loss_buffer_pct', 0.001),  # Widened from 0.0005
                "tp_pct": self.config.get('take_profit_pct', 0.001),  # Quicker from 0.0015
                "max_position_pct": self.config.get('max_position_pct', 2.0) * self.config.get('position_size_reduction', 0.7),
                "risk_reward_ratio": 3.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_risk_parameters: {str(e)}")
            return {
                "sl_pct": 0.001,  # Widened from 0.0005
                "tp_pct": 0.001,  # Quicker from 0.0015
                "max_position_pct": 1.4,
                "risk_reward_ratio": 3.0
            } 