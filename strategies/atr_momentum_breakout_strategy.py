"""
ATR Momentum Breakout Scalper Strategy

Strategy #5 in the Strategy Matrix

Market Conditions: Best fit for HIGH_VOLATILITY markets
Description: Uses ATR-based breakout detection for high-volatility momentum trading
"""

import logging
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

from .strategy_template import StrategyTemplate

class StrategyATRMomentumBreakout(StrategyTemplate):
    """
    ATR Momentum Breakout Scalper Strategy
    
    A high-volatility momentum strategy designed for breakout trading during volatile
    market conditions. The strategy identifies volatility surges and jumps in with
    momentum, using ATR-based dynamic risk management and quick scalping exits.
    
    Market Type:
    -----------
    - High-volatility and trending or breakout markets
    - Sudden, strong moves with expanded range
    - News releases or impulse moves when volatility jumps
    - Scenarios where price breaks out of consolidation with volume
    
    Strategy Logic:
    --------------
    1. Volatility Regime Detection:
       - ATR above threshold (1.5x recent average)
       - Current candle range 2x recent average range
       - Minimum volatility threshold (0.2% of price)
       - High-volatility environment confirmation
    
    2. Consolidation and Breakout Detection:
       - Donchian Channel breakout levels (20-period)
       - Recent consolidation pattern identification
       - Clean breakout above/below key levels
       - Trend alignment with fast EMA (21-period)
    
    3. Volume Confirmation:
       - Volume surge 2.5x average (10-period)
       - Breakout candle volume validation
       - Momentum move confirmation
       - Avoid false breakouts with low volume
    
    4. Quick Scalping Management:
       - Rapid profit taking (0.5% scalp target)
       - Partial profit taking (70% on scalp hit)
       - ATR-based trailing stops for runners
       - Time-based exits for failed breakouts
    
    Entry Conditions:
    ----------------
    1. Volatility Prerequisites:
       - High volatility regime confirmed
       - ATR > 1.5x recent average
       - Current candle range > 2x average
       - Minimum volatility threshold met
    
    2. Breakout Triggers:
       - Clean breakout above Donchian high (long)
       - Clean breakdown below Donchian low (short)
       - Price closes beyond breakout level
       - Trend alignment with EMA direction
    
    3. Volume Confirmation:
       - Volume > 2.5x recent average
       - Breakout candle volume surge
       - True momentum move validation
    
    4. Optional Consolidation:
       - Recent tight consolidation detected
       - Range compression before breakout
       - Better breakout quality identification
    
    Exit Conditions:
    ---------------
    1. Quick Scalp Exits:
       - 0.5% profit target hit within 3 bars
       - Take 70% profits on scalp target
       - Keep 30% for potential runners
    
    2. Trailing Stop Management:
       - ATR-based trailing stops (1.0x ATR)
       - Update high/low water marks
       - Trail behind recent price action
    
    3. Failed Breakout Exits:
       - Exit after 3 bars if no immediate profit
       - Time-based exit for stalled momentum
       - Maximum 10 bars position holding
    
    4. Momentum Exhaustion:
       - Trail stops tighten after scalp target
       - Exit on momentum fade signals
       - Protect profits from volatility reversals
    
    Risk Management:
    --------------
    - ATR-based dynamic stops (1.5x ATR)
    - ATR-based profit targets (2.0x ATR)
    - Quick scalping with partial exits
    - Time stops for failed breakouts
    - Volatility-adjusted position sizing
    """
    
    MARKET_TYPE_TAGS: List[str] = ['HIGH_VOLATILITY']
    SHOW_IN_SELECTION: bool = True
    
    def __init__(self, data: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(data, config, logger)
        
        # Range tracking for volatility
        self.recent_ranges = []
        
        # Consolidation and breakout tracking
        self.consolidation_high = 0
        self.consolidation_low = 0
        self.consolidation_start = 0
        self.is_in_consolidation = False
        
        # Trade management
        self.entry_bar = None
        self.entry_side = None
        self.entry_price = None
        self.scalp_target_hit = False
        self.trailing_stop_price = None
        self.high_water_mark = None
        self.low_water_mark = None
        self.last_trade_bar = -self.config.get('cooldown_bars', 3)
        
        self.logger.info("ATR Momentum Breakout strategy initialized")
    
    def init_indicators(self) -> None:
        """Initialize ATR and volume indicators"""
        try:
            # Import pandas_ta with fallback
            ta = None
            try:
                import pandas_ta as ta
                self.has_pandas_ta = True
            except ImportError:
                self.logger.warning("pandas_ta not available, using manual calculations")
                self.has_pandas_ta = False
            
            # Get strategy parameters
            atr_period = self.config.get('atr_period', 14)
            donchian_period = self.config.get('donchian_period', 20)
            trend_ema_period = self.config.get('trend_ema_period', 21)
            volume_avg_period = self.config.get('volume_avg_period', 10)
            
            # ATR for volatility measurement
            if self.has_pandas_ta and ta is not None:
                self.data['atr'] = ta.atr(self.data['high'], self.data['low'], self.data['close'], length=atr_period)
            else:
                self.data['atr'] = self._calculate_atr_manual(atr_period)
            
            # ATR moving average for volatility regime detection
            self.data['atr_sma'] = self.data['atr'].rolling(window=atr_period).mean()
            
            # Donchian Channels for breakout levels
            self.data['donchian_high'] = self.data['high'].rolling(window=donchian_period).max()
            self.data['donchian_low'] = self.data['low'].rolling(window=donchian_period).min()
            
            # Trend filter EMA
            if self.has_pandas_ta and ta is not None:
                self.data['trend_ema'] = ta.ema(self.data['close'], length=trend_ema_period)
            else:
                self.data['trend_ema'] = self.data['close'].ewm(span=trend_ema_period).mean()
            
            # Volume analysis
            self.data['volume_sma'] = self.data['volume'].rolling(window=volume_avg_period).mean()
            
            self.logger.info("All ATR momentum indicators initialized successfully")
            
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
    
    def is_high_volatility_regime(self, idx: int) -> bool:
        """Check if we're in a high volatility environment"""
        try:
            if idx < 1:
                return False
            
            current_atr = self.data['atr'].iloc[idx]
            avg_atr = self.data['atr_sma'].iloc[idx]
            current_price = self.data['close'].iloc[idx]
            
            if pd.isna(current_atr) or pd.isna(avg_atr):
                return False
            
            # ATR must be above threshold compared to recent average
            atr_volatility_threshold = self.config.get('atr_volatility_threshold', 1.5)
            atr_condition = current_atr > avg_atr * atr_volatility_threshold
            
            # Current candle range must be significantly larger than recent average
            current_range = self.data['high'].iloc[idx] - self.data['low'].iloc[idx]
            
            # Calculate recent average range
            if len(self.recent_ranges) >= 10:
                avg_range = sum(self.recent_ranges[-10:]) / len(self.recent_ranges[-10:])
                range_multiplier = self.config.get('range_multiplier', 2.0)
                range_condition = current_range > avg_range * range_multiplier
            else:
                range_condition = True  # Not enough data, assume ok
            
            # Minimum volatility threshold
            min_volatility_threshold = self.config.get('min_volatility_threshold', 0.002)
            volatility_condition = current_atr / current_price > min_volatility_threshold
            
            return bool(atr_condition and range_condition and volatility_condition)
            
        except Exception as e:
            self.logger.error(f"Error in is_high_volatility_regime: {str(e)}")
            return False
    
    def detect_consolidation(self, idx: int) -> bool:
        """Detect if price has been in consolidation before potential breakout"""
        try:
            consolidation_period = self.config.get('consolidation_period', 5)
            if idx < consolidation_period:
                return False
            
            # Look at recent highs and lows
            start_idx = max(0, idx - consolidation_period + 1)
            end_idx = idx + 1
            
            recent_highs = self.data['high'].iloc[start_idx:end_idx].tolist()
            recent_lows = self.data['low'].iloc[start_idx:end_idx].tolist()
            
            consolidation_high = max(recent_highs)
            consolidation_low = min(recent_lows)
            
            # Check if range is tight relative to ATR
            range_size = consolidation_high - consolidation_low
            atr_value = self.data['atr'].iloc[idx] if not pd.isna(self.data['atr'].iloc[idx]) else range_size
            
            # Consolidation if range is less than 1.5x ATR
            if range_size < atr_value * 1.5:
                self.consolidation_high = consolidation_high
                self.consolidation_low = consolidation_low
                self.is_in_consolidation = True
                return True
            
            self.is_in_consolidation = False
            return False
            
        except Exception as e:
            self.logger.error(f"Error in detect_consolidation: {str(e)}")
            return False
    
    def is_volume_surge(self, idx: int) -> bool:
        """Check for volume surge confirmation"""
        try:
            current_volume = self.data['volume'].iloc[idx]
            avg_volume = self.data['volume_sma'].iloc[idx]
            
            if pd.isna(avg_volume) or avg_volume == 0:
                return True  # Assume volume is ok if no volume data
            
            volume_surge_multiplier = self.config.get('volume_surge_multiplier', 2.5)
            return bool(current_volume > avg_volume * volume_surge_multiplier)
            
        except Exception:
            return True  # Assume volume is ok on error
    
    def is_trend_aligned(self, idx: int, direction: str) -> bool:
        """Check if breakout direction aligns with trend"""
        try:
            current_price = self.data['close'].iloc[idx]
            ema_value = self.data['trend_ema'].iloc[idx]
            
            if pd.isna(ema_value):
                return True  # Assume trend is ok if no EMA data
            
            if direction == 'long':
                return bool(current_price > ema_value)
            elif direction == 'short':
                return bool(current_price < ema_value)
            
            return False
            
        except Exception:
            return True  # Assume trend is ok on error
    
    def detect_breakout(self, idx: int) -> Tuple[Optional[str], Optional[float]]:
        """Detect clean breakout with all confirmations"""
        try:
            current_price = self.data['close'].iloc[idx]
            current_high = self.data['high'].iloc[idx]
            current_low = self.data['low'].iloc[idx]
            
            # Check for breakout above recent highs
            if idx > 0:
                resistance_level = self.data['donchian_high'].iloc[idx-1]  # Previous bar's resistance
                
                if not pd.isna(resistance_level):
                    # Bullish breakout conditions
                    if (current_high > resistance_level and 
                        current_price > resistance_level and
                        self.is_volume_surge(idx) and
                        self.is_trend_aligned(idx, 'long')):
                        return 'long', resistance_level
            
            # Check for breakdown below recent lows
            if idx > 0:
                support_level = self.data['donchian_low'].iloc[idx-1]  # Previous bar's support
                
                if not pd.isna(support_level):
                    # Bearish breakdown conditions
                    if (current_low < support_level and 
                        current_price < support_level and
                        self.is_volume_surge(idx) and
                        self.is_trend_aligned(idx, 'short')):
                        return 'short', support_level
            
            return None, None
            
        except Exception as e:
            self.logger.error(f"Error in detect_breakout: {str(e)}")
            return None, None
    
    def update_trailing_stop(self, idx: int) -> None:
        """Update trailing stop based on ATR"""
        try:
            if not hasattr(self, 'entry_side') or self.entry_side is None:
                return
            
            current_price = self.data['close'].iloc[idx]
            atr_value = self.data['atr'].iloc[idx]
            
            if pd.isna(atr_value):
                return
            
            trailing_atr_multiplier = self.config.get('trailing_atr_multiplier', 1.0)
            
            if self.entry_side == 'long':
                # Update high water mark
                if self.high_water_mark is None or current_price > self.high_water_mark:
                    self.high_water_mark = current_price
                
                # Calculate trailing stop
                new_trailing_stop = self.high_water_mark - (atr_value * trailing_atr_multiplier)
                
                # Update trailing stop if it's higher than current
                if self.trailing_stop_price is None or new_trailing_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_trailing_stop
            
            elif self.entry_side == 'short':
                # Update low water mark
                if self.low_water_mark is None or current_price < self.low_water_mark:
                    self.low_water_mark = current_price
                
                # Calculate trailing stop
                new_trailing_stop = self.low_water_mark + (atr_value * trailing_atr_multiplier)
                
                # Update trailing stop if it's lower than current
                if self.trailing_stop_price is None or new_trailing_stop < self.trailing_stop_price:
                    self.trailing_stop_price = new_trailing_stop
                    
        except Exception as e:
            self.logger.error(f"Error in update_trailing_stop: {str(e)}")
    
    def check_scalp_target(self, idx: int) -> bool:
        """Check if quick scalp target is hit"""
        try:
            if not hasattr(self, 'entry_price') or self.entry_price is None:
                return False
            
            current_price = self.data['close'].iloc[idx]
            scalp_target_pct = self.config.get('scalp_target_pct', 0.005)
            
            if self.entry_side == 'long':
                target_price = self.entry_price * (1 + scalp_target_pct)
                return bool(current_price >= target_price)
            elif self.entry_side == 'short':
                target_price = self.entry_price * (1 - scalp_target_pct)
                return bool(current_price <= target_price)
            
            return False
            
        except Exception:
            return False
    
    def check_exit_conditions(self, idx: int) -> Tuple[bool, Optional[str]]:
        """Check various exit conditions"""
        try:
            if not hasattr(self, 'entry_bar') or self.entry_bar is None:
                return False, None
            
            current_price = self.data['close'].iloc[idx]
            bars_held = idx - self.entry_bar
            
            # Time-based exit for failed breakouts
            failed_breakout_time = self.config.get('failed_breakout_time', 3)
            if bars_held >= failed_breakout_time and not self.scalp_target_hit:
                # Check if trade is not profitable
                if self.entry_side == 'long' and current_price <= self.entry_price:
                    return True, "failed_breakout_long"
                elif self.entry_side == 'short' and current_price >= self.entry_price:
                    return True, "failed_breakout_short"
            
            # Maximum position time
            max_position_time = self.config.get('max_position_time', 10)
            if bars_held >= max_position_time:
                return True, "max_time"
            
            # Quick scalp target hit
            if self.check_scalp_target(idx) and not self.scalp_target_hit:
                self.scalp_target_hit = True
                return True, "scalp_target"
            
            # Trailing stop hit (after scalp target achieved)
            if self.scalp_target_hit and self.trailing_stop_price is not None:
                if self.entry_side == 'long' and current_price <= self.trailing_stop_price:
                    return True, "trailing_stop_long"
                elif self.entry_side == 'short' and current_price >= self.trailing_stop_price:
                    return True, "trailing_stop_short"
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error in check_exit_conditions: {str(e)}")
            return False, None
    
    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for ATR breakout entry opportunities"""
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
                self.config.get('atr_period', 14),
                self.config.get('donchian_period', 20),
                self.config.get('trend_ema_period', 21),
                self.config.get('volume_avg_period', 10)
            )
            if idx < min_bars:
                return None
            
            current_price = self.data['close'].iloc[idx]
            if pd.isna(current_price):
                return None
            
            # Update range tracking
            current_range = self.data['high'].iloc[idx] - self.data['low'].iloc[idx]
            self.recent_ranges.append(current_range)
            if len(self.recent_ranges) > 20:
                self.recent_ranges.pop(0)
            
            # Step 1: Check for high volatility regime
            if not self.is_high_volatility_regime(idx):
                return None
            
            # Step 2: Detect consolidation (optional - helps identify better breakouts)
            self.detect_consolidation(idx)
            
            # Step 3: Look for breakout signal
            breakout_direction, breakout_level = self.detect_breakout(idx)
            
            if breakout_direction is not None:
                # Initialize trade state
                self.entry_side = breakout_direction
                self.entry_bar = idx
                self.entry_price = current_price
                self.scalp_target_hit = False
                self.trailing_stop_price = None
                
                if breakout_direction == 'long':
                    self.high_water_mark = None
                    self.logger.info(f"ATR momentum breakout long entry above {breakout_level:.4f}")
                    
                    return {
                        'action': 'long',
                        'price': current_price,
                        'confidence': 0.85,
                        'reason': f'atr_momentum_breakout_above_{breakout_level:.4f}'
                    }
                
                elif breakout_direction == 'short':
                    self.low_water_mark = None
                    self.logger.info(f"ATR momentum breakout short entry below {breakout_level:.4f}")
                    
                    return {
                        'action': 'short',
                        'price': current_price,
                        'confidence': 0.85,
                        'reason': f'atr_momentum_breakdown_below_{breakout_level:.4f}'
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _check_entry_conditions: {str(e)}")
            return None
    
    def check_exit(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check various exit conditions for momentum breakout trades"""
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
                if exit_reason == "scalp_target":
                    # Take partial profits on scalp target
                    partial_profit_pct = self.config.get('partial_profit_pct', 0.7)
                    self.logger.info(f"ATR momentum scalp target hit - taking {partial_profit_pct*100}% profits")
                    
                    return {
                        'action': 'partial_exit',
                        'price': current_price,
                        'partial_pct': partial_profit_pct,
                        'reason': exit_reason
                    }
                else:
                    # Exit completely
                    self.logger.info(f"ATR momentum breakout exit: {exit_reason}")
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
            self.scalp_target_hit = False
            self.trailing_stop_price = None
            self.high_water_mark = None
            self.low_water_mark = None
            
            # Log trade result
            exit_reason = trade_result.get('reason', 'unknown')
            pnl = trade_result.get('pnl', 0)
            
            self.logger.info(f"ATR momentum trade closed - {exit_reason}, PnL: {pnl}")
            
        except Exception as e:
            self.logger.error(f"Error in on_trade_closed: {str(e)}")
    
    def get_risk_parameters(self) -> Dict[str, Any]:
        """Get ATR-based dynamic risk management parameters"""
        try:
            # Get current ATR for dynamic risk calculation
            if len(self.data) > 0:
                current_price = self.data['close'].iloc[-1]
                atr_value = self.data['atr'].iloc[-1]
                
                if not pd.isna(atr_value) and current_price > 0:
                    # Calculate ATR-based stops and targets
                    atr_stop_multiplier = self.config.get('atr_stop_multiplier', 1.5)
                    atr_target_multiplier = self.config.get('atr_target_multiplier', 2.0)
                    
                    sl_pct = (atr_value * atr_stop_multiplier) / current_price
                    tp_pct = (atr_value * atr_target_multiplier) / current_price
                    
                    return {
                        "sl_pct": sl_pct,
                        "tp_pct": tp_pct,
                        "max_position_pct": self.config.get('max_position_pct', 3.0),
                        "risk_reward_ratio": tp_pct / sl_pct if sl_pct > 0 else 1.33
                    }
            
            # Fallback to config defaults
            return {
                "sl_pct": self.config.get('sl_pct', 0.025),
                "tp_pct": self.config.get('tp_pct', 0.05),
                "max_position_pct": self.config.get('max_position_pct', 3.0),
                "risk_reward_ratio": 2.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_risk_parameters: {str(e)}")
            return {
                "sl_pct": 0.025,
                "tp_pct": 0.05,
                "max_position_pct": 3.0,
                "risk_reward_ratio": 2.0
            } 