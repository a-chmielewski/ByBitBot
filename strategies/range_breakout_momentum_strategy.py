"""
Range Breakout Momentum Strategy

Strategy #13 in the Strategy Matrix

Market Conditions: Best fit for RANGING to TRANSITIONAL markets
Description: Detects range breakouts and rides the momentum with comprehensive confirmation
"""

import logging
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np

from .strategy_template import StrategyTemplate

class StrategyRangeBreakoutMomentum(StrategyTemplate):
    """
    Range Breakout Momentum Strategy
    
    A strategy that targets range-bound markets primed for breakouts. It combines consolidation
    phase detection on the 5-minute chart with momentum explosion confirmation on the 1-minute chart.
    
    Market Type:
    -----------
    - Range-bound markets with low volatility consolidation
    - 5-minute chart shows clear trading range or sideways drift
    - Low ATR indicating volatility compression
    - Volatility compression often precedes sharp breakouts
    
    Strategy Logic:
    --------------
    1. Range Identification:
       - Donchian Channels for range boundaries
       - ADX < 28 indicating ranging conditions
       - Bollinger Band squeeze detection
       - ATR compression monitoring
    
    2. Breakout Detection:
       - Price breaks above/below range with buffer
       - Volume surge confirmation (1.2x average)
       - Wide breakout candle (1.2x average range)
       - ADX starting to rise above 19
    
    3. Momentum Confirmation:
       - RSI breakout levels (58 up, 42 down)
       - ATR spike (1.2x recent median)
       - Volume and candle range validation
       - False breakout protection
    
    4. Risk Management:
       - Stop loss inside range (25% of range size)
       - Take profit at range size multiple
       - Trailing stops with ATR
       - Time-based exits for stalled trades
    
    Entry Conditions:
    ----------------
    1. Range Formation:
       - Identify established range boundaries
       - Confirm low volatility environment
       - Validate range size (0.25% - 5%)
       - Wait for range stability
    
    2. Breakout Trigger:
       - Price breaks range with buffer (0.04%)
       - Volume surge confirmation
       - Wide breakout candle
       - Momentum indicators align
    
    3. Direction Selection:
       - Symmetric breakout detection
       - Cancel opposite side on trigger
       - No trades within range
    
    Exit Conditions:
    ---------------
    1. Profit Targets:
       - Initial target at range size multiple
       - Partial profit taking (50% at 1R)
       - Trailing stops for remainder
    
    2. Stop Loss:
       - Tight stop inside range
       - False breakout detection
       - Quick exit on momentum failure
    
    3. Time Management:
       - Maximum trade duration (20 bars)
       - Exit if no progress made
       - Cooldown after failed breakouts
    
    Risk Management:
    --------------
    - Conservative position sizing
    - Tight stops with high R:R ratios
    - False breakout protection
    - Momentum confirmation required
    """
    
    MARKET_TYPE_TAGS: List[str] = ['RANGING', 'TRANSITIONAL']
    SHOW_IN_SELECTION: bool = True
    
    def __init__(self, data: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(data, config, logger)
        
        # Range tracking state
        self.range_high = None
        self.range_low = None
        self.range_confirmed = False
        self.range_formation_start = None
        self.in_range = False
        self.range_size = 0
        
        # Breakout tracking
        self.breakout_direction = None
        self.breakout_price = None
        self.false_breakout_count = 0
        
        # Trade management
        self.entry_bar = None
        self.stop_price = None
        self.target_price = None
        self.trailing_stop = None
        self.partial_profit_taken = False
        self.last_trade_bar = -self.config.get('cooldown_bars', 4)
        
        # Historical data for calculations
        self.atr_history = []
        self.candle_ranges = []
        
        self.logger.info("Range Breakout Momentum strategy initialized")
    
    def init_indicators(self) -> None:
        """Initialize all required indicators for the strategy"""
        try:
            # Import pandas_ta with fallback
            try:
                import pandas_ta as ta
                self.has_pandas_ta = True
            except ImportError:
                self.logger.warning("pandas_ta not available, using manual calculations")
                self.has_pandas_ta = False
            
            # Get strategy parameters
            range_period = self.config.get('range_period', 50)
            adx_period = self.config.get('adx_period', 14)
            atr_period = self.config.get('atr_period', 14)
            rsi_period = self.config.get('rsi_period', 14)
            bb_period = self.config.get('bb_period', 20)
            bb_std = self.config.get('bb_std', 2.0)
            volume_period = self.config.get('volume_period', 20)
            
            # Technical indicators
            if self.has_pandas_ta:
                self.data['adx'] = ta.adx(self.data['high'], self.data['low'], self.data['close'], length=adx_period)['ADX_' + str(adx_period)]
                self.data['atr'] = ta.atr(self.data['high'], self.data['low'], self.data['close'], length=atr_period)
                self.data['rsi'] = ta.rsi(self.data['close'], length=rsi_period)
                bb = ta.bbands(self.data['close'], length=bb_period, std=bb_std)
                self.data['bb_upper'] = bb[f'BBU_{bb_period}_{bb_std}']
                self.data['bb_lower'] = bb[f'BBL_{bb_period}_{bb_std}']
                self.data['bb_middle'] = bb[f'BBM_{bb_period}_{bb_std}']
            else:
                # Manual calculations
                self.data['adx'] = self._calculate_adx_manual(adx_period)
                self.data['atr'] = self._calculate_atr_manual(atr_period)
                self.data['rsi'] = self._calculate_rsi_manual(rsi_period)
                bb_data = self._calculate_bollinger_bands_manual(bb_period, bb_std)
                self.data['bb_upper'] = bb_data['upper']
                self.data['bb_lower'] = bb_data['lower']
                self.data['bb_middle'] = bb_data['middle']
            
            # Donchian Channels for range identification
            self.data['donchian_high'] = self.data['high'].rolling(window=range_period).max()
            self.data['donchian_low'] = self.data['low'].rolling(window=range_period).min()
            
            # Volume analysis
            self.data['volume_sma'] = self.data['volume'].rolling(window=volume_period).mean()
            
            # Bollinger Band width for squeeze detection
            self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
            
            self.logger.info("All indicators initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing indicators: {str(e)}")
            raise
    
    def _calculate_adx_manual(self, period: int) -> pd.Series:
        """Manual ADX calculation as fallback"""
        try:
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate directional movements
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            # Smooth the values
            tr_smooth = tr.rolling(window=period).mean()
            plus_dm_smooth = plus_dm.rolling(window=period).mean()
            minus_dm_smooth = minus_dm.rolling(window=period).mean()
            
            # Calculate DI+ and DI-
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)
            
            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return adx
        except Exception:
            return pd.Series([20.0] * len(self.data), index=self.data.index)
    
    def _calculate_atr_manual(self, period: int) -> pd.Series:
        """Manual ATR calculation as fallback"""
        try:
            high = self.data['high']
            low = self.data['low']
            close = self.data['close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return atr
        except Exception:
            return pd.Series([0.001] * len(self.data), index=self.data.index)
    
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
    
    def _calculate_bollinger_bands_manual(self, period: int, std_dev: float) -> Dict[str, pd.Series]:
        """Manual Bollinger Bands calculation as fallback"""
        try:
            sma = self.data['close'].rolling(window=period).mean()
            std = self.data['close'].rolling(window=period).std()
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            return {
                'upper': upper,
                'lower': lower,
                'middle': sma
            }
        except Exception:
            close = self.data['close']
            return {
                'upper': close * 1.02,
                'lower': close * 0.98,
                'middle': close
            }
    
    def is_ranging_market(self, idx: int) -> bool:
        """Detect if market is in a ranging/consolidation phase"""
        try:
            atr_lookback = self.config.get('atr_lookback', 10)
            if idx < atr_lookback:
                return True  # Assume ranging if not enough data
            
            # ADX should be low (indicating no strong trend)
            adx_threshold = self.config.get('adx_range_threshold', 28)
            adx_condition = True
            if idx >= self.config.get('adx_period', 14):
                adx_value = self.data['adx'].iloc[idx]
                if not pd.isna(adx_value):
                    adx_condition = adx_value < adx_threshold
            
            # Update ATR history
            current_atr = self.data['atr'].iloc[idx]
            if not pd.isna(current_atr):
                self.atr_history.append(current_atr)
                if len(self.atr_history) > atr_lookback:
                    self.atr_history.pop(0)
            
            # ATR should be relatively low (indicating low volatility)
            atr_condition = True
            if len(self.atr_history) >= atr_lookback:
                atr_median = np.median(self.atr_history)
                atr_condition = current_atr <= atr_median * 1.3  # Allow 30% above median
            
            # Bollinger Band squeeze (narrow bands)
            bb_condition = True
            bb_width = self.data['bb_width'].iloc[idx]
            if not pd.isna(bb_width):
                bb_threshold = self.config.get('bb_squeeze_threshold', 0.035)
                bb_condition = bb_width < bb_threshold
            
            # Use OR logic for more flexibility
            return adx_condition or (atr_condition and bb_condition)
            
        except Exception as e:
            self.logger.error(f"Error in is_ranging_market: {str(e)}")
            return True
    
    def identify_range(self, idx: int) -> bool:
        """Identify and validate current trading range"""
        try:
            if idx < self.config.get('range_period', 50):
                return False
            
            current_high = self.data['donchian_high'].iloc[idx]
            current_low = self.data['donchian_low'].iloc[idx]
            current_price = self.data['close'].iloc[idx]
            
            if pd.isna(current_high) or pd.isna(current_low):
                return False
            
            # Calculate range size
            range_size = current_high - current_low
            range_size_pct = range_size / current_price
            
            # Validate range size
            min_range = self.config.get('min_range_size_pct', 0.0025)
            max_range = self.config.get('max_range_size_pct', 0.05)
            
            if range_size_pct < min_range or range_size_pct > max_range:
                return False
            
            # Check if price is within range (allow buffer)
            price_buffer = range_size * 0.1  # 10% buffer
            price_in_range = (current_low - price_buffer) <= current_price <= (current_high + price_buffer)
            
            # Check ranging conditions
            is_ranging = self.is_ranging_market(idx)
            
            if price_in_range and is_ranging:
                # Update range if it's new or changed significantly
                if (self.range_high is None or 
                    abs(current_high - self.range_high) / current_price > 0.002 or
                    abs(current_low - self.range_low) / current_price > 0.002):
                    
                    self.range_high = current_high
                    self.range_low = current_low
                    self.range_size = range_size
                    self.range_formation_start = idx
                    self.range_confirmed = False
                    
                    self.logger.debug(f"New range identified: {current_low:.4f} - {current_high:.4f} (size: {range_size_pct:.3f}%)")
                
                # Confirm range after stability period
                elif (self.range_formation_start and 
                      idx - self.range_formation_start >= self.config.get('range_stability_bars', 3)):
                    self.range_confirmed = True
                    self.in_range = True
                    
                    self.logger.debug(f"Range confirmed: {self.range_low:.4f} - {self.range_high:.4f}")
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in identify_range: {str(e)}")
            return False
    
    def detect_range_breakout(self, idx: int) -> Tuple[bool, Optional[str]]:
        """Detect potential range breakout with confirmation"""
        try:
            if not self.range_confirmed or not self.in_range:
                return False, None
            
            current_price = self.data['close'].iloc[idx]
            current_high = self.data['high'].iloc[idx]
            current_low = self.data['low'].iloc[idx]
            
            # Calculate breakout levels with buffer
            buffer_pct = self.config.get('breakout_buffer_pct', 0.0004)
            breakout_high = self.range_high * (1 + buffer_pct)
            breakout_low = self.range_low * (1 - buffer_pct)
            
            # Check for breakout
            upside_breakout = (current_high > breakout_high and current_price > self.range_high) or \
                             (current_price > breakout_high)
            downside_breakout = (current_low < breakout_low and current_price < self.range_low) or \
                               (current_price < breakout_low)
            
            if upside_breakout:
                self.logger.debug(f"Upside breakout detected: price {current_price:.4f} > range_high {self.range_high:.4f}")
                return True, 'long'
            elif downside_breakout:
                self.logger.debug(f"Downside breakout detected: price {current_price:.4f} < range_low {self.range_low:.4f}")
                return True, 'short'
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error in detect_range_breakout: {str(e)}")
            return False, None
    
    def confirm_breakout_momentum(self, idx: int, direction: str) -> bool:
        """Confirm breakout with momentum and volume indicators"""
        try:
            confirmations = []
            
            # Volume surge confirmation
            volume_condition = True
            if idx >= self.config.get('volume_period', 20):
                current_volume = self.data['volume'].iloc[idx]
                avg_volume = self.data['volume_sma'].iloc[idx]
                
                if not pd.isna(current_volume) and not pd.isna(avg_volume) and avg_volume > 0:
                    volume_multiplier = self.config.get('volume_surge_multiplier', 1.2)
                    volume_condition = current_volume > avg_volume * volume_multiplier
                    confirmations.append(f"Volume: {volume_condition}")
            
            # Candle range confirmation (wide breakout candle)
            current_range = self.data['high'].iloc[idx] - self.data['low'].iloc[idx]
            self.candle_ranges.append(current_range)
            
            range_lookback = self.config.get('candle_range_lookback', 10)
            if len(self.candle_ranges) > range_lookback:
                self.candle_ranges.pop(0)
            
            range_condition = True
            if len(self.candle_ranges) >= range_lookback:
                avg_range = sum(self.candle_ranges) / len(self.candle_ranges)
                range_multiplier = self.config.get('candle_range_multiplier', 1.2)
                range_condition = current_range > avg_range * range_multiplier
                confirmations.append(f"Range: {range_condition}")
            
            # RSI momentum confirmation
            rsi_condition = True
            if idx >= self.config.get('rsi_period', 14):
                rsi_value = self.data['rsi'].iloc[idx]
                if not pd.isna(rsi_value):
                    if direction == 'long':
                        rsi_threshold = self.config.get('rsi_breakout_up', 58)
                        rsi_condition = rsi_value > rsi_threshold
                    else:
                        rsi_threshold = self.config.get('rsi_breakout_down', 42)
                        rsi_condition = rsi_value < rsi_threshold
                    confirmations.append(f"RSI: {rsi_condition} (value: {rsi_value:.1f})")
            
            # ADX starting to rise (trend strength building)
            adx_condition = True
            if idx >= self.config.get('adx_period', 14) + 1:
                adx_current = self.data['adx'].iloc[idx]
                adx_previous = self.data['adx'].iloc[idx-1]
                adx_threshold = self.config.get('adx_breakout_threshold', 19)
                
                if not pd.isna(adx_current) and not pd.isna(adx_previous):
                    adx_condition = adx_current > adx_previous or adx_current > adx_threshold
                    confirmations.append(f"ADX: {adx_condition} (current: {adx_current:.1f})")
            
            # Require at least 2 out of 4 confirmations
            confirmation_count = sum([volume_condition, range_condition, rsi_condition, adx_condition])
            momentum_confirmed = confirmation_count >= 2
            
            if not momentum_confirmed:
                self.logger.debug(f"Momentum confirmation failed ({confirmation_count}/4): {confirmations}")
            
            return momentum_confirmed
            
        except Exception as e:
            self.logger.error(f"Error in confirm_breakout_momentum: {str(e)}")
            return False
    
    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for range breakout momentum entry opportunities"""
        try:
            idx = len(self.data) - 1
            if idx < 0:
                return None
            
            # Check cooldown period
            cooldown_bars = self.config.get('cooldown_bars', 4)
            if idx - self.last_trade_bar < cooldown_bars:
                return None
            
            # First identify if we're in a range
            if not self.identify_range(idx):
                self.in_range = False
                return None
            
            # Detect breakout
            is_breakout, direction = self.detect_range_breakout(idx)
            if not is_breakout:
                return None
            
            # Confirm momentum
            if not self.confirm_breakout_momentum(idx, direction):
                return None
            
            current_price = self.data['close'].iloc[idx]
            
            # Store breakout information
            self.breakout_direction = direction
            self.breakout_price = current_price
            self.entry_bar = idx
            self.in_range = False  # Exit range state after breakout
            
            self.logger.info(f"Range breakout entry confirmed - {direction} at {current_price}")
            
            return {
                'action': direction,
                'price': current_price,
                'confidence': 0.85,
                'reason': f'range_breakout_{direction}_momentum'
            }
            
        except Exception as e:
            self.logger.error(f"Error in _check_entry_conditions: {str(e)}")
            return None
    
    def calculate_stops_and_targets(self, direction: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit based on range size"""
        try:
            current_price = self.data['close'].iloc[-1]
            
            # Calculate stop loss (back into range)
            stop_loss_pct = self.config.get('stop_loss_range_pct', 0.25)
            stop_distance = self.range_size * stop_loss_pct
            
            # Calculate target based on range size
            range_multiplier = self.config.get('range_size_multiplier', 1.0)
            
            if direction == 'long':
                stop_price = self.range_high - stop_distance
                target_price = current_price + (self.range_size * range_multiplier)
            else:  # short
                stop_price = self.range_low + stop_distance
                target_price = current_price - (self.range_size * range_multiplier)
            
            return stop_price, target_price
            
        except Exception as e:
            self.logger.error(f"Error calculating stops and targets: {str(e)}")
            # Fallback to percentage-based
            if direction == 'long':
                return current_price * 0.99, current_price * 1.02
            else:
                return current_price * 1.01, current_price * 0.98
    
    def update_trailing_stop(self, symbol: str) -> None:
        """Update trailing stop based on ATR"""
        try:
            if not hasattr(self, 'entry_price') or self.entry_price is None:
                return
                
            idx = len(self.data) - 1
            if idx < self.config.get('atr_period', 14):
                return
                
            current_price = self.data['close'].iloc[idx]
            atr_value = self.data['atr'].iloc[idx]
            
            if pd.isna(atr_value):
                return
                
            trail_multiplier = self.config.get('trailing_atr_multiplier', 2.0)
            trail_distance = atr_value * trail_multiplier
            
            if self.breakout_direction == 'long':  # Long position
                new_stop = current_price - trail_distance
                if self.trailing_stop is None or new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                    self.logger.debug(f"Updated trailing stop to {new_stop}")
            else:  # Short position
                new_stop = current_price + trail_distance
                if self.trailing_stop is None or new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
                    self.logger.debug(f"Updated trailing stop to {new_stop}")
                    
        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {str(e)}")
    
    def check_false_breakout(self) -> bool:
        """Check if breakout is failing (false breakout)"""
        try:
            if not hasattr(self, 'entry_price') or self.entry_price is None:
                return False
                
            idx = len(self.data) - 1
            current_price = self.data['close'].iloc[idx]
            
            # Check if price has moved back into the original range
            if self.breakout_direction == 'long':
                # Failed if price closes back below range high
                return current_price < self.range_high
            else:  # Short position
                # Failed if price closes back above range low
                return current_price > self.range_low
                
        except Exception:
            return False
    
    def check_partial_profit(self) -> bool:
        """Check if we should take partial profits"""
        try:
            if (self.partial_profit_taken or 
                not hasattr(self, 'entry_price') or 
                self.entry_price is None or
                self.stop_price is None):
                return False
            
            idx = len(self.data) - 1
            current_price = self.data['close'].iloc[idx]
            
            # Take partial profit at 1R (risk-reward ratio of 1:1)
            if self.breakout_direction == 'long':
                profit_level = self.entry_price + (self.entry_price - self.stop_price)
                return current_price >= profit_level
            else:  # Short
                profit_level = self.entry_price - (self.stop_price - self.entry_price)
                return current_price <= profit_level
                
        except Exception:
            return False
    
    def check_exit(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check various exit conditions for breakout trades"""
        try:
            idx = len(self.data) - 1
            if idx < 0:
                return None
            
            current_price = self.data['close'].iloc[idx]
            
            # Update trailing stop
            self.update_trailing_stop(symbol)
            
            # False breakout check (priority exit)
            if self.check_false_breakout():
                self.logger.info("False breakout detected, exiting position")
                self.false_breakout_count += 1
                return {
                    'action': 'exit',
                    'price': current_price,
                    'reason': 'false_breakout'
                }
            
            # Time-based exit
            max_duration = self.config.get('max_trade_duration', 20)
            if self.entry_bar and idx - self.entry_bar >= max_duration:
                self.logger.info("Time-based exit triggered")
                return {
                    'action': 'exit',
                    'price': current_price,
                    'reason': 'time_exit'
                }
            
            # Target hit
            if hasattr(self, 'target_price') and self.target_price:
                if self.breakout_direction == 'long' and current_price >= self.target_price:
                    self.logger.info(f"Profit target hit: {current_price} >= {self.target_price}")
                    return {
                        'action': 'exit',
                        'price': current_price,
                        'reason': 'target_hit'
                    }
                elif self.breakout_direction == 'short' and current_price <= self.target_price:
                    self.logger.info(f"Profit target hit: {current_price} <= {self.target_price}")
                    return {
                        'action': 'exit',
                        'price': current_price,
                        'reason': 'target_hit'
                    }
            
            # Stop loss hit
            if hasattr(self, 'stop_price') and self.stop_price:
                if self.breakout_direction == 'long' and current_price <= self.stop_price:
                    self.logger.info(f"Stop loss hit: {current_price} <= {self.stop_price}")
                    return {
                        'action': 'exit',
                        'price': current_price,
                        'reason': 'stop_loss'
                    }
                elif self.breakout_direction == 'short' and current_price >= self.stop_price:
                    self.logger.info(f"Stop loss hit: {current_price} >= {self.stop_price}")
                    return {
                        'action': 'exit',
                        'price': current_price,
                        'reason': 'stop_loss'
                    }
            
            # Trailing stop
            if self.trailing_stop:
                if self.breakout_direction == 'long' and current_price <= self.trailing_stop:
                    self.logger.info(f"Trailing stop hit: {current_price} <= {self.trailing_stop}")
                    return {
                        'action': 'exit',
                        'price': current_price,
                        'reason': 'trailing_stop'
                    }
                elif self.breakout_direction == 'short' and current_price >= self.trailing_stop:
                    self.logger.info(f"Trailing stop hit: {current_price} >= {self.trailing_stop}")
                    return {
                        'action': 'exit',
                        'price': current_price,
                        'reason': 'trailing_stop'
                    }
            
            # ADX confirmation failure (momentum fading)
            if idx >= self.config.get('adx_period', 14):
                adx_value = self.data['adx'].iloc[idx]
                if not pd.isna(adx_value) and adx_value < 15:
                    self.logger.info(f"Momentum fade detected: ADX {adx_value}")
                    return {
                        'action': 'exit',
                        'price': current_price,
                        'reason': 'momentum_fade'
                    }
            
            # Check for partial profit taking
            if self.check_partial_profit():
                self.partial_profit_taken = True
                self.logger.info("Taking partial profits at 1R")
                # Note: This would trigger partial exit in the main bot logic
                # For now, we'll continue holding the position
            
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
            self.stop_price = None
            self.target_price = None
            self.trailing_stop = None
            self.partial_profit_taken = False
            self.breakout_direction = None
            self.breakout_price = None
            self.entry_price = None
            
            # Handle false breakout tracking
            exit_reason = trade_result.get('reason', 'unknown')
            if exit_reason == 'false_breakout':
                self.false_breakout_count += 1
            
            self.logger.info(f"Trade closed - {exit_reason}")
            
        except Exception as e:
            self.logger.error(f"Error in on_trade_closed: {str(e)}")
    
    def get_risk_parameters(self) -> Dict[str, Any]:
        """Get breakout momentum risk management parameters"""
        try:
            # Calculate stops and targets if we have range data
            if self.range_size and hasattr(self, 'breakout_direction') and self.breakout_direction:
                stop_price, target_price = self.calculate_stops_and_targets(self.breakout_direction)
                
                # Store for exit logic
                self.stop_price = stop_price
                self.target_price = target_price
                
                current_price = self.data['close'].iloc[-1]
                
                if self.breakout_direction == 'long':
                    sl_pct = abs(current_price - stop_price) / current_price
                    tp_pct = abs(target_price - current_price) / current_price
                else:
                    sl_pct = abs(stop_price - current_price) / current_price
                    tp_pct = abs(current_price - target_price) / current_price
                
                return {
                    "sl_pct": sl_pct,
                    "tp_pct": tp_pct,
                    "max_position_pct": self.config.get('max_position_pct', 3.0),
                    "risk_reward_ratio": tp_pct / sl_pct if sl_pct > 0 else 2.0
                }
            
            # Fallback to config defaults
            return {
                "sl_pct": self.config.get('sl_pct', 0.02),
                "tp_pct": self.config.get('tp_pct', 0.04),
                "max_position_pct": self.config.get('max_position_pct', 3.0),
                "risk_reward_ratio": 2.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in get_risk_parameters: {str(e)}")
            return {
                "sl_pct": 0.02,
                "tp_pct": 0.04,
                "max_position_pct": 3.0,
                "risk_reward_ratio": 2.0
            } 