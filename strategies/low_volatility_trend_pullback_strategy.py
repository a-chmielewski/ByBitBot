"""
Low-Volatility Trend Pullback Scalper Strategy

Strategy #12 in the Strategy Matrix

Market Conditions: Best fit for LOW_VOLATILITY and TRENDING markets
Description: Scalps pullbacks in low volatility trending markets with tight stops and quick profit targets
"""

import logging
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np

from .strategy_template import StrategyTemplate

class StrategyLowVolatilityTrendPullback(StrategyTemplate):
    """
    Low-Volatility Trend Pullback Scalper Strategy
    
    A scalping strategy designed for steady trending markets with low volatility. The strategy
    identifies gentle trends in low-volatility conditions, waits for minor pullbacks against
    the trend, enters on pullback completion with tight stops, and uses small profit targets
    for consistent gains.

    Market Type:
    -----------
    - Steady trending markets with low volatility
    - 5-minute chart shows smooth, gradual trend with small bodies and low ATR
    - 1-minute chart shows same directional bias with mild oscillations
    - Targets quick, modest gains from minor pullbacks in trend direction

    Indicators and Parameters:
    ------------------------
    1. Trend Identification:
       - EMA(100/200) on 5-minute chart
       - ADX(14) > 20 to confirm trend
       - Avoid ADX < 15 (flat market)

    2. Volatility Check:
       - ATR(14) below threshold or 20-bar average
       - Bollinger Band width moderately tight
       - Avoid completely stagnant markets

    3. Pullback Trigger:
       - RSI(14) or Stochastic(14,3,3) on 1-minute
       - RSI < 30 (oversold) in uptrend
       - RSI > 70 (overbought) in downtrend
       - EMA(10/20) on 1-minute for pullback confirmation

    4. Volume Filter:
       - Minimum 50% of 20-bar average volume
       - Avoid extremely low volume periods

    Entry Conditions:
    ----------------
    1. With-trend Bias:
       - Long only in uptrend (price above EMA)
       - Short only in downtrend (price below EMA)

    2. Pullback Recognition:
       - Counter-trend movement on 1-minute
       - RSI/Stochastic reaching extreme levels
       - Price touching short-term EMA

    3. Entry Trigger:
       - Reversal candle pattern
       - RSI crossing back from extreme
       - MA crossover confirmation

    4. Volatility Filter:
       - Minimum 0.05% candle range
       - Avoid ultra-flat periods

    Exit Conditions:
    ---------------
    1. Quick Profit:
       - Target 0.2-0.5% move
       - Exit on new high/low past pullback

    2. Oscillator Signal:
       - RSI reaching opposite extreme
       - Exit on counter-swing completion

    3. Time-based:
       - Exit after 5-10 minutes if no progress
       - Avoid stalled trades

    4. Trend Change:
       - Exit on 5-minute trend reversal
       - Close on ADX breakdown

    Risk Management:
    --------------
    1. Stop Loss:
       - Tight stop beyond pullback extreme
       - 0.1-0.2% or 1x ATR(14)
       - Minimize losses on deeper pullbacks

    2. Take Profit:
       - Fixed 0.2-0.5% target
       - Optional trailing stop
       - Regular profit taking

    Implementation Notes:
    -------------------
    1. Market Selection:
       - ADX between 15-30
       - Avoid sideways markets
       - Monitor volatility shifts

    2. Execution:
       - Use limit orders
       - Minimize slippage
       - Quick execution critical

    3. Risk Control:
       - Modest position sizing
       - Consider trading fees
       - One trade per asset

    4. Optimization:
       - Test RSI thresholds
       - Fine-tune profit targets
       - Backtest across conditions
    """
    
    MARKET_TYPE_TAGS: List[str] = ['LOW_VOLATILITY', 'TRENDING']
    SHOW_IN_SELECTION: bool = True
    
    def __init__(self, data: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(data, config, logger)
        
        # Strategy state tracking
        self.trend_direction = None
        self.recent_highs = []
        self.recent_lows = []
        self.last_trade_bar = -self.config.get('cooldown_bars', 3)
        self.entry_bar = None
        self.trailing_stop = None
        
        # Volatility regime tracking
        self.atr_percentile_window = []
        
        self.logger.info("Low-Volatility Trend Pullback Scalper strategy initialized")
    
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
            trend_ema_period = self.config.get('trend_ema_period', 100)
            fast_ema_period = self.config.get('fast_ema_period', 20)
            adx_period = self.config.get('adx_period', 14)
            rsi_period = self.config.get('rsi_period', 14)
            atr_period = self.config.get('atr_period', 14)
            atr_lookback = self.config.get('atr_volatility_lookback', 20)
            bb_period = self.config.get('bb_period', 20)
            bb_std = self.config.get('bb_std', 2.0)
            volume_period = self.config.get('volume_period', 20)
            
            # Trend identification indicators
            if self.has_pandas_ta:
                self.data['trend_ema'] = ta.ema(self.data['close'], length=trend_ema_period)
                self.data['fast_ema'] = ta.ema(self.data['close'], length=fast_ema_period)
                self.data['adx'] = ta.adx(self.data['high'], self.data['low'], self.data['close'], length=adx_period)['ADX_' + str(adx_period)]
            else:
                # Manual EMA calculation
                self.data['trend_ema'] = self.data['close'].ewm(span=trend_ema_period).mean()
                self.data['fast_ema'] = self.data['close'].ewm(span=fast_ema_period).mean()
                # Simplified ADX calculation
                self.data['adx'] = self._calculate_adx_manual(adx_period)
            
            # Momentum oscillator
            if self.has_pandas_ta:
                self.data['rsi'] = ta.rsi(self.data['close'], length=rsi_period)
            else:
                self.data['rsi'] = self._calculate_rsi_manual(rsi_period)
            
            # Volatility indicators
            if self.has_pandas_ta:
                self.data['atr'] = ta.atr(self.data['high'], self.data['low'], self.data['close'], length=atr_period)
                bb = ta.bbands(self.data['close'], length=bb_period, std=bb_std)
                self.data['bb_upper'] = bb[f'BBU_{bb_period}_{bb_std}']
                self.data['bb_lower'] = bb[f'BBL_{bb_period}_{bb_std}']
                self.data['bb_middle'] = bb[f'BBM_{bb_period}_{bb_std}']
            else:
                self.data['atr'] = self._calculate_atr_manual(atr_period)
                bb_data = self._calculate_bollinger_bands_manual(bb_period, bb_std)
                self.data['bb_upper'] = bb_data['upper']
                self.data['bb_lower'] = bb_data['lower']
                self.data['bb_middle'] = bb_data['middle']
            
            # ATR average for volatility regime detection
            self.data['atr_sma'] = self.data['atr'].rolling(window=atr_lookback).mean()
            
            # Volume analysis
            self.data['volume_sma'] = self.data['volume'].rolling(window=volume_period).mean()
            
            # Bollinger Band width for volatility assessment
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
            # Return a series of moderate values if calculation fails
            return pd.Series([20.0] * len(self.data), index=self.data.index)
    
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
    
    def is_low_volatility_regime(self, idx: int) -> bool:
        """Check if we're in a low volatility environment"""
        try:
            if idx < self.config.get('atr_period', 14):
                return False
            
            current_atr = self.data['atr'].iloc[idx]
            avg_atr = self.data['atr_sma'].iloc[idx]
            
            if pd.isna(current_atr) or pd.isna(avg_atr) or avg_atr == 0:
                return False
            
            # ATR should be relatively low (below or near average)
            atr_condition = current_atr <= avg_atr * 1.1  # Allow 10% above average
            
            # Check Bollinger Band width (should be moderate, not too tight or too wide)
            bb_width = self.data['bb_width'].iloc[idx]
            if pd.isna(bb_width):
                bb_condition = True
            else:
                bb_condition = 0.02 < bb_width < 0.06  # Between 2% and 6% width
            
            return atr_condition and bb_condition
            
        except Exception as e:
            self.logger.error(f"Error in is_low_volatility_regime: {str(e)}")
            return False
    
    def is_trending_market(self, idx: int) -> tuple[bool, Optional[str]]:
        """Check if market is in a low-volatility trend"""
        try:
            if idx < max(self.config.get('adx_period', 14), self.config.get('trend_ema_period', 100)):
                return False, None
            
            adx_value = self.data['adx'].iloc[idx]
            adx_min = self.config.get('adx_min_threshold', 15)
            adx_max = self.config.get('adx_max_threshold', 30)
            
            if pd.isna(adx_value):
                return False, None
            
            # ADX should be moderate (trending but not too volatile)
            adx_condition = adx_min < adx_value < adx_max
            
            if not adx_condition:
                return False, None
            
            # Check trend direction and consistency
            current_price = self.data['close'].iloc[idx]
            ema_value = self.data['trend_ema'].iloc[idx]
            
            if pd.isna(ema_value):
                return False, None
            
            # Update trend tracking
            current_high = self.data['high'].iloc[idx]
            current_low = self.data['low'].iloc[idx]
            
            self.recent_highs.append(current_high)
            self.recent_lows.append(current_low)
            
            consistency_bars = self.config.get('trend_consistency_bars', 20)
            if len(self.recent_highs) > consistency_bars:
                self.recent_highs.pop(0)
                self.recent_lows.pop(0)
            
            # Determine trend direction
            if current_price > ema_value:
                trend_direction = 'up'
                # Check for consistent higher highs and higher lows
                if len(self.recent_highs) >= 10:
                    recent_high_trend = self.recent_highs[-1] > self.recent_highs[-10]
                    recent_low_trend = self.recent_lows[-1] > self.recent_lows[-10]
                    consistency = recent_high_trend and recent_low_trend
                else:
                    consistency = True
            elif current_price < ema_value:
                trend_direction = 'down'
                # Check for consistent lower highs and lower lows
                if len(self.recent_highs) >= 10:
                    recent_high_trend = self.recent_highs[-1] < self.recent_highs[-10]
                    recent_low_trend = self.recent_lows[-1] < self.recent_lows[-10]
                    consistency = recent_high_trend and recent_low_trend
                else:
                    consistency = True
            else:
                trend_direction = None
                consistency = False
            
            return consistency and trend_direction is not None, trend_direction
            
        except Exception as e:
            self.logger.error(f"Error in is_trending_market: {str(e)}")
            return False, None
    
    def is_volume_adequate(self, idx: int) -> bool:
        """Check if volume is adequate (not too low)"""
        try:
            if idx < self.config.get('volume_period', 20):
                return True  # Assume OK if no volume data
            
            current_volume = self.data['volume'].iloc[idx]
            avg_volume = self.data['volume_sma'].iloc[idx]
            
            if pd.isna(current_volume) or pd.isna(avg_volume) or avg_volume == 0:
                return True
            
            min_ratio = self.config.get('volume_min_ratio', 0.5)
            return current_volume >= avg_volume * min_ratio
            
        except Exception:
            return True
    
    def has_sufficient_range(self, idx: int) -> bool:
        """Check if recent candles have sufficient range (avoid ultra-flat periods)"""
        try:
            current_high = self.data['high'].iloc[idx]
            current_low = self.data['low'].iloc[idx]
            current_close = self.data['close'].iloc[idx]
            
            current_range = current_high - current_low
            min_range = current_close * self.config.get('min_candle_range_pct', 0.0005)
            
            return current_range >= min_range
            
        except Exception:
            return True
    
    def detect_pullback_entry(self, idx: int, trend_direction: str) -> tuple[bool, Optional[str]]:
        """Detect pullback entry opportunities"""
        try:
            if idx < max(self.config.get('rsi_period', 14), self.config.get('fast_ema_period', 20)):
                return False, None
            
            current_price = self.data['close'].iloc[idx]
            rsi_value = self.data['rsi'].iloc[idx]
            fast_ema_value = self.data['fast_ema'].iloc[idx]
            
            if pd.isna(rsi_value) or pd.isna(fast_ema_value):
                return False, None
            
            rsi_oversold = self.config.get('rsi_oversold', 30)
            rsi_overbought = self.config.get('rsi_overbought', 70)
            
            if trend_direction == 'up':
                # Look for oversold condition in uptrend
                oversold_condition = rsi_value < rsi_oversold
                
                # Check if price is near or below fast EMA (pullback)
                pullback_condition = current_price <= fast_ema_value * 1.002  # Allow 0.2% above EMA
                
                if oversold_condition or pullback_condition:
                    # Look for reversal signals
                    # RSI turning up from oversold
                    if idx > 0:
                        prev_rsi = self.data['rsi'].iloc[idx-1]
                        rsi_turning_up = (rsi_value > prev_rsi and 
                                        rsi_value < rsi_oversold + 5)
                    else:
                        rsi_turning_up = False
                    
                    # Bullish candle (close > open)
                    current_open = self.data['open'].iloc[idx]
                    bullish_candle = current_price > current_open
                    
                    # Price bouncing off fast EMA
                    ema_bounce = current_price > fast_ema_value
                    
                    return rsi_turning_up or (bullish_candle and ema_bounce), 'long'
            
            elif trend_direction == 'down':
                # Look for overbought condition in downtrend
                overbought_condition = rsi_value > rsi_overbought
                
                # Check if price is near or above fast EMA (pullback)
                pullback_condition = current_price >= fast_ema_value * 0.998  # Allow 0.2% below EMA
                
                if overbought_condition or pullback_condition:
                    # Look for reversal signals
                    # RSI turning down from overbought
                    if idx > 0:
                        prev_rsi = self.data['rsi'].iloc[idx-1]
                        rsi_turning_down = (rsi_value < prev_rsi and 
                                          rsi_value > rsi_overbought - 5)
                    else:
                        rsi_turning_down = False
                    
                    # Bearish candle (close < open)
                    current_open = self.data['open'].iloc[idx]
                    bearish_candle = current_price < current_open
                    
                    # Price rejecting fast EMA
                    ema_reject = current_price < fast_ema_value
                    
                    return rsi_turning_down or (bearish_candle and ema_reject), 'short'
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error in detect_pullback_entry: {str(e)}")
            return False, None
    
    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for pullback scalping entry opportunities"""
        try:
            idx = len(self.data) - 1
            if idx < 0:
                return None
            
            # Check cooldown period
            cooldown_bars = self.config.get('cooldown_bars', 3)
            if idx - self.last_trade_bar < cooldown_bars:
                return None
            
            # Check low volatility regime
            if not self.is_low_volatility_regime(idx):
                self.logger.debug("Not in low volatility regime")
                return None
            
            # Check trending market
            is_trending, trend_direction = self.is_trending_market(idx)
            if not is_trending:
                self.logger.debug("Market not trending appropriately")
                return None
            
            # Check volume adequacy
            if not self.is_volume_adequate(idx):
                self.logger.debug("Volume inadequate")
                return None
            
            # Check sufficient range
            if not self.has_sufficient_range(idx):
                self.logger.debug("Insufficient candle range")
                return None
            
            # Check for pullback entry
            pullback_signal, entry_side = self.detect_pullback_entry(idx, trend_direction)
            
            if pullback_signal:
                current_price = self.data['close'].iloc[idx]
                
                # Store trend direction for exit logic
                self.trend_direction = trend_direction
                self.entry_bar = idx
                
                self.logger.info(f"Pullback entry signal detected - {entry_side} at {current_price}")
                
                return {
                    'action': entry_side,
                    'price': current_price,
                    'confidence': 0.8,
                    'reason': f'pullback_scalp_{trend_direction}_trend'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in _check_entry_conditions: {str(e)}")
            return None
    
    def update_trailing_stop(self, symbol: str) -> None:
        """Update trailing stop for profitable trades"""
        try:
            if not hasattr(self, 'entry_price') or self.entry_price is None:
                return
                
            idx = len(self.data) - 1
            if idx < 0:
                return
                
            current_price = self.data['close'].iloc[idx]
            trailing_threshold = self.config.get('trailing_profit_threshold', 0.001)
            trailing_step = self.config.get('trailing_step', 0.0005)
            
            # Determine position direction from trend
            if self.trend_direction == 'up':  # Long position
                unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
                
                if unrealized_pnl_pct >= trailing_threshold:
                    new_stop = current_price * (1 - trailing_step)
                    if self.trailing_stop is None or new_stop > self.trailing_stop:
                        self.trailing_stop = new_stop
                        self.logger.debug(f"Updated trailing stop to {new_stop}")
            
            elif self.trend_direction == 'down':  # Short position
                unrealized_pnl_pct = (self.entry_price - current_price) / self.entry_price
                
                if unrealized_pnl_pct >= trailing_threshold:
                    new_stop = current_price * (1 + trailing_step)
                    if self.trailing_stop is None or new_stop < self.trailing_stop:
                        self.trailing_stop = new_stop
                        self.logger.debug(f"Updated trailing stop to {new_stop}")
                        
        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {str(e)}")
    
    def check_exit(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check various exit conditions for scalping trades"""
        try:
            idx = len(self.data) - 1
            if idx < 0:
                return None
            
            current_price = self.data['close'].iloc[idx]
            
            # Update trailing stop
            self.update_trailing_stop(symbol)
            
            # Time-based exit
            max_duration = self.config.get('max_trade_duration', 10)
            if self.entry_bar and idx - self.entry_bar >= max_duration:
                self.logger.info("Time-based exit triggered")
                return {
                    'action': 'exit',
                    'price': current_price,
                    'reason': 'time_exit'
                }
            
            # Target hit (calculated from entry price and risk parameters)
            if hasattr(self, 'entry_price') and self.entry_price:
                profit_target_pct = self.config.get('profit_target_pct', 0.003)
                
                if self.trend_direction == 'up':  # Long position
                    target_price = self.entry_price * (1 + profit_target_pct)
                    if current_price >= target_price:
                        self.logger.info(f"Profit target hit: {current_price} >= {target_price}")
                        return {
                            'action': 'exit',
                            'price': current_price,
                            'reason': 'target_hit'
                        }
                elif self.trend_direction == 'down':  # Short position
                    target_price = self.entry_price * (1 - profit_target_pct)
                    if current_price <= target_price:
                        self.logger.info(f"Profit target hit: {current_price} <= {target_price}")
                        return {
                            'action': 'exit',
                            'price': current_price,
                            'reason': 'target_hit'
                        }
                
                # Stop loss hit
                stop_loss_pct = self.config.get('stop_loss_pct', 0.0015)
                
                if self.trend_direction == 'up':  # Long position
                    stop_price = self.entry_price * (1 - stop_loss_pct)
                    if current_price <= stop_price:
                        self.logger.info(f"Stop loss hit: {current_price} <= {stop_price}")
                        return {
                            'action': 'exit',
                            'price': current_price,
                            'reason': 'stop_loss'
                        }
                elif self.trend_direction == 'down':  # Short position
                    stop_price = self.entry_price * (1 + stop_loss_pct)
                    if current_price >= stop_price:
                        self.logger.info(f"Stop loss hit: {current_price} >= {stop_price}")
                        return {
                            'action': 'exit',
                            'price': current_price,
                            'reason': 'stop_loss'
                        }
            
            # Trailing stop
            if self.trailing_stop:
                if self.trend_direction == 'up' and current_price <= self.trailing_stop:
                    self.logger.info(f"Trailing stop hit: {current_price} <= {self.trailing_stop}")
                    return {
                        'action': 'exit',
                        'price': current_price,
                        'reason': 'trailing_stop'
                    }
                elif self.trend_direction == 'down' and current_price >= self.trailing_stop:
                    self.logger.info(f"Trailing stop hit: {current_price} >= {self.trailing_stop}")
                    return {
                        'action': 'exit',
                        'price': current_price,
                        'reason': 'trailing_stop'
                    }
            
            # Oscillator exit signals
            if idx >= self.config.get('rsi_period', 14):
                rsi_value = self.data['rsi'].iloc[idx]
                rsi_overbought = self.config.get('rsi_overbought', 70)
                rsi_oversold = self.config.get('rsi_oversold', 30)
                
                if not pd.isna(rsi_value):
                    if self.trend_direction == 'up' and rsi_value >= rsi_overbought:
                        self.logger.info(f"RSI overbought exit: {rsi_value}")
                        return {
                            'action': 'exit',
                            'price': current_price,
                            'reason': 'rsi_overbought'
                        }
                    elif self.trend_direction == 'down' and rsi_value <= rsi_oversold:
                        self.logger.info(f"RSI oversold exit: {rsi_value}")
                        return {
                            'action': 'exit',
                            'price': current_price,
                            'reason': 'rsi_oversold'
                        }
            
            # Trend change exit
            is_trending, current_trend = self.is_trending_market(idx)
            if not is_trending or (current_trend != self.trend_direction):
                self.logger.info("Trend change detected, exiting position")
                return {
                    'action': 'exit',
                    'price': current_price,
                    'reason': 'trend_change'
                }
            
            # Volatility spike (exit low-vol regime)
            if not self.is_low_volatility_regime(idx):
                self.logger.info("Volatility spike detected, exiting position")
                return {
                    'action': 'exit',
                    'price': current_price,
                    'reason': 'volatility_spike'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in check_exit: {str(e)}")
            return None
    
    def on_trade_closed(self, symbol: str, trade_result: Dict[str, Any]) -> None:
        """Handle trade closure cleanup"""
        try:
            self.last_trade_bar = len(self.data) - 1
            self.trend_direction = None
            self.entry_bar = None
            self.trailing_stop = None
            self.entry_price = None
            
            self.logger.info(f"Trade closed - {trade_result.get('reason', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Error in on_trade_closed: {str(e)}")
    
    def get_risk_parameters(self) -> Dict[str, Any]:
        """Get tight pullback scalping risk management parameters"""
        return {
            "sl_pct": self.config.get('stop_loss_pct', 0.0015),    # 0.15% tight stop
            "tp_pct": self.config.get('profit_target_pct', 0.003), # 0.3% quick target
            "max_position_pct": self.config.get('max_position_pct', 2.0),  # Conservative position sizing
            "risk_reward_ratio": self.config.get('profit_target_pct', 0.003) / self.config.get('stop_loss_pct', 0.0015)  # 2:1 ratio
        } 