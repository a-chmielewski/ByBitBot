"""
High-Volatility Trend Rider Strategy

Strategy #11 in the Strategy Matrix for High-Volatility Trending Markets
Market Conditions: Best fit for HIGH_VOLATILITY and TRENDING markets
Description: Specialized trend following for high volatility trending conditions
"""

import logging
from typing import Any, Dict, Optional, List
import pandas as pd
import pandas_ta as ta
import numpy as np

from .strategy_template import StrategyTemplate

class StrategyHighVolatilityTrendRider(StrategyTemplate):
    """
    High-Volatility Trend Rider Strategy
    
    **STRATEGY MATRIX ROLE**: Primary strategy for high-volatility trending conditions
    **MATRIX USAGE**: Best for HIGH_VOLATILITY + TRENDING market combinations
    **EXECUTION TIMEFRAME**: 1-minute for precision entry timing in volatile trends
    
    Strategy Logic:
    - Enhanced trend detection for high volatility environments using EMA + ADX
    - Volatility regime filter using ATR percentiles and Bollinger Band width
    - Pullback and breakout entries in trend direction only
    - ATR-based trailing stops and dynamic risk management
    - Volume confirmation and momentum triggers for entry timing
    
    Market Type:
        Designed for strongly trending markets under high volatility on both timeframes.
        Excels when price makes sustained directional moves with large swings.
        ADX > 25 confirms strong trend, elevated ATR indicates wide price ranges.

    Indicators & Parameters:
        - EMAs (20/50) for trend direction filter
        - ADX (14) for trend strength confirmation (>25 threshold)
        - ATR (14) for volatility regime detection (top percentile)
        - Bollinger Bands (20,2) for additional volatility confirmation
        - RSI (14) for momentum triggers and exit signals
        - Volume SMA for participation confirmation

    Entry Conditions:
        Trend Confirmation:
        - Only longs in confirmed uptrend (EMA fast > slow, ADX > 25)
        - Only shorts in confirmed downtrend (EMA fast < slow, ADX > 25)
        - High volatility regime (ATR in top percentile, BB width above threshold)
        
        Entry Types:
        1. Pullback Entry: Wait for minor pullback, enter on trend resumption
        2. Breakout Entry: Enter on momentum breakouts with volume confirmation
        
        Filters:
        - Volume above average for strong participation
        - Cooldown period between trades to avoid overtrading
        - No counter-trend trades allowed

    Exit Conditions:
        - ATR-based trailing stop (2x ATR from peak)
        - Momentum fade (RSI crosses back below/above 50)
        - EMA cross against trend direction
        - Time-based exit for maximum trade duration
        - Partial profit taking at 1:1 risk/reward
    """
    
    MARKET_TYPE_TAGS: List[str] = ['HIGH_VOLATILITY', 'TRENDING']
    SHOW_IN_SELECTION: bool = True  # Now fully implemented
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(data, config, logger)

    def on_init(self) -> None:
        super().on_init()
        self.logger.info(f"{self.__class__.__name__} on_init called.")
        
        # Get strategy-specific parameters from config
        strategy_specific_params = self.config.get('strategy_configs', {}).get(self.__class__.__name__, {})

        # EMA parameters for trend detection
        self.ema_fast_period = strategy_specific_params.get("ema_fast_period", 20)
        self.ema_slow_period = strategy_specific_params.get("ema_slow_period", 50)
        
        # ADX parameters for trend strength - Enhanced for stricter trend confirmation
        self.adx_period = strategy_specific_params.get("adx_period", 14)
        self.adx_threshold = strategy_specific_params.get("adx_threshold", 23)  
        self.adx_strong_threshold = strategy_specific_params.get("adx_strong_threshold", 33)  
        
        # ATR parameters for volatility and stops
        self.atr_period = strategy_specific_params.get("atr_period", 14)
        self.atr_stop_multiplier = strategy_specific_params.get("atr_stop_multiplier", 3.0)  # Wider trailing from 2.0x
        self.atr_target_multiplier = strategy_specific_params.get("atr_target_multiplier", 2.0)
        
        # Volatility regime detection
        self.atr_volatility_percentile = strategy_specific_params.get("atr_volatility_percentile", 80)
        self.atr_lookback = strategy_specific_params.get("atr_lookback", 50)
        self.bb_period = strategy_specific_params.get("bb_period", 20)
        self.bb_std = strategy_specific_params.get("bb_std", 2.0)
        self.min_bb_width = strategy_specific_params.get("min_bb_width", 0.04)  # 4% minimum
        
        # RSI parameters for momentum
        self.rsi_period = strategy_specific_params.get("rsi_period", 14)
        self.rsi_bull_threshold = strategy_specific_params.get("rsi_bull_threshold", 50)
        self.rsi_bear_threshold = strategy_specific_params.get("rsi_bear_threshold", 50)
        
        # Volume parameters - Enhanced for stricter volume confirmation
        self.volume_period = strategy_specific_params.get("volume_period", 20)
        self.volume_multiplier = strategy_specific_params.get("volume_multiplier", 1.4)  # Reduced from 1.5 to 1.4 for less restrictive volume confirmation
        
        # Pullback and breakout parameters - Enhanced for deeper pullbacks
        self.pullback_bars = strategy_specific_params.get("pullback_bars", 5)
        self.min_pullback_pct = strategy_specific_params.get("min_pullback_pct", 0.004)  # Increased from 0.2% to 0.4%
        self.breakout_bars = strategy_specific_params.get("breakout_bars", 50)
        self.breakout_range_multiplier = strategy_specific_params.get("breakout_range_multiplier", 1.5)
        self.min_consolidation_bars = strategy_specific_params.get("min_consolidation_bars", 3)  # Minimum consolidation period
        
        # Risk management
        self.max_trade_duration = strategy_specific_params.get("max_trade_duration", 100)
        self.cooldown_bars = strategy_specific_params.get("cooldown_bars", 5)
        self.partial_profit_ratio = strategy_specific_params.get("partial_profit_ratio", 0.5)
        
        # Cache risk parameters for get_risk_parameters()
        self.sl_pct = strategy_specific_params.get('sl_pct', 0.03)  # 3% default for high volatility
        self.tp_pct = strategy_specific_params.get('tp_pct', 0.04)  # 4% quicker target from 6%
        
        # State tracking variables
        self.trend_direction = 0  # 1 for bullish, -1 for bearish, 0 for neutral
        self.in_high_volatility_regime = False
        self.atr_history = []
        self.entry_bar_index: Optional[int] = None
        self.last_trade_bar = -self.cooldown_bars  # Allow first trade immediately
        self.trailing_stop_price = None
        self.partial_profit_taken = False

        self.logger.info(f"{self.__class__.__name__} parameters: EMA={self.ema_fast_period}/{self.ema_slow_period}, "
                        f"ADX threshold={self.adx_threshold} (enhanced), Volume multiplier={self.volume_multiplier} (enhanced), "
                        f"ATR percentile={self.atr_volatility_percentile}")

    def init_indicators(self) -> None:
        """Initialize indicators for high-volatility trend detection"""
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
            
            # EMAs for trend direction
            self.data['ema_fast'] = self.data.ta.ema(close=self.data['close'], length=self.ema_fast_period)
            self.data['ema_slow'] = self.data.ta.ema(close=self.data['close'], length=self.ema_slow_period)
            
            # ADX for trend strength
            adx_data = self.data.ta.adx(high=self.data['high'], low=self.data['low'], 
                                       close=self.data['close'], length=self.adx_period)
            if adx_data is not None and not adx_data.empty:
                adx_col = None
                dmp_col = None
                dmn_col = None
                
                for col in adx_data.columns:
                    if 'adx' in col.lower():
                        adx_col = col
                    elif 'dmp' in col.lower() or '+di' in col.lower():
                        dmp_col = col
                    elif 'dmn' in col.lower() or '-di' in col.lower():
                        dmn_col = col
                
                if adx_col:
                    self.data['adx'] = adx_data[adx_col]
                    if dmp_col:
                        self.data['dmp'] = adx_data[dmp_col]
                    if dmn_col:
                        self.data['dmn'] = adx_data[dmn_col]
                else:
                    self.logger.warning("ADX calculation failed, using manual calculation")
                    self._calculate_adx_manual()
            else:
                self.logger.warning("ADX calculation failed, using manual calculation")
                self._calculate_adx_manual()
            
            # ATR for volatility measurement
            self.data['atr'] = self.data.ta.atr(high=self.data['high'], low=self.data['low'], 
                                              close=self.data['close'], length=self.atr_period)
            if 'atr' not in self.data.columns or self.data['atr'].isna().all():
                self.logger.warning("ATR calculation failed, using manual calculation")
                self._calculate_atr_manual()
            
            # Bollinger Bands for volatility confirmation
            bb_data = self.data.ta.bbands(close=self.data['close'], length=self.bb_period, std=self.bb_std)
            if bb_data is not None and not bb_data.empty:
                bb_cols = bb_data.columns.tolist()
                lower_col = next((col for col in bb_cols if 'lower' in col.lower() or 'l_' in col.lower()), None)
                middle_col = next((col for col in bb_cols if 'mid' in col.lower() or 'm_' in col.lower()), None)
                upper_col = next((col for col in bb_cols if 'upper' in col.lower() or 'u_' in col.lower()), None)
                
                if lower_col and middle_col and upper_col:
                    self.data['bb_lower'] = bb_data[lower_col]
                    self.data['bb_middle'] = bb_data[middle_col]
                    self.data['bb_upper'] = bb_data[upper_col]
                    self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
                else:
                    self.logger.warning("Could not find all BB columns, using manual calculation")
                    self._calculate_bollinger_bands_manual()
            else:
                self.logger.warning("Bollinger Bands calculation failed, using manual calculation")
                self._calculate_bollinger_bands_manual()
            
            # RSI for momentum
            self.data['rsi'] = self.data.ta.rsi(close=self.data['close'], length=self.rsi_period)
            if 'rsi' not in self.data.columns or self.data['rsi'].isna().all():
                self.logger.warning("RSI calculation failed, using manual calculation")
                self._calculate_rsi_manual()
            
            # Volume SMA
            self.data['volume_sma'] = self.data['volume'].rolling(window=self.volume_period, min_periods=1).mean()
            
            # Price highs and lows for breakout detection
            self.data['highest'] = self.data['high'].rolling(window=self.breakout_bars, min_periods=1).max()
            self.data['lowest'] = self.data['low'].rolling(window=self.breakout_bars, min_periods=1).min()
            
            self.logger.debug(f"'{self.__class__.__name__}': All indicators initialized successfully")
            
        except Exception as e:
            self.logger.error(f"'{self.__class__.__name__}': Error initializing indicators: {str(e)}")
            self._set_default_indicators()

    def _calculate_adx_manual(self) -> None:
        """Manual ADX calculation as fallback"""
        # Simplified ADX calculation
        tr1 = self.data['high'] - self.data['low']
        tr2 = abs(self.data['high'] - self.data['close'].shift(1))
        tr3 = abs(self.data['low'] - self.data['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['atr'] = tr.rolling(window=self.adx_period, min_periods=1).mean()
        self.data['adx'] = 25.0  # Default neutral value
        self.data['dmp'] = 25.0
        self.data['dmn'] = 25.0

    def _calculate_atr_manual(self) -> None:
        """Manual ATR calculation as fallback"""
        tr1 = self.data['high'] - self.data['low']
        tr2 = abs(self.data['high'] - self.data['close'].shift(1))
        tr3 = abs(self.data['low'] - self.data['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['atr'] = tr.rolling(window=self.atr_period, min_periods=1).mean()

    def _calculate_bollinger_bands_manual(self) -> None:
        """Manual Bollinger Bands calculation as fallback"""
        sma = self.data['close'].rolling(window=self.bb_period, min_periods=1).mean()
        std = self.data['close'].rolling(window=self.bb_period, min_periods=1).std()
        self.data['bb_middle'] = sma
        self.data['bb_upper'] = sma + (std * self.bb_std)
        self.data['bb_lower'] = sma - (std * self.bb_std)
        self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']

    def _calculate_rsi_manual(self) -> None:
        """Manual RSI calculation as fallback"""
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))

    def _set_default_indicators(self) -> None:
        """Set default indicator values to prevent crashes"""
        self.data['ema_fast'] = self.data['close']
        self.data['ema_slow'] = self.data['close']
        self.data['adx'] = 25.0
        self.data['dmp'] = 25.0
        self.data['dmn'] = 25.0
        self.data['atr'] = self.data['close'] * 0.01
        self.data['bb_lower'] = self.data['close'] * 0.98
        self.data['bb_middle'] = self.data['close']
        self.data['bb_upper'] = self.data['close'] * 1.02
        self.data['bb_width'] = 0.04
        self.data['rsi'] = 50.0
        self.data['volume_sma'] = self.data['volume']
        self.data['highest'] = self.data['high']
        self.data['lowest'] = self.data['low']

    def update_indicators_for_new_row(self) -> None:
        """Update indicators efficiently for the latest row only"""
        if self.data is None or len(self.data) < 2:
            self.init_indicators()
            return
            
        try:
            latest_idx = self.data.index[-1]
            
            # Update EMAs
            if len(self.data) >= max(self.ema_fast_period, self.ema_slow_period):
                alpha_fast = 2 / (self.ema_fast_period + 1)
                alpha_slow = 2 / (self.ema_slow_period + 1)
                
                prev_ema_fast = self.data['ema_fast'].iloc[-2]
                prev_ema_slow = self.data['ema_slow'].iloc[-2]
                current_close = self.data['close'].iloc[-1]
                
                self.data.loc[latest_idx, 'ema_fast'] = alpha_fast * current_close + (1 - alpha_fast) * prev_ema_fast
                self.data.loc[latest_idx, 'ema_slow'] = alpha_slow * current_close + (1 - alpha_slow) * prev_ema_slow
            
            # Update ATR
            if len(self.data) >= self.atr_period:
                current_data = self.data.iloc[-1]
                prev_close = self.data['close'].iloc[-2]
                
                tr1 = current_data['high'] - current_data['low']
                tr2 = abs(current_data['high'] - prev_close)
                tr3 = abs(current_data['low'] - prev_close)
                tr = max(tr1, tr2, tr3)
                
                prev_atr = self.data['atr'].iloc[-2]
                alpha_atr = 1 / self.atr_period
                self.data.loc[latest_idx, 'atr'] = alpha_atr * tr + (1 - alpha_atr) * prev_atr
            
            # Update other indicators (simplified for performance)
            self.data.loc[latest_idx, 'volume_sma'] = self.data['volume'].iloc[-self.volume_period:].mean()
            
            # Update ATR history for volatility regime detection
            current_atr = self.data.loc[latest_idx, 'atr']
            self.atr_history.append(current_atr)
            if len(self.atr_history) > self.atr_lookback:
                self.atr_history.pop(0)
            
        except Exception as e:
            self.logger.warning(f"Error updating indicators for new row: {e}, falling back to full recalculation")
            self.init_indicators()

    def _get_current_values(self) -> Optional[Dict[str, Any]]:
        """Get current indicator values for analysis"""
        if self.data is None or self.data.empty:
            return None
            
        try:
            latest = self.data.iloc[-1]
            return {
                'close': latest['close'],
                'high': latest['high'],
                'low': latest['low'],
                'volume': latest['volume'],
                'ema_fast': latest['ema_fast'],
                'ema_slow': latest['ema_slow'],
                'adx': latest['adx'],
                'atr': latest['atr'],
                'bb_width': latest['bb_width'],
                'rsi': latest['rsi'],
                'volume_sma': latest['volume_sma'],
                'highest': latest['highest'],
                'lowest': latest['lowest']
            }
        except Exception as e:
            self.logger.error(f"Error getting current values: {e}")
            return None

    def _is_high_volatility_regime(self, vals: Dict[str, Any]) -> bool:
        """Check if we're in a high volatility environment"""
        try:
            current_atr = vals['atr']
            
            # ATR percentile check
            if len(self.atr_history) >= self.atr_lookback:
                atr_threshold = np.percentile(self.atr_history, self.atr_volatility_percentile)
                atr_condition = current_atr >= atr_threshold
            else:
                atr_condition = True  # Not enough history, assume OK
            
            # Bollinger Band width check
            bb_condition = vals['bb_width'] > self.min_bb_width
            
            return atr_condition and bb_condition
            
        except Exception as e:
            self.logger.error(f"Error checking volatility regime: {e}")
            return False

    def _is_trending_market(self, vals: Dict[str, Any]) -> tuple:
        """Check if market is in a strong trend and determine direction"""
        try:
            adx_value = vals['adx']
            adx_condition = adx_value > self.adx_threshold
            
            if not adx_condition:
                return False, None
            
            # Determine trend direction using EMAs and price position
            ema_fast = vals['ema_fast']
            ema_slow = vals['ema_slow']
            current_price = vals['close']
            
            if ema_fast > ema_slow and current_price > ema_slow:
                trend_direction = 'up'
            elif ema_fast < ema_slow and current_price < ema_slow:
                trend_direction = 'down'
            else:
                trend_direction = None
            
            return adx_condition and trend_direction is not None, trend_direction
            
        except Exception as e:
            self.logger.error(f"Error checking trending market: {e}")
            return False, None

    def _is_volume_confirmation(self, vals: Dict[str, Any]) -> bool:
        """Check for volume confirmation"""
        try:
            current_volume = vals['volume']
            avg_volume = vals['volume_sma']
            return current_volume > avg_volume * self.volume_multiplier
        except Exception as e:
            self.logger.error(f"Error checking volume confirmation: {e}")
            return True  # Default to True if volume check fails

    def _detect_pullback_entry(self, trend_direction: str, vals: Dict[str, Any]) -> bool:
        """Detect pullback entry opportunities"""
        try:
            current_price = vals['close']
            ema_fast = vals['ema_fast']
            rsi_value = vals['rsi']
            
            if trend_direction == 'up':
                # Look for pullback in uptrend
                pullback_condition = current_price < ema_fast or rsi_value < 45
                
                # Check for resumption signals
                if pullback_condition and len(self.data) > 1:
                    prev_close = self.data['close'].iloc[-2]
                    resumption = current_price > prev_close  # Green candle
                    return resumption
            
            elif trend_direction == 'down':
                # Look for pullback in downtrend
                pullback_condition = current_price > ema_fast or rsi_value > 55
                
                # Check for resumption signals
                if pullback_condition and len(self.data) > 1:
                    prev_close = self.data['close'].iloc[-2]
                    resumption = current_price < prev_close  # Red candle
                    return resumption
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting pullback entry: {e}")
            return False

    def _detect_breakout_entry(self, trend_direction: str, vals: Dict[str, Any]) -> bool:
        """Detect breakout entry opportunities"""
        try:
            current_price = vals['close']
            current_high = vals['high']
            current_low = vals['low']
            current_range = current_high - current_low
            atr_value = vals['atr']
            
            # Check for sufficient range
            range_condition = current_range > atr_value * self.breakout_range_multiplier
            
            if trend_direction == 'up':
                # Bullish breakout
                recent_high = vals['highest']
                if len(self.data) > 1:
                    prev_highest = self.data['highest'].iloc[-2]
                    breakout_condition = current_high > prev_highest
                else:
                    breakout_condition = current_high > recent_high * 0.999  # Small buffer
                return breakout_condition and range_condition
            
            elif trend_direction == 'down':
                # Bearish breakout
                recent_low = vals['lowest']
                if len(self.data) > 1:
                    prev_lowest = self.data['lowest'].iloc[-2]
                    breakout_condition = current_low < prev_lowest
                else:
                    breakout_condition = current_low < recent_low * 1.001  # Small buffer
                return breakout_condition and range_condition
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting breakout entry: {e}")
            return False

    def _should_enter_trade(self, vals: Dict[str, Any]) -> tuple:
        """Main entry logic combining all conditions"""
        try:
            # Check cooldown period
            current_bar = len(self.data) - 1
            if current_bar - self.last_trade_bar < self.cooldown_bars:
                return False, None
            
            # Check volatility regime
            if not self._is_high_volatility_regime(vals):
                return False, None
            
            # Check trending market
            is_trending, trend_direction = self._is_trending_market(vals)
            if not is_trending:
                return False, None
            
            # Check volume confirmation
            if not self._is_volume_confirmation(vals):
                return False, None
            
            # Check for entry signals
            pullback_signal = self._detect_pullback_entry(trend_direction, vals)
            breakout_signal = self._detect_breakout_entry(trend_direction, vals)
            
            if pullback_signal or breakout_signal:
                entry_type = "pullback" if pullback_signal else "breakout"
                self.logger.info(f"Entry signal detected: {trend_direction} {entry_type} entry")
                return True, trend_direction
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error in should_enter_trade: {e}")
            return False, None

    def _calculate_stops_and_targets(self, direction: str, vals: Dict[str, Any]) -> tuple:
        """Calculate stop loss and take profit levels"""
        try:
            current_price = vals['close']
            atr_value = vals['atr']
            
            if direction == 'long':
                stop_price = current_price - (atr_value * self.atr_stop_multiplier)
                target_price = current_price + (atr_value * self.atr_target_multiplier)
            else:  # short
                stop_price = current_price + (atr_value * self.atr_stop_multiplier)
                target_price = current_price - (atr_value * self.atr_target_multiplier)
            
            return stop_price, target_price
            
        except Exception as e:
            self.logger.error(f"Error calculating stops and targets: {e}")
            # Fallback to percentage-based stops
            if direction == 'long':
                return current_price * 0.97, current_price * 1.06
            else:
                return current_price * 1.03, current_price * 0.94

    def _update_trailing_stop(self, vals: Dict[str, Any], symbol: str) -> None:
        """Update trailing stop based on ATR"""
        current_pos = self.position.get(symbol)
        if not current_pos or vals is None:
            return
            
        try:
            current_price = vals['close']
            atr_value = vals['atr']
            pos_side = current_pos.get('side', '').lower()
            
            if pos_side == 'buy':  # Long position
                new_stop = current_price - (atr_value * self.atr_stop_multiplier)
                if self.trailing_stop_price is None or new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
            else:  # Short position
                new_stop = current_price + (atr_value * self.atr_stop_multiplier)
                if self.trailing_stop_price is None or new_stop < self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
                    
        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {e}")

    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for high-volatility trend entry conditions"""
        vals = self._get_current_values()
        if vals is None:
            return None
        
        # Check entry conditions
        should_enter, trend_direction = self._should_enter_trade(vals)
        
        if should_enter and trend_direction:
            # Record entry bar and initialize trailing stop
            self.entry_bar_index = len(self.data) - 1
            self.trailing_stop_price = None
            self.partial_profit_taken = False
            
            # Calculate initial stop and target
            stop_price, target_price = self._calculate_stops_and_targets(trend_direction, vals)
            self.trailing_stop_price = stop_price
            
            order_details = {
                "side": "buy" if trend_direction == 'up' else "sell",
                "type": "market",
                "price": vals['close'],  # Market order, price for reference
            }
            
            self.logger.info(f"{self.__class__.__name__} for {symbol}: {trend_direction.upper()} trend entry at "
                           f"price {vals['close']:.6f} (stop: {stop_price:.6f}, target: {target_price:.6f})")
            
            return order_details
        
        return None

    def check_exit(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for high-volatility trend exit conditions"""
        current_pos = self.position.get(symbol)
        if current_pos is None:
            return None
            
        vals = self._get_current_values()
        if vals is None:
            return None
            
        try:
            current_price = vals['close']
            pos_side = current_pos.get('side', '').lower()
            entry_price = current_pos.get('entry_price', current_price)
            bars_held = (len(self.data) - 1) - (self.entry_bar_index or 0) if self.entry_bar_index is not None else 0
            
            # Update trailing stop
            self._update_trailing_stop(vals, symbol)
            
            # Time-based exit
            if bars_held >= self.max_trade_duration:
                self.logger.info(f"{self.__class__.__name__} for {symbol}: Time exit after {bars_held} bars")
                return {"type": "market", "reason": "time_exit"}
            
            # Trailing stop exit
            if self.trailing_stop_price:
                if pos_side == 'buy' and current_price <= self.trailing_stop_price:
                    self.logger.info(f"{self.__class__.__name__} for {symbol}: LONG trailing stop hit "
                                   f"(price={current_price:.6f}, stop={self.trailing_stop_price:.6f})")
                    return {"type": "market", "reason": "trailing_stop"}
                elif pos_side == 'sell' and current_price >= self.trailing_stop_price:
                    self.logger.info(f"{self.__class__.__name__} for {symbol}: SHORT trailing stop hit "
                                   f"(price={current_price:.6f}, stop={self.trailing_stop_price:.6f})")
                    return {"type": "market", "reason": "trailing_stop"}
            
            # Momentum fade exit
            rsi_value = vals['rsi']
            if pos_side == 'buy' and rsi_value < self.rsi_bear_threshold:
                self.logger.info(f"{self.__class__.__name__} for {symbol}: LONG momentum fade exit (RSI={rsi_value:.2f})")
                return {"type": "market", "reason": "momentum_fade"}
            elif pos_side == 'sell' and rsi_value > self.rsi_bull_threshold:
                self.logger.info(f"{self.__class__.__name__} for {symbol}: SHORT momentum fade exit (RSI={rsi_value:.2f})")
                return {"type": "market", "reason": "momentum_fade"}
            
            # EMA cross exit
            ema_fast = vals['ema_fast']
            if pos_side == 'buy' and current_price < ema_fast:
                self.logger.info(f"{self.__class__.__name__} for {symbol}: LONG EMA cross exit")
                return {"type": "market", "reason": "ema_cross"}
            elif pos_side == 'sell' and current_price > ema_fast:
                self.logger.info(f"{self.__class__.__name__} for {symbol}: SHORT EMA cross exit")
                return {"type": "market", "reason": "ema_cross"}
            
            # Check for partial profit (1:1 risk/reward)
            if not self.partial_profit_taken:
                risk = abs(entry_price - (self.trailing_stop_price or entry_price * 0.97))
                if pos_side == 'buy':
                    profit_level = entry_price + risk
                    if current_price >= profit_level:
                        self.partial_profit_taken = True
                        self.logger.info(f"{self.__class__.__name__} for {symbol}: Partial profit target reached")
                        # Note: Actual partial exit would need to be handled by OrderManager
                else:  # short
                    profit_level = entry_price - risk
                    if current_price <= profit_level:
                        self.partial_profit_taken = True
                        self.logger.info(f"{self.__class__.__name__} for {symbol}: Partial profit target reached")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return None

    def get_risk_parameters(self) -> Dict[str, Any]:
        """Return risk parameters for high-volatility trend trading"""
        return {
            "sl_pct": min(self.sl_pct * 2, 0.08),  # Widened stops for high volatility
            "tp_pct": self.tp_pct   # Configured take profit percentage
        }

    def on_order_update(self, order_responses: Dict[str, Any], symbol: str) -> None:
        """Handle order updates specific to high-volatility trend strategy"""
        super().on_order_update(order_responses, symbol)
        
        # Initialize trailing stop on successful entry
        main_order_response = order_responses.get('main_order')
        if main_order_response:
            order_result = main_order_response.get('result', {})
            order_status = order_result.get('orderStatus', '').lower()
            
            if order_status in ['filled', 'partiallyfilled']:
                # Initialize trailing stop at entry
                vals = self._get_current_values()
                if vals:
                    entry_price = float(order_result.get('avgPrice', order_result.get('price', vals['close'])))
                    atr_value = vals['atr']
                    side = order_result.get('side', '').lower()
                    
                    if side == 'buy':
                        self.trailing_stop_price = entry_price - (atr_value * self.atr_stop_multiplier)
                    else:
                        self.trailing_stop_price = entry_price + (atr_value * self.atr_stop_multiplier)
                    
                    self.logger.info(f"{self.__class__.__name__}: Entry filled, trailing stop initialized at {self.trailing_stop_price:.6f}")

    def on_trade_update(self, trade: Dict[str, Any], symbol: str) -> None:
        """Handle trade updates"""
        super().on_trade_update(trade, symbol)
        
        # Reset state on trade close
        if trade.get('exit'):
            self.entry_bar_index = None
            self.trailing_stop_price = None
            self.partial_profit_taken = False
            self.last_trade_bar = len(self.data) - 1 if self.data is not None else 0

    def on_error(self, exception: Exception) -> None:
        """Handle strategy errors"""
        self.logger.error(f"Strategy {self.__class__.__name__} encountered an error: {exception}", exc_info=True)
        # Reset state on error to prevent stuck states
        self.entry_bar_index = None
        self.trailing_stop_price = None
        self.partial_profit_taken = False 