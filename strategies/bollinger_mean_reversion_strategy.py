"""
Bollinger Band Mean Reversion Strategy

Strategy #3 in the Strategy Matrix for Ranging Markets
Market Conditions: Best fit for RANGING markets
Description: Uses Bollinger Bands for mean reversion trading in ranging conditions
"""

import logging
from typing import Any, Dict, Optional, List
import pandas as pd
import pandas_ta as ta
import numpy as np

from .strategy_template import StrategyTemplate

class StrategyBollingerMeanReversion(StrategyTemplate):
    """
    Bollinger Band Mean Reversion Strategy
    
    **STRATEGY MATRIX ROLE**: Primary strategy for ranging market conditions
    **MATRIX USAGE**: Best for RANGING(5m) + RANGING(1m) or mixed ranging conditions
    **EXECUTION TIMEFRAME**: 1-minute for precision entry/exit timing
    
    Strategy Logic:
    - Use Bollinger Bands to identify overbought/oversold conditions in ranging markets
    - Enter on mean reversion signals when price touches bands with RSI confirmation
    - Exit when price returns to middle band (SMA) or on stop loss
    - Uses volume confirmation and reversal patterns for better entry timing
    
    Market Type:
        Range-bound conditions, especially in low to medium volatility ranges. 
        Price moving sideways between support and resistance. Ideal when there's 
        no strong trend and candles frequently alternate (indecisive direction).

    Indicators & Parameters:
        - Bollinger Bands (20-period SMA, 2 standard deviations)
        - RSI (14) for overbought/oversold confirmation
        - Volume SMA for volume analysis
        - Market regime detection to avoid trending periods

    Entry Conditions:
        Long Entry:
        - Price tests lower Bollinger Band AND RSI < 30 (oversold)
        - Volume below average (exhaustion at extreme)
        - Confirmation: price closes back inside bands or bullish reversal pattern
        
        Short Entry:
        - Price tests upper Bollinger Band AND RSI > 70 (overbought)
        - Volume below average (lack of breakout force)
        - Confirmation: price closes back inside bands or bearish reversal pattern

    Exit Conditions:
        - Primary target: Middle band (20 SMA) for mean reversion
        - Stop loss: Just outside the band (tight stops for range trading)
        - Time stop: Exit if held too long without mean reversion
        - Emergency exit: If market regime changes to trending
    """
    
    MARKET_TYPE_TAGS: List[str] = ['RANGING', 'LOW_VOLATILITY']
    SHOW_IN_SELECTION: bool = True  # Now fully implemented
    
    def __init__(self, data: pd.DataFrame, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(data, config, logger)

    def on_init(self) -> None:
        super().on_init()
        self.logger.info(f"{self.__class__.__name__} on_init called.")
        
        # Get strategy-specific parameters from config
        strategy_specific_params = self.config.get('strategy_configs', {}).get(self.__class__.__name__, {})

        # Bollinger Band parameters
        self.bb_period = strategy_specific_params.get("bb_period", 20)
        self.bollinger_period = self.bb_period  # Alias for test compatibility
        self.bb_std = strategy_specific_params.get("bb_std", 2.0)
        self.bollinger_std_dev = self.bb_std  # Alias for test compatibility
        
        # RSI parameters  
        self.rsi_period = strategy_specific_params.get("rsi_period", 14)
        self.rsi_oversold = strategy_specific_params.get("rsi_oversold", 30)
        self.rsi_overbought = strategy_specific_params.get("rsi_overbought", 70)
        
        # Volume parameters
        self.volume_period = strategy_specific_params.get("volume_period", 20)
        self.min_volume_ratio = strategy_specific_params.get("min_volume_ratio", 0.8)
        self.volume_spike_multiplier = strategy_specific_params.get("volume_spike_multiplier", 1.5)
        
        # Market regime detection
        self.trend_detection_period = strategy_specific_params.get("trend_detection_period", 50)
        self.max_trend_slope = strategy_specific_params.get("max_trend_slope", 0.002)
        self.min_range_width = strategy_specific_params.get("min_range_width", 0.01)
        
        # Reversal confirmation
        self.reversal_confirmation_bars = strategy_specific_params.get("reversal_confirmation_bars", 2)
        
        # ATR parameters for tests
        self.atr_period = strategy_specific_params.get("atr_period", 14)
        self.atr_stop_multiplier = strategy_specific_params.get("atr_stop_multiplier", 1.5)
        self.atr_target_multiplier = strategy_specific_params.get("atr_target_multiplier", 2.0)
        
        # Risk management
        self.stop_loss_pct = strategy_specific_params.get("stop_loss_pct", 0.005)
        self.time_stop_bars = strategy_specific_params.get("time_stop_bars", 30)
        
        # Cache risk parameters for get_risk_parameters()
        self.sl_pct = strategy_specific_params.get('sl_pct', 0.015)  # 1.5% default for ranging
        self.tp_pct = strategy_specific_params.get('tp_pct', 0.02)   # 2% default target
        
        # State tracking variables
        self.in_range_mode = True
        self.band_touch_detected = None  # Will be 'upper'/'lower' or None for test compatibility
        self.touch_direction = 0  # 1 for upper band, -1 for lower band
        self.touch_bar = None
        self.band_touch_bar = None  # Alias for test compatibility
        self.waiting_for_reversal = False
        self.reversal_confirmed = False
        self.mean_reversion_setup = False  # For test compatibility
        
        # Entry tracking
        self.entry_bar_index: Optional[int] = None

        self.logger.info(f"{self.__class__.__name__} parameters: BB period={self.bb_period}, "
                        f"BB std={self.bb_std}, RSI thresholds={self.rsi_oversold}/{self.rsi_overbought}")

    def init_indicators(self) -> None:
        """Initialize Bollinger Bands, RSI, and other indicators"""
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
            
            # Bollinger Bands
            bb_data = self.data.ta.bbands(close=self.data['close'], length=self.bb_period, std=self.bb_std)
            if bb_data is not None and not bb_data.empty:
                # Find the correct column names
                bb_cols = bb_data.columns.tolist()
                lower_col = next((col for col in bb_cols if 'lower' in col.lower() or 'l_' in col.lower()), None)
                middle_col = next((col for col in bb_cols if 'mid' in col.lower() or 'm_' in col.lower()), None)
                upper_col = next((col for col in bb_cols if 'upper' in col.lower() or 'u_' in col.lower()), None)
                
                if lower_col and middle_col and upper_col:
                    self.data['bb_lower'] = bb_data[lower_col]
                    self.data['bb_middle'] = bb_data[middle_col]
                    self.data['bb_upper'] = bb_data[upper_col]
                else:
                    self.logger.warning("Could not find all BB columns, using manual calculation")
                    self._calculate_bollinger_bands_manual()
            else:
                self.logger.warning("Bollinger Bands calculation failed, using manual calculation")
                self._calculate_bollinger_bands_manual()
            
            # RSI
            self.data['rsi'] = self.data.ta.rsi(close=self.data['close'], length=self.rsi_period)
            if 'rsi' not in self.data.columns or self.data['rsi'].isna().all():
                self.logger.warning("RSI calculation failed, using manual calculation")
                self._calculate_rsi_manual()
            
            # Volume SMA
            self.data['volume_sma'] = self.data['volume'].rolling(window=self.volume_period, min_periods=1).mean()
            self.data['volume_ma'] = self.data['volume_sma']  # Alias for test compatibility
            
            # ATR calculation for tests
            self.data['atr'] = self.data.ta.atr(high=self.data['high'], low=self.data['low'], 
                                              close=self.data['close'], length=self.atr_period)
            if 'atr' not in self.data.columns or self.data['atr'].isna().all():
                self._calculate_atr_manual()
            
            # Trend SMA for market regime detection
            self.data['trend_sma'] = self.data['close'].rolling(window=self.trend_detection_period, min_periods=1).mean()
            
            # Calculate BB width for regime detection
            self.data['bb_width'] = (self.data['bb_upper'] - self.data['bb_lower']) / self.data['bb_middle']
            
            self.logger.debug(f"'{self.__class__.__name__}': All indicators initialized successfully")
            
        except Exception as e:
            self.logger.error(f"'{self.__class__.__name__}': Error initializing indicators: {str(e)}")
            # Set default values to prevent crashes
            self.data['bb_lower'] = self.data['close'] * 0.98
            self.data['bb_middle'] = self.data['close']
            self.data['bb_upper'] = self.data['close'] * 1.02
            self.data['rsi'] = 50.0
            self.data['volume_sma'] = self.data['volume']
            self.data['trend_sma'] = self.data['close']
            self.data['bb_width'] = 0.02

    def _calculate_bollinger_bands_manual(self) -> None:
        """Manual Bollinger Bands calculation as fallback"""
        sma = self.data['close'].rolling(window=self.bb_period, min_periods=1).mean()
        std = self.data['close'].rolling(window=self.bb_period, min_periods=1).std()
        self.data['bb_middle'] = sma
        self.data['bb_upper'] = sma + (std * self.bb_std)
        self.data['bb_lower'] = sma - (std * self.bb_std)

    def _calculate_rsi_manual(self) -> None:
        """Manual RSI calculation as fallback"""
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period, min_periods=1).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))

    def _calculate_atr_manual(self) -> None:
        """Manual ATR calculation as fallback"""
        high_low = self.data['high'] - self.data['low']
        high_close_prev = abs(self.data['high'] - self.data['close'].shift())
        low_close_prev = abs(self.data['low'] - self.data['close'].shift())
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        self.data['atr'] = true_range.rolling(window=self.atr_period, min_periods=1).mean()

    def update_indicators_for_new_row(self) -> None:
        """Update indicators efficiently for the latest row only"""
        if self.data is None or len(self.data) < 2:
            self.init_indicators()
            return
            
        try:
            latest_idx = self.data.index[-1]
            
            # Update Bollinger Bands for latest row
            if len(self.data) >= self.bb_period:
                recent_closes = self.data['close'].iloc[-self.bb_period:].values
                sma_val = np.mean(recent_closes)
                std_val = np.std(recent_closes, ddof=1)
                self.data.loc[latest_idx, 'bb_middle'] = sma_val
                self.data.loc[latest_idx, 'bb_upper'] = sma_val + (std_val * self.bb_std)
                self.data.loc[latest_idx, 'bb_lower'] = sma_val - (std_val * self.bb_std)
                self.data.loc[latest_idx, 'bb_width'] = (self.data.loc[latest_idx, 'bb_upper'] - 
                                                        self.data.loc[latest_idx, 'bb_lower']) / sma_val
            
            # Update RSI for latest row
            if len(self.data) >= self.rsi_period + 1:
                recent_closes = self.data['close'].iloc[-self.rsi_period-1:].values
                deltas = np.diff(recent_closes)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi_val = 100 - (100 / (1 + rs))
                else:
                    rsi_val = 100
                self.data.loc[latest_idx, 'rsi'] = rsi_val
            
            # Update volume SMA
            if len(self.data) >= self.volume_period:
                recent_volume = self.data['volume'].iloc[-self.volume_period:].values
                self.data.loc[latest_idx, 'volume_sma'] = np.mean(recent_volume)
            
            # Update trend SMA
            if len(self.data) >= self.trend_detection_period:
                recent_closes = self.data['close'].iloc[-self.trend_detection_period:].values
                self.data.loc[latest_idx, 'trend_sma'] = np.mean(recent_closes)
                
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
                'bb_upper': latest['bb_upper'],
                'bb_middle': latest['bb_middle'],
                'bb_lower': latest['bb_lower'],
                'bb_width': latest['bb_width'],
                'rsi': latest['rsi'],
                'volume_sma': latest['volume_sma'],
                'volume_ma': latest.get('volume_ma', latest['volume_sma']),  # Test compatibility
                'atr': latest.get('atr', 0.0),  # Test compatibility
                'trend_sma': latest['trend_sma']
            }
        except Exception as e:
            self.logger.error(f"Error getting current values: {e}")
            return None

    def _detect_market_regime(self, vals: Dict[str, Any]) -> bool:
        """Detect if market is in ranging mode (suitable for mean reversion)"""
        try:
            # Check trend slope
            if len(self.data) >= self.trend_detection_period + 10:
                current_sma = vals['trend_sma']
                past_sma = self.data['trend_sma'].iloc[-10]
                slope = (current_sma - past_sma) / past_sma if past_sma != 0 else 0
                is_ranging = abs(slope) < self.max_trend_slope
            else:
                is_ranging = True  # Assume ranging until we have enough data
            
            # Check Bollinger Band width (sufficient volatility for trades)
            sufficient_width = vals['bb_width'] > self.min_range_width
            
            return is_ranging and sufficient_width
            
        except Exception as e:
            self.logger.error(f"Error detecting market regime: {e}")
            return True  # Default to ranging

    def _detect_band_touch(self, vals: Dict[str, Any]) -> bool:
        """Detect when price touches Bollinger Bands with proper conditions"""
        try:
            current_price = vals['close']
            current_high = vals['high']
            current_low = vals['low']
            current_rsi = vals['rsi']
            current_volume = vals['volume']
            avg_volume = vals['volume_sma']
            
            # Volume confirmation (lower volume at extremes suggests exhaustion)
            volume_ok = current_volume < avg_volume * self.min_volume_ratio
            
            # Check for upper band touch with overbought RSI
            if (current_high >= vals['bb_upper'] or current_price > vals['bb_upper']) and current_rsi > self.rsi_overbought:
                if volume_ok:
                    self.band_touch_detected = 'upper'  # Changed to string for test compatibility
                    self.band_touch_bar = len(self.data) - 1  # Update both attributes
                    self.touch_direction = 1  # Upper band touch
                    self.touch_bar = len(self.data) - 1
                    self.waiting_for_reversal = True
                    self.logger.info(f"Upper band touch detected: price={current_price:.6f}, "
                                   f"bb_upper={vals['bb_upper']:.6f}, rsi={current_rsi:.2f}")
                    return True
                    
            # Check for lower band touch with oversold RSI
            elif (current_low <= vals['bb_lower'] or current_price < vals['bb_lower']) and current_rsi < self.rsi_oversold:
                if volume_ok:
                    self.band_touch_detected = 'lower'  # Changed to string for test compatibility
                    self.band_touch_bar = len(self.data) - 1  # Update both attributes
                    self.touch_direction = -1  # Lower band touch
                    self.touch_bar = len(self.data) - 1
                    self.waiting_for_reversal = True
                    self.logger.info(f"Lower band touch detected: price={current_price:.6f}, "
                                   f"bb_lower={vals['bb_lower']:.6f}, rsi={current_rsi:.2f}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detecting band touch: {e}")
            return False

    def _detect_bollinger_band_touch(self, vals: Dict[str, Any]) -> Optional[str]:
        """Alias for _detect_band_touch to match test expectations"""
        result = self._detect_band_touch(vals)
        if result:
            return self.band_touch_detected  # Return 'upper' or 'lower'
        return None

    def _check_volume_confirmation(self, vals: Dict[str, Any]) -> bool:
        """Check volume confirmation for trade signals"""
        try:
            current_volume = vals['volume']
            avg_volume = vals.get('volume_ma', vals.get('volume_sma', 0))
            
            if avg_volume == 0:
                return False
                
            # Volume should be above average * multiplier for confirmation
            return current_volume > (avg_volume * self.volume_spike_multiplier)
            
        except Exception as e:
            self.logger.error(f"Error checking volume confirmation: {e}")
            return False

    def _detect_mean_reversion_setup(self, vals: Dict[str, Any]) -> bool:
        """Detect mean reversion setup after band touch"""
        try:
            if not self.band_touch_detected or self.band_touch_bar is None:
                return False
                
            current_price = vals['close']
            current_rsi = vals['rsi']
            bb_middle = vals['bb_middle']
            
            # Check if enough bars have passed since band touch
            bars_since_touch = (len(self.data) - 1) - self.band_touch_bar
            if bars_since_touch < 1:
                return False
                
            # For upper band touch (expecting mean reversion down)
            if self.band_touch_detected == 'upper':
                # Price should be moving back towards middle, RSI moving down from overbought
                price_reverting = current_price < vals['bb_upper']
                rsi_reverting = current_rsi < 65  # Moving away from overbought
                self.mean_reversion_setup = price_reverting and rsi_reverting
                
            # For lower band touch (expecting mean reversion up)  
            elif self.band_touch_detected == 'lower':
                # Price should be moving back towards middle, RSI moving up from oversold
                price_reverting = current_price > vals['bb_lower']
                rsi_reverting = current_rsi > 35  # Moving away from oversold
                self.mean_reversion_setup = price_reverting and rsi_reverting
            
            return self.mean_reversion_setup
            
        except Exception as e:
            self.logger.error(f"Error detecting mean reversion setup: {e}")
            return False

    def _check_reversal_confirmation(self, vals: Dict[str, Any]) -> tuple:
        """Check for reversal confirmation after band touch"""
        if not self.waiting_for_reversal or self.touch_bar is None:
            return False, None
            
        try:
            current_price = vals['close']
            bars_since_touch = (len(self.data) - 1) - self.touch_bar
            
            # Don't wait too long for reversal
            if bars_since_touch > self.reversal_confirmation_bars + 2:
                self._reset_reversal_state()
                return False, None
            
            if self.touch_direction == 1:  # Upper band touch, looking for bearish reversal
                # Check if price closes back inside bands
                price_inside = current_price < vals['bb_upper']
                
                # Look for reversal patterns
                reversal_pattern = self._detect_bearish_reversal_patterns()
                
                if price_inside and (reversal_pattern or bars_since_touch >= self.reversal_confirmation_bars):
                    self.logger.info(f"Bearish reversal confirmed: price back inside bands at {current_price:.6f}")
                    return True, 'short'
                    
            elif self.touch_direction == -1:  # Lower band touch, looking for bullish reversal
                # Check if price closes back inside bands
                price_inside = current_price > vals['bb_lower']
                
                # Look for reversal patterns
                reversal_pattern = self._detect_bullish_reversal_patterns()
                
                if price_inside and (reversal_pattern or bars_since_touch >= self.reversal_confirmation_bars):
                    self.logger.info(f"Bullish reversal confirmed: price back inside bands at {current_price:.6f}")
                    return True, 'long'
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error checking reversal confirmation: {e}")
            return False, None

    def _detect_bearish_reversal_patterns(self) -> bool:
        """Detect bearish reversal candlestick patterns"""
        if len(self.data) < 2:
            return False
            
        try:
            current = self.data.iloc[-1]
            previous = self.data.iloc[-2]
            
            # Bearish engulfing
            bearish_engulfing = (previous['close'] > previous['open'] and 
                               current['close'] < current['open'] and 
                               current['close'] < previous['open'] and 
                               current['open'] > previous['close'])
            
            # Shooting star
            shooting_star = (current['open'] < current['close'] < current['high'] and 
                           (current['high'] - current['close']) > 2 * (current['close'] - current['open']))
            
            return bearish_engulfing or shooting_star
            
        except Exception as e:
            self.logger.error(f"Error detecting bearish reversal patterns: {e}")
            return False

    def _detect_bullish_reversal_patterns(self) -> bool:
        """Detect bullish reversal candlestick patterns"""
        if len(self.data) < 2:
            return False
            
        try:
            current = self.data.iloc[-1]
            previous = self.data.iloc[-2]
            
            # Bullish engulfing
            bullish_engulfing = (previous['close'] < previous['open'] and 
                               current['close'] > current['open'] and 
                               current['close'] > previous['open'] and 
                               current['open'] < previous['close'])
            
            # Hammer
            hammer = (current['open'] > current['close'] > current['low'] and 
                     (current['open'] - current['low']) > 2 * (current['open'] - current['close']))
            
            return bullish_engulfing or hammer
            
        except Exception as e:
            self.logger.error(f"Error detecting bullish reversal patterns: {e}")
            return False

    def _reset_reversal_state(self) -> None:
        """Reset reversal detection state"""
        self.waiting_for_reversal = False
        self.band_touch_detected = None
        self.reversal_confirmed = False
        self.touch_direction = 0
        self.touch_bar = None
        self.band_touch_bar = None
        self.mean_reversion_setup = False

    def _reset_strategy_state(self) -> None:
        """Alias for _reset_reversal_state to match test expectations"""
        self._reset_reversal_state()

    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for Bollinger Band mean reversion entry conditions"""
        vals = self._get_current_values()
        if vals is None:
            return None
        
        # Step 1: Check if market is in ranging mode
        self.in_range_mode = self._detect_market_regime(vals)
        
        if not self.in_range_mode:
            self.logger.debug(f"Market not in ranging mode, skipping mean reversion entry")
            self._reset_reversal_state()  # Clear any pending signals
            return None
        
        # Step 2: Look for band touches if not already detected
        if not self.band_touch_detected:
            self._detect_band_touch(vals)
        
        # Step 3: Check for reversal confirmation
        reversal_signal, direction = self._check_reversal_confirmation(vals)
        
        if reversal_signal and direction:
            # Reset state after signal generation
            self._reset_reversal_state()
            
            # Record entry bar for exit timing
            self.entry_bar_index = len(self.data) - 1
            
            order_details = {
                "side": "buy" if direction == 'long' else "sell",
                "type": "market",
                "price": vals['close'],  # Market order, price for reference
            }
            
            self.logger.info(f"{self.__class__.__name__} for {symbol}: {direction.upper()} entry signal at "
                           f"price {vals['close']:.6f} after {direction} reversal confirmation")
            
            return order_details
        
        return None

    def check_exit(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check for mean reversion exit conditions"""
        current_pos = self.position.get(symbol)
        if current_pos is None:
            return None
            
        vals = self._get_current_values()
        if vals is None:
            return None
            
        try:
            current_price = vals['close']
            pos_side = current_pos.get('side', '').lower()
            bars_held = (len(self.data) - 1) - (self.entry_bar_index or 0) if self.entry_bar_index is not None else 0
            
            # Time stop
            if bars_held >= self.time_stop_bars:
                self.logger.info(f"{self.__class__.__name__} for {symbol}: Time stop triggered after {bars_held} bars")
                return {"type": "market", "reason": "time_stop"}
            
            # Check if market regime changed to trending (emergency exit)
            if not self._detect_market_regime(vals):
                self.logger.info(f"{self.__class__.__name__} for {symbol}: Market regime changed to trending, emergency exit")
                return {"type": "market", "reason": "regime_change"}
            
            # Mean reversion exits
            if pos_side == 'buy':  # Long position
                # Take profit at middle band (mean reversion target)
                if current_price >= vals['bb_middle']:
                    self.logger.info(f"{self.__class__.__name__} for {symbol}: LONG take profit at middle band "
                                   f"(price={current_price:.6f}, middle={vals['bb_middle']:.6f})")
                    return {"type": "market", "reason": "mean_reversion_target"}
                
                # Stop loss if breaks below lower band again (breakout against us)
                elif current_price < vals['bb_lower']:
                    self.logger.info(f"{self.__class__.__name__} for {symbol}: LONG stop loss, price below lower band "
                                   f"(price={current_price:.6f}, lower={vals['bb_lower']:.6f})")
                    return {"type": "market", "reason": "stop_loss"}
                    
            elif pos_side == 'sell':  # Short position
                # Take profit at middle band
                if current_price <= vals['bb_middle']:
                    self.logger.info(f"{self.__class__.__name__} for {symbol}: SHORT take profit at middle band "
                                   f"(price={current_price:.6f}, middle={vals['bb_middle']:.6f})")
                    return {"type": "market", "reason": "mean_reversion_target"}
                
                # Stop loss if breaks above upper band again
                elif current_price > vals['bb_upper']:
                    self.logger.info(f"{self.__class__.__name__} for {symbol}: SHORT stop loss, price above upper band "
                                   f"(price={current_price:.6f}, upper={vals['bb_upper']:.6f})")
                    return {"type": "market", "reason": "stop_loss"}
        
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return None

    def get_risk_parameters(self) -> Dict[str, Any]:
        """Return risk parameters for mean reversion trading"""
        return {
            "sl_pct": self.sl_pct,  # Configured stop loss percentage
            "tp_pct": self.tp_pct   # Configured take profit percentage
        }

    def on_order_update(self, order_responses: Dict[str, Any], symbol: str) -> None:
        """Handle order updates specific to mean reversion strategy"""
        super().on_order_update(order_responses, symbol)
        
        # Reset reversal state on successful entry
        main_order_response = order_responses.get('main_order')
        if main_order_response:
            order_result = main_order_response.get('result', {})
            order_status = order_result.get('orderStatus', '').lower()
            
            if order_status in ['filled', 'partiallyfilled']:
                self._reset_reversal_state()
                self.logger.info(f"{self.__class__.__name__}: Entry filled, reversal state reset")

    def on_trade_update(self, trade: Dict[str, Any], symbol: str) -> None:
        """Handle trade updates"""
        super().on_trade_update(trade, symbol)
        
        # Reset entry tracking on trade close
        if trade.get('exit'):
            self.entry_bar_index = None
            self._reset_reversal_state()

    def on_error(self, exception: Exception) -> None:
        """Handle strategy errors"""
        self.logger.error(f"Strategy {self.__class__.__name__} encountered an error: {exception}", exc_info=True)
        # Reset state on error to prevent stuck states
        self._reset_reversal_state()
        self.entry_bar_index = None