import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Any, Dict, Optional, List
import logging

from .strategy_template import StrategyTemplate

class StrategyEMATrendRider(StrategyTemplate):
    """
    EMA Trend Rider with ADX Filter Strategy
    
    **STRATEGY MATRIX ROLE**: Primary strategy for stable trending conditions
    **MATRIX USAGE**: TRENDING(5m) + TRENDING(1m) → 5-minute execution (only 5m execution in matrix)
    **EXECUTION TIMEFRAME**: 5-minute for stable trend-following
    
    Market Type: Strong trending conditions with clear directional bias on both timeframes.
    
    Quick Strategy Logic:
    1. Wait for EMA crossover (20/50) with ADX > threshold for trend confirmation
    2. Wait for pullback against the trend for better entry price
    3. Enter on trend resumption with volume confirmation
    4. Use ATR-based stops and targets with trailing functionality

    Market Type:
    - Strong trending conditions (price making higher highs/higher lows in uptrend or 
      lower highs/lower lows in downtrend)

    Indicators & Parameters:
    - Fast and slow Exponential Moving Averages (20-period EMA and 50-period EMA)
    - Average Directional Index (ADX) 14-period as trend-strength filter
    - Optional: RSI (14) for momentum confirmation (above 50 for uptrend, below 50 for downtrend)

    Entry Conditions:
    Only trade in direction of the dominant trend:

    1. Trend Detection:
       - Wait for fast EMA to cross above slow EMA for bullish trend (or below for bearish)
       - Confirm ADX > 25-30 indicating strong trend (filters out weak/sideways moves)
       - Example: ADX rising above 30 with EMA bull-cross implies robust uptrend

    2. Pullback Entry:
       - Wait for brief pullback against trend for better price
       - Uptrend: If price above EMAs, wait for dip toward 20 EMA or red candles
       - Enter long on upward resumption (bullish engulfing off 20 EMA or 50 EMA bounce)
       - Downtrend: Wait for bounce toward EMAs, enter short on bearish reversal

    3. Volume Confirmation:
       - Prefer entries with increasing volume in trend direction
       - Example: Uptrend pullback - volume tapers on dip, spikes on rally

    Exit Conditions:
    Two-pronged approach to minimize drawdown:

    1. Take Profit:
       - Exit at nearest significant price swing level or pivot
       - Uptrend: Exit approaching last swing high or round-number resistance
       - Alternative: Fixed risk-reward (1.5:1 or 2:1) if backtests show high win rate

    2. Trailing Stop:
       - Trail stop behind short-term EMA or recent swing lows/highs
       - Stay in if trend continues, exit with locked-in profit if trend reverses
       - Example: Move stop loss up below each higher low in uptrend

    Stop Loss / Take Profit:
    - Stop Loss: Place beyond pullback's extreme
      * Long entry: Few ticks below recent swing low or 50 EMA
      * Keeps risk tight (0.2-0.5% on liquid pairs)
    - Take Profit: Split approach
      * First target at prior swing high
      * Runner position to trail for larger moves
    - ATR-based stops effective:
      * Stop at 2×ATR(14) from entry
      * Initial target at 2×ATR in profit
      * Adjust for desired risk/reward

    Notes (Leverage & Timing):
    - High win rates in strong trends due to momentum trading
    - ADX filter crucial for high leverage (avoids choppy periods)
    - Best during clear market direction (crypto: London/NY session overlap)
    - Avoid new entries if:
      * Volatility drops
      * ADX falls below 20-25
    - Always use stop losses (critical for leveraged positions)
    """
    
    # Market type tags - Active strategy for double trending conditions
    MARKET_TYPE_TAGS: List[str] = ['TRENDING']
    
    # Strategy Matrix integration
    SHOW_IN_SELECTION: bool = True  # Available for manual selection and automatic matrix selection

    def __init__(self,
                 data: pd.DataFrame,
                 config: Dict[str, Any],
                 logger: logging.Logger):
        super().__init__(data=data, config=config, logger=logger)

    def on_init(self) -> None:
        super().on_init()
        self.logger.info(f"{self.__class__.__name__} on_init called.")
        
        # Get strategy-specific parameters from config
        strategy_specific_params = self.config.get('strategy_configs', {}).get(self.__class__.__name__, {})

        # EMA parameters
        self.ema_fast_period = strategy_specific_params.get("ema_fast_period", 20)
        self.ema_slow_period = strategy_specific_params.get("ema_slow_period", 50)
        
        # ADX parameters
        self.adx_period = strategy_specific_params.get("adx_period", 14)
        self.adx_threshold = strategy_specific_params.get("adx_threshold", 25)
        self.adx_strong_threshold = strategy_specific_params.get("adx_strong_threshold", 30)
        
        # RSI parameters
        self.rsi_period = strategy_specific_params.get("rsi_period", 14)
        self.rsi_bull_threshold = strategy_specific_params.get("rsi_bull_threshold", 50)
        self.rsi_bear_threshold = strategy_specific_params.get("rsi_bear_threshold", 50)
        
        # ATR parameters
        self.atr_period = strategy_specific_params.get("atr_period", 14)
        self.atr_stop_multiplier = strategy_specific_params.get("atr_stop_multiplier", 2.0)
        self.atr_target_multiplier = strategy_specific_params.get("atr_target_multiplier", 2.0)
        
        # Pullback parameters
        self.pullback_bars = strategy_specific_params.get("pullback_bars", 5)
        self.min_pullback_pct = strategy_specific_params.get("min_pullback_pct", 0.002)  # 0.2%
        self.pullback_timeout_bars = strategy_specific_params.get("pullback_timeout_bars", 20)
        
        # Volume parameters
        self.volume_period = strategy_specific_params.get("volume_period", 20)
        self.volume_spike_multiplier = strategy_specific_params.get("volume_spike_multiplier", 1.2)
        
        # Risk parameters
        self.time_stop_bars = strategy_specific_params.get("time_stop_bars", 50)
        
        # Cache risk parameters for get_risk_parameters()
        self.sl_pct = strategy_specific_params.get('sl_pct', 0.005)  # 0.5% default
        self.tp_pct = strategy_specific_params.get('tp_pct', 0.01)   # 1% default
        
        # State tracking variables
        self.trend_direction = 0  # 1 for bullish, -1 for bearish, 0 for neutral
        self.trend_confirmed = False
        self.trend_start_bar = None
        self.waiting_for_pullback = False
        self.pullback_detected = False
        self.pullback_extreme_price = 0
        self.pullback_start_bar = None
        self.recent_high = 0
        self.recent_low = float('inf')
        
        # Entry tracking
        self.entry_bar_index: Optional[int] = None

        self.logger.info(f"{self.__class__.__name__} parameters: EMA Fast={self.ema_fast_period}, "
                        f"EMA Slow={self.ema_slow_period}, ADX threshold={self.adx_threshold}, "
                        f"ATR multipliers: SL={self.atr_stop_multiplier}, TP={self.atr_target_multiplier}")

    def init_indicators(self) -> None:
        """Initialize indicators required by the strategy."""
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
            
            # EMA Fast and Slow
            self.data['ema_fast'] = self.data.ta.ema(close=self.data['close'], length=self.ema_fast_period)
            self.data['ema_slow'] = self.data.ta.ema(close=self.data['close'], length=self.ema_slow_period)
            
            # ADX and Directional Movement
            adx_data = self.data.ta.adx(high=self.data['high'], low=self.data['low'], 
                                       close=self.data['close'], length=self.adx_period)
            if adx_data is not None and not adx_data.empty:
                # Find ADX column (different naming conventions)
                adx_col = None
                for col in adx_data.columns:
                    if 'adx' in col.lower():
                        adx_col = col
                        break
                
                if adx_col:
                    self.data['adx'] = adx_data[adx_col]
                else:
                    self.logger.warning("Could not find ADX column in pandas_ta output, using manual calculation")
                    self.data['adx'] = 25.0  # Default neutral value
            else:
                self.logger.warning("ADX calculation failed, using default values")
                self.data['adx'] = 25.0
            
            # RSI
            self.data['rsi'] = self.data.ta.rsi(close=self.data['close'], length=self.rsi_period)
            if 'rsi' not in self.data.columns or self.data['rsi'].isna().all():
                self.logger.warning("RSI calculation failed, using default values")
                self.data['rsi'] = 50.0
            
            # ATR for stop loss and take profit calculations
            self.data['atr'] = self.data.ta.atr(high=self.data['high'], low=self.data['low'], 
                                               close=self.data['close'], length=self.atr_period)
            if 'atr' not in self.data.columns or self.data['atr'].isna().all():
                self.logger.warning("ATR calculation failed, using default values")
                self.data['atr'] = self.data['close'] * 0.01  # 1% of price as fallback
            
            # Volume SMA for volume confirmation
            self.data['volume_sma'] = self.data['volume'].rolling(window=self.volume_period, min_periods=1).mean()
            
            # EMA crossover signal
            self.data['ema_cross'] = 0
            for i in range(1, len(self.data)):
                if (self.data['ema_fast'].iloc[i] > self.data['ema_slow'].iloc[i] and 
                    self.data['ema_fast'].iloc[i-1] <= self.data['ema_slow'].iloc[i-1]):
                    self.data.iloc[i, self.data.columns.get_loc('ema_cross')] = 1  # Bullish cross
                elif (self.data['ema_fast'].iloc[i] < self.data['ema_slow'].iloc[i] and 
                      self.data['ema_fast'].iloc[i-1] >= self.data['ema_slow'].iloc[i-1]):
                    self.data.iloc[i, self.data.columns.get_loc('ema_cross')] = -1  # Bearish cross
            
            self.logger.debug(f"{self.__class__.__name__} indicators initialized successfully.")
            
        except Exception as e:
            self.logger.error(f"Error initializing indicators: {e}", exc_info=True)

    def update_indicators_for_new_row(self) -> None:
        """Update indicators for the latest row efficiently."""
        if self.data is None or self.data.empty or len(self.data) < 2:
            self.logger.debug(f"'{self.__class__.__name__}': Not enough data for incremental update.")
            return

        # Check if indicator columns exist, if not initialize them first
        required_indicator_cols = ['ema_fast', 'ema_slow', 'adx', 'rsi', 'atr', 'ema_cross', 'volume_sma']
        missing_cols = [col for col in required_indicator_cols if col not in self.data.columns]
        
        if missing_cols:
            self.logger.info(f"'{self.__class__.__name__}': Missing indicator columns {missing_cols}. Initializing indicators first.")
            self.init_indicators()
            return

        try:
            latest_idx = self.data.index[-1]
            prev_idx = self.data.index[-2]
            
            # Update EMAs incrementally
            current_close = self.data.loc[latest_idx, 'close']
            
            # EMA Fast
            prev_ema_fast = self.data.loc[prev_idx, 'ema_fast']
            if pd.isna(prev_ema_fast):
                if len(self.data) >= self.ema_fast_period:
                    prev_ema_fast = self.data['close'].iloc[-self.ema_fast_period:].mean()
                else:
                    prev_ema_fast = current_close
            alpha_fast = 2 / (self.ema_fast_period + 1)
            new_ema_fast = (current_close * alpha_fast) + (prev_ema_fast * (1 - alpha_fast))
            self.data.loc[latest_idx, 'ema_fast'] = new_ema_fast

            # EMA Slow
            prev_ema_slow = self.data.loc[prev_idx, 'ema_slow']
            if pd.isna(prev_ema_slow):
                if len(self.data) >= self.ema_slow_period:
                    prev_ema_slow = self.data['close'].iloc[-self.ema_slow_period:].mean()
                else:
                    prev_ema_slow = current_close
            alpha_slow = 2 / (self.ema_slow_period + 1)
            new_ema_slow = (current_close * alpha_slow) + (prev_ema_slow * (1 - alpha_slow))
            self.data.loc[latest_idx, 'ema_slow'] = new_ema_slow
            
            # Update crossover signal
            prev_ema_fast_val = self.data.loc[prev_idx, 'ema_fast']
            prev_ema_slow_val = self.data.loc[prev_idx, 'ema_slow']
            
            cross_signal = 0
            if (new_ema_fast > new_ema_slow and prev_ema_fast_val <= prev_ema_slow_val):
                cross_signal = 1  # Bullish cross
            elif (new_ema_fast < new_ema_slow and prev_ema_fast_val >= prev_ema_slow_val):
                cross_signal = -1  # Bearish cross
            self.data.loc[latest_idx, 'ema_cross'] = cross_signal
            
            # Update other indicators on tail data for accuracy
            min_tail_data = max(self.adx_period, self.rsi_period, self.atr_period) + 5
            if len(self.data) >= min_tail_data:
                tail_data = self.data.tail(min_tail_data)
                
                # ADX
                adx_tail = tail_data.ta.adx(high='high', low='low', close='close', length=self.adx_period)
                if adx_tail is not None and not adx_tail.empty:
                    adx_col = None
                    for col in adx_tail.columns:
                        if 'adx' in col.lower():
                            adx_col = col
                            break
                    if adx_col:
                        self.data.loc[latest_idx, 'adx'] = adx_tail[adx_col].iloc[-1]
                    else:
                        self.data.loc[latest_idx, 'adx'] = 25.0
                else:
                    self.data.loc[latest_idx, 'adx'] = 25.0
                
                # RSI
                rsi_tail = tail_data.ta.rsi(close='close', length=self.rsi_period)
                if rsi_tail is not None and not rsi_tail.empty:
                    self.data.loc[latest_idx, 'rsi'] = rsi_tail.iloc[-1]
                else:
                    self.data.loc[latest_idx, 'rsi'] = 50.0
                
                # ATR
                atr_tail = tail_data.ta.atr(high='high', low='low', close='close', length=self.atr_period)
                if atr_tail is not None and not atr_tail.empty:
                    self.data.loc[latest_idx, 'atr'] = atr_tail.iloc[-1]
                else:
                    self.data.loc[latest_idx, 'atr'] = current_close * 0.01
            else:
                # Use default values if not enough data
                self.data.loc[latest_idx, 'adx'] = 25.0
                self.data.loc[latest_idx, 'rsi'] = 50.0
                self.data.loc[latest_idx, 'atr'] = current_close * 0.01
            
            # Volume SMA
            if len(self.data) >= self.volume_period:
                volume_sma = self.data['volume'].iloc[-self.volume_period:].mean()
                self.data.loc[latest_idx, 'volume_sma'] = volume_sma
            else:
                self.data.loc[latest_idx, 'volume_sma'] = self.data['volume'].iloc[-len(self.data):].mean()
                
        except Exception as e:
            self.logger.error(f"'{self.__class__.__name__}': Error in update_indicators_for_new_row: {e}", exc_info=True)
            # Fallback to full initialization
            self.init_indicators()

    def _get_current_values(self) -> Optional[Dict[str, Any]]:
        """Helper to get current market values for analysis."""
        if self.data is None or self.data.empty or len(self.data) < 2:
            return None

        latest = self.data.iloc[-1]
        previous = self.data.iloc[-2]

        required_cols = ['close', 'high', 'low', 'volume', 'ema_fast', 'ema_slow', 'adx', 'rsi', 'atr', 'ema_cross']
        if not all(col in self.data.columns for col in required_cols):
            return None

        # Check for NaN values in critical indicators
        if latest[required_cols].isnull().any():
            return None

        return {
            "current_price": latest['close'],
            "current_high": latest['high'],
            "current_low": latest['low'],
            "current_volume": latest['volume'],
            "ema_fast": latest['ema_fast'],
            "ema_slow": latest['ema_slow'],
            "adx": latest['adx'],
            "rsi": latest['rsi'],
            "atr": latest['atr'],
            "ema_cross": latest['ema_cross'],
            "volume_sma": latest.get('volume_sma', 0),
            "prev_close": previous['close'],
            "current_bar_datetime": latest.name
        }

    def _detect_trend_and_strength(self, vals: Dict[str, Any]) -> bool:
        """Detect trend direction and strength using EMA cross and ADX."""
        current_bar = len(self.data) - 1
        
        # Check for fresh EMA cross
        if vals["ema_cross"] == 1:  # Bullish cross
            if vals["adx"] >= self.adx_threshold:
                self.trend_direction = 1
                self.trend_confirmed = True
                self.waiting_for_pullback = True
                self.trend_start_bar = current_bar
                self.recent_high = vals["current_price"]
                self.recent_low = vals["current_price"]
                self.logger.info(f"{self.__class__.__name__}: Bullish trend detected - EMA cross with ADX {vals['adx']:.1f}")
                return True
        elif vals["ema_cross"] == -1:  # Bearish cross
            if vals["adx"] >= self.adx_threshold:
                self.trend_direction = -1
                self.trend_confirmed = True
                self.waiting_for_pullback = True
                self.trend_start_bar = current_bar
                self.recent_high = vals["current_price"]
                self.recent_low = vals["current_price"]
                self.logger.info(f"{self.__class__.__name__}: Bearish trend detected - EMA cross with ADX {vals['adx']:.1f}")
                return True
        
        # Check if existing trend is still valid
        if self.trend_confirmed:
            # Trend remains valid if ADX is still strong and EMAs maintain order
            if vals["adx"] < self.adx_threshold - 5:  # Give some buffer
                self.logger.info(f"{self.__class__.__name__}: Trend weakening - ADX {vals['adx']:.1f} below threshold")
                self._reset_trend_state()
                return False
            
            # Check if EMA order is maintained
            if self.trend_direction == 1 and vals["ema_fast"] < vals["ema_slow"]:
                self.logger.info(f"{self.__class__.__name__}: Bullish trend invalidated - EMA order reversed")
                self._reset_trend_state()
                return False
            elif self.trend_direction == -1 and vals["ema_fast"] > vals["ema_slow"]:
                self.logger.info(f"{self.__class__.__name__}: Bearish trend invalidated - EMA order reversed")
                self._reset_trend_state()
                return False
        
        return self.trend_confirmed

    def _detect_pullback(self, vals: Dict[str, Any]) -> bool:
        """Detect valid pullback for entry."""
        if not self.waiting_for_pullback:
            return False
            
        current_price = vals["current_price"]
        current_bar = len(self.data) - 1
        
        # Update recent highs and lows
        if self.trend_direction == 1:  # Bullish trend
            if current_price > self.recent_high:
                self.recent_high = current_price
            
            # Check for pullback (price dipping from recent high)
            pullback_pct = (self.recent_high - current_price) / self.recent_high
            if pullback_pct >= self.min_pullback_pct:
                if not self.pullback_detected:
                    self.pullback_detected = True
                    self.pullback_extreme_price = current_price
                    self.pullback_start_bar = current_bar
                    self.logger.info(f"{self.__class__.__name__}: Pullback detected in uptrend - {pullback_pct:.3f}% from high")
                elif current_price < self.pullback_extreme_price:
                    self.pullback_extreme_price = current_price
                return True
                
        elif self.trend_direction == -1:  # Bearish trend
            if current_price < self.recent_low:
                self.recent_low = current_price
            
            # Check for pullback (price bouncing from recent low)
            pullback_pct = (current_price - self.recent_low) / self.recent_low
            if pullback_pct >= self.min_pullback_pct:
                if not self.pullback_detected:
                    self.pullback_detected = True
                    self.pullback_extreme_price = current_price
                    self.pullback_start_bar = current_bar
                    self.logger.info(f"{self.__class__.__name__}: Pullback detected in downtrend - {pullback_pct:.3f}% from low")
                elif current_price > self.pullback_extreme_price:
                    self.pullback_extreme_price = current_price
                return True
        
        return False

    def _check_pullback_entry(self, vals: Dict[str, Any]) -> tuple:
        """Check if conditions are right for entry after pullback."""
        if not self.pullback_detected:
            return False, None
            
        current_price = vals["current_price"]
        current_volume = vals["current_volume"]
        volume_sma = vals["volume_sma"]
        
        # Check for timeout
        current_bar = len(self.data) - 1
        if (self.pullback_start_bar is not None and 
            current_bar - self.pullback_start_bar > self.pullback_timeout_bars):
            self.logger.info(f"{self.__class__.__name__}: Pullback timeout - resetting trend state")
            self._reset_trend_state()
            return False, None
        
        if self.trend_direction == 1:  # Bullish trend
            # Look for bounce from pullback
            bounce_threshold = self.pullback_extreme_price * 1.001  # 0.1% bounce
            if current_price > bounce_threshold:
                # Additional confirmations
                rsi_ok = vals["rsi"] > self.rsi_bull_threshold
                price_above_fast_ema = current_price > vals["ema_fast"] * 0.999  # Small tolerance
                adx_strong = vals["adx"] > self.adx_strong_threshold
                volume_ok = volume_sma > 0 and current_volume > volume_sma * self.volume_spike_multiplier
                
                if rsi_ok and (price_above_fast_ema or adx_strong) and volume_ok:
                    self.logger.info(f"{self.__class__.__name__}: Long entry conditions met - bounce from pullback")
                    return True, 'long'
                    
        elif self.trend_direction == -1:  # Bearish trend
            # Look for reversal from pullback
            reversal_threshold = self.pullback_extreme_price * 0.999  # 0.1% reversal
            if current_price < reversal_threshold:
                # Additional confirmations
                rsi_ok = vals["rsi"] < self.rsi_bear_threshold
                price_below_fast_ema = current_price < vals["ema_fast"] * 1.001  # Small tolerance
                adx_strong = vals["adx"] > self.adx_strong_threshold
                volume_ok = volume_sma > 0 and current_volume > volume_sma * self.volume_spike_multiplier
                
                if rsi_ok and (price_below_fast_ema or adx_strong) and volume_ok:
                    self.logger.info(f"{self.__class__.__name__}: Short entry conditions met - reversal from pullback")
                    return True, 'short'
        
        return False, None

    def _reset_trend_state(self) -> None:
        """Reset all trend and pullback state variables."""
        self.trend_direction = 0
        self.trend_confirmed = False
        self.trend_start_bar = None
        self.waiting_for_pullback = False
        self.pullback_detected = False
        self.pullback_extreme_price = 0
        self.pullback_start_bar = None
        self.recent_high = 0
        self.recent_low = float('inf')

    def _check_entry_conditions(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Check entry conditions for the EMA-ADX strategy."""
        vals = self._get_current_values()
        if not vals:
            return None

        # Step 1: Detect trend and strength
        trend_valid = self._detect_trend_and_strength(vals)
        
        # Step 2: If trend is valid, detect pullback
        if trend_valid:
            self._detect_pullback(vals)
        
        # Step 3: Check for entry after pullback
        entry_signal, direction = self._check_pullback_entry(vals)
        
        if entry_signal and direction:
            self.logger.info(f"{self.__class__.__name__}: Entry signal confirmed for {direction} at {vals['current_price']}")
            
            strat_order_size = self.config.get('strategy_configs', {}).get(self.__class__.__name__, {}).get('order_size')
            
            order_details = {
                "side": "buy" if direction == 'long' else "sell",
                "price": vals["current_price"],
                "size": strat_order_size
            }
            
            # Reset pullback state after entry signal
            self.waiting_for_pullback = False
            self.pullback_detected = False
            
            return order_details
        
        return None

    def check_exit(self, symbol: str) -> bool:
        """Check for strategy-specific exit conditions."""
        # Get current position
        current_position = self.position.get(symbol)
        if not current_position:
            return False
            
        vals = self._get_current_values()
        if not vals:
            return False
        
        # Time-based exit
        if self.entry_bar_index is not None:
            current_bar = len(self.data) - 1
            bars_held = current_bar - self.entry_bar_index
            
            if bars_held >= self.time_stop_bars:
                self.logger.info(f"{self.__class__.__name__}: Time stop reached ({bars_held} bars)")
                return True
        
        # Trend weakness exit
        if vals["adx"] < self.adx_threshold - 5:
            self.logger.info(f"{self.__class__.__name__}: Trend weakness exit - ADX {vals['adx']:.1f}")
            return True
        
        # EMA trend reversal exit
        position_side = current_position.get('side', '').lower()
        if position_side == 'buy' and vals["current_price"] < vals["ema_fast"]:
            self.logger.info(f"{self.__class__.__name__}: Long position exit - price below fast EMA")
            return True
        elif position_side == 'sell' and vals["current_price"] > vals["ema_fast"]:
            self.logger.info(f"{self.__class__.__name__}: Short position exit - price above fast EMA")
            return True
        
        return False

    def get_risk_parameters(self) -> Dict[str, Any]:
        """Return risk parameters for the strategy."""
        # Try to use ATR-based risk if available
        vals = self._get_current_values()
        if vals and vals["atr"] > 0:
            # ATR-based stops and targets
            atr_sl_pct = (vals["atr"] * self.atr_stop_multiplier) / vals["current_price"]
            atr_tp_pct = (vals["atr"] * self.atr_target_multiplier) / vals["current_price"]
            
            # Use ATR-based if reasonable, otherwise fall back to fixed
            sl_pct = min(atr_sl_pct, self.sl_pct * 2)  # Cap at 2x fixed SL
            tp_pct = max(atr_tp_pct, self.tp_pct * 0.5)  # Minimum 0.5x fixed TP
            
            return {
                "sl_pct": sl_pct,
                "tp_pct": tp_pct
            }
        
        # Fallback to fixed percentages
        return {
            "sl_pct": self.sl_pct,
            "tp_pct": self.tp_pct
        }

    def on_order_update(self, order_responses: Dict[str, Any], symbol: str) -> None:
        """Handle order updates and set entry bar index."""
        super().on_order_update(order_responses, symbol)
        
        # Set entry bar index when position is opened
        if self.position.get(symbol) and not self.order_pending.get(symbol, False):
            if self.data is not None and not self.data.empty:
                self.entry_bar_index = len(self.data) - 1
                self.logger.info(f"{self.__class__.__name__}: Entry bar index set to {self.entry_bar_index}")

    def on_trade_update(self, trade: Dict[str, Any], symbol: str) -> None:
        """Handle trade updates and reset entry bar index."""
        super().on_trade_update(trade, symbol)
        
        # Reset entry bar index on trade close
        if trade.get('exit'):
            self.entry_bar_index = None
            self.logger.info(f"{self.__class__.__name__}: Entry bar index reset on trade close")

    def on_error(self, exception: Exception) -> None:
        """Handle strategy errors."""
        self.logger.error(f"Strategy {self.__class__.__name__} encountered an error: {exception}", exc_info=True) 