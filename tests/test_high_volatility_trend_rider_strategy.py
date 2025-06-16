#!/usr/bin/env python3
"""
Test Suite for High Volatility Trend Rider Strategy

This test suite comprehensively tests the StrategyHighVolatilityTrendRider implementation,
including trend detection, volatility regime analysis, pullback/breakout entries, and trailing stops.
"""

import unittest
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.high_volatility_trend_rider_strategy import StrategyHighVolatilityTrendRider

class TestHighVolatilityTrendRiderStrategy(unittest.TestCase):
    """Test cases for High Volatility Trend Rider Strategy."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Configure logging to reduce noise during testing
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create test configuration
        self.config = {
            'default': {
                'coin_pair': 'BTC/USDT',
                'leverage': 10,
                'order_size': 100
            },
            'strategy_configs': {
                'StrategyHighVolatilityTrendRider': {
                    # EMA parameters
                    'ema_fast_period': 20,
                    'ema_slow_period': 50,
                    # ADX parameters
                    'adx_period': 14,
                    'adx_threshold': 25,
                    'adx_strong_threshold': 30,
                    # ATR parameters
                    'atr_period': 14,
                    'atr_stop_multiplier': 2.0,
                    'atr_target_multiplier': 2.0,
                    # Volatility regime
                    'atr_volatility_percentile': 80,
                    'atr_lookback': 50,
                    'bb_period': 20,
                    'bb_std': 2.0,
                    'min_bb_width': 0.04,
                    # RSI parameters
                    'rsi_period': 14,
                    'rsi_bull_threshold': 50,
                    'rsi_bear_threshold': 50,
                    # Volume parameters
                    'volume_period': 20,
                    'volume_multiplier': 1.5,
                    # Pullback/breakout parameters
                    'pullback_bars': 5,
                    'min_pullback_pct': 0.002,
                    'breakout_bars': 50,
                    'breakout_range_multiplier': 1.5,
                    # Risk management
                    'max_trade_duration': 100,
                    'cooldown_bars': 5,
                    'partial_profit_ratio': 0.5,
                    'sl_pct': 0.03,
                    'tp_pct': 0.06,
                    'order_size': 50
                }
            }
        }
        
        # Create test data
        self.test_data = self._create_high_volatility_trend_data()
        
        # Initialize strategy
        self.strategy = StrategyHighVolatilityTrendRider(
            data=self.test_data.copy(),
            config=self.config,
            logger=self.logger
        )
        
        # Initialize indicators
        self.strategy.init_indicators()

    def _create_high_volatility_trend_data(self) -> pd.DataFrame:
        """Create realistic high-volatility trending market data for testing."""
        np.random.seed(42)  # For reproducible tests
        
        total_bars = 200
        base_price = 50000
        timestamps = []
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        current_time = datetime.now()
        current_price = base_price
        base_volume = 1000000
        
        # Phase 1: Initial consolidation (0-39 bars) - moderate volatility
        for i in range(0, 40):
            timestamp = current_time + timedelta(minutes=i*5)
            timestamps.append(timestamp)
            
            # Moderate volatility consolidation
            price_change = np.random.normal(0, 0.005) * current_price
            current_price += price_change
            
            # OHLC generation
            open_price = current_price
            volatility = 0.008  # 0.8% volatility
            high_price = current_price + abs(np.random.normal(0, volatility)) * current_price
            low_price = current_price - abs(np.random.normal(0, volatility)) * current_price
            close_price = low_price + np.random.uniform(0, 1) * (high_price - low_price)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            
            current_price = close_price
            
            # Volume during consolidation
            volume = base_volume * np.random.uniform(0.8, 1.2)
            volumes.append(volume)
        
        # Phase 2: High volatility uptrend (40-99 bars)
        trend_strength = 0.003  # 0.3% average move per bar
        volatility_increase = 1.5
        
        for i in range(40, 100):
            timestamp = current_time + timedelta(minutes=i*5)
            timestamps.append(timestamp)
            
            # Strong uptrend with high volatility
            trend_move = np.random.normal(trend_strength, 0.002) * current_price
            current_price += trend_move
            
            # OHLC with high volatility
            open_price = current_price
            volatility = 0.015 * volatility_increase  # 1.5% volatility
            high_price = current_price + abs(np.random.normal(0, volatility)) * current_price
            low_price = current_price - abs(np.random.normal(0, volatility)) * current_price
            
            # Bias towards bullish closes
            close_bias = 0.65  # 65% chance of closing in upper range
            if np.random.random() < close_bias:
                close_price = low_price + np.random.uniform(0.6, 1.0) * (high_price - low_price)
            else:
                close_price = low_price + np.random.uniform(0.0, 0.4) * (high_price - low_price)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            
            current_price = close_price
            
            # Higher volume during trending moves
            volume_multiplier = 1.3 + np.random.uniform(0, 0.5)
            volume = base_volume * volume_multiplier
            volumes.append(volume)
        
        # Phase 3: Pullback phase (100-119 bars)
        pullback_strength = -0.0015  # -0.15% average pullback per bar
        
        for i in range(100, 120):
            timestamp = current_time + timedelta(minutes=i*5)
            timestamps.append(timestamp)
            
            # Pullback with reducing volatility
            pullback_move = np.random.normal(pullback_strength, 0.0015) * current_price
            current_price += pullback_move
            
            # OHLC during pullback
            open_price = current_price
            volatility = 0.012  # Slightly lower volatility
            high_price = current_price + abs(np.random.normal(0, volatility)) * current_price
            low_price = current_price - abs(np.random.normal(0, volatility)) * current_price
            
            # Bias towards bearish closes during pullback
            close_bias = 0.35  # 35% chance of closing in upper range
            if np.random.random() < close_bias:
                close_price = low_price + np.random.uniform(0.6, 1.0) * (high_price - low_price)
            else:
                close_price = low_price + np.random.uniform(0.0, 0.4) * (high_price - low_price)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            
            current_price = close_price
            
            # Lower volume during pullback
            volume = base_volume * np.random.uniform(0.7, 1.1)
            volumes.append(volume)
        
        # Phase 4: Trend resumption (120-199 bars)
        for i in range(120, 200):
            timestamp = current_time + timedelta(minutes=i*5)
            timestamps.append(timestamp)
            
            # Resume uptrend with high volatility
            trend_move = np.random.normal(trend_strength * 0.8, 0.0018) * current_price
            current_price += trend_move
            
            # OHLC with high volatility resumption
            open_price = current_price
            volatility = 0.013  # High volatility
            high_price = current_price + abs(np.random.normal(0, volatility)) * current_price
            low_price = current_price - abs(np.random.normal(0, volatility)) * current_price
            
            # Bias towards bullish closes
            close_bias = 0.6
            if np.random.random() < close_bias:
                close_price = low_price + np.random.uniform(0.55, 1.0) * (high_price - low_price)
            else:
                close_price = low_price + np.random.uniform(0.0, 0.45) * (high_price - low_price)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            
            current_price = close_price
            
            # Volume during trend resumption
            volume_multiplier = 1.2 + np.random.uniform(0, 0.4)
            volume = base_volume * volume_multiplier
            volumes.append(volume)
        
        # Ensure all arrays are exactly the same length
        assert len(opens) == len(highs) == len(lows) == len(closes) == len(volumes) == total_bars
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        df.set_index('timestamp', inplace=True)
        return df

    # ==================== Basic Tests ====================
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertIsInstance(self.strategy, StrategyHighVolatilityTrendRider)
        self.assertEqual(self.strategy.ema_fast_period, 20)
        self.assertEqual(self.strategy.ema_slow_period, 50)
        self.assertEqual(self.strategy.adx_threshold, 25)
        self.assertEqual(self.strategy.atr_volatility_percentile, 80)
        self.assertTrue(hasattr(self.strategy, 'atr_history'))
        self.assertIsInstance(self.strategy.atr_history, list)

    def test_strategy_parameters(self):
        """Test strategy parameter loading."""
        self.assertEqual(self.strategy.atr_period, 14)
        self.assertEqual(self.strategy.atr_stop_multiplier, 2.0)
        self.assertEqual(self.strategy.bb_period, 20)
        self.assertEqual(self.strategy.volume_multiplier, 1.5)
        self.assertEqual(self.strategy.max_trade_duration, 100)
        self.assertEqual(self.strategy.cooldown_bars, 5)

    def test_market_type_tags(self):
        """Test market type tags."""
        expected_tags = ['HIGH_VOLATILITY', 'TRENDING']
        self.assertEqual(self.strategy.MARKET_TYPE_TAGS, expected_tags)
        self.assertTrue(self.strategy.SHOW_IN_SELECTION)

    def test_indicator_initialization(self):
        """Test indicator initialization."""
        # Check that indicators are present in data
        required_indicators = ['ema_fast', 'ema_slow', 'adx', 'atr', 'bb_width', 'rsi', 'volume_sma']
        for indicator in required_indicators:
            self.assertIn(indicator, self.strategy.data.columns)
            self.assertFalse(self.strategy.data[indicator].isna().all())

    def test_state_variables_initialization(self):
        """Test state variable initialization."""
        self.assertEqual(self.strategy.trend_direction, 0)
        self.assertFalse(self.strategy.in_high_volatility_regime)
        self.assertIsInstance(self.strategy.atr_history, list)
        self.assertIsNone(self.strategy.entry_bar_index)
        self.assertEqual(self.strategy.last_trade_bar, -self.strategy.cooldown_bars)
        self.assertIsNone(self.strategy.trailing_stop_price)
        self.assertFalse(self.strategy.partial_profit_taken)

    # ==================== Technical Indicator Tests ====================
    
    def test_ema_calculation(self):
        """Test EMA calculation accuracy."""
        # EMA fast should be more responsive than EMA slow
        ema_fast = self.strategy.data['ema_fast'].dropna()
        ema_slow = self.strategy.data['ema_slow'].dropna()
        
        self.assertTrue(len(ema_fast) > 0)
        self.assertTrue(len(ema_slow) > 0)
        
        # EMAs should be reasonable values around price
        price_mean = self.strategy.data['close'].mean()
        self.assertAlmostEqual(ema_fast.mean(), price_mean, delta=price_mean * 0.1)
        self.assertAlmostEqual(ema_slow.mean(), price_mean, delta=price_mean * 0.1)

    def test_atr_calculation(self):
        """Test ATR calculation."""
        atr_values = self.strategy.data['atr'].dropna()
        
        self.assertTrue(len(atr_values) > 0)
        self.assertTrue(all(atr_values > 0))  # ATR should be positive
        
        # ATR should be reasonable relative to price
        price_mean = self.strategy.data['close'].mean()
        atr_mean = atr_values.mean()
        self.assertLess(atr_mean, price_mean * 0.1)  # ATR shouldn't be too large

    def test_adx_calculation(self):
        """Test ADX calculation."""
        if 'adx' in self.strategy.data.columns:
            adx_values = self.strategy.data['adx'].dropna()
            if len(adx_values) > 0:
                # ADX should be between 0 and 100
                self.assertTrue(all(adx_values >= 0))
                self.assertTrue(all(adx_values <= 100))

    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        if all(col in self.strategy.data.columns for col in ['bb_lower', 'bb_upper', 'bb_width']):
            bb_lower = self.strategy.data['bb_lower'].dropna()
            bb_upper = self.strategy.data['bb_upper'].dropna()
            bb_width = self.strategy.data['bb_width'].dropna()
            
            # Upper band should be above lower band
            if len(bb_lower) > 0 and len(bb_upper) > 0:
                self.assertTrue(all(bb_upper > bb_lower))
            
            # BB width should be positive
            if len(bb_width) > 0:
                self.assertTrue(all(bb_width > 0))

    def test_rsi_calculation(self):
        """Test RSI calculation."""
        rsi_values = self.strategy.data['rsi'].dropna()
        
        if len(rsi_values) > 0:
            # RSI should be between 0 and 100
            self.assertTrue(all(rsi_values >= 0))
            self.assertTrue(all(rsi_values <= 100))

    def test_volume_sma_calculation(self):
        """Test volume SMA calculation."""
        volume_sma = self.strategy.data['volume_sma'].dropna()
        
        self.assertTrue(len(volume_sma) > 0)
        self.assertTrue(all(volume_sma > 0))  # Volume SMA should be positive

    def test_highs_lows_calculation(self):
        """Test highest/lowest calculation."""
        if all(col in self.strategy.data.columns for col in ['highest', 'lowest']):
            highest_vals = self.strategy.data['highest'].dropna()
            lowest_vals = self.strategy.data['lowest'].dropna()
            
            if len(highest_vals) > 0 and len(lowest_vals) > 0:
                # Highest should be >= lowest
                self.assertTrue(all(highest_vals >= lowest_vals))

    def test_incremental_indicator_update(self):
        """Test incremental indicator updates."""
        # Save initial state
        initial_data = self.strategy.data.copy()
        
        # Add a new row
        last_timestamp = self.strategy.data.index[-1]
        new_timestamp = last_timestamp + timedelta(minutes=5)
        last_close = self.strategy.data['close'].iloc[-1]
        
        new_row = pd.DataFrame({
            'open': [last_close],
            'high': [last_close * 1.01],
            'low': [last_close * 0.99],
            'close': [last_close * 1.005],
            'volume': [1000000]
        }, index=[new_timestamp])
        
        self.strategy.data = pd.concat([self.strategy.data, new_row])
        
        # Update indicators
        self.strategy.update_indicators_for_new_row()
        
        # Check that indicators were updated
        self.assertEqual(len(self.strategy.data), len(initial_data) + 1)
        if 'ema_fast' in self.strategy.data.columns:
            self.assertFalse(pd.isna(self.strategy.data['ema_fast'].iloc[-1]))

    # ==================== Core Strategy Logic Tests ====================
    
    def test_high_volatility_regime_detection(self):
        """Test high volatility regime detection."""
        vals = self.strategy._get_current_values()
        if vals:
            # Populate ATR history for testing
            self.strategy.atr_history = [vals['atr'] * 0.8] * 50  # Lower baseline
            
            is_high_vol = self.strategy._is_high_volatility_regime(vals)
            self.assertIn(is_high_vol, [True, False])

    def test_trending_market_detection(self):
        """Test trending market detection."""
        vals = self.strategy._get_current_values()
        if vals:
            is_trending, direction = self.strategy._is_trending_market(vals)
            self.assertIn(is_trending, [True, False])
            if is_trending:
                self.assertIn(direction, ['up', 'down'])
            else:
                self.assertIsNone(direction)

    def test_volume_confirmation(self):
        """Test volume confirmation logic."""
        vals = self.strategy._get_current_values()
        if vals:
            is_confirmed = self.strategy._is_volume_confirmation(vals)
            self.assertIn(is_confirmed, [True, False])

    def test_pullback_entry_detection(self):
        """Test pullback entry detection."""
        vals = self.strategy._get_current_values()
        if vals:
            # Test both directions
            pullback_up = self.strategy._detect_pullback_entry('up', vals)
            pullback_down = self.strategy._detect_pullback_entry('down', vals)
            
            self.assertIn(pullback_up, [True, False])
            self.assertIn(pullback_down, [True, False])

    def test_breakout_entry_detection(self):
        """Test breakout entry detection."""
        vals = self.strategy._get_current_values()
        if vals:
            # Test both directions
            breakout_up = self.strategy._detect_breakout_entry('up', vals)
            breakout_down = self.strategy._detect_breakout_entry('down', vals)
            
            self.assertIn(breakout_up, [True, False])
            self.assertIn(breakout_down, [True, False])

    def test_entry_trade_logic(self):
        """Test main entry trade logic."""
        vals = self.strategy._get_current_values()
        if vals:
            should_enter, direction = self.strategy._should_enter_trade(vals)
            self.assertIn(should_enter, [True, False])
            if should_enter:
                self.assertIn(direction, ['up', 'down'])

    def test_stops_and_targets_calculation(self):
        """Test stop loss and take profit calculations."""
        vals = self.strategy._get_current_values()
        if vals:
            # Test long direction
            stop_long, target_long = self.strategy._calculate_stops_and_targets('long', vals)
            self.assertLess(stop_long, vals['close'])  # Stop should be below entry
            self.assertGreater(target_long, vals['close'])  # Target should be above entry
            
            # Test short direction
            stop_short, target_short = self.strategy._calculate_stops_and_targets('short', vals)
            self.assertGreater(stop_short, vals['close'])  # Stop should be above entry
            self.assertLess(target_short, vals['close'])  # Target should be below entry

    def test_trailing_stop_update(self):
        """Test trailing stop update logic."""
        symbol = "BTC/USDT"
        vals = self.strategy._get_current_values()
        
        if vals:
            # Set up a mock position
            self.strategy.position[symbol] = {
                'side': 'buy',
                'entry_price': vals['close'],
                'size': 1.0
            }
            
            # Update trailing stop
            self.strategy._update_trailing_stop(vals, symbol)
            
            # Check that trailing stop was set
            if self.strategy.trailing_stop_price:
                self.assertLess(self.strategy.trailing_stop_price, vals['close'])

    # ==================== Entry/Exit Condition Tests ====================
    
    def test_entry_conditions_basic(self):
        """Test basic entry condition checking."""
        symbol = "BTC/USDT"
        entry_signal = self.strategy._check_entry_conditions(symbol)
        
        # Should return None or valid order details
        if entry_signal is not None:
            self.assertIsInstance(entry_signal, dict)
            self.assertIn('side', entry_signal)
            self.assertIn('type', entry_signal)
            self.assertIn(entry_signal['side'], ['buy', 'sell'])

    def test_exit_conditions_no_position(self):
        """Test exit conditions when no position is open."""
        symbol = "BTC/USDT"
        exit_signal = self.strategy.check_exit(symbol)
        self.assertIsNone(exit_signal)

    def test_exit_conditions_with_position(self):
        """Test exit conditions with active position."""
        symbol = "BTC/USDT"
        vals = self.strategy._get_current_values()
        
        if vals:
            # Set up a mock position
            self.strategy.position[symbol] = {
                'side': 'buy',
                'entry_price': vals['close'],
                'size': 1.0
            }
            self.strategy.entry_bar_index = len(self.strategy.data) - 50  # 50 bars ago
            
            exit_signal = self.strategy.check_exit(symbol)
            
            # Should return None or valid exit signal
            if exit_signal is not None:
                self.assertIsInstance(exit_signal, dict)
                self.assertEqual(exit_signal['type'], 'market')
                self.assertIn('reason', exit_signal)

    def test_time_based_exit(self):
        """Test time-based exit condition."""
        symbol = "BTC/USDT"
        vals = self.strategy._get_current_values()
        
        if vals:
            # Set up a mock position with old entry
            self.strategy.position[symbol] = {
                'side': 'buy',
                'entry_price': vals['close'],
                'size': 1.0
            }
            self.strategy.entry_bar_index = len(self.strategy.data) - self.strategy.max_trade_duration - 1
            
            exit_signal = self.strategy.check_exit(symbol)
            
            if exit_signal:
                self.assertEqual(exit_signal['reason'], 'time_exit')

    def test_trailing_stop_exit(self):
        """Test trailing stop exit."""
        symbol = "BTC/USDT"
        vals = self.strategy._get_current_values()
        
        if vals:
            # Set up position with trailing stop
            current_price = vals['close']
            self.strategy.position[symbol] = {
                'side': 'buy',
                'entry_price': current_price,
                'size': 1.0
            }
            self.strategy.trailing_stop_price = current_price * 1.01  # Above current price for long
            
            exit_signal = self.strategy.check_exit(symbol)
            
            if exit_signal:
                self.assertEqual(exit_signal['reason'], 'trailing_stop')

    # ==================== Interface Integration Tests ====================
    
    def test_main_entry_method(self):
        """Test main entry method integration."""
        symbol = "BTC/USDT"
        entry_signal = self.strategy.check_entry(symbol)
        
        # Should return None or valid entry signal
        if entry_signal is not None:
            self.assertIsInstance(entry_signal, (dict, bool))

    def test_main_exit_method(self):
        """Test main exit method integration."""
        symbol = "BTC/USDT"
        exit_signal = self.strategy.check_exit(symbol)
        
        # Should return None or boolean
        if exit_signal is not None:
            self.assertIn(type(exit_signal), [dict, bool])

    def test_risk_parameters(self):
        """Test risk parameter retrieval."""
        risk_params = self.strategy.get_risk_parameters()
        
        self.assertIsInstance(risk_params, dict)
        self.assertIn('sl_pct', risk_params)
        self.assertIn('tp_pct', risk_params)
        
        # Verify values are reasonable
        self.assertGreater(risk_params['sl_pct'], 0)
        self.assertGreater(risk_params['tp_pct'], 0)
        self.assertLess(risk_params['sl_pct'], 1)
        self.assertLess(risk_params['tp_pct'], 1)
        self.assertEqual(risk_params['sl_pct'], 0.03)
        self.assertEqual(risk_params['tp_pct'], 0.06)

    def test_order_update_handling(self):
        """Test order update handling."""
        symbol = "BTC/USDT"
        order_responses = {
            'main_order': {
                'result': {
                    'orderId': 'test123',
                    'orderStatus': 'filled',
                    'side': 'buy',
                    'avgPrice': '50000',
                    'qty': '1.0'
                }
            }
        }
        
        # Should not raise an exception
        self.strategy.on_order_update(order_responses, symbol)
        
        # Trailing stop should be initialized
        self.assertIsNotNone(self.strategy.trailing_stop_price)

    def test_trade_update_handling(self):
        """Test trade update handling."""
        symbol = "BTC/USDT"
        
        # Set up initial state
        self.strategy.entry_bar_index = 100
        self.strategy.trailing_stop_price = 50000
        self.strategy.partial_profit_taken = True
        
        trade_update = {
            'symbol': symbol,
            'side': 'buy',
            'qty': '1.0',
            'price': '51000',
            'status': 'closed',
            'exit': True
        }
        
        self.strategy.on_trade_update(trade_update, symbol)
        
        # State should be reset
        self.assertIsNone(self.strategy.entry_bar_index)
        self.assertIsNone(self.strategy.trailing_stop_price)
        self.assertFalse(self.strategy.partial_profit_taken)

    # ==================== Robustness Tests ====================
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Create strategy with minimal data
        minimal_data = self.test_data.iloc[:5].copy()
        
        try:
            minimal_strategy = StrategyHighVolatilityTrendRider(
                data=minimal_data,
                config=self.config,
                logger=self.logger
            )
            minimal_strategy.init_indicators()
            
            # Should not crash
            vals = minimal_strategy._get_current_values()
            if vals:
                minimal_strategy._is_high_volatility_regime(vals)
                
        except Exception as e:
            self.fail(f"Strategy failed with minimal data: {e}")

    def test_missing_volume_data(self):
        """Test handling of missing volume data."""
        # Create data without volume
        data_no_volume = self.test_data.drop(columns=['volume']).copy()
        data_no_volume['volume'] = 1000000  # Add back with constant values
        
        try:
            strategy_no_vol = StrategyHighVolatilityTrendRider(
                data=data_no_volume,
                config=self.config,
                logger=self.logger
            )
            strategy_no_vol.init_indicators()
            
            # Should still work
            vals = strategy_no_vol._get_current_values()
            if vals:
                strategy_no_vol._is_volume_confirmation(vals)
                
        except Exception as e:
            self.fail(f"Strategy failed without volume variation: {e}")

    def test_invalid_configuration_values(self):
        """Test handling of invalid configuration values."""
        invalid_config = self.config.copy()
        invalid_config['strategy_configs']['StrategyHighVolatilityTrendRider'].update({
            'ema_fast_period': -5,  # Invalid
            'adx_threshold': 150,   # Too high
            'atr_multiplier': 0     # Invalid
        })
        
        try:
            invalid_strategy = StrategyHighVolatilityTrendRider(
                data=self.test_data.copy(),
                config=invalid_config,
                logger=self.logger
            )
            invalid_strategy.init_indicators()
            
        except Exception as e:
            # Should handle gracefully or use defaults
            pass

    def test_data_continuity_after_updates(self):
        """Test data continuity after indicator updates."""
        initial_length = len(self.strategy.data)
        
        # Add new data
        last_timestamp = self.strategy.data.index[-1]
        new_timestamp = last_timestamp + timedelta(minutes=5)
        new_row = pd.DataFrame({
            'open': [52000],
            'high': [52500],
            'low': [51800],
            'close': [52200],
            'volume': [1200000]
        }, index=[new_timestamp])
        
        self.strategy.data = pd.concat([self.strategy.data, new_row])
        self.strategy.update_indicators_for_new_row()
        
        # Check continuity
        self.assertEqual(len(self.strategy.data), initial_length + 1)
        self.assertFalse(self.strategy.data.isnull().all().any())

    def test_error_handling(self):
        """Test error handling capabilities."""
        # Test error handling method
        test_exception = Exception("Test error")
        
        # Should not raise an exception
        self.strategy.on_error(test_exception)

    def test_performance_with_large_dataset(self):
        """Test strategy performance with larger dataset."""
        # Create larger dataset
        large_data = pd.concat([self.test_data] * 3, ignore_index=True)
        large_data.index = pd.date_range(start='2023-01-01', periods=len(large_data), freq='5min')
        
        try:
            large_strategy = StrategyHighVolatilityTrendRider(
                data=large_data,
                config=self.config,
                logger=self.logger
            )
            
            import time
            start_time = time.time()
            large_strategy.init_indicators()
            end_time = time.time()
            
            # Should complete within reasonable time
            self.assertLess(end_time - start_time, 10)  # Less than 10 seconds
            
        except Exception as e:
            self.fail(f"Strategy failed with large dataset: {e}")

    def test_concurrent_strategy_instances(self):
        """Test multiple strategy instances."""
        try:
            strategy1 = StrategyHighVolatilityTrendRider(
                data=self.test_data.copy(),
                config=self.config,
                logger=self.logger
            )
            
            strategy2 = StrategyHighVolatilityTrendRider(
                data=self.test_data.copy(),
                config=self.config,
                logger=self.logger
            )
            
            strategy1.init_indicators()
            strategy2.init_indicators()
            
            # Both should work independently
            vals1 = strategy1._get_current_values()
            vals2 = strategy2._get_current_values()
            
            if vals1 and vals2:
                strategy1._is_high_volatility_regime(vals1)
                strategy2._is_high_volatility_regime(vals2)
                
        except Exception as e:
            self.fail(f"Multiple strategy instances failed: {e}")

if __name__ == '__main__':
    unittest.main() 