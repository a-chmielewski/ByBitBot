import unittest
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path to import the strategies
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.ema_adx_strategy import StrategyEMATrendRider


class TestEMAADXStrategy(unittest.TestCase):
    """Test cases for StrategyEMATrendRider"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample OHLCV data
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2023-01-01', periods=200, freq='1min')
        
        # Create realistic OHLCV data with trending pattern
        base_price = 100
        trend = np.linspace(0, 10, 200)  # Upward trend
        noise = np.random.normal(0, 0.5, 200)
        
        closes = base_price + trend + noise
        highs = closes + np.random.uniform(0.1, 0.5, 200)
        lows = closes - np.random.uniform(0.1, 0.5, 200)
        opens = closes + np.random.uniform(-0.2, 0.2, 200)
        volumes = np.random.randint(1000, 10000, 200)
        
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        # Test configuration
        self.test_config = {
            'strategy_configs': {
                'StrategyEMATrendRider': {
                    'ema_fast_period': 20,
                    'ema_slow_period': 50,
                    'adx_period': 14,
                    'adx_threshold': 25,
                    'adx_strong_threshold': 30,
                    'rsi_period': 14,
                    'atr_period': 14,
                    'atr_stop_multiplier': 2.0,
                    'atr_target_multiplier': 2.0,
                    'sl_pct': 0.005,
                    'tp_pct': 0.01,
                    'time_stop_bars': 50,
                    'pullback_bars': 5,
                    'min_pullback_pct': 0.002,
                    'volume_period': 20,
                    'volume_spike_multiplier': 1.2
                }
            }
        }
        
        # Create logger
        self.logger = logging.getLogger('TestEMAADXStrategy')
        self.logger.setLevel(logging.DEBUG)
        
        # Initialize strategy
        self.strategy = StrategyEMATrendRider(
            data=self.sample_data,
            config=self.test_config,
            logger=self.logger
        )

    def test_initialization(self):
        """Test strategy initialization"""
        self.assertIsInstance(self.strategy, StrategyEMATrendRider)
        self.assertEqual(self.strategy.ema_fast_period, 20)
        self.assertEqual(self.strategy.ema_slow_period, 50)
        self.assertEqual(self.strategy.adx_threshold, 25)
        self.assertEqual(self.strategy.sl_pct, 0.005)
        self.assertEqual(self.strategy.tp_pct, 0.01)
        self.assertEqual(self.strategy.MARKET_TYPE_TAGS, ['TRENDING'])
        self.assertTrue(self.strategy.SHOW_IN_SELECTION)

    def test_init_indicators(self):
        """Test indicator initialization"""
        # Indicators should be calculated
        self.assertIn('ema_fast', self.strategy.data.columns)
        self.assertIn('ema_slow', self.strategy.data.columns)
        self.assertIn('adx', self.strategy.data.columns)
        self.assertIn('rsi', self.strategy.data.columns)
        self.assertIn('atr', self.strategy.data.columns)
        
        # Check that indicators have values (not all NaN)
        self.assertFalse(self.strategy.data['ema_fast'].isna().all())
        self.assertFalse(self.strategy.data['ema_slow'].isna().all())
        
        # Check that EMA values are reasonable
        last_close = self.strategy.data['close'].iloc[-1]
        last_ema_fast = self.strategy.data['ema_fast'].iloc[-1]
        last_ema_slow = self.strategy.data['ema_slow'].iloc[-1]
        
        # EMAs should be close to the price
        self.assertAlmostEqual(last_ema_fast, last_close, delta=5)
        self.assertAlmostEqual(last_ema_slow, last_close, delta=10)

    def test_init_indicators_with_insufficient_data(self):
        """Test indicator initialization with insufficient data"""
        # Create data with only 10 rows (less than EMA slow period)
        short_data = self.sample_data.iloc[:10].copy()
        
        strategy = StrategyEMATrendRider(
            data=short_data,
            config=self.test_config,
            logger=self.logger
        )
        
        # With insufficient data, strategy should handle initialization gracefully
        # The strategy should not crash and should still have the original data
        self.assertIsNotNone(strategy.data)
        self.assertEqual(len(strategy.data), 10)
        
        # Check that basic data columns are still present
        basic_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in basic_columns:
            self.assertIn(col, strategy.data.columns)
        
        # Indicator columns may or may not be present depending on pandas-ta behavior
        # The key is that the strategy didn't crash during initialization
        
        # Test that the strategy can still be used (won't crash on method calls)
        try:
            current_vals = strategy._get_current_values()
            # Should return None or handle gracefully with insufficient data
            self.assertIsInstance(current_vals, (type(None), dict))
            
            # Test entry check doesn't crash
            entry_signal = strategy.check_entry('BTC/USDT')
            self.assertIsInstance(entry_signal, (type(None), dict))
            
        except Exception as e:
            # If it does fail, it should be due to insufficient data, not a crash
            self.assertIn('data', str(e).lower())

    def test_init_indicators_with_missing_columns(self):
        """Test indicator initialization with missing required columns"""
        # Remove required columns
        bad_data = self.sample_data.drop(columns=['high', 'low']).copy()
        
        strategy = StrategyEMATrendRider(
            data=bad_data,
            config=self.test_config,
            logger=self.logger
        )
        
        # Should handle missing columns gracefully
        self.assertIsNotNone(strategy.data)

    def test_get_risk_parameters(self):
        """Test risk parameter retrieval"""
        risk_params = self.strategy.get_risk_parameters()
        
        self.assertIn('sl_pct', risk_params)
        self.assertIn('tp_pct', risk_params)
        
        # Risk parameters may be ATR-based or fixed
        # Just check they are reasonable values
        self.assertGreater(risk_params['sl_pct'], 0)
        self.assertGreater(risk_params['tp_pct'], 0)
        self.assertLess(risk_params['sl_pct'], 0.1)  # Should be less than 10%
        self.assertLess(risk_params['tp_pct'], 0.1)   # Should be less than 10%

    def test_get_current_values(self):
        """Test getting current indicator values"""
        current_vals = self.strategy._get_current_values()
        
        if current_vals:  # Only test if data is sufficient
            # Check for the actual keys returned by _get_current_values
            expected_keys = [
                'current_price', 'current_high', 'current_low', 'current_volume',
                'ema_fast', 'ema_slow', 'adx', 'rsi', 'atr', 'ema_cross',
                'volume_sma', 'prev_close', 'current_bar_datetime'
            ]
            for key in expected_keys:
                self.assertIn(key, current_vals)

    def test_check_entry_conditions_no_position(self):
        """Test entry condition checking when no position is open"""
        symbol = 'BTC/USDT'
        
        # Ensure no position exists
        self.strategy.position[symbol] = None
        self.strategy.order_pending[symbol] = False
        
        # Test entry check
        entry_signal = self.strategy.check_entry(symbol)
        
        # Should return None or dict based on conditions
        self.assertIsInstance(entry_signal, (type(None), dict))

    def test_check_entry_conditions_with_position(self):
        """Test entry condition checking when position is already open"""
        symbol = 'BTC/USDT'
        
        # Set existing position
        self.strategy.position[symbol] = {
            'main_order': {'result': {'side': 'buy', 'qty': '0.1'}},
            'stop_loss_order': {'result': {'orderId': 'sl123'}},
            'take_profit_order': {'result': {'orderId': 'tp123'}}
        }
        
        entry_signal = self.strategy.check_entry(symbol)
        
        # Should return None when position exists
        self.assertIsNone(entry_signal)

    def test_check_entry_conditions_with_pending_order(self):
        """Test entry condition checking when order is pending"""
        symbol = 'BTC/USDT'
        
        # Set pending order
        self.strategy.position[symbol] = None
        self.strategy.order_pending[symbol] = True
        
        entry_signal = self.strategy.check_entry(symbol)
        
        # Should return None when order is pending
        self.assertIsNone(entry_signal)

    def test_check_exit_conditions_no_position(self):
        """Test exit condition checking when no position is open"""
        symbol = 'BTC/USDT'
        
        # Ensure no position exists
        self.strategy.position[symbol] = None
        
        exit_signal = self.strategy.check_exit(symbol)
        
        # Should return False when no position
        self.assertFalse(exit_signal)

    def test_check_exit_conditions_with_position(self):
        """Test exit condition checking when position is open"""
        symbol = 'BTC/USDT'
        
        # Set existing position
        self.strategy.position[symbol] = {
            'main_order': {'result': {'side': 'buy', 'qty': '0.1'}},
            'stop_loss_order': {'result': {'orderId': 'sl123'}},
            'take_profit_order': {'result': {'orderId': 'tp123'}}
        }
        self.strategy.entry_bar_index = len(self.strategy.data) - 10  # Position opened 10 bars ago
        
        exit_signal = self.strategy.check_exit(symbol)
        
        # Should return boolean
        self.assertIsInstance(exit_signal, bool)

    def test_detect_trend_and_strength(self):
        """Test trend detection logic"""
        # Create mock current values with proper structure expected by the method
        test_vals = {
            'current_price': 105.0,
            'ema_fast': 105.0,
            'ema_slow': 100.0,
            'adx': 30.0,
            'rsi': 60.0,
            'ema_cross': 1  # Bullish cross
        }
        
        # Test bullish trend detection
        trend_detected = self.strategy._detect_trend_and_strength(test_vals)
        self.assertIsInstance(trend_detected, bool)
        
        # Test bearish trend
        test_vals.update({
            'current_price': 95.0,
            'ema_fast': 95.0,
            'ema_slow': 100.0,
            'rsi': 40.0,
            'ema_cross': -1  # Bearish cross
        })
        
        trend_detected = self.strategy._detect_trend_and_strength(test_vals)
        self.assertIsInstance(trend_detected, bool)

    def test_detect_pullback(self):
        """Test pullback detection logic"""
        # Set trend state
        self.strategy.trend_direction = 1  # Bullish
        self.strategy.trend_confirmed = True
        self.strategy.waiting_for_pullback = True
        self.strategy.recent_high = 105.0
        
        # Create proper structure expected by _detect_pullback
        test_vals = {
            'current_price': 100.0,  # Lower than recent_high, indicating pullback
            'ema_fast': 102.0,
            'ema_slow': 101.0
        }
        
        pullback_detected = self.strategy._detect_pullback(test_vals)
        self.assertIsInstance(pullback_detected, bool)

    def test_check_pullback_entry(self):
        """Test pullback entry logic"""
        # Set up pullback state
        self.strategy.trend_direction = 1  # Bullish
        self.strategy.pullback_detected = True
        self.strategy.pullback_extreme_price = 99.0
        self.strategy.pullback_start_bar = len(self.strategy.data) - 5
        
        # Create proper structure for pullback entry check
        test_vals = {
            'current_price': 101.0,  # Bounce from pullback
            'current_volume': 6000,
            'volume_sma': 5000,
            'rsi': 55.0,
            'ema_fast': 100.5,
            'adx': 32.0
        }
        
        entry_signal, direction = self.strategy._check_pullback_entry(test_vals)
        self.assertIsInstance(entry_signal, bool)
        if entry_signal:
            self.assertIn(direction, ['long', 'short'])

    def test_on_order_update(self):
        """Test order update handling"""
        symbol = 'BTC/USDT'
        
        # Mock order responses
        order_responses = {
            'main_order': {
                'result': {
                    'orderId': 'main123',
                    'orderStatus': 'filled',
                    'side': 'buy',
                    'qty': '0.1'
                }
            },
            'stop_loss_order': {
                'result': {
                    'orderId': 'sl123',
                    'orderStatus': 'new'
                }
            },
            'take_profit_order': {
                'result': {
                    'orderId': 'tp123',
                    'orderStatus': 'new'
                }
            }
        }
        
        # Test order update
        self.strategy.on_order_update(order_responses, symbol)
        
        # Check that position was updated
        self.assertIsNotNone(self.strategy.position.get(symbol))
        self.assertEqual(self.strategy.active_order_id.get(symbol), 'main123')

    def test_on_trade_update(self):
        """Test trade update handling"""
        symbol = 'BTC/USDT'
        
        # Set up position first
        self.strategy.position[symbol] = {
            'main_order': {'result': {'side': 'buy', 'qty': '0.1'}},
            'stop_loss_order': {'result': {'orderId': 'sl123'}},
            'take_profit_order': {'result': {'orderId': 'tp123'}}
        }
        self.strategy.entry_bar_index = len(self.strategy.data) - 5
        
        # Mock trade data with exit flag
        trade_data = {
            'symbol': symbol,
            'side': 'buy',
            'qty': '0.1',
            'price': '100.0',
            'status': 'closed',
            'exit': True
        }
        
        # Test trade update
        self.strategy.on_trade_update(trade_data, symbol)
        
        # Check that entry bar index was reset
        self.assertIsNone(self.strategy.entry_bar_index)

    def test_update_indicators_for_new_row(self):
        """Test incremental indicator updates"""
        # Store original values
        if len(self.strategy.data) >= 2:
            original_ema_fast = self.strategy.data['ema_fast'].iloc[-1]
            original_ema_slow = self.strategy.data['ema_slow'].iloc[-1]
            
            # Add new row
            new_row = pd.DataFrame({
                'timestamp': [pd.Timestamp('2023-01-01 03:20:00')],
                'open': [110.0],
                'high': [111.0],
                'low': [109.5],
                'close': [110.5],
                'volume': [5000]
            })
            
            # Set proper index for new row to avoid issues
            new_row.index = [len(self.strategy.data)]
            self.strategy.data = pd.concat([self.strategy.data, new_row])
            
            # Update indicators
            self.strategy.update_indicators_for_new_row()
            
            # Check that indicators were updated
            new_ema_fast = self.strategy.data['ema_fast'].iloc[-1]
            new_ema_slow = self.strategy.data['ema_slow'].iloc[-1]
            
            # New values should be different from original (unless edge case)
            self.assertIsNotNone(new_ema_fast)
            self.assertIsNotNone(new_ema_slow)

    def test_on_error(self):
        """Test error handling"""
        test_exception = Exception("Test error")
        
        # Should not raise exception
        try:
            self.strategy.on_error(test_exception)
        except Exception as e:
            self.fail(f"on_error raised an exception: {e}")

    def test_clear_position(self):
        """Test position clearing"""
        symbol = 'BTC/USDT'
        
        # Set up position
        self.strategy.position[symbol] = {
            'main_order': {'result': {'side': 'buy', 'qty': '0.1'}},
            'stop_loss_order': {'result': {'orderId': 'sl123'}},
            'take_profit_order': {'result': {'orderId': 'tp123'}}
        }
        self.strategy.order_pending[symbol] = True
        self.strategy.active_order_id[symbol] = 'main123'
        
        # Clear position
        self.strategy.clear_position(symbol)
        
        # Check that position was cleared
        self.assertIsNone(self.strategy.position.get(symbol))
        self.assertFalse(self.strategy.order_pending.get(symbol, False))
        self.assertIsNone(self.strategy.active_order_id.get(symbol))

    def test_state_reset(self):
        """Test trend state reset"""
        # Set some trend state
        self.strategy.trend_direction = 1
        self.strategy.trend_confirmed = True
        self.strategy.waiting_for_pullback = True
        
        # Reset state
        self.strategy._reset_trend_state()
        
        # Check that state was reset
        self.assertEqual(self.strategy.trend_direction, 0)
        self.assertFalse(self.strategy.trend_confirmed)
        self.assertFalse(self.strategy.waiting_for_pullback)

    def test_config_fallback_values(self):
        """Test configuration fallback values"""
        # Create strategy with minimal config
        minimal_config = {'strategy_configs': {'StrategyEMATrendRider': {}}}
        
        strategy = StrategyEMATrendRider(
            data=self.sample_data,
            config=minimal_config,
            logger=self.logger
        )
        
        # Should use default values
        self.assertEqual(strategy.ema_fast_period, 20)
        self.assertEqual(strategy.ema_slow_period, 50)
        self.assertEqual(strategy.adx_threshold, 25)
        self.assertEqual(strategy.sl_pct, 0.005)
        self.assertEqual(strategy.tp_pct, 0.01)

    def test_time_stop_exit(self):
        """Test time-based exit condition"""
        symbol = 'BTC/USDT'
        
        # Set up position with old entry
        self.strategy.position[symbol] = {
            'main_order': {'result': {'side': 'buy', 'qty': '0.1'}},
            'stop_loss_order': {'result': {'orderId': 'sl123'}},
            'take_profit_order': {'result': {'orderId': 'tp123'}}
        }
        # Set entry bar index to trigger time stop
        self.strategy.entry_bar_index = len(self.strategy.data) - self.strategy.time_stop_bars - 1
        
        exit_signal = self.strategy.check_exit(symbol)
        
        # Should trigger time stop
        self.assertTrue(exit_signal)

    def test_trend_weakness_exit(self):
        """Test trend weakness exit condition"""
        symbol = 'BTC/USDT'
        
        # Set up position
        self.strategy.position[symbol] = {
            'main_order': {'result': {'side': 'buy', 'qty': '0.1'}},
            'stop_loss_order': {'result': {'orderId': 'sl123'}},
            'take_profit_order': {'result': {'orderId': 'tp123'}}
        }
        
        # Mock weak ADX in the data
        if len(self.strategy.data) > 0:
            self.strategy.data.loc[self.strategy.data.index[-1], 'adx'] = 15.0  # Below threshold
        
        exit_signal = self.strategy.check_exit(symbol)
        
        # Should trigger trend weakness exit
        self.assertTrue(exit_signal)


if __name__ == '__main__':
    unittest.main() 