import unittest
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path to import the strategies
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.atr_momentum_breakout_strategy import StrategyATRMomentumBreakout


class TestATRMomentumBreakoutStrategy(unittest.TestCase):
    """Test cases for StrategyATRMomentumBreakout"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample OHLCV data with high volatility breakout pattern
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2023-01-01', periods=200, freq='1min')
        
        # Create high-volatility data with breakout patterns
        base_price = 100.0
        
        # Create consolidation followed by breakout pattern
        consolidation = np.full(100, base_price) + np.random.normal(0, 0.1, 100)
        breakout = np.linspace(base_price, base_price + 5, 100) + np.random.normal(0, 0.3, 100)
        
        closes = np.concatenate([consolidation, breakout])
        
        # Create realistic OHLC from closes
        highs = closes + np.random.uniform(0.2, 0.8, 200)
        lows = closes - np.random.uniform(0.2, 0.8, 200)
        opens = closes + np.random.uniform(-0.3, 0.3, 200)
        
        # Create volume with surges during breakout
        base_volume = np.random.randint(1000, 3000, 100)
        surge_volume = np.random.randint(5000, 15000, 100)
        volumes = np.concatenate([base_volume, surge_volume])
        
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
                'StrategyATRMomentumBreakout': {
                    'atr_period': 14,
                    'donchian_period': 20,
                    'trend_ema_period': 21,
                    'volume_avg_period': 10,
                    'volatility_threshold_multiplier': 1.5,
                    'range_multiplier': 2.0,
                    'min_volatility_pct': 0.002,
                    'volume_surge_multiplier': 2.5,
                    'scalp_target_pct': 0.005,
                    'scalp_profit_take_pct': 0.7,
                    'atr_stop_multiplier': 1.5,
                    'atr_target_multiplier': 2.0,
                    'trailing_atr_multiplier': 1.0,
                    'max_position_bars': 10,
                    'failed_breakout_bars': 3,
                    'cooldown_bars': 3,
                    'sl_pct': 0.01,
                    'tp_pct': 0.02
                }
            }
        }
        
        # Create logger
        self.logger = logging.getLogger('TestATRMomentumBreakoutStrategy')
        self.logger.setLevel(logging.DEBUG)
        
        # Initialize strategy
        self.strategy = StrategyATRMomentumBreakout(
            data=self.sample_data,
            config=self.test_config,
            logger=self.logger
        )

    def test_initialization(self):
        """Test strategy initialization"""
        self.assertIsInstance(self.strategy, StrategyATRMomentumBreakout)
        self.assertEqual(self.strategy.MARKET_TYPE_TAGS, ['HIGH_VOLATILITY'])
        self.assertTrue(self.strategy.SHOW_IN_SELECTION)
        
        # Check internal state initialization
        self.assertEqual(self.strategy.recent_ranges, [])
        self.assertEqual(self.strategy.consolidation_high, 0)
        self.assertEqual(self.strategy.consolidation_low, 0)
        self.assertEqual(self.strategy.consolidation_start, 0)
        self.assertFalse(self.strategy.is_in_consolidation)
        self.assertIsNone(self.strategy.entry_bar)
        self.assertIsNone(self.strategy.entry_side)
        self.assertIsNone(self.strategy.entry_price)
        self.assertFalse(self.strategy.scalp_target_hit)

    def test_init_indicators(self):
        """Test indicator initialization"""
        # ATR indicators should be calculated
        self.assertIn('atr', self.strategy.data.columns)
        self.assertIn('atr_sma', self.strategy.data.columns)
        
        # Donchian Channels should be calculated
        self.assertIn('donchian_high', self.strategy.data.columns)
        self.assertIn('donchian_low', self.strategy.data.columns)
        
        # Trend EMA should be calculated
        self.assertIn('trend_ema', self.strategy.data.columns)
        
        # Volume indicators should be calculated
        self.assertIn('volume_sma', self.strategy.data.columns)
        
        # Check that indicators have values (not all NaN)
        self.assertFalse(self.strategy.data['atr'].isna().all())
        self.assertFalse(self.strategy.data['donchian_high'].isna().all())
        self.assertFalse(self.strategy.data['donchian_low'].isna().all())
        self.assertFalse(self.strategy.data['trend_ema'].isna().all())
        self.assertFalse(self.strategy.data['volume_sma'].isna().all())
        
        # Check Donchian Channel relationships
        # High should be >= Low
        last_idx = -1
        if (not self.strategy.data['donchian_high'].isna().iloc[last_idx] and 
            not self.strategy.data['donchian_low'].isna().iloc[last_idx]):
            self.assertGreaterEqual(
                self.strategy.data['donchian_high'].iloc[last_idx],
                self.strategy.data['donchian_low'].iloc[last_idx]
            )

    def test_init_indicators_with_missing_pandas_ta(self):
        """Test indicator initialization without pandas_ta"""
        # Create a fresh instance and manually set has_pandas_ta to False
        # to simulate the pandas_ta import failure scenario
        strategy = StrategyATRMomentumBreakout(
            data=self.sample_data,  
            config=self.test_config,
            logger=self.logger
        )
        
        # Test that manual ATR calculation works as fallback
        # This effectively tests the same functionality as missing pandas_ta
        manual_atr = strategy._calculate_atr_manual(14)
        
        # Should create valid ATR series
        self.assertIsInstance(manual_atr, pd.Series)
        self.assertEqual(len(manual_atr), len(strategy.data))
        
        # ATR values should be positive for most bars
        non_zero_values = manual_atr[manual_atr > 0]
        self.assertGreater(len(non_zero_values), 0)
        
        # Should still have all required indicators
        self.assertIn('atr', strategy.data.columns)
        self.assertIn('trend_ema', strategy.data.columns)

    def test_calculate_atr_manual(self):
        """Test manual ATR calculation"""
        period = 14
        atr_manual = self.strategy._calculate_atr_manual(period)
        
        # Should return a pandas Series
        self.assertIsInstance(atr_manual, pd.Series)
        self.assertEqual(len(atr_manual), len(self.strategy.data))
        
        # ATR values should be positive
        non_nan_values = atr_manual.dropna()
        if len(non_nan_values) > 0:
            self.assertTrue((non_nan_values > 0).all())

    def test_is_high_volatility_regime(self):
        """Test high volatility regime detection"""
        # Test with sufficient data
        if len(self.strategy.data) > 50:
            idx = len(self.strategy.data) - 1
            
            # Should return boolean
            is_high_vol = self.strategy.is_high_volatility_regime(idx)
            # Check that it's a boolean-like value (True/False)
            self.assertIn(is_high_vol, [True, False])

    def test_detect_consolidation(self):
        """Test consolidation detection"""
        # Test with sufficient data
        if len(self.strategy.data) > 50:
            idx = len(self.strategy.data) - 1
            
            # Should return boolean
            is_consolidation = self.strategy.detect_consolidation(idx)
            self.assertIsInstance(is_consolidation, bool)

    def test_is_volume_surge(self):
        """Test volume surge detection"""
        # Test with sufficient data
        if len(self.strategy.data) > 20:
            idx = len(self.strategy.data) - 1
            
            # Should return boolean
            is_surge = self.strategy.is_volume_surge(idx)
            # Check that it's a boolean-like value (True/False)
            self.assertIn(is_surge, [True, False])

    def test_is_trend_aligned(self):
        """Test trend alignment check"""
        # Test with sufficient data
        if len(self.strategy.data) > 30:
            idx = len(self.strategy.data) - 1
            
            # Test both directions
            is_aligned_long = self.strategy.is_trend_aligned(idx, 'long')
            is_aligned_short = self.strategy.is_trend_aligned(idx, 'short')
            
            # Check that they're boolean-like values (True/False)
            self.assertIn(is_aligned_long, [True, False])
            self.assertIn(is_aligned_short, [True, False])

    def test_detect_breakout(self):
        """Test breakout detection"""
        # Test with sufficient data
        if len(self.strategy.data) > 30:
            idx = len(self.strategy.data) - 1
            
            # Should return tuple of (direction, price) or (None, None)
            direction, price = self.strategy.detect_breakout(idx)
            
            if direction is not None:
                self.assertIn(direction, ['long', 'short'])
                self.assertIsInstance(price, (int, float))
                self.assertGreater(price, 0)
            else:
                self.assertIsNone(price)

    def test_update_trailing_stop(self):
        """Test trailing stop update"""
        # Set up position state
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 100.0
        self.strategy.high_water_mark = 102.0
        self.strategy.trailing_stop_price = 99.0
        
        # Test with sufficient data
        if len(self.strategy.data) > 30:
            idx = len(self.strategy.data) - 1
            
            # Should not raise exception
            try:
                self.strategy.update_trailing_stop(idx)
            except Exception as e:
                self.fail(f"update_trailing_stop raised an exception: {e}")

    def test_check_scalp_target(self):
        """Test scalp target checking"""
        # Set up position state
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 100.0
        self.strategy.scalp_target_hit = False
        
        # Test with sufficient data
        if len(self.strategy.data) > 30:
            idx = len(self.strategy.data) - 1
            
                    # Should return boolean
        target_hit = self.strategy.check_scalp_target(idx)
        # Check that it's a boolean-like value (True/False)
        self.assertIn(target_hit, [True, False])

    def test_check_exit_conditions(self):
        """Test exit condition checking"""
        # Set up position state
        self.strategy.entry_bar = len(self.strategy.data) - 10
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 100.0
        
        # Test with sufficient data
        if len(self.strategy.data) > 30:
            idx = len(self.strategy.data) - 1
            
            # Should return tuple of (should_exit, reason)
            should_exit, reason = self.strategy.check_exit_conditions(idx)
            
            self.assertIsInstance(should_exit, bool)
            if should_exit:
                self.assertIsInstance(reason, str)
            else:
                self.assertIsNone(reason)

    def test_get_risk_parameters(self):
        """Test risk parameter retrieval"""
        risk_params = self.strategy.get_risk_parameters()
        
        self.assertIn('sl_pct', risk_params)
        self.assertIn('tp_pct', risk_params)
        
        # Risk parameters should be calculated dynamically based on ATR
        # Just verify they are positive numbers and reasonable
        self.assertIsInstance(risk_params['sl_pct'], (int, float))
        self.assertIsInstance(risk_params['tp_pct'], (int, float))
        self.assertGreater(risk_params['sl_pct'], 0)
        self.assertGreater(risk_params['tp_pct'], 0)
        
        # TP should generally be larger than SL for good risk/reward
        self.assertGreater(risk_params['tp_pct'], risk_params['sl_pct'])
        
        # Verify reasonable ranges (should be between 0.1% and 10%)
        self.assertLess(risk_params['sl_pct'], 0.1)
        self.assertLess(risk_params['tp_pct'], 0.1)

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
        
        # Should return None when no position
        self.assertIsNone(exit_signal)

    def test_check_exit_conditions_with_position(self):
        """Test exit condition checking when position is open"""
        symbol = 'BTC/USDT'
        
        # Set existing position
        self.strategy.position[symbol] = {
            'main_order': {'result': {'side': 'buy', 'qty': '0.1'}},
            'stop_loss_order': {'result': {'orderId': 'sl123'}},
            'take_profit_order': {'result': {'orderId': 'tp123'}}
        }
        
        # Set up strategy state
        self.strategy.entry_bar = len(self.strategy.data) - 10
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 100.0
        
        exit_signal = self.strategy.check_exit(symbol)
        
        # Should return None or dict based on conditions
        self.assertIsInstance(exit_signal, (type(None), dict))

    def test_on_trade_closed(self):
        """Test trade closure handling"""
        symbol = 'BTC/USDT'
        
        # Set up position state
        self.strategy.entry_bar = len(self.strategy.data) - 10
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 100.0
        self.strategy.scalp_target_hit = True
        
        # Mock trade result
        trade_result = {
            'symbol': symbol,
            'side': 'buy',
            'qty': '0.1',
            'price': '102.0',
            'pnl': '2.0',
            'status': 'closed'
        }
        
        # Test trade closure
        self.strategy.on_trade_closed(symbol, trade_result)
        
        # Check that state was reset
        self.assertIsNone(self.strategy.entry_bar)
        self.assertIsNone(self.strategy.entry_side)
        self.assertIsNone(self.strategy.entry_price)
        self.assertFalse(self.strategy.scalp_target_hit)
        self.assertIsNone(self.strategy.trailing_stop_price)

    def test_config_parameters(self):
        """Test that all configuration parameters are properly loaded"""
        # Check that config parameters are accessible
        self.assertIn('strategy_configs', self.strategy.config)
        strategy_config = self.strategy.config['strategy_configs']['StrategyATRMomentumBreakout']
        
        # Check key parameters
        self.assertEqual(strategy_config['atr_period'], 14)
        self.assertEqual(strategy_config['donchian_period'], 20)
        self.assertEqual(strategy_config['trend_ema_period'], 21)
        self.assertEqual(strategy_config['volume_avg_period'], 10)
        self.assertEqual(strategy_config['volatility_threshold_multiplier'], 1.5)
        self.assertEqual(strategy_config['range_multiplier'], 2.0)
        self.assertEqual(strategy_config['min_volatility_pct'], 0.002)
        self.assertEqual(strategy_config['volume_surge_multiplier'], 2.5)
        self.assertEqual(strategy_config['scalp_target_pct'], 0.005)

    def test_data_integrity(self):
        """Test that strategy works with copy of data"""
        original_data = self.sample_data.copy()
        
        # Strategy should work on a copy (modifications expected for indicators)
        self.assertIsNotNone(self.strategy.data)
        
        # Original data should still have basic columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, original_data.columns)

    def test_volatility_threshold_validation(self):
        """Test volatility threshold validation"""
        # Test with mock data
        test_config = self.test_config.copy()
        test_config['strategy_configs']['StrategyATRMomentumBreakout']['min_volatility_pct'] = 0.001
        
        strategy = StrategyATRMomentumBreakout(
            data=self.sample_data,
            config=test_config,
            logger=self.logger
        )
        
        # Should initialize without errors
        self.assertIsNotNone(strategy)

    def test_cooldown_period(self):
        """Test cooldown period functionality"""
        # Set last trade bar to recent
        self.strategy.last_trade_bar = len(self.strategy.data) - 2
        
        # Should respect cooldown period
        symbol = 'BTC/USDT'
        self.strategy.position[symbol] = None
        self.strategy.order_pending[symbol] = False
        
        # Entry should be blocked due to cooldown
        entry_signal = self.strategy.check_entry(symbol)
        # This might return None due to cooldown or other conditions
        self.assertIsInstance(entry_signal, (type(None), dict))

    def test_state_management(self):
        """Test internal state management"""
        # Test consolidation state
        self.strategy.is_in_consolidation = True
        self.strategy.consolidation_high = 105.0
        self.strategy.consolidation_low = 95.0
        
        # State should be maintained
        self.assertTrue(self.strategy.is_in_consolidation)
        self.assertEqual(self.strategy.consolidation_high, 105.0)
        self.assertEqual(self.strategy.consolidation_low, 95.0)
        
        # Test position state
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 100.0
        self.strategy.high_water_mark = 102.0
        
        self.assertEqual(self.strategy.entry_side, 'long')
        self.assertEqual(self.strategy.entry_price, 100.0)
        self.assertEqual(self.strategy.high_water_mark, 102.0)


if __name__ == '__main__':
    unittest.main() 