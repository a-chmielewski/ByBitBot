#!/usr/bin/env python3
"""
Unit tests for enhanced StrategyTemplate with risk management hooks
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.strategy_template import StrategyTemplate
import pandas as pd
import numpy as np
import logging

class TestEnhancedStrategyTemplate(unittest.TestCase):
    """Test cases for enhanced StrategyTemplate functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = self._create_test_data()
        self.enhanced_config = {
            'strategy_configs': {
                'TestStrategy': {
                    'risk_management': {
                        'stop_loss_mode': 'atr_mult',
                        'stop_loss_atr_multiplier': 2.0,
                        'take_profit_mode': 'progressive_levels',
                        'take_profit_progressive_levels': [0.02, 0.04, 0.06],
                        'trailing_stop_enabled': True,
                        'position_sizing_mode': 'vol_normalized',
                        'position_risk_per_trade': 0.01,
                        'leverage_by_regime': {'low': 1.2, 'normal': 1.0, 'high': 0.8}
                    }
                }
            }
        }
        self.minimal_config = {}
        
    def _create_test_data(self, periods=200):
        """Create test OHLCV data"""
        np.random.seed(42)
        base_price = 50000.0
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='1min')
        returns = np.random.normal(0, 0.01, periods)
        prices = base_price * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(100, 1000, periods)
        })
    
    def test_enhanced_configuration_loading(self):
        """Test that enhanced configuration is loaded correctly"""
        class TestStrategy(StrategyTemplate):
            def init_indicators(self): pass
            def _check_entry_conditions(self, symbol): return None
            def check_exit(self, symbol): return None
        
        strategy = TestStrategy(self.test_data, self.enhanced_config)
        
        # Test configuration loading
        self.assertEqual(strategy.stop_loss_mode, 'atr_mult')
        self.assertEqual(strategy.take_profit_mode, 'progressive_levels')
        self.assertTrue(strategy.trailing_stop_enabled)
        self.assertEqual(strategy.position_sizing_mode, 'vol_normalized')
        self.assertEqual(strategy.leverage_by_regime['normal'], 1.0)
    
    def test_position_sizing_calculation(self):
        """Test position sizing calculation methods"""
        class TestStrategy(StrategyTemplate):
            def init_indicators(self): pass
            def _check_entry_conditions(self, symbol): return None
            def check_exit(self, symbol): return None
        
        strategy = TestStrategy(self.test_data, self.enhanced_config)
        
        # Test fixed notional sizing
        strategy.position_sizing_mode = 'fixed_notional'
        strategy.position_fixed_notional = 1000.0
        
        size = strategy.calculate_position_size('BTCUSDT', 50000.0, 10000.0)
        self.assertAlmostEqual(size, 0.02, places=3)  # $1000 / $50000
        
        # Test volatility regime adjustment
        size_high_vol = strategy.calculate_position_size('BTCUSDT', 50000.0, 10000.0, 'high')
        self.assertLess(size_high_vol, size)  # Should be smaller due to 0.8 multiplier
    
    def test_stop_loss_calculation(self):
        """Test stop loss price calculation"""
        class TestStrategy(StrategyTemplate):
            def init_indicators(self): pass
            def _check_entry_conditions(self, symbol): return None
            def check_exit(self, symbol): return None
        
        strategy = TestStrategy(self.test_data, self.enhanced_config)
        
        # Test fixed percentage stop loss
        strategy.stop_loss_mode = 'fixed_pct'
        strategy.stop_loss_fixed_pct = 0.02
        
        sl_price_long = strategy.calculate_stop_loss_price('BTCUSDT', 50000.0, 'long')
        self.assertAlmostEqual(sl_price_long, 49000.0, places=2)
        
        sl_price_short = strategy.calculate_stop_loss_price('BTCUSDT', 50000.0, 'short')
        self.assertAlmostEqual(sl_price_short, 51000.0, places=2)
    
    def test_take_profit_calculation(self):
        """Test take profit price calculation"""
        class TestStrategy(StrategyTemplate):
            def init_indicators(self): pass
            def _check_entry_conditions(self, symbol): return None
            def check_exit(self, symbol): return None
        
        strategy = TestStrategy(self.test_data, self.enhanced_config)
        
        # Test progressive levels
        tp_prices = strategy.calculate_take_profit_prices('BTCUSDT', 50000.0, 'long')
        
        self.assertEqual(len(tp_prices), 3)
        self.assertAlmostEqual(tp_prices[0], 51000.0, places=2)  # 2%
        self.assertAlmostEqual(tp_prices[1], 52000.0, places=2)  # 4%
        self.assertAlmostEqual(tp_prices[2], 53000.0, places=2)  # 6%
    
    def test_liquidity_filter(self):
        """Test liquidity filtering functionality"""
        class TestStrategy(StrategyTemplate):
            def init_indicators(self): pass
            def _check_entry_conditions(self, symbol): return None
            def check_exit(self, symbol): return None
        
        strategy = TestStrategy(self.test_data, self.enhanced_config)
        
        # Test passing liquidity conditions
        good_market_data = {
            'bid': 49999.0,
            'ask': 50001.0,
            'bid_size': 1.0,  # $50k liquidity
            'ask_size': 1.0   # $50k liquidity
        }
        
        self.assertTrue(strategy.check_liquidity_filter('BTCUSDT', good_market_data))
        
        # Test failing liquidity conditions (low volume)
        bad_market_data = {
            'bid': 49999.0,
            'ask': 50001.0,
            'bid_size': 0.1,  # Only $5k liquidity
            'ask_size': 0.1   # Only $5k liquidity
        }
        
        self.assertFalse(strategy.check_liquidity_filter('BTCUSDT', bad_market_data))
    
    def test_spread_slippage_guard(self):
        """Test spread and slippage guard functionality"""
        class TestStrategy(StrategyTemplate):
            def init_indicators(self): pass
            def _check_entry_conditions(self, symbol): return None
            def check_exit(self, symbol): return None
        
        strategy = TestStrategy(self.test_data, self.enhanced_config)
        
        # Test acceptable spread
        good_spread_data = {
            'bid': 49995.0,
            'ask': 50005.0  # 0.02% spread
        }
        
        self.assertTrue(strategy.check_spread_slippage_guard('BTCUSDT', good_spread_data))
        
        # Test unacceptable spread
        bad_spread_data = {
            'bid': 49900.0,
            'ask': 50100.0  # 0.4% spread (too wide)
        }
        
        self.assertFalse(strategy.check_spread_slippage_guard('BTCUSDT', bad_spread_data))
    
    def test_enhanced_risk_parameters(self):
        """Test enhanced risk parameter calculation"""
        class TestStrategy(StrategyTemplate):
            def init_indicators(self): pass
            def _check_entry_conditions(self, symbol): return None
            def check_exit(self, symbol): return None
        
        strategy = TestStrategy(self.test_data, self.enhanced_config)
        
        risk_params = strategy.get_risk_parameters(
            symbol='BTCUSDT',
            entry_price=50000.0,
            side='long',
            account_equity=10000.0
        )
        
        # Check that all required parameters are present
        self.assertIn('sl_pct', risk_params)
        self.assertIn('tp_pct', risk_params)
        self.assertIn('position_size', risk_params)
        self.assertIn('trailing_stop_enabled', risk_params)
        self.assertIn('stop_loss_mode', risk_params)
        self.assertIn('take_profit_mode', risk_params)
        
        # Check that position size is calculated
        self.assertIsNotNone(risk_params['position_size'])
        self.assertGreater(risk_params['position_size'], 0)
    
    def test_backward_compatibility(self):
        """Test backward compatibility with legacy strategies"""
        class LegacyStrategy(StrategyTemplate):
            def init_indicators(self): pass
            def _check_entry_conditions(self, symbol): return None
            def check_exit(self, symbol): return None
        
        # Test with minimal config
        legacy_strategy = LegacyStrategy(self.test_data, self.minimal_config)
        
        # Should still work with basic risk parameters
        risk_params = legacy_strategy.get_risk_parameters()
        self.assertIn('sl_pct', risk_params)
        self.assertIn('tp_pct', risk_params)
        
        # Should use default values
        self.assertEqual(legacy_strategy.stop_loss_mode, 'fixed_pct')
        self.assertEqual(legacy_strategy.take_profit_mode, 'fixed_pct')
        self.assertFalse(legacy_strategy.trailing_stop_enabled)
    
    def test_trailing_stop_functionality(self):
        """Test trailing stop update functionality"""
        class TestStrategy(StrategyTemplate):
            def init_indicators(self): pass
            def _check_entry_conditions(self, symbol): return None
            def check_exit(self, symbol): return None
        
        strategy = TestStrategy(self.test_data, self.enhanced_config)
        
        # Test that trailing stop can be calculated
        if hasattr(strategy, 'update_trailing_stop_if_enabled'):
            trail_price = strategy.update_trailing_stop_if_enabled('BTCUSDT', 51000.0, 'long')
            # If RiskUtilities available, should return a price, otherwise None
            if trail_price is not None:
                self.assertIsInstance(trail_price, float)
                self.assertGreater(trail_price, 0)
    
    def test_utility_methods(self):
        """Test utility methods for enhanced functionality"""
        class TestStrategy(StrategyTemplate):
            def init_indicators(self): pass
            def _check_entry_conditions(self, symbol): return None
            def check_exit(self, symbol): return None
        
        strategy = TestStrategy(self.test_data, self.enhanced_config)
        
        # Test account equity setting
        strategy.set_account_equity(10000.0)
        self.assertEqual(strategy._get_account_equity(), 10000.0)
        
        # Test market data setting
        market_data = {'bid': 50000.0, 'ask': 50001.0}
        strategy.set_market_data('BTCUSDT', market_data)
        retrieved_data = strategy._get_market_data('BTCUSDT')
        self.assertEqual(retrieved_data, market_data)
        
        # Test current price retrieval
        current_price = strategy._get_current_price('BTCUSDT')
        self.assertIsNotNone(current_price)
        self.assertGreater(current_price, 0)

if __name__ == '__main__':
    unittest.main()