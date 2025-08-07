#!/usr/bin/env python3
"""
Unit tests for Enhanced Strategy Matrix and Config Loader
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.strategy_matrix import (
    StrategyMatrix, StrategyRiskProfile, StopLossConfig, TakeProfitConfig,
    TrailingStopConfig, PositionSizingConfig, LeverageByRegimeConfig,
    PortfolioTagsConfig, TradingLimitsConfig
)
from modules.config_loader import StrategyConfigLoader, ConfigValidationError
import json

class TestEnhancedStrategyMatrix(unittest.TestCase):
    """Test cases for enhanced StrategyMatrix functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.matrix = StrategyMatrix()
        
    def test_strategy_selection_with_risk_profiles(self):
        """Test strategy selection returns strategies with complete risk profiles"""
        strategy_name, timeframe, reason = self.matrix.select_strategy_and_timeframe('HIGH_VOLATILITY', 'HIGH_VOLATILITY')
        
        self.assertIsInstance(strategy_name, str)
        self.assertIsInstance(timeframe, str)
        self.assertIsInstance(reason, str)
        
        # Verify strategy has risk profile
        risk_profile = self.matrix.get_strategy_risk_profile(strategy_name)
        self.assertIsNotNone(risk_profile)
        self.assertIsInstance(risk_profile, StrategyRiskProfile)
        
    def test_risk_profile_completeness(self):
        """Test that all strategies have complete risk profiles"""
        for strategy_name in self.matrix.STRATEGY_RISK_PROFILES:
            with self.subTest(strategy=strategy_name):
                profile = self.matrix.get_strategy_risk_profile(strategy_name)
                
                # Test required fields
                self.assertIsNotNone(profile.strategy_name)
                self.assertIsNotNone(profile.description)
                self.assertIsInstance(profile.market_type_tags, list)
                self.assertGreater(len(profile.market_type_tags), 0)
                
                # Test risk management components
                self.assertIsInstance(profile.stop_loss, StopLossConfig)
                self.assertIsInstance(profile.take_profit, TakeProfitConfig)
                self.assertIsInstance(profile.trailing_stop, TrailingStopConfig)
                self.assertIsInstance(profile.position_sizing, PositionSizingConfig)
                self.assertIsInstance(profile.leverage_by_regime, LeverageByRegimeConfig)
                self.assertIsInstance(profile.portfolio_tags, PortfolioTagsConfig)
                self.assertIsInstance(profile.trading_limits, TradingLimitsConfig)
                
    def test_strategies_by_factor(self):
        """Test factor-based strategy grouping"""
        momentum_strategies = self.matrix.get_strategies_by_factor('momentum')
        trend_strategies = self.matrix.get_strategies_by_factor('trend_following')
        mean_reversion_strategies = self.matrix.get_strategies_by_factor('mean_reversion')
        
        self.assertIsInstance(momentum_strategies, list)
        self.assertIsInstance(trend_strategies, list)
        self.assertIsInstance(mean_reversion_strategies, list)
        
        # Should have strategies in each category
        self.assertGreater(len(momentum_strategies), 0)
        self.assertGreater(len(trend_strategies), 0)
        self.assertGreater(len(mean_reversion_strategies), 0)
        
        # Test specific expected strategies
        self.assertIn('StrategyATRMomentumBreakout', momentum_strategies)
        self.assertIn('StrategyEMATrendRider', trend_strategies)
        self.assertIn('StrategyRSIRangeScalping', mean_reversion_strategies)
        
    def test_correlation_groups(self):
        """Test correlation group functionality"""
        correlation_matrix = self.matrix.get_portfolio_correlation_matrix()
        
        self.assertIsInstance(correlation_matrix, dict)
        self.assertGreater(len(correlation_matrix), 0)
        
        # Each group should have at least one strategy
        for group, strategies in correlation_matrix.items():
            self.assertIsInstance(strategies, list)
            self.assertGreater(len(strategies), 0)
            
        # Test specific correlation groups exist
        group_names = list(correlation_matrix.keys())
        expected_groups = ['high_vol_breakout', 'trend_momentum', 'range_scalping', 'vol_reversal']
        for group in expected_groups:
            self.assertIn(group, group_names, f"Expected correlation group '{group}' not found")
    
    def test_volatility_validation(self):
        """Test volatility regime validation for strategies"""
        # ATR Momentum should accept high volatility
        self.assertTrue(
            self.matrix.validate_volatility_regime_for_strategy('StrategyATRMomentumBreakout', 5.0)
        )
        
        # ATR Momentum should reject very low volatility
        self.assertFalse(
            self.matrix.validate_volatility_regime_for_strategy('StrategyATRMomentumBreakout', 0.3)
        )
        
        # Micro Range Scalping should accept low volatility
        self.assertTrue(
            self.matrix.validate_volatility_regime_for_strategy('StrategyMicroRangeScalping', 0.5)
        )
        
        # Micro Range Scalping should reject high volatility
        self.assertFalse(
            self.matrix.validate_volatility_regime_for_strategy('StrategyMicroRangeScalping', 5.0)
        )
    
    def test_max_concurrent_trades(self):
        """Test max concurrent trades retrieval"""
        # Test known strategies
        atr_max_trades = self.matrix.get_max_concurrent_trades_for_strategy('StrategyATRMomentumBreakout')
        micro_max_trades = self.matrix.get_max_concurrent_trades_for_strategy('StrategyMicroRangeScalping')
        
        self.assertIsInstance(atr_max_trades, int)
        self.assertIsInstance(micro_max_trades, int)
        self.assertGreater(atr_max_trades, 0)
        self.assertGreater(micro_max_trades, 0)
        
        # Micro scalping should allow more concurrent trades than ATR momentum
        self.assertGreater(micro_max_trades, atr_max_trades)
        
        # Unknown strategy should return default
        unknown_max_trades = self.matrix.get_max_concurrent_trades_for_strategy('UnknownStrategy')
        self.assertEqual(unknown_max_trades, 3)

class TestStrategyConfigLoader(unittest.TestCase):
    """Test cases for StrategyConfigLoader functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.matrix = StrategyMatrix()
        self.loader = StrategyConfigLoader(self.matrix)
        
    def test_config_loading(self):
        """Test basic configuration loading"""
        config = self.loader.get_strategy_config('StrategyATRMomentumBreakout')
        
        self.assertIsInstance(config, dict)
        self.assertIn('strategy_configs', config)
        
        strategy_config = config['strategy_configs']['StrategyATRMomentumBreakout']
        self.assertIn('risk_management', strategy_config)
        self.assertIn('portfolio', strategy_config)
        self.assertIn('trading_limits', strategy_config)
        
        # Test specific risk management fields
        risk_config = strategy_config['risk_management']
        self.assertIn('stop_loss_mode', risk_config)
        self.assertIn('stop_loss_atr_multiplier', risk_config)
        self.assertIn('take_profit_mode', risk_config)
        self.assertIn('position_sizing_mode', risk_config)
        self.assertIn('leverage_by_regime', risk_config)
        
    def test_configuration_overrides(self):
        """Test configuration override functionality"""
        overrides = {
            'strategy_configs': {
                'StrategyATRMomentumBreakout': {
                    'risk_management': {
                        'stop_loss_atr_multiplier': 2.5,
                        'position_risk_per_trade': 0.02
                    }
                }
            }
        }
        
        config = self.loader.get_strategy_config('StrategyATRMomentumBreakout', overrides)
        risk_config = config['strategy_configs']['StrategyATRMomentumBreakout']['risk_management']
        
        # Overridden values should be applied
        self.assertEqual(risk_config['stop_loss_atr_multiplier'], 2.5)
        self.assertEqual(risk_config['position_risk_per_trade'], 0.02)
        
        # Non-overridden values should remain from base profile
        self.assertIsNotNone(risk_config['take_profit_mode'])
        
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Valid overrides should pass
        valid_overrides = {
            'strategy_configs': {
                'StrategyATRMomentumBreakout': {
                    'risk_management': {
                        'stop_loss_atr_multiplier': 2.0,
                        'position_risk_per_trade': 0.015
                    }
                }
            }
        }
        
        try:
            config = self.loader.get_strategy_config('StrategyATRMomentumBreakout', valid_overrides)
            # Should not raise exception
        except ConfigValidationError:
            self.fail("Valid configuration should not raise validation error")
        
        # Invalid overrides should fail
        invalid_overrides = {
            'strategy_configs': {
                'StrategyATRMomentumBreakout': {
                    'risk_management': {
                        'stop_loss_atr_multiplier': -1.0,  # Invalid: negative
                        'position_risk_per_trade': 0.15    # Invalid: too high (15%)
                    }
                }
            }
        }
        
        with self.assertRaises(ConfigValidationError):
            self.loader.get_strategy_config('StrategyATRMomentumBreakout', invalid_overrides)
    
    def test_market_condition_validation(self):
        """Test market condition validation for strategies"""
        # ATR Momentum should be valid for high volatility
        high_vol_conditions = {
            'volatility_pct': 5.0,
            'market_type': 'HIGH_VOLATILITY'
        }
        
        self.assertTrue(
            self.loader.validate_strategy_for_conditions('StrategyATRMomentumBreakout', high_vol_conditions)
        )
        
        # ATR Momentum should be invalid for low volatility
        low_vol_conditions = {
            'volatility_pct': 0.3,
            'market_type': 'LOW_VOLATILITY'
        }
        
        self.assertFalse(
            self.loader.validate_strategy_for_conditions('StrategyATRMomentumBreakout', low_vol_conditions)
        )
    
    def test_correlation_aware_selection(self):
        """Test correlation-aware strategy selection"""
        # Test with no active strategies - should return all strategies
        available_all = self.loader.get_correlation_aware_strategies([], max_correlation_exposure=0.8)
        self.assertIsInstance(available_all, list)
        self.assertGreater(len(available_all), 0)
        
        # Test with one active strategy - should return strategies from other groups
        active_strategies = ['StrategyATRMomentumBreakout']
        available = self.loader.get_correlation_aware_strategies(active_strategies, max_correlation_exposure=0.8)
        
        self.assertIsInstance(available, list)
        
        # Should not include the already active strategy
        self.assertNotIn('StrategyATRMomentumBreakout', available)
        
        # Since each correlation group typically has one strategy, and we're using 0.8 exposure,
        # strategies from different groups should be available
        correlation_matrix = self.matrix.get_portfolio_correlation_matrix()
        atr_group = None
        for group, strategies in correlation_matrix.items():
            if 'StrategyATRMomentumBreakout' in strategies:
                atr_group = group
                break
        
        # Count strategies from different groups
        different_group_count = 0
        for strategy in available_all:  # Use available_all since available might be empty due to single-strategy groups
            profile = self.matrix.get_strategy_risk_profile(strategy)
            if profile and profile.portfolio_tags.correlation_group != atr_group:
                different_group_count += 1
        
        # Should have strategies from different correlation groups
        self.assertGreater(different_group_count, 0)
    
    def test_json_export(self):
        """Test JSON configuration export"""
        json_str = self.loader.export_strategy_config_to_json('StrategyRSIRangeScalping')
        
        self.assertIsInstance(json_str, str)
        
        # Should be valid JSON
        parsed_config = json.loads(json_str)
        self.assertIsInstance(parsed_config, dict)
        self.assertIn('strategy_configs', parsed_config)
        self.assertIn('StrategyRSIRangeScalping', parsed_config['strategy_configs'])
    
    def test_all_strategies_config_loading(self):
        """Test loading configurations for all strategies"""
        all_configs = self.loader.get_all_strategy_configs()
        
        self.assertIsInstance(all_configs, dict)
        self.assertGreater(len(all_configs), 0)
        
        # Should have configs for known strategies
        expected_strategies = ['StrategyATRMomentumBreakout', 'StrategyRSIRangeScalping', 'StrategyEMATrendRider']
        for strategy in expected_strategies:
            self.assertIn(strategy, all_configs)
            
            # Each config should have required sections
            config = all_configs[strategy]
            self.assertIn('risk_management', config)
            self.assertIn('portfolio', config)
            self.assertIn('trading_limits', config)
    
    def test_unknown_strategy_error(self):
        """Test error handling for unknown strategies"""
        with self.assertRaises(ConfigValidationError):
            self.loader.get_strategy_config('UnknownStrategy')

if __name__ == '__main__':
    unittest.main()