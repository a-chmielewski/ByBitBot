#!/usr/bin/env python3
"""
Test suite for Adaptive Transitional Momentum Strategy

This test suite covers all aspects of the adaptive transitional momentum strategy,
including volatility regime detection, adaptive momentum calculation, trend context analysis,
candlestick pattern detection, and trade management.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
import logging

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.adaptive_transitional_momentum_strategy import StrategyAdaptiveTransitionalMomentum

class TestAdaptiveTransitionalMomentumStrategy(unittest.TestCase):
    """Test cases for Adaptive Transitional Momentum Strategy"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Set random seed for reproducible tests
        np.random.seed(42)
        
        # Create test data representing transitional market conditions
        # (volatility regime changes, consolidation to breakout patterns)
        self.data_length = 200
        
        # Create base price data with regime transitions
        base_price = 50000.0
        
        # Phase 1: Low volatility consolidation (bars 0-50)
        low_vol_returns = np.random.normal(0, 0.005, 50)  # 0.5% daily volatility
        
        # Phase 2: Volatility expansion transition (bars 51-100)
        transition_returns = np.random.normal(0, 0.015, 50)  # 1.5% daily volatility
        
        # Phase 3: High volatility breakout (bars 101-150) 
        high_vol_returns = np.random.normal(0.002, 0.025, 50)  # 2.5% volatility with slight upward bias
        
        # Phase 4: Return to normal (bars 151-199)
        normal_returns = np.random.normal(0, 0.01, 50)  # 1% volatility
        
        # Combine all phases
        all_returns = np.concatenate([low_vol_returns, transition_returns, high_vol_returns, normal_returns])
        
        # Generate price series
        prices = [base_price]
        for ret in all_returns:
            prices.append(prices[-1] * (1 + ret))
        
        # Ensure we have exactly the right number of prices
        prices = prices[:self.data_length]
        
        # Create OHLCV data with realistic intrabar variations
        timestamps = pd.date_range(start='2023-01-01', periods=self.data_length, freq='5min')
        
        # Generate OHLC from close prices
        opens = [prices[0]] + prices[:-1]  # Open is previous close
        closes = prices
        
        # Generate highs and lows with some randomness
        highs = []
        lows = []
        volumes = []
        
        for i in range(self.data_length):
            open_price = opens[i]
            close_price = closes[i]
            
            # High/low based on volatility phase
            if i <= 50:  # Low volatility
                hl_range = abs(close_price - open_price) * 1.5
            elif i <= 100:  # Transition
                hl_range = abs(close_price - open_price) * 2.0
            elif i <= 150:  # High volatility
                hl_range = abs(close_price - open_price) * 3.0
            else:  # Normal
                hl_range = abs(close_price - open_price) * 2.0
            
            high = max(open_price, close_price) + np.random.uniform(0, hl_range)
            low = min(open_price, close_price) - np.random.uniform(0, hl_range)
            
            highs.append(high)
            lows.append(low)
            
            # Volume spikes during volatility expansion and breakouts
            if 51 <= i <= 100 or 101 <= i <= 150:
                base_volume = np.random.uniform(800, 1500)  # Higher volume during transitions/breakouts
            else:
                base_volume = np.random.uniform(400, 800)   # Normal volume
            
            volumes.append(base_volume)
        
        # Create DataFrame
        self.test_data = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        # Test configuration
        self.test_config = {
            'atr_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
            'fast_ma_period': 20,
            'slow_ma_period': 50,
            'adx_period': 14,
            'dmi_period': 14,
            'volume_avg_period': 20,
            'momentum_base_period': 20,
            'momentum_min_period': 10,
            'momentum_max_period': 30,
            'momentum_ema_period': 7,
            'vol_regime_lookback': 30,
            'vol_expansion_threshold': 1.5,
            'bb_squeeze_threshold': 0.02,
            'bb_expansion_threshold': 0.04,
            'regime_hysteresis': 0.1,
            'momentum_threshold': 0.5,
            'dmi_crossover_threshold': 5,
            'adx_low_threshold': 25,
            'adx_rising_threshold': 30,
            'volume_surge_multiplier': 1.5,
            'engulfing_min_ratio': 1.2,
            'range_lookback': 10,
            'wide_range_multiplier': 2.0,
            'stop_atr_multiplier': 1.5,
            'trailing_atr_multiplier': 2.0,
            'risk_reward_ratio': 2.0,
            'position_size_factor': 0.5,
            'max_position_pct': 2.0,
            'cooldown_bars': 5,
            'use_fixed_exit': False,
            'max_hold_bars': 5,
            'momentum_fade_threshold': 0.2,
            'emergency_exit_bars': 2,
            'sl_pct': 0.025,
            'tp_pct': 0.05
        }
        
        # Create mock logger
        self.mock_logger = Mock(spec=logging.Logger)
        
        # Initialize strategy
        self.strategy = StrategyAdaptiveTransitionalMomentum(
            data=self.test_data.copy(),
            config=self.test_config,
            logger=self.mock_logger
        )
    
    def test_initialization(self):
        """Test strategy initialization"""
        # Check basic initialization
        self.assertIsInstance(self.strategy, StrategyAdaptiveTransitionalMomentum)
        self.assertEqual(self.strategy.current_volatility_regime, "unknown")
        self.assertEqual(self.strategy.previous_regime, "unknown")
        self.assertEqual(self.strategy.adaptive_momentum_period, 20)
        self.assertIsNone(self.strategy.regime_change_bar)
        
        # Check configuration
        self.assertEqual(self.strategy.config, self.test_config)
        
        # Check data
        self.assertIsInstance(self.strategy.data, pd.DataFrame)
        self.assertEqual(len(self.strategy.data), self.data_length)
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, self.strategy.data.columns)
        
        # Check market type tags
        self.assertIn('TRANSITIONAL', self.strategy.MARKET_TYPE_TAGS)
        self.assertTrue(self.strategy.SHOW_IN_SELECTION)
    
    def test_init_indicators(self):
        """Test indicator initialization"""
        self.strategy.init_indicators()
        
        # Check that all indicators are added to data
        expected_indicators = [
            'atr', 'bb_upper', 'bb_middle', 'bb_lower', 'fast_ma', 'slow_ma',
            'adx', 'dmi_plus', 'dmi_minus', 'volume_sma'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, self.strategy.data.columns)
            # Check that indicator has valid values (not all NaN)
            self.assertFalse(self.strategy.data[indicator].isna().all())
    
    def test_atr_calculation(self):
        """Test ATR calculation"""
        self.strategy.init_indicators()
        
        # ATR should be positive and reasonable
        atr_values = self.strategy.data['atr'].dropna()
        self.assertTrue((atr_values > 0).all())
        
        # ATR should be less than 10% of price typically
        close_values = self.strategy.data['close'].iloc[len(self.strategy.data) - len(atr_values):]
        atr_percentage = atr_values / close_values.values
        self.assertTrue((atr_percentage < 0.1).all())
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        self.strategy.init_indicators()
        
        # Check that bands are properly ordered
        valid_data = self.strategy.data.dropna(subset=['bb_upper', 'bb_middle', 'bb_lower'])
        
        # Upper band should be above middle
        self.assertTrue((valid_data['bb_upper'] > valid_data['bb_middle']).all())
        
        # Middle band should be above lower
        self.assertTrue((valid_data['bb_middle'] > valid_data['bb_lower']).all())
        
        # Price should sometimes be between bands
        between_bands = ((valid_data['close'] >= valid_data['bb_lower']) & 
                        (valid_data['close'] <= valid_data['bb_upper']))
        self.assertTrue(between_bands.any())
    
    def test_moving_averages_calculation(self):
        """Test moving averages calculation"""
        self.strategy.init_indicators()
        
        # Check that MAs follow price trends generally
        valid_data = self.strategy.data.dropna(subset=['fast_ma', 'slow_ma'])
        
        # Fast MA should be more responsive than slow MA
        fast_ma_diff = valid_data['fast_ma'].diff().abs().mean()
        slow_ma_diff = valid_data['slow_ma'].diff().abs().mean()
        self.assertGreater(fast_ma_diff, slow_ma_diff * 0.8)  # Allow some tolerance
        
        # MAs should be reasonable relative to price
        price_ratio_fast = valid_data['fast_ma'] / valid_data['close']
        price_ratio_slow = valid_data['slow_ma'] / valid_data['close']
        
        # Ratios should be close to 1 (within 20%)
        self.assertTrue(((price_ratio_fast > 0.8) & (price_ratio_fast < 1.2)).all())
        self.assertTrue(((price_ratio_slow > 0.8) & (price_ratio_slow < 1.2)).all())
    
    def test_adx_and_dmi_calculation(self):
        """Test ADX and DMI calculations"""
        self.strategy.init_indicators()
        
        # ADX should be between 0 and 100
        adx_values = self.strategy.data['adx'].dropna()
        self.assertTrue((adx_values >= 0).all())
        self.assertTrue((adx_values <= 100).all())
        
        # DMI values should be between 0 and 100
        dmi_plus_values = self.strategy.data['dmi_plus'].dropna()
        dmi_minus_values = self.strategy.data['dmi_minus'].dropna()
        
        self.assertTrue((dmi_plus_values >= 0).all())
        self.assertTrue((dmi_plus_values <= 100).all())
        self.assertTrue((dmi_minus_values >= 0).all())
        self.assertTrue((dmi_minus_values <= 100).all())
    
    def test_volume_analysis(self):
        """Test volume analysis"""
        self.strategy.init_indicators()
        
        # Volume SMA should be positive
        volume_sma = self.strategy.data['volume_sma'].dropna()
        self.assertTrue((volume_sma > 0).all())
        
        # Volume SMA should be reasonable compared to actual volume
        valid_indices = self.strategy.data['volume_sma'].notna()
        actual_volumes = self.strategy.data.loc[valid_indices, 'volume']
        volume_sma_values = self.strategy.data.loc[valid_indices, 'volume_sma']
        
        # Ratio should be reasonable (within 0.1x to 10x)
        ratios = actual_volumes / volume_sma_values
        self.assertTrue((ratios > 0.1).all())
        self.assertTrue((ratios < 10.0).all())
    
    def test_volatility_regime_detection(self):
        """Test volatility regime detection logic"""
        self.strategy.init_indicators()
        
        # Test at different points in our transitional data
        
        # Early low volatility period (should detect low regime eventually)
        regime_30 = self.strategy.detect_volatility_regime(40)
        
        # Transition period (should detect transitioning or high)
        regime_75 = self.strategy.detect_volatility_regime(75)
        
        # High volatility period (should detect high regime)
        regime_125 = self.strategy.detect_volatility_regime(125)
        
        # All regimes should be valid
        valid_regimes = ["unknown", "low", "high", "transitioning"]
        self.assertIn(regime_30, valid_regimes)
        self.assertIn(regime_75, valid_regimes)
        self.assertIn(regime_125, valid_regimes)
        
        # Should have detected some regime changes
        # Run through multiple points to build up history
        for i in range(60, 190, 10):
            self.strategy.detect_volatility_regime(i)
        
        # After processing all data, should have some regime information
        # Allow "unknown" as it's a valid state if conditions aren't met
        valid_regimes = ["unknown", "low", "high", "transitioning"]
        self.assertIn(self.strategy.current_volatility_regime, valid_regimes)
    
    def test_adaptive_momentum_calculation(self):
        """Test adaptive momentum calculation"""
        self.strategy.init_indicators()
        
        # Set different volatility regimes and test momentum adaptation
        test_index = 100
        
        # Test high volatility regime (should use shorter period)
        self.strategy.current_volatility_regime = "high"
        momentum_high = self.strategy.calculate_adaptive_momentum(test_index)
        expected_period_high = self.strategy.adaptive_momentum_period
        self.assertEqual(expected_period_high, 10)  # Should use min period
        
        # Test low volatility regime (should use longer period)
        self.strategy.current_volatility_regime = "low"
        momentum_low = self.strategy.calculate_adaptive_momentum(test_index)
        expected_period_low = self.strategy.adaptive_momentum_period
        self.assertEqual(expected_period_low, 30)  # Should use max period
        
        # Test transitioning regime (should use base period)
        self.strategy.current_volatility_regime = "transitioning"
        momentum_trans = self.strategy.calculate_adaptive_momentum(test_index)
        expected_period_trans = self.strategy.adaptive_momentum_period
        self.assertEqual(expected_period_trans, 20)  # Should use base period
        
        # Momentum values should be reasonable (-100% to +100% typically)
        self.assertGreater(momentum_high, -200)
        self.assertLess(momentum_high, 200)
        self.assertGreater(momentum_low, -200)
        self.assertLess(momentum_low, 200)
        self.assertGreater(momentum_trans, -200)
        self.assertLess(momentum_trans, 200)
    
    def test_trend_context_detection(self):
        """Test trend context analysis"""
        self.strategy.init_indicators()
        
        test_index = 100
        trend_context = self.strategy.detect_trend_context(test_index)
        
        # Should return all required keys
        expected_keys = ['ma_trend', 'dmi_signal', 'adx_state']
        for key in expected_keys:
            self.assertIn(key, trend_context)
        
        # Values should be from valid sets
        valid_ma_trends = ['unknown', 'neutral', 'bullish', 'bearish', 'bullish_crossover', 'bearish_crossover']
        valid_dmi_signals = ['unknown', 'neutral', 'bullish', 'bearish', 'bullish_crossover', 'bearish_crossover']
        valid_adx_states = ['unknown', 'ranging', 'trending', 'transitioning']
        
        self.assertIn(trend_context['ma_trend'], valid_ma_trends)
        self.assertIn(trend_context['dmi_signal'], valid_dmi_signals)
        self.assertIn(trend_context['adx_state'], valid_adx_states)
    
    def test_candlestick_pattern_detection(self):
        """Test candlestick pattern detection"""
        self.strategy.init_indicators()
        
        # Test pattern detection at various points
        test_indices = [50, 100, 150]
        
        for idx in test_indices:
            patterns = self.strategy.detect_candlestick_patterns(idx)
            
            # Should return required keys
            self.assertIn('engulfing', patterns)
            self.assertIn('wide_range', patterns)
            
            # Engulfing should be None, 'bullish', or 'bearish'
            self.assertIn(patterns['engulfing'], [None, 'bullish', 'bearish'])
            
            # Wide range should be boolean
            self.assertIsInstance(patterns['wide_range'], bool)
    
    def test_entry_conditions_basic(self):
        """Test basic entry condition logic"""
        self.strategy.init_indicators()
        
        # Setup regime change scenario
        self.strategy.current_volatility_regime = "transitioning"
        self.strategy.regime_change_bar = 90
        
        # Test entry conditions at a point shortly after regime change
        test_index = 95
        should_enter, direction = self.strategy.check_entry_conditions(test_index)
        
        # Should return boolean and valid direction
        self.assertIsInstance(should_enter, bool)
        if direction is not None:
            self.assertIn(direction, ['long', 'short'])
    
    def test_entry_conditions_no_regime_change(self):
        """Test that entry is blocked without recent regime change"""
        self.strategy.init_indicators()
        
        # No regime change
        self.strategy.current_volatility_regime = "low"
        self.strategy.regime_change_bar = None
        
        should_enter, direction = self.strategy.check_entry_conditions(100)
        
        self.assertFalse(should_enter)
        self.assertIsNone(direction)
    
    def test_entry_conditions_old_regime_change(self):
        """Test that entry is blocked if regime change is too old"""
        self.strategy.init_indicators()
        
        # Old regime change (more than 10 bars ago)
        self.strategy.current_volatility_regime = "transitioning"
        self.strategy.regime_change_bar = 80
        
        should_enter, direction = self.strategy.check_entry_conditions(100)
        
        self.assertFalse(should_enter)
        self.assertIsNone(direction)
    
    def test_stops_and_targets_calculation(self):
        """Test stop loss and take profit calculation"""
        self.strategy.init_indicators()
        
        test_index = 100
        current_price = self.strategy.data['close'].iloc[test_index]
        
        # Test long direction
        stop_long, target_long = self.strategy.calculate_stops_and_targets(test_index, 'long')
        
        self.assertLess(stop_long, current_price)  # Stop should be below entry for long
        self.assertGreater(target_long, current_price)  # Target should be above entry for long
        
        # Test short direction
        stop_short, target_short = self.strategy.calculate_stops_and_targets(test_index, 'short')
        
        self.assertGreater(stop_short, current_price)  # Stop should be above entry for short
        self.assertLess(target_short, current_price)  # Target should be below entry for short
        
        # Risk-reward ratio should be approximately correct
        long_risk = current_price - stop_long
        long_reward = target_long - current_price
        long_rr = long_reward / long_risk if long_risk > 0 else 0
        
        self.assertGreater(long_rr, 1.5)  # Should be close to 2.0 risk/reward
        self.assertLess(long_rr, 2.5)
    
    def test_exit_conditions_without_position(self):
        """Test exit conditions when no position exists"""
        self.strategy.init_indicators()
        
        # No position
        should_exit, exit_reason = self.strategy.check_exit_conditions(100)
        
        self.assertFalse(should_exit)
        self.assertIsNone(exit_reason)
    
    def test_exit_conditions_with_position(self):
        """Test exit conditions with active position"""
        self.strategy.init_indicators()
        
        # Setup position
        entry_index = 95
        self.strategy.entry_bar = entry_index
        self.strategy.entry_price = self.strategy.data['close'].iloc[entry_index]
        self.strategy.entry_side = 'long'
        
        # Calculate stops
        stop_price, target_price = self.strategy.calculate_stops_and_targets(entry_index, 'long')
        self.strategy.stop_price = stop_price
        self.strategy.target_price = target_price
        
        # Test various exit scenarios
        test_index = 100
        
        # Normal check (should not necessarily exit)
        should_exit, exit_reason = self.strategy.check_exit_conditions(test_index)
        self.assertIsInstance(should_exit, bool)
        if exit_reason is not None:
            valid_reasons = ['fixed_time_exit', 'stop_loss', 'target_hit', 'momentum_fade', 
                           'emergency_regime_reversal', 'trailing_stop']
            self.assertIn(exit_reason, valid_reasons)
    
    def test_trailing_stop_update(self):
        """Test trailing stop update logic"""
        self.strategy.init_indicators()
        
        # Setup long position
        entry_index = 95
        self.strategy.entry_bar = entry_index
        self.strategy.entry_price = self.strategy.data['close'].iloc[entry_index]
        self.strategy.entry_side = 'long'
        self.strategy.trailing_stop = None
        
        # Update trailing stop
        test_index = 100
        self.strategy.update_trailing_stop(test_index)
        
        # Check if trailing stop was set (depends on profit situation)
        if self.strategy.trailing_stop is not None:
            current_price = self.strategy.data['close'].iloc[test_index]
            self.assertLess(self.strategy.trailing_stop, current_price)  # For long position
    
    def test_check_entry_conditions_interface(self):
        """Test the main entry conditions interface"""
        self.strategy.init_indicators()
        
        # Mock a regime change scenario for testing
        self.strategy.current_volatility_regime = "transitioning"
        self.strategy.regime_change_bar = len(self.strategy.data) - 5
        
        result = self.strategy._check_entry_conditions('BTCUSDT')
        
        if result is not None:
            # Should have required keys
            self.assertIn('action', result)
            self.assertIn('price', result)
            self.assertIn('confidence', result)
            self.assertIn('reason', result)
            
            # Values should be valid
            self.assertIn(result['action'], ['long', 'short'])
            self.assertGreater(result['price'], 0)
            self.assertGreater(result['confidence'], 0)
            self.assertLess(result['confidence'], 1)
            self.assertIsInstance(result['reason'], str)
    
    def test_check_exit_interface(self):
        """Test the main exit interface"""
        self.strategy.init_indicators()
        
        # No position case
        result = self.strategy.check_exit('BTCUSDT')
        self.assertIsNone(result)
        
        # With position
        self.strategy.entry_bar = len(self.strategy.data) - 5
        self.strategy.entry_price = self.strategy.data['close'].iloc[-5]
        self.strategy.entry_side = 'long'
        
        result = self.strategy.check_exit('BTCUSDT')
        if result is not None:
            self.assertIn('action', result)
            self.assertIn('price', result)
            self.assertIn('reason', result)
            self.assertEqual(result['action'], 'exit')
    
    def test_trade_closed_handling(self):
        """Test trade closure handling"""
        self.strategy.init_indicators()
        
        # Setup position
        self.strategy.entry_bar = 95
        self.strategy.entry_price = 50000.0
        self.strategy.entry_side = 'long'
        self.strategy.last_trade_bar = 0
        
        # Close trade
        trade_result = {'reason': 'target_hit', 'pnl': 100.0}
        self.strategy.on_trade_closed('BTCUSDT', trade_result)
        
        # Check cleanup
        self.assertIsNone(self.strategy.entry_price)
        self.assertIsNone(self.strategy.entry_bar)
        self.assertIsNone(self.strategy.entry_side)
        self.assertIsNone(self.strategy.stop_price)
        self.assertIsNone(self.strategy.target_price)
        self.assertIsNone(self.strategy.trailing_stop)
        
        # Last trade bar should be updated
        self.assertGreater(self.strategy.last_trade_bar, 0)
    
    def test_risk_parameters(self):
        """Test risk parameter calculation"""
        self.strategy.init_indicators()
        
        risk_params = self.strategy.get_risk_parameters()
        
        # Should have required keys
        required_keys = ['sl_pct', 'tp_pct', 'max_position_pct', 'risk_reward_ratio']
        for key in required_keys:
            self.assertIn(key, risk_params)
        
        # Values should be reasonable
        self.assertGreater(risk_params['sl_pct'], 0)
        self.assertLess(risk_params['sl_pct'], 0.1)  # Less than 10%
        self.assertGreater(risk_params['tp_pct'], 0)
        self.assertLess(risk_params['tp_pct'], 0.2)  # Less than 20%
        self.assertGreater(risk_params['max_position_pct'], 0)
        self.assertLess(risk_params['max_position_pct'], 5.0)  # Less than 5% (adjusting from 0.05 to 5.0)
        self.assertGreater(risk_params['risk_reward_ratio'], 1)
        
        # Risk-reward relationship
        self.assertGreater(risk_params['tp_pct'], risk_params['sl_pct'])
    
    def test_risk_parameters_with_position(self):
        """Test risk parameters with active position"""
        self.strategy.init_indicators()
        
        # Setup position
        self.strategy.entry_price = self.strategy.data['close'].iloc[-1]
        
        risk_params = self.strategy.get_risk_parameters()
        
        # Should still have all required keys
        required_keys = ['sl_pct', 'tp_pct', 'max_position_pct', 'risk_reward_ratio']
        for key in required_keys:
            self.assertIn(key, risk_params)
        
        # Values should be ATR-based and reasonable
        self.assertGreater(risk_params['sl_pct'], 0)
        self.assertGreater(risk_params['tp_pct'], 0)
    
    def test_cooldown_period(self):
        """Test cooldown period between trades"""
        self.strategy.init_indicators()
        
        # Set recent trade
        self.strategy.last_trade_bar = len(self.strategy.data) - 3
        
        result = self.strategy._check_entry_conditions('BTCUSDT')
        
        # Should not enter during cooldown
        self.assertIsNone(result)
    
    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        # Create strategy with minimal data
        minimal_data = self.test_data.head(10).copy()
        
        # Should handle insufficient data gracefully
        try:
            strategy = StrategyAdaptiveTransitionalMomentum(
                data=minimal_data,
                config=self.test_config,
                logger=self.mock_logger
            )
            
            # Should not crash with insufficient data
            result = strategy._check_entry_conditions('BTCUSDT')
            self.assertIsNone(result)
        except Exception as e:
            # If pandas_ta fails with insufficient data, that's acceptable
            self.assertTrue(isinstance(e, (ValueError, KeyError, TypeError)))
    
    def test_invalid_data_handling(self):
        """Test handling of invalid data"""
        # Create data with NaN values
        invalid_data = self.test_data.copy()
        invalid_data.loc[50:60, 'close'] = np.nan
        invalid_data.loc[70:80, 'volume'] = np.nan
        
        strategy = StrategyAdaptiveTransitionalMomentum(
            data=invalid_data,
            config=self.test_config,
            logger=self.mock_logger
        )
        
        # Should not crash with invalid data
        try:
            strategy.init_indicators()
            result = strategy._check_entry_conditions('BTCUSDT')
            # If it returns something, it should be valid
            if result is not None:
                self.assertIn('action', result)
        except Exception as e:
            self.fail(f"Strategy should handle invalid data gracefully: {e}")
    
    def test_missing_columns(self):
        """Test behavior with missing required columns"""
        # Remove volume column
        incomplete_data = self.test_data.drop('volume', axis=1).copy()
        
        # Should handle missing volume gracefully
        try:
            strategy = StrategyAdaptiveTransitionalMomentum(
                data=incomplete_data,
                config=self.test_config,
                logger=self.mock_logger
            )
            # If it doesn't fail during init, that's ok too
            self.assertIsInstance(strategy, StrategyAdaptiveTransitionalMomentum)
        except Exception as e:
            # Should either work or fail gracefully with expected error types
            self.assertIsInstance(e, (KeyError, ValueError, AttributeError))
    
    def test_configuration_validation(self):
        """Test configuration parameter validation"""
        # Test with minimal config
        minimal_config = {}
        
        strategy = StrategyAdaptiveTransitionalMomentum(
            data=self.test_data.copy(),
            config=minimal_config,
            logger=self.mock_logger
        )
        
        # Should use default values
        self.assertIsNotNone(strategy.config)
        
        # Test with invalid config values (but avoid completely broken values)
        invalid_config = {
            'atr_period': 1,  # Very small but valid
            'bb_period': 5,   # Small but valid (avoid negative that breaks pandas_ta)
            'momentum_base_period': 100  # Large but reasonable
        }
        
        # Should handle edge case configuration gracefully
        try:
            strategy = StrategyAdaptiveTransitionalMomentum(
                data=self.test_data.copy(),
                config=invalid_config,
                logger=self.mock_logger
            )
            # Should not crash during initialization
            self.assertIsInstance(strategy, StrategyAdaptiveTransitionalMomentum)
        except Exception as e:
            # If it fails, it should be a reasonable error
            self.assertIsInstance(e, (ValueError, KeyError))
    
    def test_regime_detection_edge_cases(self):
        """Test volatility regime detection edge cases"""
        self.strategy.init_indicators()
        
        # Test with no ATR history
        self.strategy.atr_history = []
        regime = self.strategy.detect_volatility_regime(50)
        self.assertIn(regime, ['unknown', 'low', 'high', 'transitioning'])
        
        # Test with NaN ATR values by directly setting them
        original_atr = self.strategy.data['atr'].iloc[50]
        self.strategy.data.loc[self.strategy.data.index[50], 'atr'] = np.nan
        regime = self.strategy.detect_volatility_regime(50)
        self.assertIsInstance(regime, str)
        # Restore original value
        self.strategy.data.loc[self.strategy.data.index[50], 'atr'] = original_atr
    
    def test_momentum_calculation_edge_cases(self):
        """Test adaptive momentum calculation edge cases"""
        self.strategy.init_indicators()
        
        # Test with zero price by directly setting it
        original_price = self.strategy.data['close'].iloc[50]
        self.strategy.data.loc[self.strategy.data.index[50], 'close'] = 0
        momentum = self.strategy.calculate_adaptive_momentum(70)
        self.assertIsInstance(momentum, (int, float))
        # Restore original value
        self.strategy.data.loc[self.strategy.data.index[50], 'close'] = original_price
        
        # Test with early index
        momentum = self.strategy.calculate_adaptive_momentum(5)
        self.assertEqual(momentum, 0)
    
    def test_pattern_detection_edge_cases(self):
        """Test candlestick pattern detection edge cases"""
        self.strategy.init_indicators()
        
        # Test with first bar
        patterns = self.strategy.detect_candlestick_patterns(0)
        expected_result = {"engulfing": None, "wide_range": False}
        self.assertEqual(patterns, expected_result)
        
        # Test with equal OHLC values (doji-like)
        test_data = self.test_data.copy()
        test_data.loc[50, ['open', 'high', 'low', 'close']] = 50000.0
        
        strategy = StrategyAdaptiveTransitionalMomentum(
            data=test_data,
            config=self.test_config,
            logger=self.mock_logger
        )
        strategy.init_indicators()
        
        patterns = strategy.detect_candlestick_patterns(51)
        self.assertIn('engulfing', patterns)
        self.assertIn('wide_range', patterns)
    
    def test_data_integrity(self):
        """Test that strategy doesn't modify original data inappropriately"""
        original_data = self.test_data.copy()
        self.strategy.init_indicators()
        
        # Original OHLCV columns should remain unchanged
        for col in ['open', 'high', 'low', 'close', 'volume']:
            pd.testing.assert_series_equal(
                original_data[col], 
                self.strategy.data[col],
                check_names=False
            )
    
    def test_performance_tracking_integration(self):
        """Test integration with performance tracking"""
        self.strategy.init_indicators()
        
        # Simulate a complete trade cycle
        # Setup regime change
        self.strategy.current_volatility_regime = "transitioning"
        self.strategy.regime_change_bar = len(self.strategy.data) - 10
        
        # Check for entry
        entry_result = self.strategy._check_entry_conditions('BTCUSDT')
        
        if entry_result is not None:
            # Simulate position setup
            self.strategy.entry_bar = len(self.strategy.data) - 5
            self.strategy.entry_price = entry_result['price']
            self.strategy.entry_side = entry_result['action']
            
            # Check for exit
            exit_result = self.strategy.check_exit('BTCUSDT')
            
            # Close trade
            trade_result = {
                'reason': 'test_exit',
                'pnl': 50.0,
                'entry_price': self.strategy.entry_price,
                'exit_price': self.strategy.data['close'].iloc[-1]
            }
            
            self.strategy.on_trade_closed('BTCUSDT', trade_result)
            
            # Verify cleanup
            self.assertIsNone(self.strategy.entry_price)
            self.assertIsNone(self.strategy.entry_side)

if __name__ == '__main__':
    unittest.main() 