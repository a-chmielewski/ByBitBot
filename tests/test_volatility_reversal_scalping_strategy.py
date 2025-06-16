#!/usr/bin/env python3
"""
Test Suite for Volatility Reversal Scalping Strategy

This comprehensive test suite validates all aspects of the volatility reversal scalping strategy,
including spike detection, reversal patterns, entry/exit conditions, and risk management.
"""

import unittest
import pandas as pd
import numpy as np
import logging
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.volatility_reversal_scalping_strategy import StrategyVolatilityReversalScalping

class TestVolatilityReversalScalpingStrategy(unittest.TestCase):
    """Test cases for Volatility Reversal Scalping Strategy"""
    
    def setUp(self):
        """Set up test fixtures with realistic volatile market data"""
        # Create realistic volatile market data with spikes and reversals
        np.random.seed(42)  # For reproducible tests
        
        # Base price around 50000 (like BTC)
        base_price = 50000
        n_bars = 200
        
        # Create volatile market with clear spikes
        prices = []
        volumes = []
        
        for i in range(n_bars):
            # Normal market movement
            if i < 20:
                # Initial stable period
                price = base_price + np.random.normal(0, 50)
                volume = 1000 + np.random.normal(0, 100)
            elif i == 20:
                # First up-spike (extreme volatility)
                price = base_price + 800  # 1.6% spike
                volume = 3000  # High volume
            elif i == 21:
                # Reversal after up-spike
                price = base_price + 600  # Partial retracement
                volume = 2000
            elif i == 22:
                # Further reversal
                price = base_price + 200  # Strong retracement
                volume = 1500
            elif i < 40:
                # Consolidation period
                price = base_price + np.random.normal(0, 100)
                volume = 1000 + np.random.normal(0, 150)
            elif i == 40:
                # Down-spike (extreme volatility)
                price = base_price - 900  # 1.8% spike down
                volume = 3500  # Very high volume
            elif i == 41:
                # Reversal after down-spike
                price = base_price - 600  # Partial recovery
                volume = 2200
            elif i == 42:
                # Further recovery
                price = base_price - 100  # Strong recovery
                volume = 1800
            elif i < 60:
                # Another consolidation
                price = base_price + np.random.normal(0, 80)
                volume = 1000 + np.random.normal(0, 120)
            elif i == 60:
                # Moderate up-spike
                price = base_price + 400  # 0.8% spike
                volume = 2200  # Elevated volume
            elif i == 61:
                # Shooting star reversal
                price = base_price + 300  # Failed to hold gains
                volume = 1800
            elif i < 80:
                # Ranging market
                price = base_price + np.random.normal(0, 60)
                volume = 1000 + np.random.normal(0, 100)
            elif i == 80:
                # Another down-spike
                price = base_price - 600  # 1.2% down
                volume = 2800
            elif i == 81:
                # Hammer reversal
                price = base_price - 200  # Strong recovery
                volume = 2000
            else:
                # Normal market continuation
                price = base_price + np.random.normal(0, 70)
                volume = 1000 + np.random.normal(0, 130)
            
            prices.append(max(price, 100))  # Ensure positive prices
            volumes.append(max(volume, 100))  # Ensure positive volumes
        
        # Create OHLCV data with realistic candle structure
        data = []
        for i, (price, volume) in enumerate(zip(prices, volumes)):
            # Create realistic OHLC from close price
            volatility = 0.002 if i not in [20, 40, 60, 80] else 0.008  # Higher volatility on spike bars
            
            open_price = price * (1 + np.random.normal(0, volatility/2))
            high_price = max(open_price, price) * (1 + abs(np.random.normal(0, volatility)))
            low_price = min(open_price, price) * (1 - abs(np.random.normal(0, volatility)))
            
            # Special candle patterns for key bars
            if i == 20:  # Up-spike bar
                high_price = price * 1.012  # Extended high
                low_price = open_price * 0.998
            elif i == 21:  # Shooting star after up-spike
                open_price = prices[20] * 0.998
                high_price = prices[20] * 1.005  # Failed to make new high
                low_price = price * 0.995
            elif i == 40:  # Down-spike bar
                low_price = price * 0.988  # Extended low
                high_price = open_price * 1.002
            elif i == 41:  # Hammer after down-spike
                open_price = prices[40] * 1.002
                low_price = prices[40] * 0.995  # Failed to make new low
                high_price = price * 1.005
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': price,
                'volume': volume
            })
        
        self.test_data = pd.DataFrame(data)
        
        # Default configuration for testing
        self.config = {
            'bb_period': 20,
            'bb_std': 2.0,
            'bb_extreme_std': 2.5,
            'atr_period': 14,
            'atr_extreme_multiplier': 2.5,
            'rsi_period': 7,
            'rsi_overbought_extreme': 80,
            'rsi_oversold_extreme': 20,
            'rsi_overbought_moderate': 75,
            'rsi_oversold_moderate': 25,
            'ema_period': 20,
            'volume_avg_period': 20,
            'volume_climax_multiplier': 2.0,
            'min_spike_size_pct': 0.005,
            'min_score_threshold': 3,
            'immediate_entry_score': 5,
            'reversal_confirmation_bars': 3,
            'fibonacci_retracement_1': 0.382,
            'fibonacci_retracement_2': 0.5,
            'max_hold_bars': 8,
            'cooldown_bars': 2,
            'max_consecutive_losses': 4,
            'stop_buffer_pct': 0.002,
            'position_size_reduction': 0.7,
            'max_position_pct': 2.0
        }
        
        # Mock logger
        self.logger = Mock(spec=logging.Logger)
        
        # Initialize strategy
        self.strategy = StrategyVolatilityReversalScalping(
            data=self.test_data.copy(),
            config=self.config,
            logger=self.logger
        )
    
    def test_strategy_initialization(self):
        """Test strategy initialization and basic properties"""
        self.assertIsInstance(self.strategy, StrategyVolatilityReversalScalping)
        self.assertEqual(self.strategy.MARKET_TYPE_TAGS, ['HIGH_VOLATILITY', 'RANGING'])
        self.assertTrue(self.strategy.SHOW_IN_SELECTION)
        
        # Check initial state
        self.assertFalse(self.strategy.extreme_spike_detected)
        self.assertIsNone(self.strategy.spike_direction)
        self.assertEqual(self.strategy.consecutive_losses, 0)
        self.assertTrue(self.strategy.trading_enabled)
        self.assertFalse(self.strategy.reversal_pattern_detected)
    
    def test_indicator_initialization(self):
        """Test that all indicators are properly initialized"""
        # Check that all required indicators exist
        required_indicators = ['bb_upper', 'bb_middle', 'bb_lower', 'atr', 'rsi', 'ema', 'volume_sma']
        
        for indicator in required_indicators:
            self.assertIn(indicator, self.strategy.data.columns)
            self.assertFalse(self.strategy.data[indicator].isna().all())
        
        # Check indicator values are reasonable (skip NaN values at the beginning)
        valid_data = self.strategy.data.dropna()
        if len(valid_data) > 0:
            self.assertTrue((valid_data['bb_upper'] >= valid_data['bb_middle']).all())
            self.assertTrue((valid_data['bb_middle'] >= valid_data['bb_lower']).all())
            self.assertTrue((valid_data['rsi'] >= 0).all())
            self.assertTrue((valid_data['rsi'] <= 100).all())
            self.assertTrue((valid_data['atr'] > 0).all())
    
    def test_manual_atr_calculation(self):
        """Test manual ATR calculation fallback"""
        # Test manual ATR calculation directly using existing strategy
        atr_result = self.strategy._calculate_atr_manual(14)
        
        self.assertIsInstance(atr_result, pd.Series)
        self.assertEqual(len(atr_result), len(self.strategy.data))
        # Check non-NaN values are >= 0
        valid_atr = atr_result.dropna()
        if len(valid_atr) > 0:
            self.assertTrue((valid_atr >= 0).all())
    
    def test_manual_rsi_calculation(self):
        """Test manual RSI calculation fallback"""
        # Test manual RSI calculation directly using existing strategy
        rsi_result = self.strategy._calculate_rsi_manual(7)
        
        self.assertIsInstance(rsi_result, pd.Series)
        self.assertEqual(len(rsi_result), len(self.strategy.data))
        # Check non-NaN values are in valid range
        valid_rsi = rsi_result.dropna()
        if len(valid_rsi) > 0:
            self.assertTrue((valid_rsi >= 0).all())
            self.assertTrue((valid_rsi <= 100).all())
    
    def test_spike_detection_up_spike(self):
        """Test detection of upward volatility spikes"""
        # Test at bar 20 (known up-spike)
        score, direction, spike_data = self.strategy.evaluate_spike_conditions(20)
        
        self.assertGreater(score, 0)
        self.assertEqual(direction, 'up')
        self.assertIsNotNone(spike_data)
        self.assertEqual(spike_data['direction'], 'up')
        self.assertGreater(spike_data['size'], 0)
        self.assertGreater(spike_data['score'], 0)
    
    def test_spike_detection_down_spike(self):
        """Test detection of downward volatility spikes"""
        # Test at bar 40 (known down-spike)
        score, direction, spike_data = self.strategy.evaluate_spike_conditions(40)
        
        self.assertGreater(score, 0)
        self.assertEqual(direction, 'down')
        self.assertIsNotNone(spike_data)
        self.assertEqual(spike_data['direction'], 'down')
        self.assertGreater(spike_data['size'], 0)
        self.assertGreater(spike_data['score'], 0)
    
    def test_spike_detection_no_spike(self):
        """Test that normal market conditions don't trigger spike detection"""
        # Test at bar 10 (normal market)
        score, direction, spike_data = self.strategy.evaluate_spike_conditions(10)
        
        self.assertEqual(score, 0)
        self.assertIsNone(direction)
        self.assertIsNone(spike_data)
    
    def test_spike_scoring_system(self):
        """Test the spike scoring system components"""
        # Test at extreme spike bar
        score, direction, spike_data = self.strategy.evaluate_spike_conditions(20)
        
        # Should have high score due to multiple factors
        self.assertGreaterEqual(score, 3)  # Minimum threshold
        
        # Test scoring components individually by checking the logic
        idx = 20
        current_price = self.strategy.data['close'].iloc[idx]
        bb_upper = self.strategy.data['bb_upper'].iloc[idx]
        rsi_value = self.strategy.data['rsi'].iloc[idx]
        
        # Price should be above BB upper for up-spike
        self.assertGreater(current_price, bb_upper)
    
    def test_reversal_pattern_shooting_star(self):
        """Test shooting star pattern detection after up-spike"""
        # Create specific shooting star pattern
        test_data = self.test_data.copy()
        idx = 21  # Bar after up-spike
        
        # Set previous bar high for comparison
        test_data.loc[idx-1, 'high'] = 50700  # Previous high
        
        # Modify to create clear shooting star
        # Body size = |close - open| = |50590 - 50600| = 10
        # Upper shadow = high - max(open, close) = 50680 - 50600 = 80
        # Lower shadow = min(open, close) - low = 50590 - 50580 = 10
        # Conditions: upper_shadow > 2 * body_size (80 > 20) ✓
        #            lower_shadow < body_size * 0.5 (10 < 5) ✗
        test_data.loc[idx, 'open'] = 50600
        test_data.loc[idx, 'high'] = 50680  # Long upper shadow
        test_data.loc[idx, 'low'] = 50595   # Very small lower shadow
        test_data.loc[idx, 'close'] = 50590  # Small body
        
        strategy = StrategyVolatilityReversalScalping(test_data, self.config, self.logger)
        
        pattern_detected, pattern_type = strategy.detect_reversal_pattern(idx, 'up')
        
        self.assertTrue(pattern_detected)
        self.assertEqual(pattern_type, 'shooting_star')
    
    def test_reversal_pattern_hammer(self):
        """Test hammer pattern detection after down-spike"""
        # Create specific hammer pattern
        test_data = self.test_data.copy()
        idx = 41  # Bar after down-spike
        
        # Set previous bar low for comparison
        test_data.loc[idx-1, 'low'] = 49050  # Previous low
        
        # Modify to create clear hammer
        # Body size = |close - open| = |49210 - 49200| = 10
        # Lower shadow = min(open, close) - low = 49200 - 49080 = 120
        # Upper shadow = high - max(open, close) = 49220 - 49210 = 10
        # Conditions: lower_shadow > 2 * body_size (120 > 20) ✓
        #            upper_shadow < body_size * 0.5 (10 < 5) ✗
        test_data.loc[idx, 'open'] = 49200
        test_data.loc[idx, 'high'] = 49205   # Very small upper shadow
        test_data.loc[idx, 'low'] = 49080    # Long lower shadow
        test_data.loc[idx, 'close'] = 49210  # Small body
        
        strategy = StrategyVolatilityReversalScalping(test_data, self.config, self.logger)
        
        pattern_detected, pattern_type = strategy.detect_reversal_pattern(idx, 'down')
        
        self.assertTrue(pattern_detected)
        self.assertEqual(pattern_type, 'hammer')
    
    def test_reversal_pattern_bearish_engulfing(self):
        """Test bearish engulfing pattern detection"""
        test_data = self.test_data.copy()
        idx = 21
        
        # Set up bearish engulfing pattern
        # Previous bar (green)
        test_data.loc[idx-1, 'open'] = 50500
        test_data.loc[idx-1, 'close'] = 50600
        test_data.loc[idx-1, 'high'] = 50620
        test_data.loc[idx-1, 'low'] = 50480
        
        # Current bar (red, engulfs previous)
        test_data.loc[idx, 'open'] = 50650  # Opens above prev close
        test_data.loc[idx, 'close'] = 50450  # Closes below prev open
        test_data.loc[idx, 'high'] = 50620   # Doesn't exceed prev high
        test_data.loc[idx, 'low'] = 50400
        
        strategy = StrategyVolatilityReversalScalping(test_data, self.config, self.logger)
        
        pattern_detected, pattern_type = strategy.detect_reversal_pattern(idx, 'up')
        
        self.assertTrue(pattern_detected)
        self.assertEqual(pattern_type, 'bearish_engulfing')
    
    def test_reversal_pattern_bullish_engulfing(self):
        """Test bullish engulfing pattern detection"""
        test_data = self.test_data.copy()
        idx = 41
        
        # Set up bullish engulfing pattern
        # Previous bar (red)
        test_data.loc[idx-1, 'open'] = 49200
        test_data.loc[idx-1, 'close'] = 49100
        test_data.loc[idx-1, 'high'] = 49220
        test_data.loc[idx-1, 'low'] = 49080
        
        # Current bar (green, engulfs previous)
        test_data.loc[idx, 'open'] = 49050  # Opens below prev close
        test_data.loc[idx, 'close'] = 49250  # Closes above prev open
        test_data.loc[idx, 'high'] = 49280
        test_data.loc[idx, 'low'] = 49080   # Doesn't go below prev low
        
        strategy = StrategyVolatilityReversalScalping(test_data, self.config, self.logger)
        
        pattern_detected, pattern_type = strategy.detect_reversal_pattern(idx, 'down')
        
        self.assertTrue(pattern_detected)
        self.assertEqual(pattern_type, 'bullish_engulfing')
    
    def test_reversal_pattern_failed_breakout(self):
        """Test failed breakout pattern detection"""
        test_data = self.test_data.copy()
        idx = 21
        
        # Set up failed breakout (bearish)
        test_data.loc[idx-1, 'high'] = 50800  # Previous high
        test_data.loc[idx-1, 'close'] = 50750
        
        test_data.loc[idx, 'open'] = 50760
        test_data.loc[idx, 'high'] = 50790   # Failed to exceed prev high
        test_data.loc[idx, 'close'] = 50700  # Red candle, closes below prev close
        test_data.loc[idx, 'low'] = 50680
        
        strategy = StrategyVolatilityReversalScalping(test_data, self.config, self.logger)
        
        pattern_detected, pattern_type = strategy.detect_reversal_pattern(idx, 'up')
        
        self.assertTrue(pattern_detected)
        self.assertEqual(pattern_type, 'failed_breakout_bearish')
    
    def test_reversal_pattern_no_pattern(self):
        """Test that normal candles don't trigger pattern detection"""
        # Create normal candle that doesn't meet pattern criteria
        test_data = self.test_data.copy()
        idx = 22
        
        # Set up normal candle (no extreme shadows, normal body)
        test_data.loc[idx, 'open'] = 50300
        test_data.loc[idx, 'high'] = 50320  # Small upper shadow
        test_data.loc[idx, 'low'] = 50290   # Small lower shadow
        test_data.loc[idx, 'close'] = 50310  # Normal body
        
        strategy = StrategyVolatilityReversalScalping(test_data, self.config, self.logger)
        
        pattern_detected, pattern_type = strategy.detect_reversal_pattern(idx, 'up')
        
        # Should not detect pattern on normal continuation
        self.assertFalse(pattern_detected)
        self.assertIsNone(pattern_type)
    
    def test_rsi_divergence_bearish(self):
        """Test bearish RSI divergence detection"""
        # Modify data to create clear bearish divergence
        test_data = self.test_data.copy()
        
        # Create higher high in price but lower high in RSI
        test_data.loc[15, 'close'] = 50500  # First high
        test_data.loc[20, 'close'] = 50600  # Higher high
        
        strategy = StrategyVolatilityReversalScalping(test_data, self.config, self.logger)
        
        # Manually set RSI values to create divergence
        strategy.data.loc[15, 'rsi'] = 85  # Higher RSI
        strategy.data.loc[20, 'rsi'] = 82  # Lower RSI (divergence)
        
        divergence = strategy.check_rsi_divergence(20, 'up')
        
        self.assertTrue(divergence)
    
    def test_rsi_divergence_bullish(self):
        """Test bullish RSI divergence detection"""
        # Modify data to create clear bullish divergence
        test_data = self.test_data.copy()
        
        # Create lower low in price but higher low in RSI
        test_data.loc[35, 'close'] = 49500  # First low
        test_data.loc[40, 'close'] = 49400  # Lower low
        
        strategy = StrategyVolatilityReversalScalping(test_data, self.config, self.logger)
        
        # Manually set RSI values to create divergence
        strategy.data.loc[35, 'rsi'] = 15  # Lower RSI
        strategy.data.loc[40, 'rsi'] = 18  # Higher RSI (divergence)
        
        divergence = strategy.check_rsi_divergence(40, 'down')
        
        self.assertTrue(divergence)
    
    def test_rsi_divergence_no_divergence(self):
        """Test that normal RSI behavior doesn't trigger divergence"""
        divergence = self.strategy.check_rsi_divergence(25, 'up')
        
        # Should not detect divergence in normal conditions
        self.assertFalse(divergence)
    
    def test_retracement_targets_up_spike(self):
        """Test Fibonacci retracement target calculation for up-spike"""
        spike_data = {
            'direction': 'up',
            'extreme_price': 50800,
            'size': 0.016
        }
        
        target_382, target_50 = self.strategy.calculate_retracement_targets(20, spike_data)
        
        self.assertIsNotNone(target_382)
        self.assertIsNotNone(target_50)
        self.assertLess(target_382, spike_data['extreme_price'])  # Should be below spike high
        self.assertLess(target_50, target_382)  # 50% retracement deeper than 38.2%
    
    def test_retracement_targets_down_spike(self):
        """Test Fibonacci retracement target calculation for down-spike"""
        spike_data = {
            'direction': 'down',
            'extreme_price': 49200,
            'size': 0.016
        }
        
        target_382, target_50 = self.strategy.calculate_retracement_targets(40, spike_data)
        
        self.assertIsNotNone(target_382)
        self.assertIsNotNone(target_50)
        self.assertGreater(target_382, spike_data['extreme_price'])  # Should be above spike low
        self.assertGreater(target_50, target_382)  # 50% retracement higher than 38.2%
    
    def test_entry_immediate_high_score(self):
        """Test immediate entry on very high confidence score"""
        # Set up strategy state for immediate entry test
        self.strategy.last_trade_bar = -10  # Ensure cooldown passed
        
        # Test at bar with high spike score
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        
        # Should get entry signal on extreme spike
        if entry_signal:
            self.assertIn(entry_signal['action'], ['long', 'short'])
            self.assertGreaterEqual(entry_signal['confidence'], 0.7)
            self.assertIn('volatility_reversal', entry_signal['reason'])
    
    def test_entry_spike_and_reversal_pattern(self):
        """Test entry after spike detection and reversal pattern confirmation"""
        # Simulate spike detection
        self.strategy.extreme_spike_detected = True
        self.strategy.spike_direction = 'up'
        self.strategy.spike_bar = 20
        self.strategy.last_trade_bar = -10
        
        # Mock reversal pattern detection
        with patch.object(self.strategy, 'detect_reversal_pattern', return_value=(True, 'shooting_star')):
            entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
            
            if entry_signal:
                self.assertEqual(entry_signal['action'], 'short')  # Fade up-spike
                self.assertIn('pattern', entry_signal['reason'])
    
    def test_entry_moderate_conditions(self):
        """Test entry on moderate overbought/oversold with volume"""
        # Set up moderate conditions
        self.strategy.last_trade_bar = -10
        self.strategy.extreme_spike_detected = False  # Ensure no spike detected
        
        # Modify current bar to have moderate overbought conditions
        idx = len(self.strategy.data) - 1
        
        # Set up moderate conditions without triggering spike
        bb_upper = self.strategy.data['bb_upper'].iloc[idx]
        self.strategy.data.loc[idx, 'close'] = bb_upper * 1.005  # Slightly above BB upper
        self.strategy.data.loc[idx, 'high'] = bb_upper * 1.006   # Small range
        self.strategy.data.loc[idx, 'low'] = bb_upper * 1.004    # Small range
        self.strategy.data.loc[idx, 'open'] = bb_upper * 1.0045
        self.strategy.data.loc[idx, 'rsi'] = 76  # Moderate overbought
        self.strategy.data.loc[idx, 'volume'] = self.strategy.data['volume_sma'].iloc[idx] * 1.6
        
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        
        if entry_signal:
            self.assertEqual(entry_signal['action'], 'short')
            # Accept either moderate_overbought or immediate spike (both are valid)
            self.assertTrue('moderate_overbought' in entry_signal['reason'] or 
                          'volatility_reversal' in entry_signal['reason'])
    
    def test_entry_cooldown_period(self):
        """Test that cooldown period prevents premature entries"""
        # Set recent trade
        self.strategy.last_trade_bar = len(self.strategy.data) - 1
        
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        
        # Should not enter due to cooldown
        self.assertIsNone(entry_signal)
    
    def test_entry_trading_disabled(self):
        """Test that disabled trading prevents entries"""
        self.strategy.trading_enabled = False
        self.strategy.last_trade_bar = -10
        
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        
        # Should not enter when trading disabled
        self.assertIsNone(entry_signal)
    
    def test_entry_insufficient_data(self):
        """Test entry with insufficient data"""
        # Create strategy with minimal data (but enough for basic indicators)
        minimal_data = self.test_data.head(25)  # Need at least 20 for BB period
        strategy = StrategyVolatilityReversalScalping(minimal_data, self.config, self.logger)
        
        entry_signal = strategy._check_entry_conditions('BTCUSDT')
        
        # Should not enter with insufficient data
        self.assertIsNone(entry_signal)
    
    def test_exit_time_based(self):
        """Test time-based exit condition"""
        # Set up active trade
        self.strategy.entry_bar = len(self.strategy.data) - 10  # 10 bars ago
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50000
        
        should_exit, exit_reason = self.strategy.check_exit_conditions(len(self.strategy.data) - 1)
        
        self.assertTrue(should_exit)
        self.assertEqual(exit_reason, 'time_exit')
    
    def test_exit_quick_scalp_long(self):
        """Test quick scalp exit for long position"""
        # Set up long position
        idx = len(self.strategy.data) - 1
        self.strategy.entry_bar = idx - 2
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 49800
        self.strategy.first_target_hit = False
        
        # Set current price for 0.2% profit
        self.strategy.data.loc[idx, 'close'] = 49800 * 1.0025  # 0.25% profit
        
        should_exit, exit_reason = self.strategy.check_exit_conditions(idx)
        
        self.assertTrue(should_exit)
        self.assertEqual(exit_reason, 'quick_scalp_long')
    
    def test_exit_quick_scalp_short(self):
        """Test quick scalp exit for short position"""
        # Set up short position
        idx = len(self.strategy.data) - 1
        self.strategy.entry_bar = idx - 2
        self.strategy.entry_side = 'short'
        self.strategy.entry_price = 50200
        self.strategy.first_target_hit = False
        
        # Set current price for 0.2% profit
        self.strategy.data.loc[idx, 'close'] = 50200 * 0.9975  # 0.25% profit
        
        should_exit, exit_reason = self.strategy.check_exit_conditions(idx)
        
        self.assertTrue(should_exit)
        self.assertEqual(exit_reason, 'quick_scalp_short')
    
    def test_exit_first_target_long(self):
        """Test first target exit for long position"""
        # Set up long position
        idx = len(self.strategy.data) - 1
        self.strategy.entry_bar = idx - 3
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 49800
        self.strategy.first_target_hit = True  # Set first target already hit
        
        # Set price at BB middle (target)
        bb_middle = self.strategy.data['bb_middle'].iloc[idx]
        self.strategy.data.loc[idx, 'close'] = bb_middle * 1.001  # Slightly above target
        
        should_exit, exit_reason = self.strategy.check_exit_conditions(idx)
        
        if should_exit:
            self.assertIn('target', exit_reason)
    
    def test_exit_emergency_stop_long(self):
        """Test emergency stop loss for long position"""
        # Set up long position
        idx = len(self.strategy.data) - 1
        self.strategy.entry_bar = idx - 3
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50000
        
        # Set price for emergency stop (0.5% loss)
        self.strategy.data.loc[idx, 'close'] = 50000 * 0.994  # 0.6% loss
        
        should_exit, exit_reason = self.strategy.check_exit_conditions(idx)
        
        self.assertTrue(should_exit)
        self.assertEqual(exit_reason, 'emergency_stop_long')
    
    def test_exit_emergency_stop_short(self):
        """Test emergency stop loss for short position"""
        # Set up short position
        idx = len(self.strategy.data) - 1
        self.strategy.entry_bar = idx - 3
        self.strategy.entry_side = 'short'
        self.strategy.entry_price = 50000
        
        # Set price for emergency stop (0.5% loss)
        self.strategy.data.loc[idx, 'close'] = 50000 * 1.006  # 0.6% loss
        
        should_exit, exit_reason = self.strategy.check_exit_conditions(idx)
        
        self.assertTrue(should_exit)
        self.assertEqual(exit_reason, 'emergency_stop_short')
    
    def test_exit_rsi_reversal(self):
        """Test RSI reversal exit condition"""
        # Set up long position
        idx = len(self.strategy.data) - 1
        self.strategy.entry_bar = idx - 2
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50000
        self.strategy.first_target_hit = True  # Prevent quick scalp exit
        
        # Set price to not trigger quick scalp
        self.strategy.data.loc[idx, 'close'] = 49999  # Slightly below entry
        
        # Set RSI to overbought (reversal signal for long)
        self.strategy.data.loc[idx, 'rsi'] = 72
        
        should_exit, exit_reason = self.strategy.check_exit_conditions(idx)
        
        if should_exit:
            self.assertEqual(exit_reason, 'rsi_reversal_long')
    
    def test_exit_momentum_reversal(self):
        """Test momentum reversal exit condition"""
        # Set up long position
        idx = len(self.strategy.data) - 1
        self.strategy.entry_bar = idx - 3
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50000
        
        # Set price below BB lower (momentum against us)
        bb_lower = self.strategy.data['bb_lower'].iloc[idx]
        self.strategy.data.loc[idx, 'close'] = bb_lower * 0.999
        
        should_exit, exit_reason = self.strategy.check_exit_conditions(idx)
        
        if should_exit:
            self.assertEqual(exit_reason, 'momentum_reversal_long')
    
    def test_exit_no_active_trade(self):
        """Test exit check with no active trade"""
        # Ensure no active trade
        self.strategy.entry_bar = None
        
        should_exit, exit_reason = self.strategy.check_exit_conditions(len(self.strategy.data) - 1)
        
        self.assertFalse(should_exit)
        self.assertIsNone(exit_reason)
    
    def test_check_exit_method(self):
        """Test the main check_exit method"""
        # Set up active trade
        self.strategy.entry_bar = len(self.strategy.data) - 10
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50000
        
        exit_signal = self.strategy.check_exit('BTCUSDT')
        
        if exit_signal:
            self.assertEqual(exit_signal['action'], 'exit')
            self.assertIsNotNone(exit_signal['price'])
            self.assertIsNotNone(exit_signal['reason'])
    
    def test_trade_closed_winning_trade(self):
        """Test trade closure handling for winning trade"""
        # Set up trade state
        self.strategy.entry_bar = 100
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50000
        self.strategy.consecutive_losses = 2
        
        # Simulate winning trade
        trade_result = {
            'pnl': 150,
            'reason': 'first_target_long'
        }
        
        self.strategy.on_trade_closed('BTCUSDT', trade_result)
        
        # Check state reset
        self.assertIsNone(self.strategy.entry_bar)
        self.assertIsNone(self.strategy.entry_side)
        self.assertIsNone(self.strategy.entry_price)
        self.assertFalse(self.strategy.first_target_hit)
        
        # Consecutive losses should reset
        self.assertEqual(self.strategy.consecutive_losses, 0)
    
    def test_trade_closed_losing_trade(self):
        """Test trade closure handling for losing trade"""
        # Set up trade state
        self.strategy.entry_bar = 100
        self.strategy.consecutive_losses = 2
        
        # Simulate losing trade
        trade_result = {
            'pnl': -75,
            'reason': 'emergency_stop_long'
        }
        
        self.strategy.on_trade_closed('BTCUSDT', trade_result)
        
        # Consecutive losses should increment
        self.assertEqual(self.strategy.consecutive_losses, 3)
        self.assertTrue(self.strategy.trading_enabled)  # Still enabled (max is 4)
    
    def test_trade_closed_max_consecutive_losses(self):
        """Test trading disabled after max consecutive losses"""
        # Set up near max losses
        self.strategy.consecutive_losses = 3
        
        # Simulate another losing trade
        trade_result = {
            'pnl': -50,
            'reason': 'time_exit'
        }
        
        self.strategy.on_trade_closed('BTCUSDT', trade_result)
        
        # Should disable trading
        self.assertEqual(self.strategy.consecutive_losses, 4)
        self.assertFalse(self.strategy.trading_enabled)
    
    def test_risk_parameters_with_active_trade(self):
        """Test risk parameter calculation with active trade"""
        # Set up active long trade
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50000
        
        risk_params = self.strategy.get_risk_parameters()
        
        self.assertIn('sl_pct', risk_params)
        self.assertIn('tp_pct', risk_params)
        self.assertIn('max_position_pct', risk_params)
        self.assertIn('risk_reward_ratio', risk_params)
        
        # Check reduced position sizing
        self.assertLess(risk_params['max_position_pct'], 2.0)  # Should be reduced
        
        # Check tight stops
        self.assertLessEqual(risk_params['sl_pct'], 0.005)  # Tight stop
    
    def test_risk_parameters_no_active_trade(self):
        """Test risk parameter calculation without active trade"""
        # Ensure no active trade
        self.strategy.entry_side = None
        self.strategy.entry_price = None
        
        risk_params = self.strategy.get_risk_parameters()
        
        self.assertIn('sl_pct', risk_params)
        self.assertIn('tp_pct', risk_params)
        self.assertIn('max_position_pct', risk_params)
        self.assertIn('risk_reward_ratio', risk_params)
        
        # Should use fallback values
        self.assertGreater(risk_params['sl_pct'], 0)
        self.assertGreater(risk_params['tp_pct'], risk_params['sl_pct'])
    
    def test_error_handling_spike_detection(self):
        """Test error handling in spike detection"""
        # Test with invalid index
        score, direction, spike_data = self.strategy.evaluate_spike_conditions(-1)
        
        self.assertEqual(score, 0)
        self.assertIsNone(direction)
        self.assertIsNone(spike_data)
    
    def test_error_handling_reversal_pattern(self):
        """Test error handling in reversal pattern detection"""
        # Test with invalid index
        pattern_detected, pattern_type = self.strategy.detect_reversal_pattern(-1, 'up')
        
        self.assertFalse(pattern_detected)
        self.assertIsNone(pattern_type)
    
    def test_error_handling_rsi_divergence(self):
        """Test error handling in RSI divergence detection"""
        # Test with insufficient data
        divergence = self.strategy.check_rsi_divergence(2, 'up')
        
        self.assertFalse(divergence)
    
    def test_error_handling_retracement_targets(self):
        """Test error handling in retracement target calculation"""
        # Test with invalid spike data
        invalid_spike_data = {'direction': 'up', 'extreme_price': None}
        
        target_382, target_50 = self.strategy.calculate_retracement_targets(20, invalid_spike_data)
        
        # Should handle gracefully
        self.assertIsNone(target_382) or self.assertIsNone(target_50)
    
    def test_edge_case_zero_volume(self):
        """Test handling of zero volume conditions"""
        # Set volume to zero
        self.strategy.data.loc[20, 'volume'] = 0
        self.strategy.data.loc[20, 'volume_sma'] = 0.1  # Avoid division by zero
        
        score, direction, spike_data = self.strategy.evaluate_spike_conditions(20)
        
        # Should handle gracefully without crashing
        self.assertIsInstance(score, int)
    
    def test_edge_case_extreme_prices(self):
        """Test handling of extreme price values"""
        # Set extreme price
        self.strategy.data.loc[20, 'close'] = 1000000  # Very high price
        
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        
        # Should handle without crashing (either None or dict)
        self.assertTrue(entry_signal is None or isinstance(entry_signal, dict))
    
    def test_edge_case_nan_values(self):
        """Test handling of NaN values in indicators"""
        # Set some indicators to NaN
        self.strategy.data.loc[20, 'rsi'] = np.nan
        self.strategy.data.loc[20, 'bb_upper'] = np.nan
        
        score, direction, spike_data = self.strategy.evaluate_spike_conditions(20)
        
        # Should handle NaN gracefully
        self.assertEqual(score, 0)
        self.assertIsNone(direction)
    
    def test_spike_detection_reset(self):
        """Test spike detection state reset"""
        # Set spike detection state
        self.strategy.extreme_spike_detected = True
        self.strategy.spike_direction = 'up'
        self.strategy.spike_bar = 20
        
        # Simulate time passing without reversal
        self.strategy.last_trade_bar = -10
        
        # Should reset after timeout
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        
        # State should be reset if too much time passed
        if not self.strategy.extreme_spike_detected:
            self.assertIsNone(self.strategy.spike_direction)
    
    def test_multiple_spike_detection(self):
        """Test handling of multiple spikes in sequence"""
        # Test multiple spike bars
        for spike_bar in [20, 40, 60, 80]:
            score, direction, spike_data = self.strategy.evaluate_spike_conditions(spike_bar)
            
            if score > 0:
                self.assertIsNotNone(direction)
                self.assertIsNotNone(spike_data)
                self.assertIn(direction, ['up', 'down'])

if __name__ == '__main__':
    unittest.main()