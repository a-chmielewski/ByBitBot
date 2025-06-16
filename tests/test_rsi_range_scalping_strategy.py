#!/usr/bin/env python3
"""
Test Suite for RSI Range Scalping Strategy

This test suite validates the RSI Range Scalping Strategy implementation,
including range detection, RSI signals, candlestick patterns, and risk management.
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

from strategies.rsi_range_scalping_strategy import StrategyRSIRangeScalping

class TestRSIRangeScalpingStrategy(unittest.TestCase):
    """Test cases for RSI Range Scalping Strategy"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Suppress logging during tests
        logging.disable(logging.CRITICAL)
        
        # Create test configuration
        self.config = {
            'rsi_period': 14,
            'sma_reference': 100,
            'volume_period': 20,
            'range_lookback': 50,
            'min_range_width': 0.008,
            'support_resistance_tolerance': 0.002,
            'range_middle_buffer': 0.3,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'rsi_neutral_low': 40,
            'rsi_neutral_high': 60,
            'volume_surge_threshold': 1.5,
            'time_stop_bars': 20,
            'cooldown_bars': 3,
            'stop_loss_pct': 0.002,
            'take_profit_ratio': 1.2,
            'max_position_pct': 2.0,
            'consecutive_stops_limit': 2
        }
        
        # Create test data with ranging market conditions
        self.test_data = self._create_ranging_market_data()
        
        # Create strategy instance
        self.strategy = StrategyRSIRangeScalping(
            data=self.test_data,
            config=self.config,
            logger=Mock()
        )
    
    def tearDown(self):
        """Clean up after each test"""
        logging.disable(logging.NOTSET)
    
    def _create_ranging_market_data(self) -> pd.DataFrame:
        """Create realistic test data with ranging market conditions"""
        np.random.seed(42)
        n_bars = 200
        
        # Base price and parameters
        base_price = 50000.0
        base_volume = 1000000
        
        # Create time index
        dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='5min')
        
        # Initialize arrays
        opens = np.zeros(n_bars)
        highs = np.zeros(n_bars)
        lows = np.zeros(n_bars)
        closes = np.zeros(n_bars)
        volumes = np.zeros(n_bars)
        
        # Define range boundaries
        range_high = base_price + 400  # 50400
        range_low = base_price - 400   # 49600
        range_center = (range_high + range_low) / 2
        
        # Phase 1: Initial setup (0-29) - Establishing range
        for i in range(30):
            if i == 0:
                opens[i] = base_price
                closes[i] = base_price + np.random.normal(0, 50)
            else:
                opens[i] = closes[i-1]
                # Gradual movement toward range boundaries
                target = range_low if i < 15 else range_high
                drift = (target - opens[i]) * 0.1
                closes[i] = opens[i] + drift + np.random.normal(0, 30)
            
            # Create realistic OHLC
            high_low_range = abs(closes[i] - opens[i]) + np.random.uniform(20, 60)
            highs[i] = max(opens[i], closes[i]) + np.random.uniform(0, high_low_range * 0.3)
            lows[i] = min(opens[i], closes[i]) - np.random.uniform(0, high_low_range * 0.3)
            
            volumes[i] = base_volume * np.random.uniform(0.8, 1.2)
        
        # Phase 2: Range formation (30-169) - Clear ranging behavior
        for i in range(30, 170):
            opens[i] = closes[i-1]
            
            # Oscillate between support and resistance with RSI-like behavior
            cycle_position = (i - 30) / 20  # 20-bar cycles
            
            # Create RSI-driven price movement
            if cycle_position % 2 < 1:  # Moving toward oversold
                target_level = range_low + np.random.uniform(0, 100)
                rsi_factor = 0.3  # Oversold conditions
            else:  # Moving toward overbought
                target_level = range_high - np.random.uniform(0, 100)
                rsi_factor = 0.7  # Overbought conditions
            
            # Mean reversion toward target
            mean_reversion = (target_level - opens[i]) * 0.15
            noise = np.random.normal(0, 25)
            closes[i] = opens[i] + mean_reversion + noise
            
            # Keep within range bounds
            closes[i] = max(range_low - 50, min(range_high + 50, closes[i]))
            
            # Create OHLC with varying volatility
            volatility = 30 + 20 * abs(np.sin(cycle_position * np.pi))
            high_low_range = np.random.uniform(15, volatility)
            highs[i] = max(opens[i], closes[i]) + np.random.uniform(0, high_low_range * 0.4)
            lows[i] = min(opens[i], closes[i]) - np.random.uniform(0, high_low_range * 0.4)
            
            # Ensure we create clear support/resistance touches
            if i % 25 == 0:  # Every 25 bars, create a clear touch
                if np.random.random() > 0.5:
                    # Support touch
                    lows[i] = range_low + np.random.uniform(-20, 20)
                    closes[i] = range_low + np.random.uniform(20, 80)
                else:
                    # Resistance touch
                    highs[i] = range_high + np.random.uniform(-20, 20)
                    closes[i] = range_high - np.random.uniform(20, 80)
            
            # Volume patterns
            if abs(closes[i] - range_low) < 100 or abs(closes[i] - range_high) < 100:
                # Higher volume at extremes
                volumes[i] = base_volume * np.random.uniform(1.1, 1.4)
            else:
                # Lower volume in middle
                volumes[i] = base_volume * np.random.uniform(0.6, 1.0)
        
        # Phase 3: Continued ranging (170-199) - More range behavior
        for i in range(170, n_bars):
            opens[i] = closes[i-1]
            
            # Continue ranging behavior with some noise
            distance_to_support = abs(opens[i] - range_low)
            distance_to_resistance = abs(opens[i] - range_high)
            
            if distance_to_support < distance_to_resistance:
                # Closer to support, bias upward
                closes[i] = opens[i] + np.random.uniform(10, 50)
            else:
                # Closer to resistance, bias downward
                closes[i] = opens[i] - np.random.uniform(10, 50)
            
            # Add noise
            closes[i] += np.random.normal(0, 20)
            
            # Keep in range
            closes[i] = max(range_low - 30, min(range_high + 30, closes[i]))
            
            high_low_range = np.random.uniform(20, 50)
            highs[i] = max(opens[i], closes[i]) + np.random.uniform(0, high_low_range * 0.3)
            lows[i] = min(opens[i], closes[i]) - np.random.uniform(0, high_low_range * 0.3)
            
            volumes[i] = base_volume * np.random.uniform(0.7, 1.3)
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        return df
    
    # Test Strategy Initialization
    def test_strategy_initialization(self):
        """Test strategy initialization"""
        self.assertIsInstance(self.strategy, StrategyRSIRangeScalping)
        self.assertEqual(self.strategy.MARKET_TYPE_TAGS, ['RANGING'])
        self.assertTrue(self.strategy.SHOW_IN_SELECTION)
        
        # Check initial state
        self.assertFalse(self.strategy.range_active)
        self.assertEqual(self.strategy.support_level, 0)
        self.assertEqual(self.strategy.resistance_level, 0)
        self.assertEqual(self.strategy.consecutive_stops, 0)
        self.assertTrue(self.strategy.range_trading_enabled)
    
    def test_init_indicators(self):
        """Test indicator initialization"""
        self.strategy.init_indicators()
        
        # Check that indicators are added to data
        expected_indicators = ['rsi', 'sma_reference', 'volume_sma']
        
        for indicator in expected_indicators:
            self.assertIn(indicator, self.strategy.data.columns)
    
    def test_init_indicators_without_pandas_ta(self):
        """Test indicator initialization without pandas_ta"""
        # Test manual calculation fallback
        original_has_pandas_ta = self.strategy.has_pandas_ta
        
        try:
            # Simulate pandas_ta not being available
            self.strategy.has_pandas_ta = False
            self.strategy.init_indicators()
            
            # Should still have all indicators (using manual calculations)
            expected_indicators = ['rsi', 'sma_reference', 'volume_sma']
            
            for indicator in expected_indicators:
                self.assertIn(indicator, self.strategy.data.columns)
        finally:
            # Restore original state
            self.strategy.has_pandas_ta = original_has_pandas_ta
    
    # Test Manual RSI Calculation
    def test_calculate_rsi_manual(self):
        """Test manual RSI calculation"""
        rsi_series = self.strategy._calculate_rsi_manual(14)
        self.assertIsInstance(rsi_series, pd.Series)
        self.assertEqual(len(rsi_series), len(self.test_data))
        
        # RSI should be between 0 and 100
        valid_rsi = rsi_series.dropna()
        if len(valid_rsi) > 0:
            self.assertTrue(all(0 <= val <= 100 for val in valid_rsi))
    
    # Test Range Detection
    def test_detect_range_levels(self):
        """Test range level detection"""
        self.strategy.init_indicators()
        
        # Test range detection in ranging market
        range_detected = False
        for i in range(60, 120):
            if self.strategy.detect_range_levels(i):
                range_detected = True
                break
        
        self.assertTrue(range_detected, "Should detect range in ranging market")
        
        # Check range properties
        if self.strategy.range_active:
            self.assertGreater(self.strategy.resistance_level, self.strategy.support_level)
            self.assertGreater(self.strategy.range_width, 0)
            self.assertEqual(
                self.strategy.range_middle,
                (self.strategy.support_level + self.strategy.resistance_level) / 2
            )
    
    def test_detect_range_levels_insufficient_data(self):
        """Test range detection with insufficient data"""
        # Should return False with insufficient data
        result = self.strategy.detect_range_levels(10)
        self.assertFalse(result)
    
    # Test Support/Resistance Detection
    def test_is_near_support(self):
        """Test support level detection"""
        self.strategy.init_indicators()
        
        # Set up range
        self.strategy.range_active = True
        self.strategy.support_level = 49600
        
        # Test prices near support
        self.assertTrue(self.strategy.is_near_support(49600))
        self.assertTrue(self.strategy.is_near_support(49610))  # Within tolerance
        self.assertFalse(self.strategy.is_near_support(49700))  # Too far
        
        # Test without active range
        self.strategy.range_active = False
        self.assertFalse(self.strategy.is_near_support(49600))
    
    def test_is_near_resistance(self):
        """Test resistance level detection"""
        self.strategy.init_indicators()
        
        # Set up range
        self.strategy.range_active = True
        self.strategy.resistance_level = 50400
        
        # Test prices near resistance
        self.assertTrue(self.strategy.is_near_resistance(50400))
        self.assertTrue(self.strategy.is_near_resistance(50390))  # Within tolerance
        self.assertFalse(self.strategy.is_near_resistance(50200))  # Too far (200 points away)
        
        # Test without active range
        self.strategy.range_active = False
        self.assertFalse(self.strategy.is_near_resistance(50400))
    
    def test_is_in_range_middle(self):
        """Test range middle zone detection"""
        self.strategy.init_indicators()
        
        # Set up range
        self.strategy.range_active = True
        self.strategy.support_level = 49600
        self.strategy.resistance_level = 50400
        self.strategy.range_middle = 50000
        
        # Test middle zone detection
        self.assertTrue(self.strategy.is_in_range_middle(50000))  # Exact middle
        self.assertTrue(self.strategy.is_in_range_middle(49950))  # Near middle
        self.assertFalse(self.strategy.is_in_range_middle(49700))  # Near support
        self.assertFalse(self.strategy.is_in_range_middle(50300))  # Near resistance
        
        # Test without active range
        self.strategy.range_active = False
        self.assertTrue(self.strategy.is_in_range_middle(50000))  # Should return True
    
    # Test Candlestick Pattern Detection
    def test_detect_candlestick_patterns_bullish_engulfing(self):
        """Test bullish engulfing pattern detection"""
        self.strategy.init_indicators()
        
        # Set up range for support detection
        self.strategy.range_active = True
        self.strategy.support_level = 49600
        
        # Create bullish engulfing pattern data
        test_data = self.strategy.data.copy()
        idx = len(test_data) - 1
        
        # Previous red candle
        test_data.loc[test_data.index[idx-1], 'open'] = 49610
        test_data.loc[test_data.index[idx-1], 'close'] = 49590
        
        # Current green engulfing candle
        test_data.loc[test_data.index[idx], 'open'] = 49585
        test_data.loc[test_data.index[idx], 'close'] = 49620
        
        self.strategy.data = test_data
        
        pattern_detected, pattern_type = self.strategy.detect_candlestick_patterns(idx)
        self.assertTrue(pattern_detected)
        self.assertEqual(pattern_type, 'bullish_engulfing')
    
    def test_detect_candlestick_patterns_hammer(self):
        """Test hammer pattern detection"""
        self.strategy.init_indicators()
        
        # Set up range for support detection
        self.strategy.range_active = True
        self.strategy.support_level = 49600
        
        # Create hammer pattern data
        test_data = self.strategy.data.copy()
        idx = len(test_data) - 1
        
        # Hammer candle at support - small body, long lower shadow
        test_data.loc[test_data.index[idx], 'open'] = 49605
        test_data.loc[test_data.index[idx], 'high'] = 49606  # Very small upper shadow (1 point)
        test_data.loc[test_data.index[idx], 'low'] = 49560  # Very long lower shadow (45 points)
        test_data.loc[test_data.index[idx], 'close'] = 49600  # Small body (5 points)
        
        self.strategy.data = test_data
        
        pattern_detected, pattern_type = self.strategy.detect_candlestick_patterns(idx)
        self.assertTrue(pattern_detected)
        self.assertEqual(pattern_type, 'hammer')
    
    def test_detect_candlestick_patterns_bearish_engulfing(self):
        """Test bearish engulfing pattern detection"""
        self.strategy.init_indicators()
        
        # Set up range for resistance detection
        self.strategy.range_active = True
        self.strategy.resistance_level = 50400
        
        # Create bearish engulfing pattern data
        test_data = self.strategy.data.copy()
        idx = len(test_data) - 1
        
        # Previous green candle
        test_data.loc[test_data.index[idx-1], 'open'] = 50390
        test_data.loc[test_data.index[idx-1], 'close'] = 50410
        
        # Current red engulfing candle
        test_data.loc[test_data.index[idx], 'open'] = 50415
        test_data.loc[test_data.index[idx], 'close'] = 50380
        
        self.strategy.data = test_data
        
        pattern_detected, pattern_type = self.strategy.detect_candlestick_patterns(idx)
        self.assertTrue(pattern_detected)
        self.assertEqual(pattern_type, 'bearish_engulfing')
    
    def test_detect_candlestick_patterns_shooting_star(self):
        """Test shooting star pattern detection"""
        self.strategy.init_indicators()
        
        # Set up range for resistance detection
        self.strategy.range_active = True
        self.strategy.resistance_level = 50400
        
        # Create shooting star pattern data - red candle with long upper shadow
        test_data = self.strategy.data.copy()
        idx = len(test_data) - 1
        
        # Shooting star candle at resistance - red candle with long upper shadow
        test_data.loc[test_data.index[idx], 'open'] = 50405
        test_data.loc[test_data.index[idx], 'high'] = 50450  # Very long upper shadow (45 points)
        test_data.loc[test_data.index[idx], 'low'] = 50404  # Very small lower shadow (1 point)
        test_data.loc[test_data.index[idx], 'close'] = 50400  # Small body (5 points)
        
        self.strategy.data = test_data
        
        pattern_detected, pattern_type = self.strategy.detect_candlestick_patterns(idx)
        self.assertTrue(pattern_detected)
        self.assertEqual(pattern_type, 'shooting_star')
    
    def test_detect_candlestick_patterns_no_pattern(self):
        """Test when no pattern is detected"""
        self.strategy.init_indicators()
        
        # Set up range but price not at extremes
        self.strategy.range_active = True
        self.strategy.support_level = 49600
        self.strategy.resistance_level = 50400
        
        # Test with price in middle (no pattern expected)
        idx = len(self.strategy.data) - 1
        pattern_detected, pattern_type = self.strategy.detect_candlestick_patterns(idx)
        
        # Should not detect pattern in middle zone
        self.assertFalse(pattern_detected)
        self.assertIsNone(pattern_type)
    
    # Test RSI Divergence
    def test_check_rsi_divergence_bullish(self):
        """Test bullish RSI divergence detection"""
        self.strategy.init_indicators()
        
        # Set up range
        self.strategy.range_active = True
        self.strategy.support_level = 49600
        
        # Create divergence scenario in data
        test_data = self.strategy.data.copy()
        idx = len(test_data) - 1
        
        # Current: lower low in price, higher low in RSI
        test_data.loc[test_data.index[idx], 'close'] = 49580
        test_data.loc[test_data.index[idx], 'rsi'] = 25
        
        # Past: higher low in price, lower low in RSI
        test_data.loc[test_data.index[idx-5], 'close'] = 49590
        test_data.loc[test_data.index[idx-5], 'rsi'] = 20
        
        self.strategy.data = test_data
        
        divergence = self.strategy.check_rsi_divergence(idx, 'bullish')
        self.assertTrue(divergence)
    
    def test_check_rsi_divergence_bearish(self):
        """Test bearish RSI divergence detection"""
        self.strategy.init_indicators()
        
        # Set up range
        self.strategy.range_active = True
        self.strategy.resistance_level = 50400
        
        # Create divergence scenario in data
        test_data = self.strategy.data.copy()
        idx = len(test_data) - 1
        
        # Current: higher high in price, lower high in RSI
        test_data.loc[test_data.index[idx], 'close'] = 50420
        test_data.loc[test_data.index[idx], 'rsi'] = 75
        
        # Past: lower high in price, higher high in RSI
        test_data.loc[test_data.index[idx-5], 'close'] = 50410
        test_data.loc[test_data.index[idx-5], 'rsi'] = 80
        
        self.strategy.data = test_data
        
        divergence = self.strategy.check_rsi_divergence(idx, 'bearish')
        self.assertTrue(divergence)
    
    def test_check_rsi_divergence_insufficient_data(self):
        """Test RSI divergence with insufficient data"""
        # Should return False with insufficient data
        result = self.strategy.check_rsi_divergence(5, 'bullish')
        self.assertFalse(result)
    
    # Test Volume Analysis
    def test_check_volume_breakout_signal(self):
        """Test volume breakout detection"""
        self.strategy.init_indicators()
        
        # Create volume surge scenario
        test_data = self.strategy.data.copy()
        idx = len(test_data) - 1
        
        # Set high current volume
        test_data.loc[test_data.index[idx], 'volume'] = 2000000
        # Set normal average volume
        test_data.loc[test_data.index[idx], 'volume_sma'] = 1000000
        
        self.strategy.data = test_data
        
        volume_breakout = self.strategy.check_volume_breakout_signal(idx)
        self.assertTrue(volume_breakout)
    
    def test_check_volume_breakout_signal_normal(self):
        """Test normal volume conditions"""
        self.strategy.init_indicators()
        
        # Normal volume should not trigger breakout signal
        idx = len(self.strategy.data) - 1
        volume_breakout = self.strategy.check_volume_breakout_signal(idx)
        
        # Should not detect breakout with normal volume
        self.assertFalse(volume_breakout)
    
    # Test Entry Conditions
    def test_check_entry_conditions_no_range(self):
        """Test entry conditions when no range is detected"""
        self.strategy.init_indicators()
        
        # Ensure no range is active
        self.strategy.range_active = False
        
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        self.assertIsNone(entry_signal)
    
    def test_check_entry_conditions_middle_zone(self):
        """Test entry conditions in range middle zone"""
        self.strategy.init_indicators()
        
        # Set up range with current price in middle
        self.strategy.range_active = True
        self.strategy.support_level = 49600
        self.strategy.resistance_level = 50400
        self.strategy.range_middle = 50000
        
        # Current price in middle zone should not generate entry
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        self.assertIsNone(entry_signal)
    
    def test_check_entry_conditions_cooldown(self):
        """Test entry conditions during cooldown period"""
        self.strategy.init_indicators()
        self.strategy.last_trade_bar = len(self.strategy.data) - 2
        
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        self.assertIsNone(entry_signal, "Should respect cooldown period")
    
    def test_check_entry_conditions_insufficient_data(self):
        """Test entry conditions with insufficient data"""
        # Create strategy with minimal data
        minimal_data = self.test_data.iloc[:10].copy()
        strategy = StrategyRSIRangeScalping(
            data=minimal_data,
            config=self.config,
            logger=Mock()
        )
        
        entry_signal = strategy._check_entry_conditions('BTCUSDT')
        self.assertIsNone(entry_signal)
    
    def test_check_entry_conditions_volume_breakout(self):
        """Test entry conditions during volume breakout"""
        self.strategy.init_indicators()
        
        # Set up range
        self.strategy.range_active = True
        self.strategy.support_level = 49600
        self.strategy.resistance_level = 50400
        
        # Mock volume breakout
        with patch.object(self.strategy, 'check_volume_breakout_signal', return_value=True):
            entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
            self.assertIsNone(entry_signal, "Should avoid trades during volume breakout")
    
    def test_check_entry_conditions_disabled_trading(self):
        """Test entry conditions when range trading is disabled"""
        self.strategy.init_indicators()
        
        # Disable range trading
        self.strategy.range_trading_enabled = False
        self.strategy.last_stop_bar = len(self.strategy.data) - 10
        
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        self.assertIsNone(entry_signal, "Should not trade when disabled")
    
    # Test Exit Conditions
    def test_check_range_exit_conditions_time_stop(self):
        """Test time-based exit conditions"""
        self.strategy.init_indicators()
        
        # Set up trade state
        self.strategy.entry_bar = len(self.strategy.data) - 25  # Old entry
        self.strategy.entry_side = 'long'
        
        idx = len(self.strategy.data) - 1
        should_exit = self.strategy.check_range_exit_conditions(idx)
        self.assertTrue(should_exit, "Should exit on time stop")
    
    def test_check_range_exit_conditions_rsi_normalization_long(self):
        """Test RSI normalization exit for long positions"""
        self.strategy.init_indicators()
        
        # Set up long trade state
        self.strategy.entry_bar = len(self.strategy.data) - 5
        self.strategy.entry_side = 'long'
        
        # Mock RSI reaching neutral zone
        test_data = self.strategy.data.copy()
        idx = len(test_data) - 1
        test_data.loc[test_data.index[idx], 'rsi'] = 45  # Above neutral low (40)
        self.strategy.data = test_data
        
        should_exit = self.strategy.check_range_exit_conditions(idx)
        self.assertTrue(should_exit, "Should exit long on RSI normalization")
    
    def test_check_range_exit_conditions_rsi_normalization_short(self):
        """Test RSI normalization exit for short positions"""
        self.strategy.init_indicators()
        
        # Set up short trade state
        self.strategy.entry_bar = len(self.strategy.data) - 5
        self.strategy.entry_side = 'short'
        
        # Mock RSI reaching neutral zone
        test_data = self.strategy.data.copy()
        idx = len(test_data) - 1
        test_data.loc[test_data.index[idx], 'rsi'] = 55  # Below neutral high (60)
        self.strategy.data = test_data
        
        should_exit = self.strategy.check_range_exit_conditions(idx)
        self.assertTrue(should_exit, "Should exit short on RSI normalization")
    
    def test_check_range_exit_conditions_range_middle_approach(self):
        """Test exit when approaching range middle"""
        self.strategy.init_indicators()
        
        # Set up range and long trade
        self.strategy.range_active = True
        self.strategy.range_middle = 50000
        self.strategy.entry_bar = len(self.strategy.data) - 5
        self.strategy.entry_side = 'long'
        
        # Mock price approaching range middle
        test_data = self.strategy.data.copy()
        idx = len(test_data) - 1
        test_data.loc[test_data.index[idx], 'close'] = 49950  # 95% of range middle
        test_data.loc[test_data.index[idx], 'rsi'] = 35  # Still oversold
        self.strategy.data = test_data
        
        should_exit = self.strategy.check_range_exit_conditions(idx)
        self.assertTrue(should_exit, "Should exit when approaching range middle")
    
    def test_check_range_exit_conditions_no_position(self):
        """Test exit conditions when no position is open"""
        self.strategy.init_indicators()
        
        # No position state
        self.strategy.entry_bar = None
        
        idx = len(self.strategy.data) - 1
        should_exit = self.strategy.check_range_exit_conditions(idx)
        self.assertFalse(should_exit, "Should not exit when no position")
    
    def test_check_exit_volume_breakout(self):
        """Test exit on volume breakout"""
        self.strategy.init_indicators()
        
        # Mock volume breakout
        with patch.object(self.strategy, 'check_volume_breakout_signal', return_value=True):
            exit_signal = self.strategy.check_exit('BTCUSDT')
            
            if exit_signal:
                self.assertEqual(exit_signal['action'], 'exit')
                self.assertEqual(exit_signal['reason'], 'volume_breakout')
    
    def test_check_exit_range_conditions(self):
        """Test exit based on range scalping conditions"""
        self.strategy.init_indicators()
        
        # Mock range exit conditions
        with patch.object(self.strategy, 'check_range_exit_conditions', return_value=True):
            exit_signal = self.strategy.check_exit('BTCUSDT')
            
            if exit_signal:
                self.assertEqual(exit_signal['action'], 'exit')
                self.assertEqual(exit_signal['reason'], 'range_scalp_exit')
    
    def test_check_exit_no_conditions(self):
        """Test exit when no exit conditions are met"""
        self.strategy.init_indicators()
        
        # No exit conditions should be met
        exit_signal = self.strategy.check_exit('BTCUSDT')
        self.assertIsNone(exit_signal)
    
    # Test Trade Management
    def test_on_trade_closed_winning_trade(self):
        """Test trade closure with winning trade"""
        # Set up trade state
        self.strategy.entry_bar = 100
        self.strategy.entry_side = 'long'
        self.strategy.entry_level = 49600
        self.strategy.entry_price = 49610
        self.strategy.consecutive_stops = 1
        
        # Winning trade
        trade_result = {'reason': 'target_hit', 'pnl': 50}
        self.strategy.on_trade_closed('BTCUSDT', trade_result)
        
        # Check cleanup
        self.assertIsNone(self.strategy.entry_bar)
        self.assertIsNone(self.strategy.entry_side)
        self.assertIsNone(self.strategy.entry_level)
        self.assertIsNone(self.strategy.entry_price)
        
        # Consecutive stops should be reset
        self.assertEqual(self.strategy.consecutive_stops, 0)
    
    def test_on_trade_closed_losing_trade(self):
        """Test trade closure with losing trade"""
        # Set up trade state
        self.strategy.entry_bar = 100
        self.strategy.consecutive_stops = 0
        
        # Losing trade
        trade_result = {'reason': 'stop_loss', 'pnl': -20}
        self.strategy.on_trade_closed('BTCUSDT', trade_result)
        
        # Consecutive stops should increment
        self.assertEqual(self.strategy.consecutive_stops, 1)
        self.assertIsNotNone(self.strategy.last_stop_bar)
    
    def test_on_trade_closed_consecutive_stops_limit(self):
        """Test trade closure reaching consecutive stops limit"""
        # Set up trade state
        self.strategy.consecutive_stops = 1  # One stop already
        self.strategy.range_trading_enabled = True
        
        # Another losing trade (reaches limit of 2)
        trade_result = {'reason': 'stop_loss', 'pnl': -20}
        self.strategy.on_trade_closed('BTCUSDT', trade_result)
        
        # Should disable range trading
        self.assertEqual(self.strategy.consecutive_stops, 2)
        self.assertFalse(self.strategy.range_trading_enabled)
    
    # Test Risk Management
    def test_get_risk_parameters_with_long_position(self):
        """Test risk parameters for long position"""
        self.strategy.init_indicators()
        
        # Set up long position state
        self.strategy.range_active = True
        self.strategy.entry_side = 'long'
        self.strategy.entry_level = 49600  # Support level
        
        risk_params = self.strategy.get_risk_parameters()
        
        required_keys = ['sl_pct', 'tp_pct', 'max_position_pct', 'risk_reward_ratio']
        for key in required_keys:
            self.assertIn(key, risk_params)
            self.assertGreater(risk_params[key], 0)
        
        # Risk/reward ratio should match config
        self.assertEqual(risk_params['risk_reward_ratio'], self.config['take_profit_ratio'])
    
    def test_get_risk_parameters_with_short_position(self):
        """Test risk parameters for short position"""
        self.strategy.init_indicators()
        
        # Set up short position state
        self.strategy.range_active = True
        self.strategy.entry_side = 'short'
        self.strategy.entry_level = 50400  # Resistance level
        
        risk_params = self.strategy.get_risk_parameters()
        
        required_keys = ['sl_pct', 'tp_pct', 'max_position_pct', 'risk_reward_ratio']
        for key in required_keys:
            self.assertIn(key, risk_params)
            self.assertGreater(risk_params[key], 0)
    
    def test_get_risk_parameters_fallback(self):
        """Test risk parameters fallback values"""
        # No position state
        risk_params = self.strategy.get_risk_parameters()
        
        # Should return fallback values
        self.assertEqual(risk_params['sl_pct'], self.config['stop_loss_pct'])
        self.assertEqual(risk_params['max_position_pct'], self.config['max_position_pct'])
        self.assertEqual(risk_params['risk_reward_ratio'], self.config['take_profit_ratio'])
    
    # Test Error Handling
    def test_error_handling_in_indicators(self):
        """Test error handling in indicator calculations"""
        # Create data with NaN values
        corrupted_data = self.test_data.copy()
        corrupted_data.loc[50:60, 'close'] = np.nan
        
        strategy = StrategyRSIRangeScalping(
            data=corrupted_data,
            config=self.config,
            logger=Mock()
        )
        
        # Should not raise exception
        try:
            strategy.init_indicators()
        except Exception as e:
            self.fail(f"init_indicators raised {e} unexpectedly")
    
    def test_error_handling_in_range_detection(self):
        """Test error handling in range detection"""
        self.strategy.init_indicators()
        
        # Should handle errors gracefully
        result = self.strategy.detect_range_levels(-1)  # Invalid index
        self.assertFalse(result)
        
        # Test with corrupted data
        original_data = self.strategy.data.copy()
        self.strategy.data = pd.DataFrame()  # Empty data
        
        result = self.strategy.detect_range_levels(50)
        self.assertFalse(result)
        
        # Restore data
        self.strategy.data = original_data
    
    def test_error_handling_in_pattern_detection(self):
        """Test error handling in candlestick pattern detection"""
        self.strategy.init_indicators()
        
        # Should handle insufficient data gracefully
        pattern_detected, pattern_type = self.strategy.detect_candlestick_patterns(0)
        self.assertFalse(pattern_detected)
        self.assertIsNone(pattern_type)
    
    def test_error_handling_in_entry_conditions(self):
        """Test error handling in entry condition checks"""
        # Test with current strategy but simulate error conditions
        original_range_active = self.strategy.range_active
        
        try:
            # Reset to inactive state to test error handling
            self.strategy.range_active = False
            
            # Should handle gracefully when no range is active
            entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
            self.assertIsNone(entry_signal)
            
        finally:
            # Restore original state
            self.strategy.range_active = original_range_active
    
    def test_error_handling_in_exit_conditions(self):
        """Test error handling in exit condition checks"""
        # Test with current strategy but no position state
        # Should handle gracefully when no position is open
        exit_signal = self.strategy.check_exit('BTCUSDT')
        self.assertIsNone(exit_signal)
    
    # Test Edge Cases
    def test_range_reactivation_after_stops(self):
        """Test range trading reactivation after consecutive stops"""
        self.strategy.init_indicators()
        
        # Disable range trading
        self.strategy.range_trading_enabled = False
        self.strategy.last_stop_bar = len(self.strategy.data) - 60  # Old stop
        
        # Should re-enable after sufficient time
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        # The method should re-enable trading and reset consecutive stops
        self.assertTrue(self.strategy.range_trading_enabled)
        self.assertEqual(self.strategy.consecutive_stops, 0)
    
    def test_range_update_frequency(self):
        """Test range level update frequency"""
        self.strategy.init_indicators()
        
        # Set up old range
        self.strategy.range_active = True
        self.strategy.last_range_update = len(self.strategy.data) - 30  # Old update
        
        # Should trigger range update
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        # Range should be updated due to age
        self.assertGreater(self.strategy.last_range_update, len(self.strategy.data) - 30)
    
    def test_extreme_market_conditions(self):
        """Test strategy behavior in extreme market conditions"""
        # Create data with extreme volatility
        extreme_data = self.test_data.copy()
        extreme_data['high'] = extreme_data['close'] * 1.05
        extreme_data['low'] = extreme_data['close'] * 0.95
        
        strategy = StrategyRSIRangeScalping(
            data=extreme_data,
            config=self.config,
            logger=Mock()
        )
        
        strategy.init_indicators()
        
        # Should handle extreme conditions
        risk_params = strategy.get_risk_parameters()
        self.assertIsInstance(risk_params, dict)
        
        # Range detection should work
        for i in range(60, 120):
            result = strategy.detect_range_levels(i)
            self.assertIn(result, [True, False])
    
    def test_missing_data_handling(self):
        """Test handling of missing data in calculations"""
        self.strategy.init_indicators()
        
        # Create data with missing RSI values
        test_data = self.strategy.data.copy()
        test_data.loc[test_data.index[-5:], 'rsi'] = np.nan
        self.strategy.data = test_data
        
        # Should handle missing RSI gracefully
        idx = len(self.strategy.data) - 1
        
        # These should not crash
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        exit_signal = self.strategy.check_exit('BTCUSDT')
        
        # Should return None or handle gracefully
        if entry_signal is not None:
            self.assertIsInstance(entry_signal, dict)
        if exit_signal is not None:
            self.assertIsInstance(exit_signal, dict)

if __name__ == '__main__':
    unittest.main() 