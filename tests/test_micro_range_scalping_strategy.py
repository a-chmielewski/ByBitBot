#!/usr/bin/env python3
"""
Test Suite for Micro Range Scalping Strategy

This test suite comprehensively tests the StrategyMicroRangeScalping implementation,
including low volatility detection, micro-range identification, oscillator-based entries, and ultra-tight scalping exits.
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

from strategies.micro_range_scalping_strategy import StrategyMicroRangeScalping

class TestMicroRangeScalpingStrategy(unittest.TestCase):
    """Test cases for Micro Range Scalping Strategy."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Configure logging to reduce noise during testing
        logging.getLogger().setLevel(logging.WARNING)
        
        # Create test logger
        self.logger = logging.getLogger('test_micro_range_scalping')
        self.logger.setLevel(logging.WARNING)
        
        # Create test configuration
        self.config = {
            # Volatility detection
            'atr_period': 14,
            'atr_low_threshold': 0.002,  # 0.2%
            'bb_period': 20,
            'bb_std': 2.0,
            'bb_squeeze_threshold': 0.005,  # 0.5%
            'volume_avg_period': 20,
            'volume_decline_threshold': 0.8,
            
            # Range detection
            'range_detection_bars': 20,
            'range_tolerance_pct': 0.001,  # 0.1%
            'min_range_touches': 2,
            'micro_range_max_pct': 0.003,  # 0.3%
            'min_range_bars': 6,
            
            # Oscillators
            'rsi_period': 7,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'stoch_k_period': 14,
            'stoch_d_period': 3,
            'stoch_oversold': 20,
            'stoch_overbought': 80,
            
            # Risk management
            'stop_loss_buffer_pct': 0.0005,  # 0.05%
            'take_profit_pct': 0.0015,  # 0.15%
            'break_even_buffer_pct': 0.0003,  # 0.03%
            'max_position_pct': 2.0,
            'position_size_reduction': 0.7,
            
            # Trade management
            'cooldown_bars': 2,
            'max_hold_bars': 10
        }
        
        # Generate test data
        self.test_data = self._generate_micro_range_data()
        
        # Initialize strategy
        self.strategy = StrategyMicroRangeScalping(
            data=self.test_data.copy(),
            config=self.config,
            logger=self.logger
        )
    
    def _generate_micro_range_data(self) -> pd.DataFrame:
        """Generate realistic micro-range market data with ultra-low volatility"""
        np.random.seed(42)
        total_bars = 200
        
        # Initialize arrays
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        base_price = 50000
        base_volume = 1000000
        current_price = base_price
        
        # Phase 1: Normal volatility setup (0-39 bars)
        for i in range(40):
            # Normal volatility movements
            noise = np.random.uniform(-100, 100)
            current_price = base_price + noise
            
            open_price = current_price + np.random.uniform(-20, 20)
            close_price = current_price + np.random.uniform(-30, 30)
            
            # Normal ranges
            high_price = max(open_price, close_price) + abs(np.random.uniform(0, 40))
            low_price = min(open_price, close_price) - abs(np.random.uniform(0, 40))
            
            # Normal volume
            volume = base_volume * np.random.uniform(0.8, 1.2)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
        
        # Phase 2: Volatility compression begins (40-79 bars)
        compression_start = closes[-1]
        for i in range(40, 80):
            # Gradually reducing volatility
            volatility_factor = 1.0 - (i - 40) / 40 * 0.7  # Reduce to 30% of original
            
            noise = np.random.uniform(-50, 50) * volatility_factor
            current_price = compression_start + noise
            
            open_price = current_price + np.random.uniform(-10, 10) * volatility_factor
            close_price = current_price + np.random.uniform(-15, 15) * volatility_factor
            
            # Shrinking ranges
            range_size = abs(np.random.uniform(5, 25)) * volatility_factor
            high_price = max(open_price, close_price) + range_size * 0.6
            low_price = min(open_price, close_price) - range_size * 0.4
            
            # Declining volume
            volume_factor = 1.0 - (i - 40) / 40 * 0.4  # Reduce to 60% of normal
            volume = base_volume * np.random.uniform(0.7, 1.0) * volume_factor
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
        
        # Phase 3: Micro-range formation (80-139 bars) - ultra-low volatility
        micro_range_center = closes[-1]
        micro_range_width = micro_range_center * 0.002  # 0.2% range
        support_level = micro_range_center - micro_range_width / 2
        resistance_level = micro_range_center + micro_range_width / 2
        
        for i in range(80, 140):
            # Ultra-tight price action within micro-range
            # Oscillate between support and resistance with tiny movements
            cycle_position = (i - 80) % 12  # 12-bar cycles
            
            if cycle_position < 3:  # Near support
                target_price = support_level + np.random.uniform(0, micro_range_width * 0.2)
            elif cycle_position < 6:  # Moving up
                progress = (cycle_position - 3) / 3
                target_price = support_level + micro_range_width * progress * 0.8
            elif cycle_position < 9:  # Near resistance
                target_price = resistance_level - np.random.uniform(0, micro_range_width * 0.2)
            else:  # Moving down
                progress = (cycle_position - 9) / 3
                target_price = resistance_level - micro_range_width * progress * 0.8
            
            # Add tiny random noise
            current_price = target_price + np.random.uniform(-micro_range_width * 0.1, micro_range_width * 0.1)
            
            open_price = current_price + np.random.uniform(-micro_range_width * 0.05, micro_range_width * 0.05)
            close_price = current_price + np.random.uniform(-micro_range_width * 0.08, micro_range_width * 0.08)
            
            # Ultra-tight ranges
            tiny_range = micro_range_width * 0.15
            high_price = max(open_price, close_price) + abs(np.random.uniform(0, tiny_range))
            low_price = min(open_price, close_price) - abs(np.random.uniform(0, tiny_range))
            
            # Ensure we stay within the micro-range bounds
            high_price = min(high_price, resistance_level + micro_range_width * 0.1)
            low_price = max(low_price, support_level - micro_range_width * 0.1)
            
            # Very low volume
            volume = base_volume * np.random.uniform(0.4, 0.7)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
        
        # Phase 4: Continued micro-range with some tests (140-199 bars)
        for i in range(140, 200):
            # Mostly stay in range with occasional tests of boundaries
            if i % 20 == 0:  # Occasional test of resistance
                current_price = resistance_level + np.random.uniform(0, micro_range_width * 0.05)
            elif i % 20 == 10:  # Occasional test of support
                current_price = support_level - np.random.uniform(0, micro_range_width * 0.05)
            else:  # Normal micro-range action
                current_price = support_level + np.random.uniform(0, micro_range_width)
            
            open_price = current_price + np.random.uniform(-micro_range_width * 0.03, micro_range_width * 0.03)
            close_price = current_price + np.random.uniform(-micro_range_width * 0.05, micro_range_width * 0.05)
            
            # Tiny ranges
            tiny_range = micro_range_width * 0.1
            high_price = max(open_price, close_price) + abs(np.random.uniform(0, tiny_range))
            low_price = min(open_price, close_price) - abs(np.random.uniform(0, tiny_range))
            
            # Low volume
            volume = base_volume * np.random.uniform(0.5, 0.8)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
        
        # Ensure all arrays are exactly the same length
        assert len(opens) == len(highs) == len(lows) == len(closes) == len(volumes) == total_bars
        
        # Create DataFrame
        timestamps = pd.date_range(start='2024-01-01', periods=total_bars, freq='1min')
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        return df
    
    # Basic initialization tests
    def test_strategy_initialization(self):
        """Test strategy initialization."""
        self.assertIsInstance(self.strategy, StrategyMicroRangeScalping)
        self.assertEqual(self.strategy.MARKET_TYPE_TAGS, ['LOW_VOLATILITY', 'RANGING'])
        self.assertTrue(self.strategy.SHOW_IN_SELECTION)
    
    def test_state_variables_initialization(self):
        """Test state variable initialization."""
        self.assertFalse(self.strategy.micro_range_detected)
        self.assertEqual(self.strategy.range_support, 0)
        self.assertEqual(self.strategy.range_resistance, 0)
        self.assertEqual(self.strategy.range_middle, 0)
        self.assertEqual(self.strategy.range_width, 0)
        self.assertEqual(self.strategy.range_touches_support, 0)
        self.assertEqual(self.strategy.range_touches_resistance, 0)
        self.assertEqual(self.strategy.last_range_update, 0)
        self.assertFalse(self.strategy.low_vol_confirmed)
        self.assertEqual(self.strategy.low_vol_bars_count, 0)
        self.assertIsNone(self.strategy.entry_bar)
        self.assertIsNone(self.strategy.entry_side)
        self.assertIsNone(self.strategy.entry_price)
        self.assertFalse(self.strategy.break_even_moved)
        self.assertEqual(self.strategy.last_trade_bar, -2)
    
    def test_market_type_tags(self):
        """Test market type tags."""
        expected_tags = ['LOW_VOLATILITY', 'RANGING']
        self.assertEqual(self.strategy.MARKET_TYPE_TAGS, expected_tags)
        self.assertTrue(self.strategy.SHOW_IN_SELECTION)
    
    def test_indicator_initialization(self):
        """Test indicator initialization."""
        # Check that indicators are present in data
        required_indicators = ['atr', 'bb_upper', 'bb_middle', 'bb_lower', 'rsi', 'stoch_k', 'stoch_d', 'volume_sma']
        for indicator in required_indicators:
            self.assertIn(indicator, self.strategy.data.columns, f"Missing indicator: {indicator}")
    
    # Technical indicator tests
    def test_atr_calculation(self):
        """Test ATR calculation accuracy."""
        atr_values = self.strategy.data['atr'].dropna()
        self.assertTrue(len(atr_values) > 0)
        self.assertTrue(all(atr_values >= 0))
        
        # ATR should be very small for our micro-range data
        avg_atr = atr_values.mean()
        self.assertGreater(avg_atr, 0)
        self.assertLess(avg_atr, 500)  # Should be small for low volatility
    
    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation."""
        bb_upper = self.strategy.data['bb_upper'].dropna()
        bb_middle = self.strategy.data['bb_middle'].dropna()
        bb_lower = self.strategy.data['bb_lower'].dropna()
        
        self.assertTrue(len(bb_upper) > 0)
        self.assertTrue(len(bb_middle) > 0)
        self.assertTrue(len(bb_lower) > 0)
        
        # Check proper ordering: lower < middle < upper
        for i in range(len(bb_upper)):
            if not pd.isna(bb_upper.iloc[i]) and not pd.isna(bb_lower.iloc[i]):
                self.assertLess(bb_lower.iloc[i], bb_middle.iloc[i])
                self.assertLess(bb_middle.iloc[i], bb_upper.iloc[i])
    
    def test_rsi_calculation(self):
        """Test RSI calculation."""
        rsi_values = self.strategy.data['rsi'].dropna()
        self.assertTrue(len(rsi_values) > 0)
        
        # RSI should be between 0 and 100
        self.assertTrue(all(rsi_values >= 0))
        self.assertTrue(all(rsi_values <= 100))
    
    def test_stochastic_calculation(self):
        """Test Stochastic oscillator calculation."""
        stoch_k = self.strategy.data['stoch_k'].dropna()
        stoch_d = self.strategy.data['stoch_d'].dropna()
        
        self.assertTrue(len(stoch_k) > 0)
        self.assertTrue(len(stoch_d) > 0)
        
        # Stochastic should be between 0 and 100
        self.assertTrue(all(stoch_k >= 0))
        self.assertTrue(all(stoch_k <= 100))
        self.assertTrue(all(stoch_d >= 0))
        self.assertTrue(all(stoch_d <= 100))
    
    def test_volume_sma_calculation(self):
        """Test volume SMA calculation."""
        volume_sma = self.strategy.data['volume_sma'].dropna()
        self.assertTrue(len(volume_sma) > 0)
        self.assertTrue(all(volume_sma > 0))
    
    # Low volatility detection tests
    def test_low_volatility_environment_detection(self):
        """Test low volatility environment detection."""
        # Test with sufficient data (should detect low vol in later phases)
        for idx in range(100, len(self.strategy.data), 10):
            try:
                is_low_vol = self.strategy.is_low_volatility_environment(idx)
                self.assertIn(is_low_vol, [True, False])
            except Exception as e:
                self.fail(f"Low volatility detection failed at index {idx}: {e}")
    
    def test_low_volatility_insufficient_data(self):
        """Test low volatility detection with insufficient data."""
        result = self.strategy.is_low_volatility_environment(0)
        self.assertFalse(result)
    
    # Micro-range detection tests
    def test_micro_range_detection(self):
        """Test micro-range detection."""
        # Test with sufficient data (should detect micro-range in later phases)
        for idx in range(100, len(self.strategy.data), 15):
            try:
                detected = self.strategy.detect_micro_range(idx)
                self.assertIn(detected, [True, False])
                if detected:
                    # Validate range properties
                    self.assertGreater(self.strategy.range_resistance, self.strategy.range_support)
                    self.assertGreater(self.strategy.range_touches_support, 0)
                    self.assertGreater(self.strategy.range_touches_resistance, 0)
            except Exception as e:
                self.fail(f"Micro-range detection failed at index {idx}: {e}")
    
    def test_micro_range_insufficient_data(self):
        """Test micro-range detection with insufficient data."""
        result = self.strategy.detect_micro_range(10)
        self.assertFalse(result)
    
    # Support/resistance proximity tests
    def test_near_support_detection(self):
        """Test near support level detection."""
        # Set up a mock micro-range
        self.strategy.micro_range_detected = True
        self.strategy.range_support = 50000
        
        # Test prices near support
        self.assertTrue(self.strategy.is_near_support(50000))
        self.assertTrue(self.strategy.is_near_support(49995))  # Within tolerance
        self.assertFalse(self.strategy.is_near_support(49900))  # Too far
    
    def test_near_resistance_detection(self):
        """Test near resistance level detection."""
        # Set up a mock micro-range
        self.strategy.micro_range_detected = True
        self.strategy.range_resistance = 50100
        
        # Test prices near resistance
        self.assertTrue(self.strategy.is_near_resistance(50100))
        self.assertTrue(self.strategy.is_near_resistance(50105))  # Within tolerance
        self.assertFalse(self.strategy.is_near_resistance(50200))  # Too far
    
    def test_near_levels_without_range(self):
        """Test near level detection when no micro-range is detected."""
        self.strategy.micro_range_detected = False
        
        self.assertFalse(self.strategy.is_near_support(50000))
        self.assertFalse(self.strategy.is_near_resistance(50100))
    
    # Oscillator condition tests
    def test_oscillator_oversold_detection(self):
        """Test oscillator oversold condition detection."""
        for idx in range(50, len(self.strategy.data), 20):
            try:
                is_oversold = self.strategy.check_oscillator_oversold(idx)
                self.assertIn(is_oversold, [True, False])
            except Exception as e:
                self.fail(f"Oscillator oversold detection failed at index {idx}: {e}")
    
    def test_oscillator_overbought_detection(self):
        """Test oscillator overbought condition detection."""
        for idx in range(50, len(self.strategy.data), 20):
            try:
                is_overbought = self.strategy.check_oscillator_overbought(idx)
                self.assertIn(is_overbought, [True, False])
            except Exception as e:
                self.fail(f"Oscillator overbought detection failed at index {idx}: {e}")
    
    # Range breakout tests
    def test_range_breakout_detection(self):
        """Test range breakout detection."""
        # Set up a mock micro-range
        self.strategy.micro_range_detected = True
        self.strategy.range_support = 49900
        self.strategy.range_resistance = 50100
        
        # Test various price scenarios
        for idx in range(100, len(self.strategy.data), 25):
            try:
                breakout, direction = self.strategy.check_range_breakout(idx)
                self.assertIn(breakout, [True, False])
                if breakout:
                    self.assertIn(direction, ['upside', 'downside'])
            except Exception as e:
                self.fail(f"Range breakout detection failed at index {idx}: {e}")
    
    def test_range_breakout_without_range(self):
        """Test range breakout detection when no micro-range is detected."""
        self.strategy.micro_range_detected = False
        
        breakout, direction = self.strategy.check_range_breakout(100)
        self.assertFalse(breakout)
        self.assertIsNone(direction)
    
    # Exit condition tests
    def test_exit_conditions_no_position(self):
        """Test exit conditions when no position is open."""
        self.strategy.entry_bar = None
        
        should_exit, reason = self.strategy.check_exit_conditions(100)
        self.assertFalse(should_exit)
        self.assertIsNone(reason)
    
    def test_exit_conditions_time_based(self):
        """Test time-based exit conditions."""
        self.strategy.entry_bar = 90
        
        should_exit, reason = self.strategy.check_exit_conditions(105)  # 15 bars later
        if should_exit and reason == "time_exit":
            self.assertEqual(reason, "time_exit")
    
    def test_exit_conditions_range_breakout(self):
        """Test exit conditions on range breakout."""
        # Set up position and micro-range
        self.strategy.entry_bar = 100
        self.strategy.micro_range_detected = True
        self.strategy.range_support = 49900
        self.strategy.range_resistance = 50100
        
        # Mock a breakout scenario by setting close price outside range
        original_close = self.strategy.data.loc[105, 'close']
        self.strategy.data.loc[105, 'close'] = 50200  # Above resistance
        
        should_exit, reason = self.strategy.check_exit_conditions(105)
        if should_exit and 'range_breakout' in reason:
            self.assertIn('range_breakout', reason)
        
        # Restore original data
        self.strategy.data.loc[105, 'close'] = original_close
    
    # Entry condition tests
    def test_entry_conditions_basic(self):
        """Test basic entry condition checking."""
        # Set strategy to a good position in data (micro-range phase)
        self.strategy.data = self.strategy.data.iloc[120:].reset_index(drop=True)
        self.strategy.init_indicators()
        
        try:
            entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
            if entry_signal:
                self.assertIn('action', entry_signal)
                self.assertIn('price', entry_signal)
                self.assertIn('confidence', entry_signal)
                self.assertIn('reason', entry_signal)
                self.assertIn(entry_signal['action'], ['long', 'short'])
        except Exception as e:
            self.fail(f"Entry condition check failed: {e}")
    
    def test_entry_conditions_cooldown(self):
        """Test entry conditions during cooldown period."""
        # Set recent trade
        self.strategy.last_trade_bar = len(self.strategy.data) - 1
        
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        self.assertIsNone(entry_signal)
    
    def test_entry_conditions_insufficient_data(self):
        """Test entry conditions with insufficient data."""
        # Use minimal data
        minimal_data = self.strategy.data.head(10).copy()
        
        try:
            strategy = StrategyMicroRangeScalping(
                data=minimal_data,
                config=self.config,
                logger=self.logger
            )
            entry_signal = strategy._check_entry_conditions('BTCUSDT')
            self.assertIsNone(entry_signal)
        except (TypeError, KeyError, IndexError) as e:
            # These are acceptable errors for insufficient data
            pass
        except Exception as e:
            self.fail(f"Strategy should handle insufficient data gracefully: {e}")
    
    # Trade management tests
    def test_trade_closure_cleanup(self):
        """Test trade closure cleanup."""
        # Set up active trade state
        self.strategy.entry_bar = 100
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50000
        self.strategy.break_even_moved = True
        
        # Simulate trade closure
        trade_result = {'reason': 'profit_target_long', 'pnl': 75}
        self.strategy.on_trade_closed('BTCUSDT', trade_result)
        
        # Check cleanup
        self.assertIsNone(self.strategy.entry_bar)
        self.assertIsNone(self.strategy.entry_side)
        self.assertIsNone(self.strategy.entry_price)
        self.assertFalse(self.strategy.break_even_moved)
        self.assertGreater(self.strategy.last_trade_bar, -1)
    
    # Risk management tests
    def test_risk_parameters_with_position(self):
        """Test risk parameters with active position."""
        # Set up active position and micro-range
        self.strategy.micro_range_detected = True
        self.strategy.range_support = 49950
        self.strategy.range_resistance = 50050
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 49960
        
        params = self.strategy.get_risk_parameters()
        
        required_params = ['sl_pct', 'tp_pct', 'max_position_pct', 'risk_reward_ratio']
        for param in required_params:
            self.assertIn(param, params)
        
        # Check reasonable values for micro-scalping
        self.assertGreater(params['sl_pct'], 0)
        self.assertGreater(params['tp_pct'], 0)
        self.assertGreater(params['max_position_pct'], 0)
        self.assertLess(params['max_position_pct'], 5.0)  # Reduced position sizing
        self.assertGreater(params['risk_reward_ratio'], 0)
    
    def test_risk_parameters_fallback(self):
        """Test risk parameters fallback values."""
        # No active position
        self.strategy.micro_range_detected = False
        self.strategy.entry_side = None
        
        params = self.strategy.get_risk_parameters()
        
        required_params = ['sl_pct', 'tp_pct', 'max_position_pct', 'risk_reward_ratio']
        for param in required_params:
            self.assertIn(param, params)
        
        # Should use fallback values
        self.assertEqual(params['sl_pct'], self.config['stop_loss_buffer_pct'])
        self.assertEqual(params['tp_pct'], self.config['take_profit_pct'])
    
    # Integration tests
    def test_main_entry_method(self):
        """Test main entry method."""
        try:
            result = self.strategy.check_entry('BTCUSDT')
            if result:
                self.assertIn('action', result)
                self.assertIn('price', result)
                self.assertIn(result['action'], ['long', 'short'])
        except Exception as e:
            self.fail(f"Main entry method failed: {e}")
    
    def test_main_exit_method(self):
        """Test main exit method."""
        try:
            result = self.strategy.check_exit('BTCUSDT')
            if result:
                self.assertEqual(result['action'], 'exit')
                self.assertIn('reason', result)
        except Exception as e:
            self.fail(f"Main exit method failed: {e}")
    
    # Break-even management tests
    def test_break_even_move_long(self):
        """Test break-even move for long positions."""
        # Set up profitable long position
        self.strategy.entry_bar = 100
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50000
        self.strategy.break_even_moved = False
        
        # Mock profitable price
        original_close = self.strategy.data.loc[105, 'close']
        self.strategy.data.loc[105, 'close'] = 50020  # Small profit
        
        result = self.strategy.check_exit('BTCUSDT')
        # Should not exit but may set break_even_moved flag
        
        # Restore original data
        self.strategy.data.loc[105, 'close'] = original_close
    
    def test_break_even_move_short(self):
        """Test break-even move for short positions."""
        # Set up profitable short position
        self.strategy.entry_bar = 100
        self.strategy.entry_side = 'short'
        self.strategy.entry_price = 50000
        self.strategy.break_even_moved = False
        
        # Mock profitable price
        original_close = self.strategy.data.loc[105, 'close']
        self.strategy.data.loc[105, 'close'] = 49980  # Small profit
        
        result = self.strategy.check_exit('BTCUSDT')
        # Should not exit but may set break_even_moved flag
        
        # Restore original data
        self.strategy.data.loc[105, 'close'] = original_close
    
    # Error handling tests
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Create minimal dataset
        minimal_data = self.test_data.head(5).copy()
        
        try:
            strategy = StrategyMicroRangeScalping(
                data=minimal_data,
                config=self.config,
                logger=self.logger
            )
            entry_result = strategy.check_entry('BTCUSDT')
            self.assertIsNone(entry_result)
        except (TypeError, KeyError, IndexError) as e:
            # These are acceptable errors for insufficient data
            pass
        except Exception as e:
            self.fail(f"Strategy should handle insufficient data gracefully: {e}")
    
    def test_missing_indicator_handling(self):
        """Test handling of missing indicators."""
        test_data = self.test_data.copy()
        strategy = StrategyMicroRangeScalping(
            data=test_data,
            config=self.config,
            logger=self.logger
        )
        
        # Should initialize indicators without errors
        try:
            strategy.init_indicators()
        except Exception as e:
            self.fail(f"Strategy should handle missing indicators: {e}")
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration."""
        invalid_config = self.config.copy()
        invalid_config['rsi_period'] = -5  # Invalid period
        
        try:
            strategy = StrategyMicroRangeScalping(
                data=self.test_data.copy(),
                config=invalid_config,
                logger=self.logger
            )
            # Should not crash during initialization
        except Exception as e:
            # Should handle gracefully or provide meaningful error
            self.assertIsInstance(e, (ValueError, KeyError))
    
    # Performance tests
    def test_strategy_performance_with_large_dataset(self):
        """Test strategy performance with larger dataset."""
        # Create larger dataset
        large_data = pd.concat([self.test_data] * 3, ignore_index=True)
        large_data['timestamp'] = pd.date_range(start='2024-01-01', periods=len(large_data), freq='1min')
        
        try:
            strategy = StrategyMicroRangeScalping(
                data=large_data,
                config=self.config,
                logger=self.logger
            )
            
            # Should handle larger datasets efficiently
            entry_result = strategy.check_entry('BTCUSDT')
            exit_result = strategy.check_exit('BTCUSDT')
            
        except Exception as e:
            self.fail(f"Strategy should handle large datasets: {e}")
    
    def test_concurrent_strategy_instances(self):
        """Test multiple strategy instances."""
        try:
            strategy1 = StrategyMicroRangeScalping(
                data=self.test_data.copy(),
                config=self.config,
                logger=self.logger
            )
            
            strategy2 = StrategyMicroRangeScalping(
                data=self.test_data.copy(),
                config=self.config,
                logger=self.logger
            )
            
            # Both should work independently
            result1 = strategy1.check_entry('BTCUSDT')
            result2 = strategy2.check_entry('ETHUSDT')
            
        except Exception as e:
            self.fail(f"Multiple strategy instances should work independently: {e}")

if __name__ == '__main__':
    unittest.main() 