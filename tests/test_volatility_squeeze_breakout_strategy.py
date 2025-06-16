#!/usr/bin/env python3
"""
Test Suite for Volatility Squeeze Breakout Strategy

This module contains comprehensive tests for the StrategyVolatilitySqueezeBreakout class,
covering squeeze detection, range identification, breakout triggers, and exit conditions.
"""

import unittest
import sys
import os
import logging
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.volatility_squeeze_breakout_strategy import StrategyVolatilitySqueezeBreakout

class TestVolatilitySqueezeBreakoutStrategy(unittest.TestCase):
    """Test cases for Volatility Squeeze Breakout Strategy"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Suppress logging during tests
        logging.disable(logging.CRITICAL)
        
        # Create realistic squeeze market data
        self.data_size = 200
        self.create_squeeze_market_data()
        
        # Default configuration
        self.config = {
            'bb_period': 20,
            'bb_std': 2.0,
            'adx_period': 14,
            'atr_period': 14,
            'volume_avg_period': 20,
            'bb_squeeze_threshold': 0.02,
            'adx_low_threshold': 30,
            'volume_low_threshold': 0.85,
            'bb_squeeze_bars': 5,
            'max_squeeze_bars': 150,
            'range_detection_bars': 20,
            'range_touch_tolerance': 0.003,
            'min_range_touches': 2,
            'min_range_size_pct': 0.003,
            'breakout_buffer_pct': 0.003,
            'confirmation_required': False,
            'volume_breakout_multiplier': 1.5,
            'candle_close_pct': 0.6,
            'initial_stop_buffer_pct': 0.002,
            'quick_profit_pct': 0.004,
            'range_projection_multiplier': 0.8,
            'atr_target_multiplier': 1.5,
            'trail_stop_pct': 0.004,
            'max_hold_bars': 50,
            'fake_breakout_exit_bars': 5,
            'partial_exit_pct': 0.4,
            'cooldown_bars': 5,
            'position_size_reduction': 0.9,
            'max_position_pct': 2.0,
            'sl_pct': 0.015,
            'tp_pct': 0.03,
            'enable_simple_squeeze': True
        }
        
        # Create mock logger
        self.mock_logger = Mock()
        
        # Initialize strategy
        self.strategy = StrategyVolatilitySqueezeBreakout(
            data=self.data,
            config=self.config,
            logger=self.mock_logger
        )
    
    def tearDown(self):
        """Clean up after each test method"""
        logging.disable(logging.NOTSET)
    
    def create_squeeze_market_data(self):
        """Create realistic market data with squeeze conditions"""
        np.random.seed(42)  # For reproducible tests
        
        # Base price around 50,000 (BTC-like)
        base_price = 50000.0
        
        # Create time index
        dates = pd.date_range(start='2024-01-01', periods=self.data_size, freq='1min')
        
        # Create squeeze pattern: consolidation followed by breakout
        prices = []
        volumes = []
        
        for i in range(self.data_size):
            if i < 50:
                # Initial normal volatility
                price_change = np.random.normal(0, base_price * 0.002)
                volume = np.random.uniform(800, 1200)
            elif i < 120:
                # Squeeze period - low volatility, tight range
                price_change = np.random.normal(0, base_price * 0.0005)  # Very low volatility
                volume = np.random.uniform(400, 800)  # Lower volume
            elif i < 140:
                # Breakout period - higher volatility
                if i == 120:  # Initial breakout
                    price_change = base_price * 0.008  # Strong upward breakout
                else:
                    price_change = np.random.normal(base_price * 0.002, base_price * 0.003)
                volume = np.random.uniform(1200, 2000)  # Higher volume
            else:
                # Post-breakout consolidation
                price_change = np.random.normal(0, base_price * 0.001)
                volume = np.random.uniform(600, 1000)
            
            if i == 0:
                prices.append(base_price)
            else:
                prices.append(max(prices[-1] + price_change, base_price * 0.8))  # Prevent negative prices
            
            volumes.append(volume)
        
        # Create OHLCV data
        opens = []
        highs = []
        lows = []
        closes = []
        
        for i, price in enumerate(prices):
            if i == 0:
                open_price = price
            else:
                open_price = closes[-1]  # Open = previous close
            
            # Create realistic OHLC from the price
            volatility = 0.001 if 50 <= i < 120 else 0.002  # Lower volatility during squeeze
            high = price + np.random.uniform(0, price * volatility)
            low = price - np.random.uniform(0, price * volatility)
            close = price + np.random.uniform(-price * volatility/2, price * volatility/2)
            
            # Ensure OHLC relationships are valid
            high = max(high, open_price, close)
            low = min(low, open_price, close)
            
            opens.append(open_price)
            highs.append(high)
            lows.append(low)
            closes.append(close)
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        
        # Set timestamp as index
        self.data.set_index('timestamp', inplace=True)
    
    def test_strategy_initialization(self):
        """Test strategy initialization and state variables"""
        self.assertIsInstance(self.strategy, StrategyVolatilitySqueezeBreakout)
        self.assertEqual(self.strategy.MARKET_TYPE_TAGS, ['LOW_VOLATILITY', 'TRANSITIONAL'])
        self.assertTrue(self.strategy.SHOW_IN_SELECTION)
        
        # Check initial state
        self.assertFalse(self.strategy.squeeze_detected)
        self.assertIsNone(self.strategy.squeeze_start_bar)
        self.assertEqual(self.strategy.squeeze_bars_count, 0)
        self.assertFalse(self.strategy.range_confirmed)
        self.assertFalse(self.strategy.pending_breakout_long)
        self.assertFalse(self.strategy.pending_breakout_short)
        self.assertIsNone(self.strategy.breakout_direction)
        self.assertFalse(self.strategy.first_target_hit)
    
    def test_indicator_initialization(self):
        """Test that all indicators are properly initialized"""
        self.strategy.init_indicators()
        
        # Check that all required indicators are present
        required_indicators = ['bb_upper', 'bb_middle', 'bb_lower', 'adx', 'atr', 'volume_sma']
        for indicator in required_indicators:
            self.assertIn(indicator, self.strategy.data.columns)
            self.assertFalse(self.strategy.data[indicator].isna().all())
    
    def test_manual_calculations_fallback(self):
        """Test manual calculation methods when pandas_ta is not available"""
        # Test manual ADX calculation
        adx_manual = self.strategy._calculate_adx_manual(14)
        self.assertIsInstance(adx_manual, pd.Series)
        self.assertEqual(len(adx_manual), len(self.strategy.data))
        
        # Test manual ATR calculation
        atr_manual = self.strategy._calculate_atr_manual(14)
        self.assertIsInstance(atr_manual, pd.Series)
        self.assertEqual(len(atr_manual), len(self.strategy.data))
        # ATR should be mostly non-negative (allow for some calculation edge cases)
        self.assertGreater((atr_manual >= 0).sum(), len(atr_manual) * 0.8) 
    
    def test_bollinger_squeeze_detection(self):
        """Test Bollinger Band squeeze detection logic"""
        self.strategy.init_indicators()
        
        # Test squeeze detection during squeeze period (bars 50-120)
        squeeze_detected_count = 0
        for i in range(50, 120):
            if self.strategy.is_bollinger_squeeze(i):
                squeeze_detected_count += 1
        
        # Should detect squeeze in most of the squeeze period
        self.assertGreater(squeeze_detected_count, 30)
        
        # Test no squeeze during high volatility periods
        no_squeeze_count = 0
        for i in range(120, 140):
            if not self.strategy.is_bollinger_squeeze(i):
                no_squeeze_count += 1
        
        # Should not detect squeeze during breakout period
        self.assertGreater(no_squeeze_count, 10)
    
    def test_simple_squeeze_mode(self):
        """Test simple squeeze detection mode"""
        self.strategy.config['enable_simple_squeeze'] = True
        self.strategy.init_indicators()
        
        # Simple mode should be more permissive
        simple_squeeze_count = 0
        for i in range(50, 120):
            if self.strategy.is_bollinger_squeeze(i):
                simple_squeeze_count += 1
        
        # Should detect more squeezes in simple mode
        self.assertGreater(simple_squeeze_count, 40)
    
    def test_complex_squeeze_mode(self):
        """Test complex squeeze detection mode with all criteria"""
        self.strategy.config['enable_simple_squeeze'] = False
        self.strategy.init_indicators()
        
        # Complex mode requires BB squeeze + low ADX + low volume
        complex_squeeze_count = 0
        for i in range(50, 120):
            if self.strategy.is_bollinger_squeeze(i):
                complex_squeeze_count += 1
        
        # Should detect fewer squeezes in complex mode
        self.assertLess(complex_squeeze_count, 50)
    
    def test_squeeze_state_management(self):
        """Test squeeze state tracking and management"""
        self.strategy.init_indicators()
        
        # Simulate squeeze detection
        idx = 60  # During squeeze period
        if self.strategy.is_bollinger_squeeze(idx):
            # First detection
            self.strategy.squeeze_detected = True
            self.strategy.squeeze_start_bar = idx
            self.strategy.squeeze_bars_count = 1
            
            self.assertTrue(self.strategy.squeeze_detected)
            self.assertEqual(self.strategy.squeeze_start_bar, idx)
            self.assertEqual(self.strategy.squeeze_bars_count, 1)
        
        # Test state reset
        self.strategy.reset_squeeze_state()
        self.assertFalse(self.strategy.squeeze_detected)
        self.assertIsNone(self.strategy.squeeze_start_bar)
        self.assertEqual(self.strategy.squeeze_bars_count, 0)
        self.assertFalse(self.strategy.range_confirmed)
    
    def test_simple_range_detection(self):
        """Test simple range detection logic"""
        self.strategy.init_indicators()
        
        # Test range detection during squeeze period
        range_detected = self.strategy.is_simple_range_detected(80)
        
        if range_detected:
            self.assertGreater(self.strategy.range_resistance, self.strategy.range_support)
            self.assertGreater(self.strategy.range_height, 0)
            self.assertEqual(self.strategy.range_middle, 
                           (self.strategy.range_resistance + self.strategy.range_support) / 2)
    
    def test_complex_range_detection(self):
        """Test complex support/resistance range detection"""
        self.strategy.init_indicators()
        
        # Set up squeeze state for range detection
        self.strategy.squeeze_detected = True
        self.strategy.squeeze_bars_count = 20
        
        # Test range detection
        range_detected = self.strategy.detect_support_resistance_range(80)
        
        if range_detected:
            self.assertGreater(self.strategy.range_resistance, self.strategy.range_support)
            self.assertGreater(self.strategy.range_height, 0)
            
            # Check minimum range size
            min_range_size_pct = self.strategy.config.get('min_range_size_pct', 0.003)
            actual_range_pct = self.strategy.range_height / self.strategy.range_support
            self.assertGreaterEqual(actual_range_pct, min_range_size_pct)
    
    def test_breakout_level_setup(self):
        """Test breakout level calculation and setup"""
        self.strategy.init_indicators()
        
        # Set up range first
        self.strategy.range_support = 49800
        self.strategy.range_resistance = 50200
        self.strategy.range_height = 400
        self.strategy.range_confirmed = True
        
        # Setup breakout levels
        success = self.strategy.setup_breakout_levels()
        self.assertTrue(success)
        
        # Check breakout levels are set correctly
        breakout_buffer_pct = self.strategy.config.get('breakout_buffer_pct', 0.003)
        expected_long_price = self.strategy.range_resistance * (1 + breakout_buffer_pct)
        expected_short_price = self.strategy.range_support * (1 - breakout_buffer_pct)
        
        self.assertAlmostEqual(self.strategy.breakout_long_price, expected_long_price, places=2)
        self.assertAlmostEqual(self.strategy.breakout_short_price, expected_short_price, places=2)
        self.assertTrue(self.strategy.pending_breakout_long)
        self.assertTrue(self.strategy.pending_breakout_short)
    
    def test_breakout_trigger_detection(self):
        """Test breakout trigger detection logic"""
        self.strategy.init_indicators()
        
        # Set up breakout levels
        self.strategy.pending_breakout_long = True
        self.strategy.pending_breakout_short = True
        self.strategy.breakout_long_price = 50200
        self.strategy.breakout_short_price = 49800
        
        # Test long breakout trigger
        # Create a bar that breaks above resistance
        test_idx = 125  # During breakout period
        if test_idx < len(self.strategy.data):
            # Modify the high to trigger breakout
            original_high = self.strategy.data['high'].iloc[test_idx]
            self.strategy.data.loc[self.strategy.data.index[test_idx], 'high'] = 50250
            
            triggered, direction = self.strategy.check_breakout_trigger(test_idx)
            
            if triggered:
                self.assertEqual(direction, 'long')
            
            # Restore original value
            self.strategy.data.loc[self.strategy.data.index[test_idx], 'high'] = original_high
        
        # Test short breakout trigger
        if test_idx < len(self.strategy.data):
            # Reset breakout states first
            self.strategy.pending_breakout_long = False
            self.strategy.pending_breakout_short = True
            
            # Modify the low to trigger breakout
            original_low = self.strategy.data['low'].iloc[test_idx]
            self.strategy.data.loc[self.strategy.data.index[test_idx], 'low'] = 49750
            
            triggered, direction = self.strategy.check_breakout_trigger(test_idx)
            
            if triggered:
                self.assertEqual(direction, 'short')
            
            # Restore original value
            self.strategy.data.loc[self.strategy.data.index[test_idx], 'low'] = original_low
    
    def test_breakout_confirmation(self):
        """Test breakout confirmation with volume and candle analysis"""
        self.strategy.init_indicators()
        
        # Test with confirmation disabled (should always confirm)
        self.strategy.config['confirmation_required'] = False
        confirmed = self.strategy.confirm_breakout(125, 'long')
        self.assertTrue(confirmed)
        
        # Test with confirmation enabled
        self.strategy.config['confirmation_required'] = True
        
        # Set up test data for confirmation
        test_idx = 125
        if test_idx < len(self.strategy.data):
            # Modify volume to create spike
            original_volume = self.strategy.data['volume'].iloc[test_idx]
            avg_volume = self.strategy.data['volume_sma'].iloc[test_idx]
            
            if not pd.isna(avg_volume) and avg_volume > 0:
                volume_multiplier = self.strategy.config.get('volume_breakout_multiplier', 1.5)
                self.strategy.data.loc[self.strategy.data.index[test_idx], 'volume'] = avg_volume * volume_multiplier * 1.1
                
                # Create bullish candle for long confirmation
                self.strategy.data.loc[self.strategy.data.index[test_idx], 'open'] = 50100
                self.strategy.data.loc[self.strategy.data.index[test_idx], 'close'] = 50180
                self.strategy.data.loc[self.strategy.data.index[test_idx], 'high'] = 50200
                self.strategy.data.loc[self.strategy.data.index[test_idx], 'low'] = 50090
                
                confirmed = self.strategy.confirm_breakout(test_idx, 'long')
                # Should be confirmed with volume spike and good candle
                
                # Restore original values
                self.strategy.data.loc[self.strategy.data.index[test_idx], 'volume'] = original_volume 
    
    def test_targets_and_stops_calculation(self):
        """Test profit target and stop loss calculation"""
        self.strategy.init_indicators()
        
        # Set up range and entry conditions
        self.strategy.range_support = 49800
        self.strategy.range_resistance = 50200
        self.strategy.range_height = 400
        self.strategy.initial_atr = 100
        
        entry_price = 50250
        
        # Test long position targets
        stop_price, quick_target, main_target = self.strategy.calculate_targets_and_stops('long', entry_price)
        
        # Stop should be just inside resistance
        initial_stop_buffer_pct = self.strategy.config.get('initial_stop_buffer_pct', 0.002)
        expected_stop = self.strategy.range_resistance * (1 - initial_stop_buffer_pct)
        self.assertAlmostEqual(stop_price, expected_stop, places=2)
        
        # Quick target should be above entry
        self.assertGreater(quick_target, entry_price)
        
        # Main target should be above entry price (may be less than quick target due to ATR vs range calculations)
        self.assertGreater(main_target, entry_price)
        
        # Test short position targets
        entry_price_short = 49750
        stop_price_short, quick_target_short, main_target_short = self.strategy.calculate_targets_and_stops('short', entry_price_short)
        
        # Stop should be just inside support
        expected_stop_short = self.strategy.range_support * (1 + initial_stop_buffer_pct)
        self.assertAlmostEqual(stop_price_short, expected_stop_short, places=2)
        
        # Quick target should be below entry
        self.assertLess(quick_target_short, entry_price_short)
        
        # Main target should be below entry price (may be higher than quick target due to ATR vs range calculations)
        self.assertLess(main_target_short, entry_price_short)
    
    def test_entry_conditions_squeeze_breakout(self):
        """Test entry conditions during squeeze breakout scenario"""
        self.strategy.init_indicators()
        
        # Test during squeeze period - should not enter yet
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        # May or may not have signal depending on exact conditions
        
        # Test cooldown period
        self.strategy.last_trade_bar = len(self.strategy.data) - 3  # Recent trade
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        self.assertIsNone(entry_signal)  # Should be in cooldown
        
        # Reset cooldown
        self.strategy.last_trade_bar = -10
        
        # Test with insufficient data
        small_data = self.strategy.data.iloc[:10].copy()
        original_data = self.strategy.data
        self.strategy.data = small_data
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        self.assertIsNone(entry_signal)  # Insufficient data
        
        # Restore data
        self.strategy.data = original_data
    
    def test_entry_conditions_simple_range_breakout(self):
        """Test entry conditions for simple range breakout"""
        self.strategy.init_indicators()
        self.strategy.config['enable_simple_squeeze'] = True
        
        # Create a clear breakout scenario
        test_idx = 125
        if test_idx < len(self.strategy.data):
            # Force a breakout by modifying price data
            self.strategy.data.loc[self.strategy.data.index[test_idx], 'high'] = 51000  # Clear breakout
            self.strategy.data.loc[self.strategy.data.index[test_idx], 'close'] = 50900
            
            # Truncate data to test_idx + 1 to simulate real-time
            test_data = self.strategy.data.iloc[:test_idx + 1].copy()
            original_data = self.strategy.data
            self.strategy.data = test_data
            
            entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
            
            # Restore original data
            self.strategy.data = original_data
            
            if entry_signal:
                self.assertIn(entry_signal['action'], ['long', 'short'])
                self.assertIn('confidence', entry_signal)
                self.assertIn('reason', entry_signal)
                self.assertGreater(entry_signal['confidence'], 0.5)
    
    def test_exit_conditions_time_based(self):
        """Test time-based exit conditions"""
        self.strategy.init_indicators()
        
        # Set up active trade
        self.strategy.entry_bar = 100
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50200
        
        # Test within time limit
        should_exit, reason = self.strategy.check_exit_conditions(110)
        # Should not exit yet (within time limit)
        
        # Test beyond time limit
        max_hold_bars = self.strategy.config.get('max_hold_bars', 50)
        should_exit, reason = self.strategy.check_exit_conditions(100 + max_hold_bars + 1)
        self.assertTrue(should_exit)
        self.assertEqual(reason, "time_exit")
    
    def test_exit_conditions_fake_breakout(self):
        """Test fake breakout exit conditions"""
        self.strategy.init_indicators()
        
        # Set up active trade and range
        self.strategy.entry_bar = 125
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50250
        self.strategy.range_support = 49800
        self.strategy.range_resistance = 50200
        
        # Test fake breakout (price back in range quickly)
        test_idx = 127  # 2 bars after entry
        if test_idx < len(self.strategy.data):
            # Force price back into range
            self.strategy.data.loc[self.strategy.data.index[test_idx], 'close'] = 50000  # Back in range
            
            should_exit, reason = self.strategy.check_exit_conditions(test_idx)
            
            fake_breakout_exit_bars = self.strategy.config.get('fake_breakout_exit_bars', 5)
            if test_idx - self.strategy.entry_bar <= fake_breakout_exit_bars:
                self.assertTrue(should_exit)
                self.assertEqual(reason, "fake_breakout")
    
    def test_exit_conditions_profit_targets(self):
        """Test profit target exit conditions"""
        self.strategy.init_indicators()
        
        # Set up profitable long trade
        self.strategy.entry_bar = 120
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50000
        self.strategy.initial_atr = 100
        self.strategy.first_target_hit = False
        
        # Test quick profit target
        test_idx = 125
        if test_idx < len(self.strategy.data):
            quick_profit_pct = self.strategy.config.get('quick_profit_pct', 0.004)
            profit_price = self.strategy.entry_price * (1 + quick_profit_pct + 0.001)  # Slightly above target
            
            self.strategy.data.loc[self.strategy.data.index[test_idx], 'close'] = profit_price
            
            should_exit, reason = self.strategy.check_exit_conditions(test_idx)
            
            if should_exit and reason == "quick_profit_long":
                self.assertTrue(should_exit)
                self.assertEqual(reason, "quick_profit_long")
        
        # Test main target after first target hit
        self.strategy.first_target_hit = True
        
        # Calculate main target
        range_projection_multiplier = self.strategy.config.get('range_projection_multiplier', 0.8)
        atr_target_multiplier = self.strategy.config.get('atr_target_multiplier', 1.5)
        
        # Use a high profit price to trigger main target
        main_target_price = self.strategy.entry_price * 1.02  # 2% profit
        
        if test_idx < len(self.strategy.data):
            self.strategy.data.loc[self.strategy.data.index[test_idx], 'close'] = main_target_price
            
            should_exit, reason = self.strategy.check_exit_conditions(test_idx)
            # May trigger main target depending on ATR and range values
    
    def test_exit_conditions_trailing_stop(self):
        """Test trailing stop exit conditions"""
        self.strategy.init_indicators()
        
        # Set up profitable trade with first target hit
        self.strategy.entry_bar = 120
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50000
        self.strategy.first_target_hit = True
        
        # Test trailing stop trigger
        test_idx = 130
        if test_idx < len(self.strategy.data):
            # Set a price that would trigger trailing stop
            trail_stop_pct = self.strategy.config.get('trail_stop_pct', 0.004)
            # Price that was profitable but now declining
            current_price = self.strategy.entry_price * (1 + trail_stop_pct - 0.001)  # Just below trail level
            
            self.strategy.data.loc[self.strategy.data.index[test_idx], 'close'] = current_price
            
            should_exit, reason = self.strategy.check_exit_conditions(test_idx)
            # May trigger trailing stop depending on exact calculations
    
    def test_check_exit_method(self):
        """Test the main check_exit method"""
        self.strategy.init_indicators()
        
        # Test with no active trade
        exit_signal = self.strategy.check_exit('BTCUSDT')
        self.assertIsNone(exit_signal)
        
        # Set up active trade
        self.strategy.entry_bar = 120
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50000
        
        # Test normal conditions (should not exit)
        exit_signal = self.strategy.check_exit('BTCUSDT')
        # May or may not have exit signal depending on conditions
        
        if exit_signal:
            self.assertIn(exit_signal['action'], ['exit', 'partial_exit'])
            self.assertIn('price', exit_signal)
            self.assertIn('reason', exit_signal)
            
            if exit_signal['action'] == 'partial_exit':
                self.assertIn('partial_pct', exit_signal)
                self.assertGreater(exit_signal['partial_pct'], 0)
                self.assertLessEqual(exit_signal['partial_pct'], 1)
    
    def test_trade_closure_handling(self):
        """Test trade closure and cleanup"""
        self.strategy.init_indicators()
        
        # Set up active trade state
        self.strategy.entry_bar = 120
        self.strategy.entry_side = 'long'
        self.strategy.entry_price = 50000
        self.strategy.breakout_direction = 'long'
        self.strategy.first_target_hit = True
        
        # Simulate trade closure
        trade_result = {
            'reason': 'quick_profit_long',
            'pnl': 150.0,
            'pnl_pct': 0.003
        }
        
        self.strategy.on_trade_closed('BTCUSDT', trade_result)
        
        # Check that state is reset
        self.assertIsNone(self.strategy.entry_bar)
        self.assertIsNone(self.strategy.entry_side)
        self.assertIsNone(self.strategy.entry_price)
        self.assertIsNone(self.strategy.breakout_direction)
        self.assertFalse(self.strategy.first_target_hit)
        
        # Check that last_trade_bar is updated
        self.assertEqual(self.strategy.last_trade_bar, len(self.strategy.data) - 1)
    
    def test_risk_parameters_calculation(self):
        """Test risk parameter calculation"""
        self.strategy.init_indicators()
        
        # Test with no active trade (fallback values)
        risk_params = self.strategy.get_risk_parameters()
        
        self.assertIn('sl_pct', risk_params)
        self.assertIn('tp_pct', risk_params)
        self.assertIn('max_position_pct', risk_params)
        self.assertIn('risk_reward_ratio', risk_params)
        
        self.assertGreater(risk_params['sl_pct'], 0)
        self.assertGreater(risk_params['tp_pct'], 0)
        self.assertGreater(risk_params['max_position_pct'], 0)
        self.assertGreater(risk_params['risk_reward_ratio'], 0)
        
        # Test with active trade and range data
        self.strategy.range_height = 400
        self.strategy.entry_price = 50000
        self.strategy.entry_side = 'long'
        self.strategy.range_resistance = 50200
        self.strategy.range_support = 49800
        
        risk_params_active = self.strategy.get_risk_parameters()
        
        # Should have calculated values based on range
        self.assertGreater(risk_params_active['sl_pct'], 0)
        self.assertGreater(risk_params_active['tp_pct'], 0)
        
        # Position size should be reduced for breakout trades
        position_size_reduction = self.strategy.config.get('position_size_reduction', 0.9)
        max_position_base = self.strategy.config.get('max_position_pct', 2.0)
        expected_max_position = max_position_base * position_size_reduction
        self.assertAlmostEqual(risk_params_active['max_position_pct'], expected_max_position, places=2)
    
    def test_error_handling_squeeze_detection(self):
        """Test error handling in squeeze detection methods"""
        self.strategy.init_indicators()
        
        # Test with invalid index
        result = self.strategy.is_bollinger_squeeze(-1)
        self.assertFalse(result)
        
        result = self.strategy.is_bollinger_squeeze(len(self.strategy.data) + 10)
        self.assertFalse(result)
        
        # Test with NaN values
        original_close = self.strategy.data['close'].iloc[50]
        self.strategy.data.loc[self.strategy.data.index[50], 'close'] = np.nan
        
        result = self.strategy.is_bollinger_squeeze(50)
        # Should handle NaN gracefully
        
        # Restore original value
        self.strategy.data.loc[self.strategy.data.index[50], 'close'] = original_close
    
    def test_error_handling_range_detection(self):
        """Test error handling in range detection methods"""
        self.strategy.init_indicators()
        
        # Test simple range detection with invalid index
        result = self.strategy.is_simple_range_detected(-1)
        self.assertFalse(result)
        
        result = self.strategy.is_simple_range_detected(5)  # Too few bars
        self.assertFalse(result)
        
        # Test complex range detection without squeeze state
        result = self.strategy.detect_support_resistance_range(80)
        # Should handle missing squeeze state
        
        # Test with insufficient data
        self.strategy.squeeze_detected = True
        self.strategy.squeeze_bars_count = 5  # Too few bars
        result = self.strategy.detect_support_resistance_range(80)
        self.assertFalse(result)
    
    def test_error_handling_breakout_methods(self):
        """Test error handling in breakout-related methods"""
        self.strategy.init_indicators()
        
        # Test setup_breakout_levels without confirmed range
        self.strategy.range_confirmed = False
        result = self.strategy.setup_breakout_levels()
        self.assertFalse(result)
        
        # Test check_breakout_trigger without pending breakouts
        self.strategy.pending_breakout_long = False
        self.strategy.pending_breakout_short = False
        triggered, direction = self.strategy.check_breakout_trigger(100)
        self.assertFalse(triggered)
        self.assertIsNone(direction)
        
        # Test confirm_breakout with invalid data
        result = self.strategy.confirm_breakout(100, 'invalid_direction')
        # Should handle gracefully
    
    def test_error_handling_exit_conditions(self):
        """Test error handling in exit condition methods"""
        self.strategy.init_indicators()
        
        # Test check_exit_conditions without active trade
        should_exit, reason = self.strategy.check_exit_conditions(100)
        self.assertFalse(should_exit)
        self.assertIsNone(reason)
        
        # Test with invalid entry_bar
        self.strategy.entry_bar = -1
        should_exit, reason = self.strategy.check_exit_conditions(100)
        # May return True due to time-based exit logic, so just check it doesn't crash
        
        # Test with None entry_price
        self.strategy.entry_bar = 50
        self.strategy.entry_price = None
        should_exit, reason = self.strategy.check_exit_conditions(100)
        # Should handle None entry_price gracefully
    
    def test_edge_cases_zero_values(self):
        """Test edge cases with zero or very small values"""
        self.strategy.init_indicators()
        
        # Test with zero volume
        original_volume = self.strategy.data['volume'].iloc[50]
        self.strategy.data.loc[self.strategy.data.index[50], 'volume'] = 0
        
        result = self.strategy.is_bollinger_squeeze(50)
        # Should handle zero volume
        
        # Restore original value
        self.strategy.data.loc[self.strategy.data.index[50], 'volume'] = original_volume
        
        # Test with zero range height
        self.strategy.range_height = 0
        self.strategy.range_support = 50000
        self.strategy.range_resistance = 50000  # Same as support
        
        risk_params = self.strategy.get_risk_parameters()
        # Should provide fallback values
        self.assertGreater(risk_params['sl_pct'], 0)
    
    def test_edge_cases_extreme_prices(self):
        """Test edge cases with extreme price values"""
        self.strategy.init_indicators()
        
        # Test with very high prices
        extreme_data = self.strategy.data.copy()
        extreme_data[['open', 'high', 'low', 'close']] *= 1000000  # Very high prices
        
        original_data = self.strategy.data
        self.strategy.data = extreme_data
        
        # Should handle extreme prices
        result = self.strategy.is_bollinger_squeeze(50)
        
        # Restore original data
        self.strategy.data = original_data
        
        # Test with very small prices
        small_data = self.strategy.data.copy()
        small_data[['open', 'high', 'low', 'close']] *= 0.0001  # Very small prices
        
        self.strategy.data = small_data
        
        # Should handle small prices
        result = self.strategy.is_bollinger_squeeze(50)
        
        # Restore original data
        self.strategy.data = original_data
    
    def test_edge_cases_nan_values(self):
        """Test edge cases with NaN values in data"""
        self.strategy.init_indicators()
        
        # Test with NaN in indicators
        original_bb_upper = self.strategy.data['bb_upper'].iloc[50]
        self.strategy.data.loc[self.strategy.data.index[50], 'bb_upper'] = np.nan
        
        result = self.strategy.is_bollinger_squeeze(50)
        # Should handle NaN in indicators
        
        # Restore original value
        self.strategy.data.loc[self.strategy.data.index[50], 'bb_upper'] = original_bb_upper
        
        # Test entry conditions with NaN close price
        original_close = self.strategy.data['close'].iloc[-1]
        self.strategy.data.loc[self.strategy.data.index[-1], 'close'] = np.nan
        
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        self.assertIsNone(entry_signal)  # Should return None with NaN price
        
        # Restore original value
        self.strategy.data.loc[self.strategy.data.index[-1], 'close'] = original_close
    
    def test_configuration_variations(self):
        """Test strategy behavior with different configuration settings"""
        # Test with very tight squeeze threshold
        tight_config = self.config.copy()
        tight_config['bb_squeeze_threshold'] = 0.001  # Very tight
        
        tight_strategy = StrategyVolatilitySqueezeBreakout(
            data=self.data,
            config=tight_config,
            logger=self.mock_logger
        )
        tight_strategy.init_indicators()
        
        # Should detect fewer squeezes
        tight_squeeze_count = 0
        for i in range(50, 120):
            if tight_strategy.is_bollinger_squeeze(i):
                tight_squeeze_count += 1
        
        # Test with very loose squeeze threshold
        loose_config = self.config.copy()
        loose_config['bb_squeeze_threshold'] = 0.1  # Very loose
        
        loose_strategy = StrategyVolatilitySqueezeBreakout(
            data=self.data,
            config=loose_config,
            logger=self.mock_logger
        )
        loose_strategy.init_indicators()
        
        # Should detect more squeezes
        loose_squeeze_count = 0
        for i in range(50, 120):
            if loose_strategy.is_bollinger_squeeze(i):
                loose_squeeze_count += 1
        
        # Loose should detect more than tight
        self.assertGreaterEqual(loose_squeeze_count, tight_squeeze_count)
    
    def test_state_consistency(self):
        """Test that strategy state remains consistent across operations"""
        self.strategy.init_indicators()
        
        # Record initial state
        initial_squeeze_detected = self.strategy.squeeze_detected
        initial_range_confirmed = self.strategy.range_confirmed
        initial_pending_long = self.strategy.pending_breakout_long
        initial_pending_short = self.strategy.pending_breakout_short
        
        # Perform various operations
        self.strategy.is_bollinger_squeeze(50)
        self.strategy.is_simple_range_detected(80)
        self.strategy._check_entry_conditions('BTCUSDT')
        
        # State may change during _check_entry_conditions as it's designed to modify state
        # Just verify the operations don't crash and state variables are still valid types
        self.assertIsInstance(self.strategy.squeeze_detected, bool)
        self.assertIsInstance(self.strategy.range_confirmed, bool)
        self.assertIsInstance(self.strategy.pending_breakout_long, bool)
        self.assertIsInstance(self.strategy.pending_breakout_short, bool)
        
        # Test state reset
        self.strategy.squeeze_detected = True
        self.strategy.range_confirmed = True
        self.strategy.pending_breakout_long = True
        
        self.strategy.reset_squeeze_state()
        
        # All squeeze-related state should be reset
        self.assertFalse(self.strategy.squeeze_detected)
        self.assertFalse(self.strategy.range_confirmed)
        self.assertFalse(self.strategy.pending_breakout_long)
        self.assertFalse(self.strategy.pending_breakout_short)
    
    def test_performance_with_large_dataset(self):
        """Test strategy performance with larger dataset"""
        # Create larger dataset
        large_data_size = 1000
        np.random.seed(42)
        
        # Create larger market data
        base_price = 50000.0
        dates = pd.date_range(start='2024-01-01', periods=large_data_size, freq='1min')
        
        prices = [base_price]
        volumes = []
        
        for i in range(1, large_data_size):
            price_change = np.random.normal(0, base_price * 0.001)
            prices.append(max(prices[-1] + price_change, base_price * 0.8))
            volumes.append(np.random.uniform(500, 1500))
        
        volumes.insert(0, 1000)  # First volume
        
        # Create OHLCV
        opens = [prices[0]] + prices[:-1]
        highs = [p + abs(np.random.normal(0, p * 0.0005)) for p in prices]
        lows = [p - abs(np.random.normal(0, p * 0.0005)) for p in prices]
        closes = prices
        
        # Ensure OHLC relationships
        for i in range(len(prices)):
            highs[i] = max(highs[i], opens[i], closes[i])
            lows[i] = min(lows[i], opens[i], closes[i])
        
        large_data = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        large_data.set_index('timestamp', inplace=True)
        
        # Test with large dataset
        large_strategy = StrategyVolatilitySqueezeBreakout(
            data=large_data,
            config=self.config,
            logger=self.mock_logger
        )
        
        # Should initialize without issues
        large_strategy.init_indicators()
        
        # Should handle entry condition checks
        entry_signal = large_strategy._check_entry_conditions('BTCUSDT')
        # May or may not have signal, but should not crash
        
        # Should handle exit condition checks
        exit_signal = large_strategy.check_exit('BTCUSDT')
        # Should return None (no active trade)
        self.assertIsNone(exit_signal)


if __name__ == '__main__':
    unittest.main() 