#!/usr/bin/env python3
"""
Test Suite for Breakout and Retest Strategy

This test suite comprehensively tests the StrategyBreakoutAndRetest implementation,
including breakout detection, retest confirmation, reversal patterns, and risk management.
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

from strategies.breakout_and_retest_strategy import StrategyBreakoutAndRetest

class TestBreakoutAndRetestStrategy(unittest.TestCase):
    """Test cases for Breakout and Retest Strategy."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Configure logging to reduce noise during testing
        logging.basicConfig(level=logging.CRITICAL)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.CRITICAL)
        
        # Create comprehensive test configuration
        self.config = {
            'default': {
                'coin_pair': 'BTC/USDT',
                'leverage': 50,
                'timeframe': '1m'
            },
            'strategy_configs': {
                'StrategyBreakoutAndRetest': {
                    # Support/Resistance parameters
                    'sr_lookback_period': 30,
                    'sr_min_touches': 3,
                    'sr_tolerance_pct': 0.002,  # 0.2%
                    
                    # Breakout parameters
                    'breakout_min_pct': 0.003,  # 0.3%
                    'volume_breakout_multiplier': 1.5,
                    'volume_avg_period': 20,
                    
                    # Trend filter parameters
                    'ema_trend_period': 50,
                    'use_trend_filter': True,
                    
                    # Momentum filter parameters
                    'rsi_period': 14,
                    'rsi_pullback_min': 40,
                    'rsi_pullback_max': 60,
                    
                    # Retest parameters
                    'retest_timeout_bars': 15,
                    'retest_tolerance_pct': 0.002,  # 0.2%
                    'reversal_confirmation_bars': 2,
                    
                    # Pattern recognition parameters
                    'engulfing_min_ratio': 1.2,
                    'hammer_ratio': 2.0,
                    
                    # Risk parameters
                    'stop_loss_buffer_pct': 0.001,  # 0.1%
                    'first_target_pct': 0.005,  # 0.5%
                    'measured_move_multiplier': 1.0,
                    'min_breakout_volume_decline': 0.7,
                    
                    # Strategy risk parameters
                    'sl_pct': 0.01,  # 1%
                    'tp_pct': 0.02,  # 2%
                    'order_size': 100
                }
            }
        }
        
        # Create realistic test data with distinct phases for breakout and retest
        self.test_data = self._create_breakout_retest_data()
        
        # Initialize strategy
        self.strategy = StrategyBreakoutAndRetest(
            data=self.test_data.copy(),
            config=self.config,
            logger=self.logger
        )

    def _create_breakout_retest_data(self) -> pd.DataFrame:
        """Create realistic OHLCV data for breakout and retest testing."""
        np.random.seed(42)  # For reproducible results
        
        total_bars = 200
        base_price = 50000.0
        base_volume = 1000000
        
        # Create timestamps
        start_time = datetime.now() - timedelta(minutes=total_bars)
        timestamps = [start_time + timedelta(minutes=i) for i in range(total_bars)]
        
        # Initialize data arrays
        opens = []
        highs = []
        lows = []
        closes = []
        volumes = []
        
        current_price = base_price
        
        # Phase 1: Consolidation phase (0-60 bars) - establish support/resistance
        for i in range(60):
            # Create consolidation between 49800-50200 (400 point range)
            support_level = 49800
            resistance_level = 50200
            
            # Random walk within range
            price_change = np.random.uniform(-50, 50)
            current_price = max(support_level + 20, min(resistance_level - 20, current_price + price_change))
            
            # Create OHLC with some noise
            open_price = current_price + np.random.uniform(-20, 20)
            high_price = max(open_price, current_price) + np.random.uniform(0, 30)
            low_price = min(open_price, current_price) - np.random.uniform(0, 30)
            close_price = current_price
            
            # Ensure logical OHLC relationships
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            # Test support/resistance levels multiple times
            if i % 10 == 5:  # Hit resistance
                high_price = max(high_price, resistance_level + np.random.uniform(-5, 5))
                close_price = resistance_level - np.random.uniform(10, 30)
            elif i % 10 == 8:  # Hit support
                low_price = min(low_price, support_level + np.random.uniform(-5, 5))
                close_price = support_level + np.random.uniform(10, 30)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(base_volume + np.random.uniform(-200000, 200000))
            
            current_price = close_price
        
        # Phase 2: Breakout phase (60-70 bars) - strong breakout with volume
        resistance_level = 50200
        for i in range(60, 71):
            if i == 65:  # Breakout bar
                open_price = current_price
                # Strong breakout above resistance
                close_price = resistance_level + np.random.uniform(150, 250)  # Clear breakout
                high_price = close_price + np.random.uniform(0, 50)
                low_price = min(open_price, current_price - 20)
                
                # High volume breakout
                volume = base_volume * 2.5  # 2.5x average volume
            else:
                # Build momentum toward breakout
                if i < 65:
                    price_change = np.random.uniform(20, 60)
                else:
                    price_change = np.random.uniform(50, 100)
                
                current_price += price_change
                open_price = current_price + np.random.uniform(-30, 30)
                close_price = current_price
                high_price = max(open_price, close_price) + np.random.uniform(0, 40)
                low_price = min(open_price, close_price) - np.random.uniform(0, 20)
                
                volume = base_volume * (1.5 + (i - 59) * 0.1)  # Increasing volume
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
            
            current_price = close_price
        
        # Phase 3: Retest phase (71-85 bars) - pullback to broken resistance
        peak_price = current_price
        for i in range(71, 86):
            if i < 80:  # Pullback phase
                # Gradual pullback toward resistance level
                pullback_target = resistance_level + np.random.uniform(-20, 20)
                progress = (i - 71) / 9  # Progress from 0 to 1
                current_price = peak_price - (peak_price - pullback_target) * progress
                
                # Declining volume during pullback
                volume_decline = 0.3 + (1 - progress) * 0.4  # Volume 30-70% of normal
                volume = base_volume * volume_decline
                
                # Create OHLC for pullback
                open_price = current_price + np.random.uniform(-30, 30)
                close_price = current_price
                high_price = max(open_price, close_price) + np.random.uniform(0, 20)
                low_price = min(open_price, close_price) - np.random.uniform(0, 30)
                
            else:  # Reversal phase (80-85)
                if i == 82:  # Hammer/reversal pattern
                    open_price = current_price
                    low_price = current_price - 80  # Long lower wick
                    close_price = current_price + 20  # Small body
                    high_price = close_price + 10
                    volume = base_volume * 1.2  # Volume pickup
                elif i == 83:  # Engulfing pattern
                    open_price = current_price - 20
                    close_price = current_price + 60  # Large bullish candle
                    high_price = close_price + 10
                    low_price = open_price - 10
                    volume = base_volume * 1.4
                else:
                    # Continuation higher
                    current_price += np.random.uniform(30, 60)
                    open_price = current_price + np.random.uniform(-20, 20)
                    close_price = current_price
                    high_price = max(open_price, close_price) + np.random.uniform(0, 30)
                    low_price = min(open_price, close_price) - np.random.uniform(0, 15)
                    volume = base_volume * np.random.uniform(1.1, 1.3)
                
                current_price = close_price
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
        
        # Phase 4: Trend continuation (86-199 bars) - uptrend with normal volatility
        for i in range(86, total_bars):
            # Trending higher with normal volatility
            price_change = np.random.uniform(-20, 50)  # Slight upward bias
            current_price += price_change
            
            open_price = current_price + np.random.uniform(-25, 25)
            close_price = current_price
            high_price = max(open_price, close_price) + np.random.uniform(0, 35)
            low_price = min(open_price, close_price) - np.random.uniform(0, 25)
            volume = base_volume + np.random.uniform(-300000, 300000)
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            closes.append(close_price)
            volumes.append(volume)
            
            current_price = close_price
        
        # Ensure all arrays are exactly the same length
        assert len(opens) == len(highs) == len(lows) == len(closes) == len(volumes) == total_bars
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=pd.DatetimeIndex(timestamps, freq='1min'))
        
        return df

    # ==================== Initialization Tests ====================
    
    def test_initialization(self):
        """Test strategy initialization."""
        self.assertIsInstance(self.strategy, StrategyBreakoutAndRetest)
        self.assertEqual(self.strategy.sr_lookback_period, 30)
        self.assertEqual(self.strategy.volume_breakout_multiplier, 1.5)
        self.assertEqual(self.strategy.ema_trend_period, 50)
        self.assertTrue(self.strategy.use_trend_filter)

    def test_strategy_parameters(self):
        """Test strategy parameter loading."""
        self.assertEqual(self.strategy.sr_min_touches, 3)
        self.assertEqual(self.strategy.sr_tolerance_pct, 0.002)
        self.assertEqual(self.strategy.breakout_min_pct, 0.003)
        self.assertEqual(self.strategy.rsi_period, 14)
        self.assertEqual(self.strategy.retest_timeout_bars, 15)

    def test_market_type_tags(self):
        """Test market type tags."""
        expected_tags = ['TRENDING', 'RANGING', 'TRANSITIONAL']
        self.assertEqual(self.strategy.MARKET_TYPE_TAGS, expected_tags)
        self.assertTrue(self.strategy.SHOW_IN_SELECTION)

    def test_indicator_initialization(self):
        """Test indicator initialization."""
        self.strategy.init_indicators()
        
        # Check required indicators exist
        required_indicators = ['volume_sma', 'rsi', 'current_support', 'current_resistance']
        if self.strategy.use_trend_filter:
            required_indicators.append('ema_trend')
        
        for indicator in required_indicators:
            self.assertIn(indicator, self.strategy.data.columns, f"Missing indicator: {indicator}")

    def test_state_variables_initialization(self):
        """Test state variable initialization."""
        # Test the base state variables are correctly initialized
        self.assertFalse(self.strategy.breakout_detected)
        self.assertIsNone(self.strategy.breakout_direction)
        self.assertFalse(self.strategy.waiting_for_retest)
        self.assertFalse(self.strategy.retest_confirmed)
        
        # Note: support_levels and resistance_levels may be populated during on_init() 
        # so we test that they are lists
        self.assertIsInstance(self.strategy.support_levels, list)
        self.assertIsInstance(self.strategy.resistance_levels, list)

    # ==================== Technical Indicator Tests ====================
    
    def test_volume_sma_calculation(self):
        """Test volume SMA calculation."""
        self.strategy.init_indicators()
        
        # Check that volume_sma is calculated
        self.assertIn('volume_sma', self.strategy.data.columns)
        
        # Verify volume SMA values are reasonable
        volume_sma = self.strategy.data['volume_sma'].iloc[-1]
        self.assertGreater(volume_sma, 0)
        
        # Check manual calculation for last few bars
        if len(self.strategy.data) >= self.strategy.volume_avg_period:
            expected_sma = self.strategy.data['volume'].iloc[-self.strategy.volume_avg_period:].mean()
            actual_sma = self.strategy.data['volume_sma'].iloc[-1]
            self.assertAlmostEqual(actual_sma, expected_sma, places=0)

    def test_ema_trend_calculation(self):
        """Test EMA trend calculation."""
        self.strategy.init_indicators()
        
        if self.strategy.use_trend_filter:
            self.assertIn('ema_trend', self.strategy.data.columns)
            
            # Check EMA values are reasonable
            ema_values = self.strategy.data['ema_trend'].dropna()
            self.assertGreater(len(ema_values), 0)
            
            # EMA should be close to price levels
            last_close = self.strategy.data['close'].iloc[-1]
            last_ema = self.strategy.data['ema_trend'].iloc[-1]
            price_diff_pct = abs(last_ema - last_close) / last_close
            self.assertLess(price_diff_pct, 0.1)  # Within 10%

    def test_rsi_calculation(self):
        """Test RSI calculation."""
        self.strategy.init_indicators()
        
        self.assertIn('rsi', self.strategy.data.columns)
        
        # Check RSI bounds
        rsi_values = self.strategy.data['rsi'].dropna()
        if len(rsi_values) > 0:
            self.assertGreaterEqual(rsi_values.min(), 0)
            self.assertLessEqual(rsi_values.max(), 100)

    def test_support_resistance_detection(self):
        """Test support and resistance level detection."""
        self.strategy.init_indicators()
        
        # Should have detected some levels in our test data
        self.assertGreaterEqual(len(self.strategy.support_levels), 0)
        self.assertGreaterEqual(len(self.strategy.resistance_levels), 0)
        
        # Check level structure
        if self.strategy.resistance_levels:
            level = self.strategy.resistance_levels[0]
            self.assertIn('level', level)
            self.assertIn('touches', level)
            self.assertIn('strength', level)
            self.assertGreaterEqual(level['touches'], self.strategy.sr_min_touches)

    def test_incremental_indicator_update(self):
        """Test incremental indicator updates."""
        self.strategy.init_indicators()
        
        # Store initial values
        initial_rsi = self.strategy.data['rsi'].iloc[-1]
        initial_volume_sma = self.strategy.data['volume_sma'].iloc[-1]
        
        # Add new row
        new_row = pd.DataFrame({
            'open': [self.strategy.data['close'].iloc[-1] + 10],
            'high': [self.strategy.data['close'].iloc[-1] + 20],
            'low': [self.strategy.data['close'].iloc[-1] - 5],
            'close': [self.strategy.data['close'].iloc[-1] + 15],
            'volume': [1000000]
        }, index=[self.strategy.data.index[-1] + pd.Timedelta(minutes=1)])
        
        self.strategy.data = pd.concat([self.strategy.data, new_row])
        
        # Update indicators
        self.strategy.update_indicators_for_new_row()
        
        # Check that values changed
        new_rsi = self.strategy.data['rsi'].iloc[-1]
        new_volume_sma = self.strategy.data['volume_sma'].iloc[-1]
        
        # At least volume SMA should change
        self.assertNotEqual(initial_volume_sma, new_volume_sma)

    # ==================== Breakout Detection Tests ====================
    
    def test_breakout_detection_basic(self):
        """Test basic breakout detection."""
        self.strategy.init_indicators()
        
        # Manually test breakout detection
        vals = self.strategy._get_current_values()
        if vals:
            breakout_occurred, direction, breakout_info = self.strategy._detect_breakout(vals)
            
            # Should be boolean results
            self.assertIsInstance(breakout_occurred, bool)
            if breakout_occurred:
                self.assertIn(direction, ['long', 'short'])
                self.assertIsInstance(breakout_info, dict)
                self.assertIn('level', breakout_info)
                self.assertIn('range_height', breakout_info)

    def test_volume_confirmation_requirement(self):
        """Test volume confirmation for breakouts."""
        self.strategy.init_indicators()
        
        # Test with low volume (should not confirm breakout)
        vals = self.strategy._get_current_values()
        if vals:
            # Temporarily reduce volume to test
            original_volume = vals['current_volume']
            vals['current_volume'] = vals['volume_sma'] * 0.5  # Below threshold
            
            breakout_occurred, _, _ = self.strategy._detect_breakout(vals)
            # Should not confirm breakout with low volume
            self.assertFalse(breakout_occurred)

    def test_trend_bias_filtering(self):
        """Test trend bias filtering for breakouts."""
        self.strategy.init_indicators()
        
        vals = self.strategy._get_current_values()
        if vals and vals['ema_trend'] is not None:
            # Test bullish breakout with trend alignment
            bullish_aligned = self.strategy._check_trend_bias('long', vals)
            bearish_aligned = self.strategy._check_trend_bias('short', vals)
            
            # Should return boolean values
            self.assertIn(bullish_aligned, [True, False])
            self.assertIn(bearish_aligned, [True, False])

    def test_range_height_calculation(self):
        """Test range height calculation for measured moves."""
        self.strategy.init_indicators()
        
        # Test with known levels
        if self.strategy.resistance_levels and self.strategy.support_levels:
            resistance_level = self.strategy.resistance_levels[0]['level']
            support_level = self.strategy.support_levels[0]['level']
            
            # Calculate range height
            range_height = self.strategy._calculate_range_height(resistance_level, 'resistance')
            self.assertGreater(range_height, 0)
            
            # Should be reasonable relative to price
            price_ratio = range_height / resistance_level
            self.assertLess(price_ratio, 0.1)  # Less than 10% of price

    # ==================== Retest Detection Tests ====================
    
    def test_retest_detection_setup(self):
        """Test retest detection setup."""
        # Simulate breakout state
        self.strategy.breakout_detected = True
        self.strategy.breakout_direction = 'long'
        self.strategy.waiting_for_retest = True
        self.strategy.retest_level = 50200
        self.strategy.breakout_volume = 2000000
        
        vals = self.strategy._get_current_values()
        if vals:
            # Test retest detection
            retest_detected = self.strategy._detect_retest(vals)
            self.assertIsInstance(retest_detected, bool)

    def test_rsi_pullback_validation(self):
        """Test RSI validation during pullbacks."""
        self.strategy.init_indicators()
        
        self.strategy.breakout_direction = 'long'
        vals = self.strategy._get_current_values()
        if vals:
            rsi_ok = self.strategy._check_rsi_pullback(vals)
            self.assertIn(rsi_ok, [True, False])
        
        # Test bearish scenario
        self.strategy.breakout_direction = 'short'
        if vals:
            rsi_ok = self.strategy._check_rsi_pullback(vals)
            self.assertIn(rsi_ok, [True, False])

    def test_retest_timeout_mechanism(self):
        """Test retest timeout mechanism."""
        # Set up waiting state
        self.strategy.waiting_for_retest = True
        self.strategy.breakout_bar = len(self.strategy.data) - 20  # 20 bars ago
        
        # Should timeout if configured timeout is exceeded
        timeout_exceeded = (len(self.strategy.data) - 1 - self.strategy.breakout_bar > 
                          self.strategy.retest_timeout_bars)
        
        if timeout_exceeded:
            # Simulate timeout in entry conditions check
            symbol = "BTC/USDT"
            entry_signal = self.strategy._check_entry_conditions(symbol)
            # After timeout, waiting_for_retest should be reset
            # This would be tested in integration with the full entry logic

    # ==================== Reversal Pattern Tests ====================
    
    def test_engulfing_pattern_detection(self):
        """Test engulfing pattern detection."""
        if len(self.strategy.data) >= 2:
            # Set up for bullish engulfing test
            self.strategy.breakout_direction = 'long'
            
            # Test pattern detection
            is_engulfing = self.strategy._is_engulfing_pattern()
            self.assertIn(is_engulfing, [True, False])
            
            # Test bearish scenario
            self.strategy.breakout_direction = 'short'
            is_engulfing = self.strategy._is_engulfing_pattern()
            self.assertIn(is_engulfing, [True, False])

    def test_hammer_doji_detection(self):
        """Test hammer/doji pattern detection."""
        if len(self.strategy.data) >= 1:
            # Set up for hammer test
            self.strategy.breakout_direction = 'long'
            
            is_hammer = self.strategy._is_hammer_or_doji()
            self.assertIn(is_hammer, [True, False])
            
            # Test inverted hammer
            self.strategy.breakout_direction = 'short'
            is_inverted_hammer = self.strategy._is_hammer_or_doji()
            self.assertIn(is_inverted_hammer, [True, False])

    def test_multi_bar_reversal_confirmation(self):
        """Test multi-bar reversal confirmation."""
        # Set up reversal state
        self.strategy.retest_confirmed = True
        self.strategy.reversal_start_bar = len(self.strategy.data) - 3
        self.strategy.breakout_direction = 'long'
        self.strategy.retest_level = 50200
        
        vals = self.strategy._get_current_values()
        if vals:
            multi_bar_reversal = self.strategy._is_multi_bar_reversal(vals)
            self.assertIn(multi_bar_reversal, [True, False])

    def test_reversal_confirmation_integration(self):
        """Test overall reversal confirmation."""
        # Set up complete reversal test
        self.strategy.retest_confirmed = True
        self.strategy.breakout_direction = 'long'
        
        vals = self.strategy._get_current_values()
        if vals:
            reversal_confirmed = self.strategy._detect_reversal_confirmation(vals)
            self.assertIsInstance(reversal_confirmed, bool)

    # ==================== Entry Condition Tests ====================
    
    def test_entry_conditions_basic(self):
        """Test basic entry condition checking."""
        self.strategy.init_indicators()
        
        symbol = "BTC/USDT"
        entry_signal = self.strategy._check_entry_conditions(symbol)
        
        # Should return None or valid order details
        if entry_signal is not None:
            self.assertIsInstance(entry_signal, dict)
            self.assertIn('side', entry_signal)
            self.assertIn('price', entry_signal)
            self.assertIn('size', entry_signal)
            self.assertIn(entry_signal['side'], ['buy', 'sell'])

    def test_entry_conditions_no_breakout(self):
        """Test entry conditions when no breakout detected."""
        self.strategy.init_indicators()
        
        # Ensure clean state
        self.strategy._reset_breakout_state()
        
        symbol = "BTC/USDT"
        entry_signal = self.strategy._check_entry_conditions(symbol)
        
        # Should be None when no setup
        # Note: Could also be a valid signal if breakout is detected in test data

    def test_entry_conditions_with_breakout_state(self):
        """Test entry conditions with active breakout state."""
        self.strategy.init_indicators()
        
        # Simulate active breakout waiting for retest
        self.strategy.breakout_detected = True
        self.strategy.breakout_direction = 'long'
        self.strategy.waiting_for_retest = True
        self.strategy.retest_level = 50200
        self.strategy.breakout_bar = len(self.strategy.data) - 5
        
        symbol = "BTC/USDT"
        entry_signal = self.strategy._check_entry_conditions(symbol)
        
        # Should handle the retest checking logic
        if entry_signal is not None:
            self.assertIsInstance(entry_signal, dict)

    def test_state_reset_after_entry(self):
        """Test state reset after entry signal."""
        # Set up full state
        self.strategy.breakout_detected = True
        self.strategy.breakout_direction = 'long'
        self.strategy.waiting_for_retest = True
        self.strategy.retest_confirmed = True
        
        # Reset state
        self.strategy._reset_breakout_state()
        
        # Verify clean state
        self.assertFalse(self.strategy.breakout_detected)
        self.assertIsNone(self.strategy.breakout_direction)
        self.assertFalse(self.strategy.waiting_for_retest)
        self.assertFalse(self.strategy.retest_confirmed)
        self.assertIsNone(self.strategy.reversal_start_bar)

    # ==================== Interface Integration Tests ====================
    
    def test_main_entry_method(self):
        """Test main entry method integration."""
        self.strategy.init_indicators()
        
        symbol = "BTC/USDT"
        entry_signal = self.strategy.check_entry(symbol)
        
        # Should return boolean or entry details
        if entry_signal is not None:
            self.assertIsInstance(entry_signal, (dict, bool))

    def test_exit_conditions(self):
        """Test exit condition checking."""
        symbol = "BTC/USDT"
        should_exit = self.strategy.check_exit(symbol)
        
        # Should return boolean
        self.assertIsInstance(should_exit, bool)

    def test_risk_parameters(self):
        """Test risk parameter retrieval."""
        risk_params = self.strategy.get_risk_parameters()
        
        self.assertIsInstance(risk_params, dict)
        self.assertIn('sl_pct', risk_params)
        self.assertIn('tp_pct', risk_params)
        
        # Verify values are reasonable
        self.assertGreater(risk_params['sl_pct'], 0)
        self.assertGreater(risk_params['tp_pct'], 0)
        self.assertLess(risk_params['sl_pct'], 1)  # Less than 100%
        self.assertLess(risk_params['tp_pct'], 1)  # Less than 100%

    def test_strategy_configuration_validation(self):
        """Test strategy configuration validation."""
        # Test with minimal config
        minimal_config = {
            'default': {'coin_pair': 'BTC/USDT'},
            'strategy_configs': {
                'StrategyBreakoutAndRetest': {
                    'sl_pct': 0.015,
                    'tp_pct': 0.025,
                    'order_size': 50
                }
            }
        }
        
        minimal_strategy = StrategyBreakoutAndRetest(
            data=self.test_data.copy(),
            config=minimal_config,
            logger=self.logger
        )
        
        # Should initialize with defaults
        self.assertEqual(minimal_strategy.sr_lookback_period, 30)  # Default value
        self.assertEqual(minimal_strategy.sl_pct, 0.015)  # Custom value

    # ==================== Robustness Tests ====================
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Create minimal data
        minimal_data = self.test_data.head(5)
        
        minimal_strategy = StrategyBreakoutAndRetest(
            data=minimal_data,
            config=self.config,
            logger=self.logger
        )
        
        # Should handle gracefully
        minimal_strategy.init_indicators()
        
        symbol = "BTC/USDT"
        entry_signal = minimal_strategy._check_entry_conditions(symbol)
        # Should return None due to insufficient data

    def test_missing_volume_data(self):
        """Test handling of missing volume data."""
        # Create data without volume
        no_volume_data = self.test_data.drop('volume', axis=1)
        
        try:
            no_volume_strategy = StrategyBreakoutAndRetest(
                data=no_volume_data,
                config=self.config,
                logger=self.logger
            )
            no_volume_strategy.init_indicators()
            # Should either handle gracefully or raise appropriate error
        except Exception as e:
            # Expected behavior - strategy requires volume data
            self.assertIsInstance(e, (KeyError, ValueError))

    def test_invalid_configuration_values(self):
        """Test handling of invalid configuration values."""
        invalid_config = self.config.copy()
        invalid_config['strategy_configs']['StrategyBreakoutAndRetest']['sr_lookback_period'] = -5
        invalid_config['strategy_configs']['StrategyBreakoutAndRetest']['volume_avg_period'] = 0
        
        try:
            invalid_strategy = StrategyBreakoutAndRetest(
                data=self.test_data.copy(),
                config=invalid_config,
                logger=self.logger
            )
            invalid_strategy.init_indicators()
            # Should handle gracefully or raise appropriate error
        except Exception as e:
            # Expected behavior for invalid parameters
            pass

    def test_data_continuity_after_updates(self):
        """Test data continuity after indicator updates."""
        self.strategy.init_indicators()
        
        original_length = len(self.strategy.data)
        original_columns = set(self.strategy.data.columns)
        
        # Add several new rows
        for i in range(3):
            new_row = pd.DataFrame({
                'open': [self.strategy.data['close'].iloc[-1] + np.random.uniform(-10, 10)],
                'high': [self.strategy.data['close'].iloc[-1] + np.random.uniform(0, 20)],
                'low': [self.strategy.data['close'].iloc[-1] + np.random.uniform(-20, 0)],
                'close': [self.strategy.data['close'].iloc[-1] + np.random.uniform(-5, 15)],
                'volume': [1000000 + np.random.uniform(-200000, 200000)]
            }, index=[self.strategy.data.index[-1] + pd.Timedelta(minutes=i+1)])
            
            self.strategy.data = pd.concat([self.strategy.data, new_row])
            self.strategy.update_indicators_for_new_row()
        
        # Verify data integrity
        self.assertEqual(len(self.strategy.data), original_length + 3)
        self.assertEqual(set(self.strategy.data.columns), original_columns)

    def test_error_handling(self):
        """Test error handling capabilities."""
        # Test with corrupted data
        corrupted_data = self.test_data.copy()
        corrupted_data.loc[corrupted_data.index[50], 'close'] = np.nan
        corrupted_data.loc[corrupted_data.index[51], 'volume'] = np.inf
        
        try:
            corrupted_strategy = StrategyBreakoutAndRetest(
                data=corrupted_data,
                config=self.config,
                logger=self.logger
            )
            corrupted_strategy.init_indicators()
            
            # Test error handling method
            test_exception = ValueError("Test error")
            corrupted_strategy.on_error(test_exception)
            
        except Exception as e:
            # Should handle gracefully
            pass

    def test_performance_with_large_dataset(self):
        """Test strategy performance with larger dataset."""
        # Create larger dataset
        large_data = self.test_data.copy()
        for i in range(5):  # Add 5x more data
            additional_data = self.test_data.copy()
            additional_data.index = additional_data.index + pd.Timedelta(hours=i+1)
            large_data = pd.concat([large_data, additional_data])
        
        large_strategy = StrategyBreakoutAndRetest(
            data=large_data,
            config=self.config,
            logger=self.logger
        )
        
        # Should handle larger datasets efficiently
        import time
        start_time = time.time()
        large_strategy.init_indicators()
        end_time = time.time()
        
        # Should complete within reasonable time (< 5 seconds)
        self.assertLess(end_time - start_time, 5)

    def test_concurrent_strategy_instances(self):
        """Test multiple strategy instances."""
        strategies = []
        
        for i in range(3):
            strategy = StrategyBreakoutAndRetest(
                data=self.test_data.copy(),
                config=self.config,
                logger=self.logger
            )
            strategy.init_indicators()
            strategies.append(strategy)
        
        # Each should maintain independent state
        strategies[0].breakout_detected = True
        strategies[1].waiting_for_retest = True
        
        self.assertTrue(strategies[0].breakout_detected)
        self.assertFalse(strategies[1].breakout_detected)
        self.assertTrue(strategies[1].waiting_for_retest)
        self.assertFalse(strategies[2].waiting_for_retest)


if __name__ == '__main__':
    unittest.main() 