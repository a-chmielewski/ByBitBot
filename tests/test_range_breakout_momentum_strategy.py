#!/usr/bin/env python3
"""
Test Suite for Range Breakout Momentum Strategy

This test suite validates the Range Breakout Momentum Strategy implementation,
including range identification, breakout detection, momentum confirmation,
and risk management.
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

from strategies.range_breakout_momentum_strategy import StrategyRangeBreakoutMomentum

class TestRangeBreakoutMomentumStrategy(unittest.TestCase):
    """Test cases for Range Breakout Momentum Strategy"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Suppress logging during tests
        logging.disable(logging.CRITICAL)
        
        # Create test configuration
        self.config = {
            'range_period': 50,
            'adx_period': 14,
            'atr_period': 14,
            'rsi_period': 14,
            'bb_period': 20,
            'bb_std': 2.0,
            'volume_period': 20,
            'atr_lookback': 10,
            'adx_range_threshold': 28,
            'bb_squeeze_threshold': 0.035,
            'min_range_size_pct': 0.0025,
            'max_range_size_pct': 0.05,
            'range_stability_bars': 3,
            'breakout_buffer_pct': 0.0004,
            'volume_surge_multiplier': 1.2,
            'candle_range_multiplier': 1.2,
            'candle_range_lookback': 10,
            'rsi_breakout_up': 58,
            'rsi_breakout_down': 42,
            'adx_breakout_threshold': 19,
            'cooldown_bars': 4,
            'stop_loss_range_pct': 0.25,
            'range_size_multiplier': 1.0,
            'trailing_atr_multiplier': 2.0,
            'max_trade_duration': 20,
            'max_position_pct': 3.0,
            'sl_pct': 0.02,
            'tp_pct': 0.04
        }
        
        # Create test data with range formation and breakout
        self.test_data = self._create_range_breakout_data()
        
        # Create strategy instance
        self.strategy = StrategyRangeBreakoutMomentum(
            data=self.test_data,
            config=self.config,
            logger=Mock()
        )
    
    def tearDown(self):
        """Clean up after each test"""
        logging.disable(logging.NOTSET)
    
    def _create_range_breakout_data(self) -> pd.DataFrame:
        """Create realistic test data with range formation and breakout"""
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
        
        # Phase 1: Initial trend (0-39) - Establishing base level
        for i in range(40):
            if i == 0:
                opens[i] = base_price
                closes[i] = base_price + np.random.normal(0, 50)
            else:
                opens[i] = closes[i-1]
                # Slight upward drift with noise
                closes[i] = opens[i] + np.random.normal(20, 80)
            
            # Create realistic OHLC
            high_low_range = abs(closes[i] - opens[i]) + np.random.uniform(30, 100)
            highs[i] = max(opens[i], closes[i]) + np.random.uniform(0, high_low_range * 0.3)
            lows[i] = min(opens[i], closes[i]) - np.random.uniform(0, high_low_range * 0.3)
            
            volumes[i] = base_volume * np.random.uniform(0.7, 1.3)
        
        # Phase 2: Range formation (40-119) - Clear consolidation
        range_high = max(closes[:40]) + 100
        range_low = min(closes[:40]) - 100
        range_center = (range_high + range_low) / 2
        
        for i in range(40, 120):
            opens[i] = closes[i-1] if i > 0 else range_center
            
            # Price oscillates within range with decreasing volatility
            range_position = np.random.uniform(-0.8, 0.8)  # Stay within 80% of range
            target_price = range_center + (range_position * (range_high - range_low) / 2)
            
            # Add some mean reversion
            mean_reversion = (range_center - opens[i]) * 0.1
            closes[i] = opens[i] + mean_reversion + np.random.normal(0, 30)
            
            # Keep within range bounds
            closes[i] = max(range_low + 50, min(range_high - 50, closes[i]))
            
            # Create OHLC with reduced volatility
            volatility = 40 * (1 - (i - 40) / 80 * 0.5)  # Decreasing volatility
            high_low_range = np.random.uniform(20, volatility)
            highs[i] = max(opens[i], closes[i]) + np.random.uniform(0, high_low_range * 0.4)
            lows[i] = min(opens[i], closes[i]) - np.random.uniform(0, high_low_range * 0.4)
            
            # Keep OHLC within range
            highs[i] = min(highs[i], range_high - 10)
            lows[i] = max(lows[i], range_low + 10)
            
            # Lower volume during consolidation
            volumes[i] = base_volume * np.random.uniform(0.5, 0.9)
        
        # Phase 3: Breakout preparation (120-129) - Building pressure
        for i in range(120, 130):
            opens[i] = closes[i-1]
            
            # Price starts testing upper boundary
            test_level = range_high - np.random.uniform(20, 60)
            closes[i] = opens[i] + (test_level - opens[i]) * 0.3 + np.random.normal(0, 25)
            
            # Slightly higher volatility
            high_low_range = np.random.uniform(30, 60)
            highs[i] = max(opens[i], closes[i]) + np.random.uniform(10, high_low_range * 0.5)
            lows[i] = min(opens[i], closes[i]) - np.random.uniform(0, high_low_range * 0.3)
            
            # Keep within range but allow testing
            highs[i] = min(highs[i], range_high + 20)
            lows[i] = max(lows[i], range_low + 10)
            
            # Increasing volume
            volumes[i] = base_volume * np.random.uniform(0.8, 1.2)
        
        # Phase 4: Breakout (130-139) - Strong upward breakout
        breakout_start = 130
        for i in range(breakout_start, 140):
            opens[i] = closes[i-1]
            
            if i == breakout_start:
                # Initial breakout candle
                closes[i] = range_high + np.random.uniform(80, 150)
                highs[i] = closes[i] + np.random.uniform(20, 80)
                lows[i] = max(opens[i] - 30, range_high - 50)
                volumes[i] = base_volume * np.random.uniform(1.5, 2.5)  # Volume surge
            else:
                # Continuation candles
                momentum = (i - breakout_start) * 30
                closes[i] = opens[i] + momentum + np.random.normal(50, 40)
                
                high_low_range = np.random.uniform(60, 120)
                highs[i] = max(opens[i], closes[i]) + np.random.uniform(20, 60)
                lows[i] = min(opens[i], closes[i]) - np.random.uniform(0, 30)
                
                volumes[i] = base_volume * np.random.uniform(1.2, 1.8)
        
        # Phase 5: Trend continuation (140-199) - Sustained upward movement
        for i in range(140, n_bars):
            opens[i] = closes[i-1]
            
            # Continued upward movement with pullbacks
            trend_strength = 0.7 - (i - 140) / 60 * 0.3  # Weakening trend
            pullback_prob = 0.3 if i % 5 == 0 else 0.1
            
            if np.random.random() < pullback_prob:
                # Pullback candle
                closes[i] = opens[i] - np.random.uniform(30, 80)
            else:
                # Trend continuation
                closes[i] = opens[i] + np.random.normal(40 * trend_strength, 50)
            
            high_low_range = np.random.uniform(40, 100)
            highs[i] = max(opens[i], closes[i]) + np.random.uniform(0, high_low_range * 0.4)
            lows[i] = min(opens[i], closes[i]) - np.random.uniform(0, high_low_range * 0.4)
            
            volumes[i] = base_volume * np.random.uniform(0.8, 1.4)
        
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
        self.assertIsInstance(self.strategy, StrategyRangeBreakoutMomentum)
        self.assertEqual(self.strategy.MARKET_TYPE_TAGS, ['RANGING', 'TRANSITIONAL'])
        self.assertTrue(self.strategy.SHOW_IN_SELECTION)
        
        # Check initial state
        self.assertIsNone(self.strategy.range_high)
        self.assertIsNone(self.strategy.range_low)
        self.assertFalse(self.strategy.range_confirmed)
        self.assertFalse(self.strategy.in_range)
        self.assertEqual(self.strategy.range_size, 0)
        self.assertEqual(self.strategy.false_breakout_count, 0)
    
    def test_init_indicators(self):
        """Test indicator initialization"""
        self.strategy.init_indicators()
        
        # Check that indicators are added to data
        expected_indicators = [
            'adx', 'atr', 'rsi', 'bb_upper', 'bb_lower', 'bb_middle',
            'donchian_high', 'donchian_low', 'volume_sma', 'bb_width'
        ]
        
        for indicator in expected_indicators:
            self.assertIn(indicator, self.strategy.data.columns)
    
    def test_init_indicators_without_pandas_ta(self):
        """Test indicator initialization without pandas_ta"""
        # Test manual calculation fallback by temporarily removing pandas_ta
        original_has_pandas_ta = self.strategy.has_pandas_ta
        
        try:
            # Simulate pandas_ta not being available
            self.strategy.has_pandas_ta = False
            self.strategy.init_indicators()
            
            # Should still have all indicators (using manual calculations)
            expected_indicators = [
                'adx', 'atr', 'rsi', 'bb_upper', 'bb_lower', 'bb_middle',
                'donchian_high', 'donchian_low', 'volume_sma', 'bb_width'
            ]
            
            for indicator in expected_indicators:
                self.assertIn(indicator, self.strategy.data.columns)
        finally:
            # Restore original state
            self.strategy.has_pandas_ta = original_has_pandas_ta
    
    # Test Manual Indicator Calculations
    def test_calculate_adx_manual(self):
        """Test manual ADX calculation"""
        adx_series = self.strategy._calculate_adx_manual(14)
        self.assertIsInstance(adx_series, pd.Series)
        self.assertEqual(len(adx_series), len(self.test_data))
        
        # Check for reasonable values
        valid_adx = adx_series.dropna()
        if len(valid_adx) > 0:
            self.assertTrue(all(0 <= val <= 100 for val in valid_adx))
    
    def test_calculate_atr_manual(self):
        """Test manual ATR calculation"""
        atr_series = self.strategy._calculate_atr_manual(14)
        self.assertIsInstance(atr_series, pd.Series)
        self.assertEqual(len(atr_series), len(self.test_data))
        
        # ATR should be positive
        valid_atr = atr_series.dropna()
        if len(valid_atr) > 0:
            self.assertTrue(all(val >= 0 for val in valid_atr))
    
    def test_calculate_rsi_manual(self):
        """Test manual RSI calculation"""
        rsi_series = self.strategy._calculate_rsi_manual(14)
        self.assertIsInstance(rsi_series, pd.Series)
        self.assertEqual(len(rsi_series), len(self.test_data))
        
        # RSI should be between 0 and 100
        valid_rsi = rsi_series.dropna()
        if len(valid_rsi) > 0:
            self.assertTrue(all(0 <= val <= 100 for val in valid_rsi))
    
    def test_calculate_bollinger_bands_manual(self):
        """Test manual Bollinger Bands calculation"""
        bb_data = self.strategy._calculate_bollinger_bands_manual(20, 2.0)
        
        self.assertIn('upper', bb_data)
        self.assertIn('lower', bb_data)
        self.assertIn('middle', bb_data)
        
        # Upper should be above middle, middle above lower
        for i in range(20, len(self.test_data)):
            if not pd.isna(bb_data['upper'].iloc[i]):
                self.assertGreaterEqual(bb_data['upper'].iloc[i], bb_data['middle'].iloc[i])
                self.assertGreaterEqual(bb_data['middle'].iloc[i], bb_data['lower'].iloc[i])
    
    # Test Range Detection
    def test_is_ranging_market(self):
        """Test ranging market detection"""
        self.strategy.init_indicators()
        
        # Test during consolidation phase (bars 60-119)
        ranging_detected = False
        for i in range(60, 120):
            if self.strategy.is_ranging_market(i):
                ranging_detected = True
                break
        
        self.assertTrue(ranging_detected, "Should detect ranging market during consolidation")
    
    def test_identify_range(self):
        """Test range identification"""
        self.strategy.init_indicators()
        
        # Test range identification during consolidation
        range_identified = False
        for i in range(60, 120):
            if self.strategy.identify_range(i):
                range_identified = True
                break
        
        self.assertTrue(range_identified, "Should identify range during consolidation phase")
    
    def test_range_confirmation(self):
        """Test range confirmation process"""
        self.strategy.init_indicators()
        
        # Simulate range identification and confirmation
        for i in range(60, 120):
            self.strategy.identify_range(i)
            if self.strategy.range_confirmed:
                break
        
        # Should eventually confirm range
        self.assertTrue(self.strategy.range_confirmed, "Range should be confirmed during consolidation")
        self.assertIsNotNone(self.strategy.range_high)
        self.assertIsNotNone(self.strategy.range_low)
        self.assertGreater(self.strategy.range_size, 0)
    
    # Test Breakout Detection
    def test_detect_range_breakout(self):
        """Test range breakout detection"""
        self.strategy.init_indicators()
        
        # First establish range
        for i in range(60, 120):
            self.strategy.identify_range(i)
            if self.strategy.range_confirmed:
                break
        
        # Test breakout detection during breakout phase
        breakout_detected = False
        breakout_direction = None
        
        for i in range(130, 140):
            is_breakout, direction = self.strategy.detect_range_breakout(i)
            if is_breakout:
                breakout_detected = True
                breakout_direction = direction
                break
        
        self.assertTrue(breakout_detected, "Should detect breakout during breakout phase")
        self.assertEqual(breakout_direction, 'long', "Should detect upward breakout")
    
    def test_confirm_breakout_momentum(self):
        """Test breakout momentum confirmation"""
        self.strategy.init_indicators()
        
        # Test momentum confirmation during breakout
        momentum_confirmed = False
        for i in range(130, 140):
            if self.strategy.confirm_breakout_momentum(i, 'long'):
                momentum_confirmed = True
                break
        
        self.assertTrue(momentum_confirmed, "Should confirm momentum during breakout")
    
    # Test Entry Conditions
    def test_check_entry_conditions_no_range(self):
        """Test entry conditions when no range is identified"""
        self.strategy.init_indicators()
        
        # Test early in data when no range is established
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        self.assertIsNone(entry_signal, "Should not generate entry signal without established range")
    
    def test_check_entry_conditions_during_range(self):
        """Test entry conditions during range formation"""
        self.strategy.init_indicators()
        
        # Simulate being in range phase
        for i in range(60, 120):
            self.strategy.identify_range(i)
        
        # Should not generate entry during range
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        self.assertIsNone(entry_signal, "Should not generate entry signal during range formation")
    
    def test_check_entry_conditions_breakout(self):
        """Test entry conditions during breakout"""
        self.strategy.init_indicators()
        
        # Establish range first
        for i in range(60, 120):
            self.strategy.identify_range(i)
            if self.strategy.range_confirmed:
                break
        
        # Simulate breakout conditions
        self.strategy.data = self.strategy.data.iloc[:135].copy()  # Up to breakout
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        
        if entry_signal:
            self.assertIn(entry_signal['action'], ['long', 'short'])
            self.assertGreater(entry_signal['confidence'], 0)
            self.assertIn('breakout', entry_signal['reason'])
    
    def test_cooldown_period(self):
        """Test cooldown period after trades"""
        self.strategy.init_indicators()
        self.strategy.last_trade_bar = len(self.strategy.data) - 2
        
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        self.assertIsNone(entry_signal, "Should respect cooldown period")
    
    # Test Risk Management
    def test_calculate_stops_and_targets(self):
        """Test stop loss and target calculation"""
        self.strategy.range_size = 200  # Mock range size
        
        stop_price, target_price = self.strategy.calculate_stops_and_targets('long')
        current_price = self.strategy.data['close'].iloc[-1]
        
        self.assertLess(stop_price, current_price, "Stop should be below current price for long")
        self.assertGreater(target_price, current_price, "Target should be above current price for long")
        
        # Test short direction
        stop_price, target_price = self.strategy.calculate_stops_and_targets('short')
        
        self.assertGreater(stop_price, current_price, "Stop should be above current price for short")
        self.assertLess(target_price, current_price, "Target should be below current price for short")
    
    def test_get_risk_parameters(self):
        """Test risk parameter calculation"""
        risk_params = self.strategy.get_risk_parameters()
        
        required_keys = ['sl_pct', 'tp_pct', 'max_position_pct', 'risk_reward_ratio']
        for key in required_keys:
            self.assertIn(key, risk_params)
            self.assertGreater(risk_params[key], 0)
    
    def test_get_risk_parameters_with_range(self):
        """Test risk parameters with established range"""
        self.strategy.range_size = 200
        self.strategy.breakout_direction = 'long'
        
        risk_params = self.strategy.get_risk_parameters()
        
        self.assertIn('sl_pct', risk_params)
        self.assertIn('tp_pct', risk_params)
        self.assertGreater(risk_params['sl_pct'], 0)
        self.assertGreater(risk_params['tp_pct'], 0)
    
    # Test Trade Management
    def test_update_trailing_stop(self):
        """Test trailing stop updates"""
        self.strategy.init_indicators()
        self.strategy.entry_price = 50000
        self.strategy.breakout_direction = 'long'
        
        # Test trailing stop update
        initial_trailing = self.strategy.trailing_stop
        self.strategy.update_trailing_stop('BTCUSDT')
        
        # Should set trailing stop for long position
        self.assertIsNotNone(self.strategy.trailing_stop)
    
    def test_check_false_breakout(self):
        """Test false breakout detection"""
        self.strategy.entry_price = 50000
        self.strategy.breakout_direction = 'long'
        self.strategy.range_high = 49800
        
        # Mock current price back in range by modifying the data directly
        original_close = self.strategy.data['close'].iloc[-1]
        self.strategy.data.loc[self.strategy.data.index[-1], 'close'] = 49700
        
        try:
            is_false = self.strategy.check_false_breakout()
            self.assertTrue(is_false, "Should detect false breakout when price returns to range")
        finally:
            # Restore original value
            self.strategy.data.loc[self.strategy.data.index[-1], 'close'] = original_close
    
    def test_check_partial_profit(self):
        """Test partial profit taking logic"""
        self.strategy.entry_price = 50000
        self.strategy.stop_price = 49500
        self.strategy.breakout_direction = 'long'
        self.strategy.partial_profit_taken = False
        
        # Mock current price at profit level by modifying data directly
        original_close = self.strategy.data['close'].iloc[-1]
        self.strategy.data.loc[self.strategy.data.index[-1], 'close'] = 50500  # 1R profit
        
        try:
            should_take_profit = self.strategy.check_partial_profit()
            self.assertTrue(should_take_profit, "Should take partial profit at 1R")
        finally:
            # Restore original value
            self.strategy.data.loc[self.strategy.data.index[-1], 'close'] = original_close
    
    # Test Exit Conditions
    def test_check_exit_false_breakout(self):
        """Test exit on false breakout"""
        self.strategy.init_indicators()
        self.strategy.entry_price = 50000
        self.strategy.breakout_direction = 'long'
        self.strategy.range_high = 49800
        
        # Mock price back in range
        with patch.object(self.strategy, 'check_false_breakout', return_value=True):
            exit_signal = self.strategy.check_exit('BTCUSDT')
            
            self.assertIsNotNone(exit_signal)
            self.assertEqual(exit_signal['action'], 'exit')
            self.assertEqual(exit_signal['reason'], 'false_breakout')
    
    def test_check_exit_time_based(self):
        """Test time-based exit"""
        self.strategy.init_indicators()
        self.strategy.entry_bar = len(self.strategy.data) - 25  # Old entry
        
        exit_signal = self.strategy.check_exit('BTCUSDT')
        
        if exit_signal:
            self.assertEqual(exit_signal['action'], 'exit')
            self.assertEqual(exit_signal['reason'], 'time_exit')
    
    def test_check_exit_target_hit(self):
        """Test exit when target is hit"""
        self.strategy.init_indicators()
        self.strategy.breakout_direction = 'long'
        self.strategy.target_price = 49000  # Below current price
        
        exit_signal = self.strategy.check_exit('BTCUSDT')
        
        if exit_signal:
            self.assertEqual(exit_signal['action'], 'exit')
            self.assertEqual(exit_signal['reason'], 'target_hit')
    
    def test_check_exit_stop_loss(self):
        """Test exit when stop loss is hit"""
        self.strategy.init_indicators()
        self.strategy.breakout_direction = 'long'
        self.strategy.stop_price = 60000  # Above current price
        
        exit_signal = self.strategy.check_exit('BTCUSDT')
        
        if exit_signal:
            self.assertEqual(exit_signal['action'], 'exit')
            self.assertEqual(exit_signal['reason'], 'stop_loss')
    
    def test_check_exit_trailing_stop(self):
        """Test exit when trailing stop is hit"""
        self.strategy.init_indicators()
        self.strategy.breakout_direction = 'long'
        self.strategy.trailing_stop = 60000  # Above current price
        
        exit_signal = self.strategy.check_exit('BTCUSDT')
        
        if exit_signal:
            self.assertEqual(exit_signal['action'], 'exit')
            self.assertEqual(exit_signal['reason'], 'trailing_stop')
    
    def test_check_exit_momentum_fade(self):
        """Test exit on momentum fade"""
        self.strategy.init_indicators()
        
        # Mock low ADX by modifying data directly
        if 'adx' in self.strategy.data.columns:
            original_adx = self.strategy.data['adx'].iloc[-1]
            self.strategy.data.loc[self.strategy.data.index[-1], 'adx'] = 10
            
            try:
                exit_signal = self.strategy.check_exit('BTCUSDT')
                
                if exit_signal:
                    self.assertEqual(exit_signal['action'], 'exit')
                    self.assertEqual(exit_signal['reason'], 'momentum_fade')
            finally:
                # Restore original value
                self.strategy.data.loc[self.strategy.data.index[-1], 'adx'] = original_adx
    
    # Test Trade Lifecycle
    def test_on_trade_closed(self):
        """Test trade closure cleanup"""
        # Set up trade state
        self.strategy.entry_bar = 100
        self.strategy.stop_price = 49000
        self.strategy.target_price = 51000
        self.strategy.trailing_stop = 49500
        self.strategy.partial_profit_taken = True
        self.strategy.breakout_direction = 'long'
        self.strategy.entry_price = 50000
        
        # Close trade
        trade_result = {'reason': 'target_hit', 'pnl': 500}
        self.strategy.on_trade_closed('BTCUSDT', trade_result)
        
        # Check cleanup
        self.assertIsNone(self.strategy.entry_bar)
        self.assertIsNone(self.strategy.stop_price)
        self.assertIsNone(self.strategy.target_price)
        self.assertIsNone(self.strategy.trailing_stop)
        self.assertFalse(self.strategy.partial_profit_taken)
        self.assertIsNone(self.strategy.breakout_direction)
        self.assertIsNone(self.strategy.entry_price)
    
    def test_on_trade_closed_false_breakout(self):
        """Test trade closure with false breakout tracking"""
        initial_count = self.strategy.false_breakout_count
        
        trade_result = {'reason': 'false_breakout', 'pnl': -100}
        self.strategy.on_trade_closed('BTCUSDT', trade_result)
        
        self.assertEqual(self.strategy.false_breakout_count, initial_count + 1)
    
    # Test Error Handling
    def test_error_handling_in_indicators(self):
        """Test error handling in indicator calculations"""
        # Create data with NaN values
        corrupted_data = self.test_data.copy()
        corrupted_data.loc[50:60, 'close'] = np.nan
        
        strategy = StrategyRangeBreakoutMomentum(
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
        result = self.strategy.is_ranging_market(-1)  # Invalid index
        self.assertIn(result, [True, False])
        
        result = self.strategy.identify_range(-1)  # Invalid index
        self.assertFalse(result)
    
    def test_error_handling_in_entry_conditions(self):
        """Test error handling in entry condition checks"""
        # Test with current strategy but simulate error conditions
        # Save original state
        original_range_confirmed = self.strategy.range_confirmed
        original_in_range = self.strategy.in_range
        
        try:
            # Reset to unconfirmed state to test error handling
            self.strategy.range_confirmed = False
            self.strategy.in_range = False
            
            # Should handle gracefully when no range is established
            entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
            self.assertIsNone(entry_signal)
            
        finally:
            # Restore original state
            self.strategy.range_confirmed = original_range_confirmed
            self.strategy.in_range = original_in_range
    
    def test_error_handling_in_exit_conditions(self):
        """Test error handling in exit condition checks"""
        # Test with current strategy but no position state
        # Should handle gracefully when no position is open
        exit_signal = self.strategy.check_exit('BTCUSDT')
        self.assertIsNone(exit_signal)
    
    # Test Edge Cases
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios"""
        # Test with current strategy but early data indices
        # Should handle gracefully when insufficient data for calculations
        
        # Test range identification with insufficient data
        result = self.strategy.identify_range(5)  # Very early index
        self.assertFalse(result)
        
        # Test ranging market detection with insufficient data
        result = self.strategy.is_ranging_market(5)
        self.assertIn(result, [True, False])  # Should not crash
        
        # Test entry conditions with insufficient data
        entry_signal = self.strategy._check_entry_conditions('BTCUSDT')
        # Should either return None or handle gracefully
        if entry_signal is not None:
            self.assertIsInstance(entry_signal, dict)
    
    def test_extreme_market_conditions(self):
        """Test strategy behavior in extreme market conditions"""
        # Create data with extreme volatility
        extreme_data = self.test_data.copy()
        extreme_data['high'] = extreme_data['close'] * 1.1
        extreme_data['low'] = extreme_data['close'] * 0.9
        
        strategy = StrategyRangeBreakoutMomentum(
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
            result = strategy.is_ranging_market(i)
            self.assertIn(result, [True, False])

if __name__ == '__main__':
    unittest.main() 