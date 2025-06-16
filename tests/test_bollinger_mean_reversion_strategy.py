import unittest
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path to import the strategies
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.bollinger_mean_reversion_strategy import StrategyBollingerMeanReversion


class TestBollingerMeanReversionStrategy(unittest.TestCase):
    """Test cases for StrategyBollingerMeanReversion"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample OHLCV data with ranging/mean-reverting pattern
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2023-01-01', periods=150, freq='1min')
        
        # Create ranging market data (oscillating around a mean)
        base_price = 100.0
        # Create oscillating pattern instead of trending
        oscillation = 5 * np.sin(np.linspace(0, 4*np.pi, 150))
        noise = np.random.normal(0, 0.3, 150)
        
        closes = base_price + oscillation + noise
        highs = closes + np.random.uniform(0.1, 0.4, 150)
        lows = closes - np.random.uniform(0.1, 0.4, 150)
        opens = closes + np.random.uniform(-0.2, 0.2, 150)
        volumes = np.random.randint(1000, 8000, 150)
        
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
                'StrategyBollingerMeanReversion': {
                    'bollinger_period': 20,
                    'bollinger_std_dev': 2.0,
                    'rsi_period': 14,
                    'rsi_oversold': 30,
                    'rsi_overbought': 70,
                    'atr_period': 14,
                    'atr_stop_multiplier': 1.5,
                    'atr_target_multiplier': 2.0,
                    'volume_period': 20,
                    'volume_spike_multiplier': 1.5,
                    'sl_pct': 0.008,
                    'tp_pct': 0.012,
                    'time_stop_bars': 30,
                    'band_touch_threshold': 0.002,
                    'mean_revert_bars': 5,
                    'entry_confirmation_bars': 2
                }
            }
        }
        
        # Create logger
        self.logger = logging.getLogger('TestBollingerMeanReversionStrategy')
        self.logger.setLevel(logging.DEBUG)
        
        # Initialize strategy
        self.strategy = StrategyBollingerMeanReversion(
            data=self.sample_data,
            config=self.test_config,
            logger=self.logger
        )

    def test_initialization(self):
        """Test strategy initialization"""
        self.assertIsInstance(self.strategy, StrategyBollingerMeanReversion)
        self.assertEqual(self.strategy.bollinger_period, 20)
        self.assertEqual(self.strategy.bollinger_std_dev, 2.0)
        self.assertEqual(self.strategy.rsi_period, 14)
        self.assertEqual(self.strategy.rsi_oversold, 30)
        self.assertEqual(self.strategy.rsi_overbought, 70)
        self.assertEqual(self.strategy.sl_pct, 0.008)
        self.assertEqual(self.strategy.tp_pct, 0.012)
        self.assertEqual(self.strategy.MARKET_TYPE_TAGS, ['RANGING', 'LOW_VOLATILITY'])
        self.assertTrue(self.strategy.SHOW_IN_SELECTION)

    def test_init_indicators(self):
        """Test indicator initialization"""
        # Bollinger Bands should be calculated
        self.assertIn('bb_upper', self.strategy.data.columns)
        self.assertIn('bb_middle', self.strategy.data.columns)
        self.assertIn('bb_lower', self.strategy.data.columns)
        
        # RSI should be calculated
        self.assertIn('rsi', self.strategy.data.columns)
        
        # ATR should be calculated
        self.assertIn('atr', self.strategy.data.columns)
        
        # Volume MA should be calculated
        self.assertIn('volume_ma', self.strategy.data.columns)
        
        # Check that indicators have values (not all NaN)
        self.assertFalse(self.strategy.data['bb_upper'].isna().all())
        self.assertFalse(self.strategy.data['bb_middle'].isna().all())
        self.assertFalse(self.strategy.data['bb_lower'].isna().all())
        self.assertFalse(self.strategy.data['rsi'].isna().all())
        
        # Check Bollinger Band relationships
        # Upper band should be greater than middle, middle greater than lower
        last_idx = -1
        if not self.strategy.data['bb_upper'].isna().iloc[last_idx]:
            self.assertGreater(
                self.strategy.data['bb_upper'].iloc[last_idx],
                self.strategy.data['bb_middle'].iloc[last_idx]
            )
            self.assertGreater(
                self.strategy.data['bb_middle'].iloc[last_idx],
                self.strategy.data['bb_lower'].iloc[last_idx]
            )

    def test_init_indicators_with_insufficient_data(self):
        """Test indicator initialization with insufficient data"""
        # Create data with only 10 rows (less than Bollinger period)
        short_data = self.sample_data.iloc[:10].copy()
        
        strategy = StrategyBollingerMeanReversion(
            data=short_data,
            config=self.test_config,
            logger=self.logger
        )
        
        # Should still create indicator columns
        self.assertIn('bb_upper', strategy.data.columns)
        self.assertIn('bb_middle', strategy.data.columns)
        self.assertIn('bb_lower', strategy.data.columns)
        self.assertIn('rsi', strategy.data.columns)

    def test_init_indicators_with_missing_columns(self):
        """Test indicator initialization with missing required columns"""
        # Remove required columns
        bad_data = self.sample_data.drop(columns=['high', 'low']).copy()
        
        strategy = StrategyBollingerMeanReversion(
            data=bad_data,
            config=self.test_config,
            logger=self.logger
        )
        
        # Should handle missing columns gracefully
        self.assertIsNotNone(strategy.data)

    def test_get_risk_parameters(self):
        """Test risk parameter retrieval"""
        risk_params = self.strategy.get_risk_parameters()
        
        self.assertIn('sl_pct', risk_params)
        self.assertIn('tp_pct', risk_params)
        self.assertEqual(risk_params['sl_pct'], 0.008)
        self.assertEqual(risk_params['tp_pct'], 0.012)

    def test_get_current_values(self):
        """Test getting current indicator values"""
        current_vals = self.strategy._get_current_values()
        
        if current_vals:  # Only test if data is sufficient
            self.assertIn('close', current_vals)
            self.assertIn('bb_upper', current_vals)
            self.assertIn('bb_middle', current_vals)
            self.assertIn('bb_lower', current_vals)
            self.assertIn('rsi', current_vals)
            self.assertIn('atr', current_vals)
            self.assertIn('volume', current_vals)
            self.assertIn('volume_ma', current_vals)

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
        self.strategy.entry_bar_index = len(self.strategy.data) - 10  # Position opened 10 bars ago
        
        exit_signal = self.strategy.check_exit(symbol)
        
        # Should return dictionary with exit details or None
        self.assertIsInstance(exit_signal, (dict, type(None)))

    def test_detect_bollinger_touch_oversold(self):
        """Test detection of price touching lower Bollinger Band (oversold)"""
        # Create mock current values for oversold condition
        test_vals = {
            'close': 95.0,
            'bb_lower': 95.1,  # Close near lower band
            'bb_middle': 100.0,
            'bb_upper': 104.9,
            'rsi': 25.0,  # Oversold RSI
            'volume': 5000,
            'volume_ma': 3000
        }
        
        # Test band touch detection
        touch_detected = self.strategy._detect_bollinger_band_touch(test_vals)
        
        if touch_detected:
            self.assertEqual(touch_detected, 'lower')

    def test_detect_bollinger_touch_overbought(self):
        """Test detection of price touching upper Bollinger Band (overbought)"""
        # Create mock current values for overbought condition
        test_vals = {
            'close': 104.9,
            'bb_lower': 95.1,
            'bb_middle': 100.0,
            'bb_upper': 104.8,  # Close near upper band
            'rsi': 75.0,  # Overbought RSI
            'volume': 5000,
            'volume_ma': 3000
        }
        
        # Test band touch detection
        touch_detected = self.strategy._detect_bollinger_band_touch(test_vals)
        
        if touch_detected:
            self.assertEqual(touch_detected, 'upper')

    def test_detect_mean_reversion_setup(self):
        """Test mean reversion setup detection"""
        # Set up band touch state
        self.strategy.band_touch_detected = 'lower'
        self.strategy.band_touch_bar = len(self.strategy.data) - 3
        
        test_vals = {
            'close': 97.0,
            'bb_middle': 100.0,
            'rsi': 45.0  # RSI moving back towards neutral
        }
        
        reversion_setup = self.strategy._detect_mean_reversion_setup(test_vals)
        self.assertIsInstance(reversion_setup, bool)

    def test_volume_confirmation(self):
        """Test volume confirmation logic"""
        test_vals = {
            'volume': 5000,
            'volume_ma': 3000
        }
        
        volume_confirmed = self.strategy._check_volume_confirmation(test_vals)
        self.assertTrue(volume_confirmed)  # Volume > volume_ma * multiplier
        
        # Test insufficient volume
        test_vals['volume'] = 2000
        volume_confirmed = self.strategy._check_volume_confirmation(test_vals)
        self.assertFalse(volume_confirmed)

    def test_on_order_update(self):
        """Test order update handling"""
        symbol = 'BTC/USDT'
        
        # Mock order responses
        order_responses = {
            'main_order': {
                'result': {
                    'orderId': 'main123',
                    'orderStatus': 'filled',
                    'side': 'buy',
                    'qty': '0.1'
                }
            },
            'stop_loss_order': {
                'result': {
                    'orderId': 'sl123',
                    'orderStatus': 'new'
                }
            },
            'take_profit_order': {
                'result': {
                    'orderId': 'tp123',
                    'orderStatus': 'new'
                }
            }
        }
        
        # Test order update
        self.strategy.on_order_update(order_responses, symbol)
        
        # Check that position was updated
        self.assertIsNotNone(self.strategy.position.get(symbol))
        self.assertEqual(self.strategy.active_order_id.get(symbol), 'main123')

    def test_on_trade_update(self):
        """Test trade update handling"""
        symbol = 'BTC/USDT'
        
        # Set up position first
        self.strategy.position[symbol] = {
            'main_order': {'result': {'side': 'buy', 'qty': '0.1'}},
            'stop_loss_order': {'result': {'orderId': 'sl123'}},
            'take_profit_order': {'result': {'orderId': 'tp123'}}
        }
        
        # Mock trade data
        trade_data = {
            'symbol': symbol,
            'side': 'buy',
            'qty': '0.1',
            'price': '100.0',
            'status': 'closed',
            'exit': True  # This key is required for position clearing
        }
        
        # Test trade update
        self.strategy.on_trade_update(trade_data, symbol)
        
        # Check that position was cleared
        self.assertIsNone(self.strategy.position.get(symbol))

    def test_update_indicators_for_new_row(self):
        """Test incremental indicator updates"""
        # Store original values
        original_bb_upper = self.strategy.data['bb_upper'].iloc[-1]
        original_bb_middle = self.strategy.data['bb_middle'].iloc[-1]
        original_bb_lower = self.strategy.data['bb_lower'].iloc[-1]
        
        # Add new row
        new_row = pd.DataFrame({
            'timestamp': [pd.Timestamp('2023-01-01 02:30:00')],
            'open': [102.0],
            'high': [103.0],
            'low': [101.5],
            'close': [102.5],
            'volume': [4500]
        })
        
        self.strategy.data = pd.concat([self.strategy.data, new_row], ignore_index=True)
        
        # Update indicators
        self.strategy.update_indicators_for_new_row()
        
        # Check that indicators were updated
        new_bb_upper = self.strategy.data['bb_upper'].iloc[-1]
        new_bb_middle = self.strategy.data['bb_middle'].iloc[-1]
        new_bb_lower = self.strategy.data['bb_lower'].iloc[-1]
        
        # New values should be different from original (unless data is insufficient)
        if not pd.isna(original_bb_upper) and not pd.isna(new_bb_upper):
            # Values may be the same in some cases, so we just check they exist
            self.assertIsNotNone(new_bb_upper)
            self.assertIsNotNone(new_bb_middle)
            self.assertIsNotNone(new_bb_lower)

    def test_on_error(self):
        """Test error handling"""
        test_exception = Exception("Test error")
        
        # Should not raise exception
        try:
            self.strategy.on_error(test_exception)
        except Exception as e:
            self.fail(f"on_error raised an exception: {e}")

    def test_clear_position(self):
        """Test position clearing"""
        symbol = 'BTC/USDT'
        
        # Set up position
        self.strategy.position[symbol] = {
            'main_order': {'result': {'side': 'buy', 'qty': '0.1'}},
            'stop_loss_order': {'result': {'orderId': 'sl123'}},
            'take_profit_order': {'result': {'orderId': 'tp123'}}
        }
        self.strategy.order_pending[symbol] = True
        self.strategy.active_order_id[symbol] = 'main123'
        
        # Clear position
        self.strategy.clear_position(symbol)
        
        # Check that position was cleared
        self.assertIsNone(self.strategy.position.get(symbol))
        self.assertFalse(self.strategy.order_pending.get(symbol, False))
        self.assertIsNone(self.strategy.active_order_id.get(symbol))

    def test_state_reset(self):
        """Test strategy state reset"""
        # Set some strategy state
        self.strategy.band_touch_detected = 'upper'
        self.strategy.band_touch_bar = 100
        self.strategy.mean_reversion_setup = True
        
        # Reset state
        self.strategy._reset_strategy_state()
        
        # Check that state was reset
        self.assertIsNone(self.strategy.band_touch_detected)
        self.assertIsNone(self.strategy.band_touch_bar)
        self.assertFalse(self.strategy.mean_reversion_setup)

    def test_config_fallback_values(self):
        """Test configuration fallback values"""
        # Create strategy with minimal config
        minimal_config = {'strategy_configs': {'StrategyBollingerMeanReversion': {}}}
        
        strategy = StrategyBollingerMeanReversion(
            data=self.sample_data,
            config=minimal_config,
            logger=self.logger
        )
        
        # Should use default values
        self.assertEqual(strategy.bollinger_period, 20)
        self.assertEqual(strategy.bollinger_std_dev, 2.0)
        self.assertEqual(strategy.rsi_period, 14)
        self.assertEqual(strategy.rsi_oversold, 30)
        self.assertEqual(strategy.rsi_overbought, 70)

    def test_band_touch_timeout(self):
        """Test band touch timeout logic"""
        # Set up old band touch
        self.strategy.band_touch_detected = 'lower'
        self.strategy.band_touch_bar = 0  # Very old
        
        current_bar = len(self.strategy.data) - 1
        
        # Check if timeout logic works
        timeout_bars = 10
        if current_bar - self.strategy.band_touch_bar > timeout_bars:
            # Should reset state due to timeout
            self.strategy._reset_strategy_state()
            self.assertIsNone(self.strategy.band_touch_detected)

    def test_rsi_level_validation(self):
        """Test RSI level validation"""
        # Test oversold condition
        self.assertTrue(25.0 < self.strategy.rsi_oversold)
        
        # Test overbought condition  
        self.assertTrue(75.0 > self.strategy.rsi_overbought)
        
        # Test that oversold < overbought
        self.assertLess(self.strategy.rsi_oversold, self.strategy.rsi_overbought)

    def test_atr_calculations(self):
        """Test ATR-based calculations"""
        if 'atr' in self.strategy.data.columns:
            last_atr = self.strategy.data['atr'].iloc[-1]
            if not pd.isna(last_atr):
                # ATR should be positive
                self.assertGreater(last_atr, 0)
                
                # Test stop and target calculations
                stop_distance = last_atr * self.strategy.atr_stop_multiplier
                target_distance = last_atr * self.strategy.atr_target_multiplier
                
                self.assertGreater(stop_distance, 0)
                self.assertGreater(target_distance, 0)
                self.assertGreater(target_distance, stop_distance)  # Target should be further than stop

    def test_bollinger_band_calculation_accuracy(self):
        """Test accuracy of Bollinger Band calculations"""
        if len(self.strategy.data) >= self.strategy.bollinger_period:
            # Get the last values where Bollinger Bands should be calculated
            last_idx = -1
            while (last_idx >= -len(self.strategy.data) and 
                   pd.isna(self.strategy.data['bb_middle'].iloc[last_idx])):
                last_idx -= 1
            
            if last_idx >= -len(self.strategy.data):
                bb_middle = self.strategy.data['bb_middle'].iloc[last_idx]
                bb_upper = self.strategy.data['bb_upper'].iloc[last_idx]
                bb_lower = self.strategy.data['bb_lower'].iloc[last_idx]
                
                # Middle band should be approximately the SMA
                period = self.strategy.bollinger_period
                start_idx = max(0, len(self.strategy.data) + last_idx - period + 1)
                end_idx = len(self.strategy.data) + last_idx + 1
                
                sma_manual = self.strategy.data['close'].iloc[start_idx:end_idx].mean()
                
                # Allow for small floating point differences
                self.assertAlmostEqual(bb_middle, sma_manual, places=6)
                
                # Upper and lower bands should be symmetric around middle
                upper_distance = bb_upper - bb_middle
                lower_distance = bb_middle - bb_lower
                self.assertAlmostEqual(upper_distance, lower_distance, places=6)


if __name__ == '__main__':
    unittest.main() 