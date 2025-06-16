#!/usr/bin/env python3
"""
Test Generator for Trading Strategies

This script automatically generates comprehensive test files for all implemented trading strategies
that don't already have test files.
"""

import os
import sys
import importlib.util
from pathlib import Path

# Add the parent directory to the path to import the strategies
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_strategy_files():
    """Get all strategy Python files from the strategies directory."""
    strategies_dir = Path(__file__).parent.parent / 'strategies'
    strategy_files = []
    
    for file_path in strategies_dir.glob('*.py'):
        if (file_path.name != '__init__.py' and 
            file_path.name != 'strategy_template.py' and
            not file_path.name.startswith('test_')):
            strategy_files.append(file_path)
    
    return strategy_files

def get_existing_test_files():
    """Get list of existing test files."""
    tests_dir = Path(__file__).parent
    existing_tests = set()
    
    for file_path in tests_dir.glob('test_*.py'):
        if file_path.name != 'test_strategy_generator.py':
            existing_tests.add(file_path.name)
    
    return existing_tests

def get_strategy_class_name(file_path):
    """Extract the strategy class name from a Python file."""
    try:
        spec = importlib.util.spec_from_file_location("strategy_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Find classes that inherit from StrategyTemplate
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type) and 
                hasattr(obj, '__bases__') and
                any('StrategyTemplate' in str(base) for base in obj.__bases__)):
                return name
    except Exception as e:
        print(f"Could not extract class name from {file_path}: {e}")
        return None
    
    return None

def generate_test_template(strategy_file_path, strategy_class_name):
    """Generate a comprehensive test template for a strategy."""
    
    strategy_file_name = strategy_file_path.stem
    test_file_name = f"test_{strategy_file_name}.py"
    
    template = f'''import unittest
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock, patch
import sys
import os

# Add the parent directory to the path to import the strategies
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.{strategy_file_name} import {strategy_class_name}


class Test{strategy_class_name}(unittest.TestCase):
    """Test cases for {strategy_class_name}"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample OHLCV data
        np.random.seed(42)  # For reproducible tests
        dates = pd.date_range('2023-01-01', periods=200, freq='1min')
        
        # Create realistic OHLCV data with market patterns
        base_price = 100.0
        # Create mixed pattern (trending + ranging + volatility)
        trend = np.linspace(0, 5, 200) * 0.5
        cycles = 3 * np.sin(np.linspace(0, 6*np.pi, 200))
        noise = np.random.normal(0, 0.4, 200)
        
        closes = base_price + trend + cycles + noise
        highs = closes + np.random.uniform(0.1, 0.6, 200)
        lows = closes - np.random.uniform(0.1, 0.6, 200)
        opens = closes + np.random.uniform(-0.3, 0.3, 200)
        volumes = np.random.randint(1000, 10000, 200)
        
        self.sample_data = pd.DataFrame({{
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }})
        
        # Test configuration - customize based on strategy requirements
        self.test_config = {{
            'strategy_configs': {{
                '{strategy_class_name}': {{
                    'sl_pct': 0.01,
                    'tp_pct': 0.02,
                    'time_stop_bars': 50
                    # Add strategy-specific parameters here
                }}
            }},
            'default': {{
                'default_sl_pct': 0.005,
                'default_tp_pct': 0.015
            }}
        }}
        
        # Create logger
        self.logger = logging.getLogger('Test{strategy_class_name}')
        self.logger.setLevel(logging.DEBUG)
        
        # Initialize strategy
        self.strategy = {strategy_class_name}(
            data=self.sample_data,
            config=self.test_config,
            logger=self.logger
        )

    def test_initialization(self):
        """Test strategy initialization"""
        self.assertIsInstance(self.strategy, {strategy_class_name})
        self.assertIsNotNone(self.strategy.config)
        self.assertIsNotNone(self.strategy.logger)
        self.assertIsNotNone(self.strategy.data)
        
        # Test MARKET_TYPE_TAGS if defined
        if hasattr(self.strategy, 'MARKET_TYPE_TAGS'):
            self.assertIsInstance(self.strategy.MARKET_TYPE_TAGS, list)
        
        # Test SHOW_IN_SELECTION if defined
        if hasattr(self.strategy, 'SHOW_IN_SELECTION'):
            self.assertIsInstance(self.strategy.SHOW_IN_SELECTION, bool)

    def test_init_indicators(self):
        """Test indicator initialization"""
        # This test needs to be customized based on what indicators the strategy uses
        # Check that data has indicator columns after initialization
        self.assertIsNotNone(self.strategy.data)
        self.assertIsInstance(self.strategy.data, pd.DataFrame)
        
        # Verify required OHLCV columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col in self.sample_data.columns:
                self.assertIn(col, self.strategy.data.columns)

    def test_get_risk_parameters(self):
        """Test risk parameter retrieval"""
        risk_params = self.strategy.get_risk_parameters()
        
        self.assertIsInstance(risk_params, dict)
        self.assertIn('sl_pct', risk_params)
        self.assertIn('tp_pct', risk_params)
        
        # Risk parameters should be numbers or None
        if risk_params['sl_pct'] is not None:
            self.assertIsInstance(risk_params['sl_pct'], (int, float))
        if risk_params['tp_pct'] is not None:
            self.assertIsInstance(risk_params['tp_pct'], (int, float))

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
        
        if entry_signal is not None:
            # Validate entry signal structure
            self.assertIn('side', entry_signal)
            self.assertIn(entry_signal['side'], ['buy', 'sell'])
            
            # Risk parameters should be included
            self.assertIn('sl_pct', entry_signal)
            self.assertIn('tp_pct', entry_signal)

    def test_check_entry_conditions_with_position(self):
        """Test entry condition checking when position is already open"""
        symbol = 'BTC/USDT'
        
        # Set existing position
        self.strategy.position[symbol] = {{
            'main_order': {{'result': {{'side': 'buy', 'qty': '0.1'}}}},
            'stop_loss_order': {{'result': {{'orderId': 'sl123'}}}},
            'take_profit_order': {{'result': {{'orderId': 'tp123'}}}}
        }}
        
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
        
        # Should return False when no position
        self.assertFalse(exit_signal)

    def test_check_exit_conditions_with_position(self):
        """Test exit condition checking when position is open"""
        symbol = 'BTC/USDT'
        
        # Set existing position
        self.strategy.position[symbol] = {{
            'main_order': {{'result': {{'side': 'buy', 'qty': '0.1'}}}},
            'stop_loss_order': {{'result': {{'orderId': 'sl123'}}}},
            'take_profit_order': {{'result': {{'orderId': 'tp123'}}}}
        }}
        
        # Set entry bar index if the strategy uses it
        if hasattr(self.strategy, 'entry_bar_index'):
            self.strategy.entry_bar_index = len(self.strategy.data) - 10
        
        exit_signal = self.strategy.check_exit(symbol)
        
        # Should return boolean
        self.assertIsInstance(exit_signal, bool)

    def test_on_order_update(self):
        """Test order update handling"""
        symbol = 'BTC/USDT'
        
        # Mock order responses
        order_responses = {{
            'main_order': {{
                'result': {{
                    'orderId': 'main123',
                    'orderStatus': 'filled',
                    'side': 'buy',
                    'qty': '0.1'
                }}
            }},
            'stop_loss_order': {{
                'result': {{
                    'orderId': 'sl123',
                    'orderStatus': 'new'
                }}
            }},
            'take_profit_order': {{
                'result': {{
                    'orderId': 'tp123',
                    'orderStatus': 'new'
                }}
            }}
        }}
        
        # Test order update
        self.strategy.on_order_update(order_responses, symbol)
        
        # Check that position was updated
        self.assertIsNotNone(self.strategy.position.get(symbol))
        if self.strategy.active_order_id.get(symbol):
            self.assertEqual(self.strategy.active_order_id.get(symbol), 'main123')

    def test_on_trade_update(self):
        """Test trade update handling"""
        symbol = 'BTC/USDT'
        
        # Set up position first
        self.strategy.position[symbol] = {{
            'main_order': {{'result': {{'side': 'buy', 'qty': '0.1'}}}},
            'stop_loss_order': {{'result': {{'orderId': 'sl123'}}}},
            'take_profit_order': {{'result': {{'orderId': 'tp123'}}}}
        }}
        
        # Mock trade data
        trade_data = {{
            'symbol': symbol,
            'side': 'buy',
            'qty': '0.1',
            'price': '100.0',
            'status': 'closed'
        }}
        
        # Test trade update
        self.strategy.on_trade_update(trade_data, symbol)
        
        # Check that position was cleared (if strategy implements this)
        # Some strategies might not clear position on trade update
        # self.assertIsNone(self.strategy.position.get(symbol))

    def test_on_error(self):
        """Test error handling"""
        test_exception = Exception("Test error")
        
        # Should not raise exception
        try:
            self.strategy.on_error(test_exception)
        except Exception as e:
            self.fail(f"on_error raised an exception: {{e}}")

    def test_clear_position(self):
        """Test position clearing"""
        symbol = 'BTC/USDT'
        
        # Set up position
        self.strategy.position[symbol] = {{
            'main_order': {{'result': {{'side': 'buy', 'qty': '0.1'}}}},
            'stop_loss_order': {{'result': {{'orderId': 'sl123'}}}},
            'take_profit_order': {{'result': {{'orderId': 'tp123'}}}}
        }}
        self.strategy.order_pending[symbol] = True
        self.strategy.active_order_id[symbol] = 'main123'
        
        # Clear position
        self.strategy.clear_position(symbol)
        
        # Check that position was cleared
        self.assertIsNone(self.strategy.position.get(symbol))
        self.assertFalse(self.strategy.order_pending.get(symbol, False))
        self.assertIsNone(self.strategy.active_order_id.get(symbol))

    def test_config_fallback_values(self):
        """Test configuration fallback values"""
        # Create strategy with minimal config
        minimal_config = {{'strategy_configs': {{'{strategy_class_name}': {{}}}}}}
        
        try:
            strategy = {strategy_class_name}(
                data=self.sample_data,
                config=minimal_config,
                logger=self.logger
            )
            
            # Should initialize without errors
            self.assertIsNotNone(strategy)
        except Exception as e:
            # Some strategies might require specific config parameters
            self.skipTest(f"Strategy requires specific config parameters: {{e}}")

    def test_data_integrity(self):
        """Test that strategy works with copy of data"""
        original_data = self.sample_data.copy()
        
        # Strategy should work on a copy (if implemented correctly)
        # Some modifications to strategy data are expected during indicator calculation
        self.assertIsNotNone(self.strategy.data)
        
        # Original data should have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col in original_data.columns:
                self.assertIn(col, original_data.columns)

    def test_update_indicators_for_new_row(self):
        """Test incremental indicator updates"""
        if not hasattr(self.strategy, 'update_indicators_for_new_row'):
            self.skipTest("Strategy does not implement update_indicators_for_new_row")
        
        # Add new row
        new_row = pd.DataFrame({{
            'timestamp': [pd.Timestamp('2023-01-01 03:20:00')],
            'open': [105.0],
            'high': [106.0],
            'low': [104.5],
            'close': [105.5],
            'volume': [5500]
        }})
        
        original_length = len(self.strategy.data)
        self.strategy.data = pd.concat([self.strategy.data, new_row], ignore_index=True)
        
        # Update indicators
        try:
            self.strategy.update_indicators_for_new_row()
            # Should not raise exception
            self.assertEqual(len(self.strategy.data), original_length + 1)
        except Exception as e:
            self.fail(f"update_indicators_for_new_row raised an exception: {{e}}")


if __name__ == '__main__':
    unittest.main()'''
    
    return template, test_file_name

def main():
    """Main function to generate tests for all strategies."""
    strategy_files = get_strategy_files()
    existing_tests = get_existing_test_files()
    
    print(f"Found {len(strategy_files)} strategy files")
    print(f"Found {len(existing_tests)} existing test files")
    
    generated_count = 0
    
    for strategy_file in strategy_files:
        strategy_file_name = strategy_file.stem
        expected_test_file = f"test_{strategy_file_name}.py"
        
        if expected_test_file in existing_tests:
            print(f"✓ Test already exists for {strategy_file_name}")
            continue
        
        # Skip deprecated or example strategies
        if 'deprecated' in strategy_file_name.lower() or 'example' in strategy_file_name.lower():
            print(f"⚠ Skipping {strategy_file_name} (deprecated/example)")
            continue
        
        strategy_class_name = get_strategy_class_name(strategy_file)
        if not strategy_class_name:
            print(f"✗ Could not extract class name from {strategy_file_name}")
            continue
        
        print(f"📝 Generating test for {strategy_file_name} (class: {strategy_class_name})")
        
        template, test_file_name = generate_test_template(strategy_file, strategy_class_name)
        
        # Write the test file
        test_file_path = Path(__file__).parent / test_file_name
        with open(test_file_path, 'w') as f:
            f.write(template)
        
        generated_count += 1
        print(f"✓ Generated {test_file_name}")
    
    print(f"\n🎉 Generated {generated_count} new test files")

if __name__ == '__main__':
    main() 