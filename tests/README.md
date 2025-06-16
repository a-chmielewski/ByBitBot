# Trading Bot Strategy Test Suite

This directory contains comprehensive tests for all trading strategies implemented in the cryptocurrency trading bot.

## Overview

The test suite provides thorough testing coverage for:

- **Strategy Initialization**: Verifying proper setup and configuration
- **Indicator Calculations**: Testing technical indicator accuracy and error handling
- **Entry/Exit Logic**: Validating signal generation and trade decision logic
- **Risk Management**: Ensuring proper stop-loss and take-profit handling
- **Position Management**: Testing order lifecycle and state management
- **Error Handling**: Verifying graceful error recovery
- **Configuration**: Testing parameter handling and defaults

## Test Structure

### Core Test Files

- `test_strategy_template.py` - Tests for the base strategy template
- `test_example_strategy.py` - Tests for the example/reference strategy
- `test_double_EMA_StochOsc.py` - Tests for the deprecated double EMA strategy

### Strategy-Specific Tests

- `test_ema_adx_strategy.py` - Tests for EMA Trend Rider with ADX filter
- `test_bollinger_mean_reversion_strategy.py` - Tests for Bollinger Band mean reversion
- `test_atr_momentum_breakout_strategy.py` - Tests for ATR momentum breakout scalper

### Module Tests

- `test_data_fetcher.py` - Tests for market data fetching and management
- `test_exchange.py` - Tests for exchange integration
- `test_order_manager.py` - Tests for order execution and management
- `test_performance_tracker.py` - Tests for performance metrics tracking
- `test_market_analyzer.py` - Tests for market condition analysis

### Utilities

- `test_strategy_generator.py` - Automated test generation for new strategies
- `run_all_strategy_tests.py` - Test runner with comprehensive reporting

## Running Tests

### Run All Strategy Tests

```bash
# Run all strategy tests with detailed output
python tests/run_all_strategy_tests.py

# Alternative using unittest
python -m unittest discover tests -p "test_*strategy*.py" -v
```

### Run Specific Strategy Tests

```bash
# Run tests for a specific strategy
python tests/run_all_strategy_tests.py run ema_adx_strategy

# Run individual test file
python -m unittest tests.test_ema_adx_strategy -v
```

### List Available Tests

```bash
# List all available strategy tests
python tests/run_all_strategy_tests.py list
```

### Run Individual Test Methods

```bash
# Run a specific test method
python -m unittest tests.test_ema_adx_strategy.TestEMAADXStrategy.test_initialization -v
```

## Test Coverage

### Strategy Interface Testing

Each strategy test verifies:

1. **Initialization**
   - Proper inheritance from StrategyTemplate
   - Configuration parameter loading
   - Market type tags and visibility settings
   - Internal state initialization

2. **Indicator Calculation**
   - All required indicators are computed
   - Indicators have valid values (not all NaN)
   - Indicator relationships are logical
   - Handles insufficient data gracefully
   - Manual calculation fallbacks work

3. **Entry Signal Generation**
   - Signals generated when conditions met
   - No signals when position already exists
   - No signals when order is pending
   - Signal structure is valid
   - Risk parameters are included

4. **Exit Signal Generation**
   - Exit signals when position is open
   - No exit signals when no position
   - Exit logic responds to market conditions
   - Time-based exits work correctly

5. **Risk Management**
   - Stop-loss and take-profit parameters
   - ATR-based dynamic risk sizing
   - Configuration fallback values
   - Risk parameter validation

6. **Order Lifecycle**
   - Order update handling
   - Trade update processing
   - Position state management
   - State cleanup on trade close

7. **Error Handling**
   - Graceful error recovery
   - Missing data handling
   - Invalid configuration handling
   - Logging without exceptions

### Data and Edge Cases

Tests include scenarios for:

- **Market Conditions**: Trending, ranging, high/low volatility
- **Data Quality**: Missing columns, insufficient data, NaN values
- **Configuration**: Missing parameters, invalid values, edge cases
- **State Management**: Position tracking, order states, cleanup
- **Performance**: Indicator calculation efficiency, memory usage

## Test Data

### Sample Data Generation

Tests use realistic market data including:

- **OHLCV Data**: Open, High, Low, Close, Volume with realistic patterns
- **Market Patterns**: Trending, ranging, breakout, mean-reversion scenarios
- **Volume Patterns**: Normal volume, volume surges, low volume periods
- **Volatility**: Low, medium, high volatility environments

### Reproducible Testing

- All tests use `np.random.seed(42)` for reproducible results
- Sample data patterns are designed to trigger strategy conditions
- Edge cases are specifically tested with crafted data

## Writing New Strategy Tests

### Automated Generation

Use the test generator for new strategies:

```bash
python tests/test_strategy_generator.py
```

This will automatically create test files for strategies that don't have them.

### Manual Test Creation

When creating tests manually, follow this structure:

1. **Import Strategy**: Import the strategy class to test
2. **Setup Data**: Create realistic OHLCV data for the strategy type
3. **Configuration**: Provide complete test configuration
4. **Test Methods**: Cover all aspects of strategy functionality
5. **Edge Cases**: Test error conditions and edge cases

### Test Template

```python
import unittest
import pandas as pd
import numpy as np
import logging

class TestYourStrategy(unittest.TestCase):
    def setUp(self):
        # Create sample data and configuration
        # Initialize strategy
        
    def test_initialization(self):
        # Test strategy setup
        
    def test_init_indicators(self):
        # Test indicator calculation
        
    def test_entry_conditions(self):
        # Test entry signal logic
        
    def test_exit_conditions(self):
        # Test exit signal logic
        
    def test_risk_parameters(self):
        # Test risk management
        
    # Add more specific tests as needed
```

## Configuration Testing

### Parameter Coverage

Tests verify:

- **Required Parameters**: Strategy fails gracefully if missing
- **Optional Parameters**: Default values are used correctly
- **Invalid Values**: Proper validation and error handling
- **Edge Values**: Boundary conditions and extreme values

### Configuration Structure

```python
test_config = {
    'strategy_configs': {
        'YourStrategyClass': {
            'parameter1': value1,
            'parameter2': value2,
            # ... strategy-specific parameters
        }
    },
    'default': {
        'default_sl_pct': 0.005,
        'default_tp_pct': 0.015
    }
}
```

## Performance Testing

### Execution Time

Tests measure:

- Indicator calculation performance
- Signal generation speed
- Memory usage patterns
- Data handling efficiency

### Scalability

Tests verify:

- Performance with large datasets
- Memory efficiency with long data histories
- Incremental update performance

## Continuous Integration

### Automated Testing

The test suite is designed for CI/CD:

- Exit codes indicate success/failure
- Detailed error reporting
- JSON output option for integration
- Parallel test execution support

### Quality Gates

Tests enforce:

- Minimum test coverage requirements
- Performance benchmarks
- Code quality standards
- Documentation completeness

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure Python path includes project root
   - Check for missing dependencies
   - Verify strategy file structure

2. **Data Issues**
   - Check pandas_ta installation
   - Verify data column names
   - Ensure sufficient data length

3. **Configuration Errors**
   - Validate configuration structure
   - Check parameter types and values
   - Ensure required parameters are present

### Debugging Tests

```bash
# Run with maximum verbosity
python -m unittest tests.test_your_strategy -v

# Run specific failing test
python -m unittest tests.test_your_strategy.TestClass.test_method -v

# Enable debug logging
PYTHONPATH=. python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
import unittest
unittest.main(module='tests.test_your_strategy', exit=False)
"
```

## Contributing

### Adding New Strategy Tests

1. Create test file following naming convention: `test_strategy_name.py`
2. Use the test generator or template as starting point
3. Cover all strategy interface methods
4. Include edge cases and error conditions
5. Add configuration tests
6. Update this README if needed

### Test Quality Standards

- **Coverage**: Aim for >90% code coverage
- **Assertions**: Use specific assertions with clear messages
- **Data**: Use realistic market data patterns
- **Documentation**: Document test purpose and expected behavior
- **Performance**: Tests should complete in <30 seconds

## Integration with Main Bot

### Testing with Real Data

```python
# Use actual market data for integration testing
from modules.data_fetcher import DataFetcher

data_fetcher = DataFetcher(config)
real_data = data_fetcher.fetch_ohlcv('BTC/USDT', '1m', 1000)

strategy = YourStrategy(real_data, config, logger)
# Test with real market conditions
```

### Configuration Validation

```python
# Validate strategy config matches main bot config
import json

with open('config.json', 'r') as f:
    main_config = json.load(f)

# Ensure test config is compatible
test_config = create_test_config()
assert all(key in main_config.get('strategy_configs', {}) 
          for key in test_config['strategy_configs'])
```

---

For questions or issues with the test suite, please refer to the project documentation or create an issue in the repository. 