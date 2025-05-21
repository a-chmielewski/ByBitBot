import pytest
import pandas as pd
from unittest.mock import MagicMock
from strategies.double_EMA_StochOsc import StrategyDoubleEMAStochOsc

class DummyLogger:
    def info(self, msg): pass
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

@pytest.fixture
def dummy_logger():
    return DummyLogger()

@pytest.fixture
def config():
    return {
        'strategy_configs': {
            'StrategyDoubleEMAStochOsc': {
                'ema_slow_period': 10,
                'ema_fast_period': 5,
                'stoch_k_period': 3,
                'stoch_d_period': 2,
                'stoch_slowing_period': 2,
                'stoch_overbought': 80,
                'stoch_oversold': 20,
                'sl_pct': 0.01,
                'tp_pct': 0.02,
                'order_size': 1
            }
        }
    }

@pytest.fixture
def ohlcv_df():
    # Create synthetic OHLCV data
    data = {
        'timestamp': pd.date_range('2024-01-01', periods=30, freq='min'),
        'open': [100 + i for i in range(30)],
        'high': [101 + i for i in range(30)],
        'low': [99 + i for i in range(30)],
        'close': [100 + i for i in range(30)],
        'volume': [10 + i for i in range(30)]
    }
    return pd.DataFrame(data)

@pytest.fixture
def strategy(config, ohlcv_df, dummy_logger):
    strat = StrategyDoubleEMAStochOsc(config, data=ohlcv_df.copy(), logger=dummy_logger)
    strat.on_init()
    return strat

def test_on_init(strategy):
    # Just ensure no exceptions and parameters are set
    assert hasattr(strategy, 'ema_fast_period')
    assert hasattr(strategy, 'ema_slow_period')
    print('test_on_init PASSED')

def test_init_indicators(strategy):
    strategy.init_indicators()
    for col in ['ema_fast', 'ema_slow', 'stoch_k', 'stoch_d']:
        assert col in strategy.data.columns
    print('test_init_indicators PASSED')

def test_update_indicators_for_new_row(strategy):
    strategy.init_indicators()
    # Add a new row
    new_row = strategy.data.iloc[-1].copy()
    new_row['timestamp'] = new_row['timestamp'] + pd.Timedelta(minutes=1)
    new_row['close'] += 1
    new_row['open'] += 1
    new_row['high'] += 1
    new_row['low'] += 1
    new_row['volume'] += 1
    strategy.data = pd.concat([strategy.data, pd.DataFrame([new_row])], ignore_index=True)
    strategy.update_indicators_for_new_row()
    assert not pd.isna(strategy.data['ema_fast'].iloc[-1])
    assert not pd.isna(strategy.data['ema_slow'].iloc[-1])
    print('test_update_indicators_for_new_row PASSED')

def test_check_entry_conditions(strategy):
    strategy.init_indicators()
    # Force indicator values to trigger a long entry
    idx = strategy.data.index[-1]
    strategy.data.at[idx, 'ema_fast'] = 110
    strategy.data.at[idx, 'ema_slow'] = 100
    strategy.data.at[idx, 'close'] = 110
    strategy.data.at[idx, 'stoch_k'] = 10
    strategy.data.at[idx, 'stoch_d'] = 5
    strategy.data.at[idx-1, 'stoch_k'] = 0
    strategy.data.at[idx-1, 'stoch_d'] = 10
    entry = strategy._check_entry_conditions('BTCUSDT')
    assert entry is not None and entry['side'] == 'buy'
    print('test_check_entry_conditions PASSED')

def test_check_exit(strategy):
    strategy.init_indicators()
    # Simulate a position
    strategy.position = {'BTCUSDT': {'side': 'buy'}}
    idx = strategy.data.index[-1]
    strategy.data.at[idx, 'close'] = 90  # Below EMA fast
    strategy.data.at[idx, 'ema_fast'] = 100
    assert strategy.check_exit('BTCUSDT') is True
    print('test_check_exit PASSED')

def test_get_risk_parameters(strategy):
    params = strategy.get_risk_parameters()
    assert 'sl_pct' in params and 'tp_pct' in params
    print('test_get_risk_parameters PASSED') 