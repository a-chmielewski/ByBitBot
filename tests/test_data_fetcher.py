import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.data_fetcher import LiveDataFetcher, DataFetchError

class DummyLogger:
    def info(self, msg): pass
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

@pytest.fixture
def mock_exchange():
    exchange = MagicMock()
    exchange.testnet = True
    return exchange

@pytest.fixture
def dummy_logger():
    return DummyLogger()

@pytest.fixture
def fetcher(mock_exchange, dummy_logger):
    return LiveDataFetcher(mock_exchange, 'BTCUSDT', '1m', window_size=5, logger=dummy_logger)

def test_fetch_initial_data_success(fetcher, mock_exchange):
    # Simulate ByBit API response
    ohlcv = [
        [1000, 1, 2, 0.5, 1.5, 10, 100],
        [2000, 1.5, 2.5, 1, 2, 20, 200],
        [3000, 2, 3, 1.5, 2.5, 30, 300],
        [4000, 2.5, 3.5, 2, 3, 40, 400],
        [5000, 3, 4, 2.5, 3.5, 50, 500],
    ]
    mock_exchange.fetch_ohlcv.return_value = {'result': {'list': ohlcv}}
    df = fetcher.fetch_initial_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    print('test_fetch_initial_data_success PASSED')

def test_fetch_initial_data_empty(mock_exchange, fetcher):
    mock_exchange.fetch_ohlcv.return_value = {'result': {'list': []}}
    with pytest.raises(DataFetchError):
        fetcher.fetch_initial_data()
    print('test_fetch_initial_data_empty PASSED')

def test_fetch_initial_data_exception(mock_exchange, fetcher):
    mock_exchange.fetch_ohlcv.side_effect = Exception('API error')
    with pytest.raises(DataFetchError):
        fetcher.fetch_initial_data()
    print('test_fetch_initial_data_exception PASSED')

def test_update_data_adds_new_row(fetcher, mock_exchange):
    # Initial data
    ohlcv = [
        [1000, 1, 2, 0.5, 1.5, 10, 100],
        [2000, 1.5, 2.5, 1, 2, 20, 200],
        [3000, 2, 3, 1.5, 2.5, 30, 300],
        [4000, 2.5, 3.5, 2, 3, 40, 400],
        [5000, 3, 4, 2.5, 3.5, 50, 500],
    ]
    mock_exchange.fetch_ohlcv.return_value = {'result': {'list': ohlcv}}
    fetcher.fetch_initial_data()
    # Newer bar
    new_ohlcv = [
        [5000, 3, 4, 2.5, 3.5, 50, 500],
        [6000, 3.5, 4.5, 3, 4, 60, 600],
    ]
    mock_exchange.fetch_ohlcv.return_value = {'result': {'list': new_ohlcv}}
    df = fetcher.update_data()
    assert len(df) == 5  # Rolling window size
    assert df['timestamp'].iloc[-1] > df['timestamp'].iloc[-2]
    print('test_update_data_adds_new_row PASSED')

def test_update_data_replaces_last_row(fetcher, mock_exchange):
    ohlcv = [
        [1710000000000, 1, 2, 0.5, 1.5, 10, 100],
        [1710000001000, 1.5, 2.5, 1, 2, 20, 200],
        [1710000002000, 2, 3, 1.5, 2.5, 30, 300],
        [1710000003000, 2.5, 3.5, 2, 3, 40, 400],
        [1710000004000, 3, 4, 2.5, 3.5, 50, 500],
    ]
    mock_exchange.fetch_ohlcv.return_value = {'result': {'list': ohlcv}}
    fetcher.fetch_initial_data()
    # Same last bar (should replace)
    new_ohlcv = [
        [1710000003000, 2.5, 3.5, 2, 3, 40, 400],
        [1710000004000, 3.1, 4.1, 2.6, 3.6, 51, 501],
    ]
    mock_exchange.fetch_ohlcv.return_value = {'result': {'list': new_ohlcv}}
    df = fetcher.update_data()
    assert df['close'].iloc[-1] == 3.6
    print('test_update_data_replaces_last_row PASSED')

def test_update_data_empty(mock_exchange, fetcher):
    mock_exchange.fetch_ohlcv.return_value = {'result': {'list': []}}
    with pytest.raises(DataFetchError):
        fetcher.update_data()
    print('test_update_data_empty PASSED')

def test_get_data_trims_to_window(fetcher, mock_exchange):
    ohlcv = [
        [1000, 1, 2, 0.5, 1.5, 10, 100],
        [2000, 1.5, 2.5, 1, 2, 20, 200],
        [3000, 2, 3, 1.5, 2.5, 30, 300],
        [4000, 2.5, 3.5, 2, 3, 40, 400],
        [5000, 3, 4, 2.5, 3.5, 50, 500],
    ]
    mock_exchange.fetch_ohlcv.return_value = {'result': {'list': ohlcv}}
    fetcher.fetch_initial_data()
    # Add extra row manually
    fetcher.data = pd.concat([fetcher.data, fetcher.data.iloc[[-1]]], ignore_index=True)
    assert len(fetcher.data) == 6
    trimmed = fetcher.get_data()
    assert len(trimmed) == 5
    print('test_get_data_trims_to_window PASSED')

def test_remove_old_data(fetcher, mock_exchange):
    ohlcv = [
        [1000, 1, 2, 0.5, 1.5, 10, 100],
        [2000, 1.5, 2.5, 1, 2, 20, 200],
        [3000, 2, 3, 1.5, 2.5, 30, 300],
        [4000, 2.5, 3.5, 2, 3, 40, 400],
        [5000, 3, 4, 2.5, 3.5, 50, 500],
    ]
    mock_exchange.fetch_ohlcv.return_value = {'result': {'list': ohlcv}}
    fetcher.fetch_initial_data()
    # Add extra rows
    fetcher.data = pd.concat([fetcher.data, fetcher.data.iloc[[-1]], fetcher.data.iloc[[-1]]], ignore_index=True)
    assert len(fetcher.data) == 7
    fetcher.remove_old_data()
    assert len(fetcher.data) == 5
    print('test_remove_old_data PASSED')

def test_normalize_timeframe(fetcher):
    assert fetcher._normalize_timeframe_to_bybit_interval('1m') == '1'
    assert fetcher._normalize_timeframe_to_bybit_interval('15m') == '15'
    assert fetcher._normalize_timeframe_to_bybit_interval('1h') == '60'
    assert fetcher._normalize_timeframe_to_bybit_interval('4h') == '240'
    assert fetcher._normalize_timeframe_to_bybit_interval('1d') == 'D'
    assert fetcher._normalize_timeframe_to_bybit_interval('1w') == 'W'
    assert fetcher._normalize_timeframe_to_bybit_interval('1M') == 'M'
    assert fetcher._normalize_timeframe_to_bybit_interval('30') == '30'
    # Unsupported format
    assert fetcher._normalize_timeframe_to_bybit_interval('weird') == 'weird'
    print('test_normalize_timeframe PASSED') 