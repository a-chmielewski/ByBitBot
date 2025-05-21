import pytest
import tempfile
import os
import shutil
import pandas as pd
from unittest.mock import MagicMock
from modules.performance_tracker import PerformanceTracker

class DummyLogger:
    def info(self, msg): pass
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

@pytest.fixture
def tmp_perf_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

@pytest.fixture
def tracker(tmp_perf_dir):
    return PerformanceTracker(log_dir=tmp_perf_dir, logger=DummyLogger())

def sample_trades():
    return [
        {'pnl': 10, 'side': 'buy', 'entry_price': 100, 'exit_price': 110, 'timestamp': '2024-01-01T00:00:00', 'exit_timestamp': '2024-01-01T01:00:00', 'strategy': 'A', 'symbol': 'BTCUSDT'},
        {'pnl': -5, 'side': 'sell', 'entry_price': 110, 'exit_price': 105, 'timestamp': '2024-01-01T02:00:00', 'exit_timestamp': '2024-01-01T03:00:00', 'strategy': 'A', 'symbol': 'BTCUSDT'},
        {'pnl': 20, 'side': 'buy', 'entry_price': 105, 'exit_price': 125, 'timestamp': '2024-01-01T04:00:00', 'exit_timestamp': '2024-01-01T05:00:00', 'strategy': 'B', 'symbol': 'ETHUSDT'},
        {'pnl': -10, 'side': 'sell', 'entry_price': 125, 'exit_price': 115, 'timestamp': '2024-01-01T06:00:00', 'exit_timestamp': '2024-01-01T07:00:00', 'strategy': 'B', 'symbol': 'ETHUSDT'},
    ]

def test_record_trade_and_metrics(tracker):
    for trade in sample_trades():
        tracker.record_trade(trade)
    assert tracker.cumulative_pnl == 15
    assert tracker.max_drawdown > 0
    print('test_record_trade_and_metrics PASSED')

def test_win_rate(tracker):
    for trade in sample_trades():
        tracker.record_trade(trade)
    assert tracker.win_rate() == 50.0
    print('test_win_rate PASSED')

def test_cumulative_return(tracker):
    for trade in sample_trades():
        tracker.record_trade(trade)
    assert tracker.cumulative_return() == 15
    print('test_cumulative_return PASSED')

def test_max_drawdown_value(tracker):
    for trade in sample_trades():
        tracker.record_trade(trade)
    assert tracker.max_drawdown_value() > 0
    print('test_max_drawdown_value PASSED')

def test_average_trade_duration(tracker):
    for trade in sample_trades():
        tracker.record_trade(trade)
    avg = tracker.average_trade_duration()
    assert avg > 0
    print('test_average_trade_duration PASSED')

def test_expectancy(tracker):
    for trade in sample_trades():
        tracker.record_trade(trade)
    val = tracker.expectancy()
    assert isinstance(val, float)
    print('test_expectancy PASSED')

def test_profit_factor(tracker):
    for trade in sample_trades():
        tracker.record_trade(trade)
    val = tracker.profit_factor()
    assert val > 0
    print('test_profit_factor PASSED')

def test_group_metrics(tracker):
    for trade in sample_trades():
        tracker.record_trade(trade)
    metrics = tracker.group_metrics(by='strategy')
    assert 'A' in metrics and 'B' in metrics
    print('test_group_metrics PASSED')

def test_rolling_drawdown_curve(tracker):
    for trade in sample_trades():
        tracker.record_trade(trade)
    curve = tracker.rolling_drawdown_curve(window=2)
    assert len(curve) == len(sample_trades())
    print('test_rolling_drawdown_curve PASSED')

def test_rolling_sharpe(tracker):
    for trade in sample_trades():
        tracker.record_trade(trade)
    sharpe = tracker.rolling_sharpe(window=2)
    assert len(sharpe) == len(sample_trades())
    print('test_rolling_sharpe PASSED')

def test_persist_to_csv_and_json(tracker, tmp_perf_dir):
    for trade in sample_trades():
        tracker.record_trade(trade)
    tracker.persist_to_csv('test_perf.csv')
    tracker.persist_to_json('test_perf.json')
    assert os.path.exists(os.path.join(tmp_perf_dir, 'test_perf.csv'))
    assert os.path.exists(os.path.join(tmp_perf_dir, 'test_perf.json'))
    print('test_persist_to_csv_and_json PASSED')

def test_close_session(tracker, tmp_perf_dir):
    for trade in sample_trades():
        tracker.record_trade(trade)
    tracker.close_session()
    assert os.path.exists(os.path.join(tmp_perf_dir, 'performance_log.csv'))
    assert os.path.exists(os.path.join(tmp_perf_dir, 'performance_log.json'))
    print('test_close_session PASSED')

def test_to_dataframe(tracker):
    for trade in sample_trades():
        tracker.record_trade(trade)
    df = tracker.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(sample_trades())
    print('test_to_dataframe PASSED')

def test_persist_on_exception(tracker, tmp_perf_dir):
    for trade in sample_trades():
        tracker.record_trade(trade)
    PerformanceTracker.persist_on_exception(tracker)
    assert os.path.exists(os.path.join(tmp_perf_dir, 'performance_log.csv'))
    assert os.path.exists(os.path.join(tmp_perf_dir, 'performance_log.json'))
    print('test_persist_on_exception PASSED') 