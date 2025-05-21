import pytest
import pandas as pd
from unittest.mock import MagicMock
from strategies.strategy_template import StrategyTemplate

class DummyLogger:
    def info(self, msg): pass
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

# Minimal concrete subclass for testing
class TestStrategy(StrategyTemplate):
    def init_indicators(self):
        self.data['dummy'] = 1
    def _check_entry_conditions(self, symbol):
        # Only signal entry if no position or order pending
        return {'side': 'buy', 'size': 1} if not self.position.get(symbol) and not self.order_pending.get(symbol, False) else None
    def check_exit(self, symbol):
        return {'type': 'market', 'reason': 'test'} if self.position.get(symbol) else None
    def get_risk_parameters(self):
        return {'sl_pct': 0.01, 'tp_pct': 0.02}

@pytest.fixture
def dummy_logger():
    return DummyLogger()

@pytest.fixture
def config():
    return {'default': {'default_order_size': 1}}

@pytest.fixture
def data():
    return pd.DataFrame({'close': [1, 2, 3, 4, 5]})

@pytest.fixture
def strategy(config, data, dummy_logger):
    return TestStrategy(data, config, logger=dummy_logger)

def test_check_entry_and_exit(strategy):
    symbol = 'BTCUSDT'
    entry = strategy.check_entry(symbol)
    assert entry is not None and entry['side'] == 'buy'
    strategy.position[symbol] = {'side': 'buy'}
    exit_signal = strategy.check_exit(symbol)
    assert exit_signal is not None and exit_signal['type'] == 'market'
    print('test_check_entry_and_exit PASSED')

def test_on_order_update_and_clear_position(strategy):
    symbol = 'BTCUSDT'
    order_responses = {'main_order': {'result': {'orderId': 'oid', 'orderStatus': 'filled', 'side': 'buy', 'cumExecQty': 1, 'avgPrice': 100}}}
    strategy.on_order_update(order_responses, symbol)
    assert strategy.position[symbol] is not None
    strategy.clear_position(symbol)
    assert strategy.position[symbol] is None
    print('test_on_order_update_and_clear_position PASSED')

def test_on_trade_update(strategy):
    symbol = 'BTCUSDT'
    strategy.position[symbol] = {'side': 'buy'}
    trade = {'exit': True}
    strategy.on_trade_update(trade, symbol)
    assert strategy.position[symbol] is None
    print('test_on_trade_update PASSED')

def test_log_state_change(strategy):
    symbol = 'BTCUSDT'
    strategy.log_state_change(symbol, 'state1', 'msg1')
    strategy.log_state_change(symbol, 'state1', 'msg2')  # Should not log again
    strategy.log_state_change(symbol, 'state2', 'msg3')
    print('test_log_state_change PASSED')

def test_on_error(strategy):
    strategy.on_error(Exception('test'))
    print('test_on_error PASSED') 