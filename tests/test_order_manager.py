import pytest
from unittest.mock import MagicMock, patch
from modules.order_manager import OrderManager, OrderExecutionError

class DummyLogger:
    def info(self, msg): pass
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

@pytest.fixture
def dummy_logger():
    return DummyLogger()

@pytest.fixture
def mock_exchange():
    exchange = MagicMock()
    exchange.get_min_order_amount.return_value = (1.0, 5.0, 0.1)
    exchange.get_qty_precision.return_value = 2
    exchange.get_price_precision.return_value = 2
    exchange.place_order.return_value = {'retCode': 0, 'retMsg': 'OK', 'result': {'orderId': 'oid', 'orderStatus': 'Filled', 'avgPrice': 100, 'cumExecQty': 1}, 'time': 1234567890}
    exchange.fetch_order.return_value = {'retCode': 0, 'retMsg': 'OK', 'result': {'orderId': 'oid', 'orderStatus': 'Filled', 'avgPrice': 100, 'cumExecQty': 1}, 'time': 1234567890}
    exchange.cancel_order.return_value = {'retCode': 0, 'retMsg': 'OK'}
    return exchange

@pytest.fixture
def manager(mock_exchange, dummy_logger):
    return OrderManager(mock_exchange, logger=dummy_logger, max_retries=2, backoff_base=0)

def test_place_order_with_risk_market_success(manager):
    # Should place main, SL, and TP orders
    result = manager.place_order_with_risk(
        symbol='BTCUSDT', side='buy', order_type='market', size=1, signal_price=100,
        sl_pct=0.01, tp_pct=0.02, params=None, reduce_only=False, time_in_force='GoodTillCancel'
    )
    assert 'main_order' in result
    assert 'stop_loss_order' in result
    assert 'take_profit_order' in result
    print('test_place_order_with_risk_market_success PASSED')

def test_place_order_with_risk_limit_success(manager):
    # Should place main, SL, and TP orders for limit
    manager.exchange.fetch_order.return_value = {'retCode': 0, 'retMsg': 'OK', 'result': {'orderId': 'oid', 'orderStatus': 'Filled', 'avgPrice': 100, 'cumExecQty': 1}, 'time': 1234567890}
    result = manager.place_order_with_risk(
        symbol='BTCUSDT', side='buy', order_type='limit', size=1, signal_price=100,
        sl_pct=0.01, tp_pct=0.02, params=None, reduce_only=False, time_in_force='GoodTillCancel'
    )
    assert 'main_order' in result
    assert 'stop_loss_order' in result
    assert 'take_profit_order' in result
    print('test_place_order_with_risk_limit_success PASSED')

def test_place_order_with_risk_main_order_error(manager):
    manager.exchange.place_order.side_effect = Exception('API error')
    with pytest.raises(OrderExecutionError):
        manager.place_order_with_risk(
            symbol='BTCUSDT', side='buy', order_type='market', size=1, signal_price=100,
            sl_pct=0.01, tp_pct=0.02, params=None, reduce_only=False, time_in_force='GoodTillCancel'
        )
    print('test_place_order_with_risk_main_order_error PASSED')

def test_retry_with_backoff_success(manager):
    func = MagicMock(return_value=42)
    result = manager._retry_with_backoff(func)
    assert result == 42
    print('test_retry_with_backoff_success PASSED')

def test_retry_with_backoff_failure(manager):
    func = MagicMock(side_effect=Exception('fail'))
    with pytest.raises(OrderExecutionError):
        manager._retry_with_backoff(func)
    print('test_retry_with_backoff_failure PASSED')

def test_log_order_status_success(manager):
    order_response = {'retCode': 0, 'result': {'orderId': 'oid', 'orderStatus': 'Filled', 'side': 'Buy', 'orderType': 'Market', 'qty': 1, 'price': 100, 'avgPrice': 100}}
    manager.log_order_status(order_response, 'Main order')
    print('test_log_order_status_success PASSED')

def test_log_order_status_error(manager):
    order_response = {'retCode': 10001, 'retMsg': 'Error', 'result': {'orderId': 'oid', 'orderStatus': 'Rejected', 'side': 'Buy', 'orderType': 'Market', 'qty': 1, 'price': 100, 'avgPrice': 0}}
    manager.log_order_status(order_response, 'Main order')
    print('test_log_order_status_error PASSED') 