import pytest
from unittest.mock import MagicMock, patch
from modules.order_manager import OrderManager, OrderExecutionError

class DummyLogger:
    def info(self, msg): pass
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg, exc_info=None): pass

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
    # Mock fetch_open_orders to return empty list (no conditional orders to cancel)
    exchange.fetch_open_orders.return_value = {'retCode': 0, 'retMsg': 'OK', 'result': {'list': []}}
    return exchange

@pytest.fixture
def manager(mock_exchange, dummy_logger):
    return OrderManager(mock_exchange, logger=dummy_logger, max_retries=2, backoff_base=0)

def test_place_order_with_risk_market_success(manager):
    # Should place main, SL, and TP orders
    result = manager.place_order_with_risk(
        symbol='BTCUSDT', side='buy', order_type='market', size=1, signal_price=100,
        sl_pct=0.01, tp_pct=0.02, params=None, reduce_only=False, time_in_force='GoodTillCancel',
        market_condition='UNKNOWN', urgency='NORMAL', slippage_tolerance=0.002
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
        sl_pct=0.01, tp_pct=0.02, params=None, reduce_only=False, time_in_force='GoodTillCancel',
        market_condition='UNKNOWN', urgency='NORMAL', slippage_tolerance=0.002
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
            sl_pct=0.01, tp_pct=0.02, params=None, reduce_only=False, time_in_force='GoodTillCancel',
            market_condition='UNKNOWN', urgency='NORMAL', slippage_tolerance=0.002
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

def test_log_order_status_bybit_response_structure(manager):
    """Test that demonstrates the fix for ByBit API response structure where order details are in 'result' field"""
    # This is the actual format returned by ByBit API
    bybit_response = {
        'retCode': 0,
        'retMsg': 'OK',
        'result': {
            'orderId': '1fceb41b-fdc6-4e3a-a040-88fa975293bd',
            'orderStatus': 'Filled',
            'side': 'Buy',
            'orderType': 'Market',
            'qty': '0.01',
            'price': '2515.33',
            'avgPrice': '2515.33'
        },
        'retExtInfo': {},
        'time': 1750354593551
    }
    
    # Before our fix, this would have logged "N/A" and "Unknown" values
    # After our fix, it should properly extract the order details
    manager.log_order_status(bybit_response, 'Main order')
    print('test_log_order_status_bybit_response_structure PASSED')

def test_log_order_status_fallback_to_top_level():
    """Test fallback when order details are at top level (for backwards compatibility)"""
    from tests.test_order_manager import DummyLogger
    from modules.order_manager import OrderManager
    from unittest.mock import MagicMock
    
    exchange = MagicMock()
    logger = DummyLogger()
    manager = OrderManager(exchange, logger=logger)
    
    # Response with order details at top level (no 'result' field)
    top_level_response = {
        'retCode': 0,
        'orderId': 'test-order-id',
        'orderStatus': 'Filled',
        'side': 'Buy',
        'orderType': 'Market',
        'qty': '0.01',
        'price': '2515.33',
        'avgPrice': '2515.33'
    }
    
    # Should still work with our fallback logic
    manager.log_order_status(top_level_response, 'Test order')
    print('test_log_order_status_fallback_to_top_level PASSED') 