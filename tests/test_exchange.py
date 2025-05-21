import pytest
from unittest.mock import MagicMock, patch
from modules.exchange import ExchangeConnector, ExchangeError

class DummyLogger:
    def info(self, msg): pass
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass

@pytest.fixture
def dummy_logger():
    return DummyLogger()

@pytest.fixture
def connector(dummy_logger):
    with patch('modules.exchange.HTTP') as MockHTTP:
        mock_client = MagicMock()
        MockHTTP.return_value = mock_client
        conn = ExchangeConnector('key', 'secret', testnet=True, logger=dummy_logger)
        conn.client = mock_client
        return conn

# Helper for a successful API response
SUCCESS_RESPONSE = {'retCode': 0, 'retMsg': 'OK', 'result': {'list': [{}]}, 'time': 1234567890}

# Helper for a failed API response
FAIL_RESPONSE = {'retCode': 10001, 'retMsg': 'Some error'}

def test_place_order_success(connector):
    connector.client.place_order.return_value = SUCCESS_RESPONSE
    result = connector.place_order(symbol='BTCUSDT', side='Buy', qty=1)
    assert result['retCode'] == 0
    print('test_place_order_success PASSED')

def test_place_order_error(connector):
    connector.client.place_order.return_value = FAIL_RESPONSE
    with pytest.raises(ExchangeError):
        connector.place_order(symbol='BTCUSDT', side='Buy', qty=1)
    print('test_place_order_error PASSED')

def test_fetch_balance_success(connector):
    connector.client.get_wallet_balance.return_value = SUCCESS_RESPONSE
    result = connector.fetch_balance()
    assert result['retCode'] == 0
    print('test_fetch_balance_success PASSED')

def test_fetch_balance_error(connector):
    connector.client.get_wallet_balance.return_value = FAIL_RESPONSE
    with pytest.raises(ExchangeError):
        connector.fetch_balance()
    print('test_fetch_balance_error PASSED')

def test_fetch_positions_success(connector):
    connector.client.get_positions.return_value = SUCCESS_RESPONSE
    result = connector.fetch_positions('BTCUSDT')
    assert result['retCode'] == 0
    print('test_fetch_positions_success PASSED')

def test_fetch_positions_error(connector):
    connector.client.get_positions.return_value = FAIL_RESPONSE
    with pytest.raises(ExchangeError):
        connector.fetch_positions('BTCUSDT')
    print('test_fetch_positions_error PASSED')

def test_fetch_ohlcv_success(connector):
    connector.client.get_kline.return_value = SUCCESS_RESPONSE
    result = connector.fetch_ohlcv('BTCUSDT')
    assert result['retCode'] == 0
    print('test_fetch_ohlcv_success PASSED')

def test_fetch_ohlcv_error(connector):
    connector.client.get_kline.return_value = FAIL_RESPONSE
    with pytest.raises(ExchangeError):
        connector.fetch_ohlcv('BTCUSDT')
    print('test_fetch_ohlcv_error PASSED')

def test_cancel_order_success(connector):
    connector.client.cancel_order.return_value = SUCCESS_RESPONSE
    result = connector.cancel_order('BTCUSDT', 'orderid')
    assert result['retCode'] == 0
    print('test_cancel_order_success PASSED')

def test_cancel_order_error(connector):
    connector.client.cancel_order.return_value = FAIL_RESPONSE
    with pytest.raises(ExchangeError):
        connector.cancel_order('BTCUSDT', 'orderid')
    print('test_cancel_order_error PASSED')

def test_fetch_open_orders_success(connector):
    connector.client.get_open_orders.return_value = SUCCESS_RESPONSE
    result = connector.fetch_open_orders('BTCUSDT')
    assert result['retCode'] == 0
    print('test_fetch_open_orders_success PASSED')

def test_fetch_open_orders_error(connector):
    connector.client.get_open_orders.return_value = FAIL_RESPONSE
    with pytest.raises(ExchangeError):
        connector.fetch_open_orders('BTCUSDT')
    print('test_fetch_open_orders_error PASSED')

def test_fetch_order_success(connector):
    # Simulate order found in history
    order_data = {'orderId': 'oid'}
    resp = {'retCode': 0, 'retMsg': 'OK', 'result': {'list': [order_data]}, 'time': 1234567890}
    connector.client.get_order_history.return_value = resp
    result = connector.fetch_order('BTCUSDT', 'oid')
    assert result['result']['orderId'] == 'oid'
    print('test_fetch_order_success PASSED')

def test_fetch_order_not_found(connector):
    # Simulate not found in history, not found in open orders
    resp_empty = {'retCode': 0, 'retMsg': 'OK', 'result': {'list': []}, 'time': 1234567890}
    connector.client.get_order_history.return_value = resp_empty
    connector.client.get_open_orders.return_value = resp_empty
    with pytest.raises(ExchangeError):
        connector.fetch_order('BTCUSDT', 'oid')
    print('test_fetch_order_not_found PASSED')

def test_get_price_precision(connector):
    # Patch _fetch_instrument_info to return a priceFilter
    connector._fetch_instrument_info = MagicMock(return_value={'priceFilter': {'tickSize': '0.001'}})
    assert connector.get_price_precision('BTCUSDT') == 3
    print('test_get_price_precision PASSED')

def test_get_qty_precision(connector):
    connector._fetch_instrument_info = MagicMock(return_value={'lotSizeFilter': {'qtyStep': '0.01'}})
    assert connector.get_qty_precision('BTCUSDT') == 2
    print('test_get_qty_precision PASSED')

def test_get_min_order_amount(connector):
    # Patch client.get_instruments_info
    connector.client.get_instruments_info.return_value = {
        'retCode': 0,
        'retMsg': 'OK',
        'result': {'list': [{'lotSizeFilter': {'minOrderQty': '1', 'minOrderAmt': '10', 'qtyStep': '0.1'}}]},
        'time': 1234567890
    }
    result = connector.get_min_order_amount('BTCUSDT')
    assert result == (1.0, 10.0, 0.1)
    print('test_get_min_order_amount PASSED') 