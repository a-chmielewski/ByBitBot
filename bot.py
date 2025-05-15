import os
import importlib
import json
from modules.logger import get_logger
from modules.exchange import ExchangeConnector
from modules.data_fetcher import LiveDataFetcher
from modules.order_manager import OrderManager
from modules.performance_tracker import PerformanceTracker
from datetime import datetime
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

CONFIG_PATH = 'config.json'
STRATEGY_DIR = 'strategies'


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def list_strategies():
    files = [f for f in os.listdir(STRATEGY_DIR) if f.endswith('.py') and not f.startswith('__') and 'template' not in f]
    return [os.path.splitext(f)[0] for f in files]

def select_strategies(available):
    print('Available strategies:')
    for i, name in enumerate(available):
        print(f"  {i+1}. {name}")
    selected = input('Select strategies (comma-separated indices, e.g. 1,2): ')
    indices = [int(i.strip())-1 for i in selected.split(',') if i.strip().isdigit() and 0 <= int(i.strip())-1 < len(available)]
    return [available[i] for i in indices]

def dynamic_import_strategy(name):
    module = importlib.import_module(f"strategies.{name}")
    # Find the class that inherits from StrategyTemplate
    for attr in dir(module):
        obj = getattr(module, attr)
        if isinstance(obj, type) and hasattr(obj, '__bases__') and any('StrategyTemplate' in str(base) for base in obj.__bases__):
            return obj
    raise ImportError(f"No valid strategy class found in {name}")

def main():
    config = load_config()
    logger = get_logger('bot')
    logger.info('Bot starting up.')
    # Exchange
    ex_cfg = config['bybit']
    exchange = ExchangeConnector(api_key=ex_cfg['api_key'], api_secret=ex_cfg['api_secret'], testnet=False, logger=logger)
    # Data fetcher
    default_cfg = config['default']
    symbol = default_cfg['coin_pair'].replace('/', '').upper()
    timeframe = default_cfg['timeframe']
    data_fetcher = LiveDataFetcher(exchange, symbol, timeframe, logger=logger)
    data = data_fetcher.fetch_initial_data()
    # Start WebSocket for live data
    data_fetcher.start_websocket()
    logger.info(f"Fetched initial OHLCV data: {len(data)} rows for {symbol} {timeframe}")
    # Order manager
    order_manager = OrderManager(exchange, logger=logger)
    # Performance tracker
    perf_tracker = PerformanceTracker(logger=logger)
    # Strategy selection
    available = list_strategies()
    selected = select_strategies(available)
    strategies = []
    for strat_name in selected:
        StratClass = dynamic_import_strategy(strat_name)
        strat_logger = get_logger(strat_name)
        strategies.append(StratClass(data, config, logger=strat_logger))
    logger.info(f"Loaded strategies: {[type(s).__name__ for s in strategies]}")
    # Main trading loop
    try:
        while True:
            data = data_fetcher.update_data()
            # Sync active orders with exchange
            order_manager.sync_active_orders_with_exchange(symbol)
            for strat in strategies:
                strat.data = data  # Ensure strategy uses latest data
                strat.update_indicators_for_new_row()  # Efficiently update indicators for new data
                # Debug: log the latest row's indicator values
                latest_row = strat.data.iloc[-1].to_dict()
                entry_signal = strat.check_entry()
                if entry_signal:
                    # Assert required keys are present
                    assert 'side' in entry_signal, f"Strategy {type(strat).__name__} did not return 'side' in entry_signal: {entry_signal}"
                    assert 'price' in entry_signal, f"Strategy {type(strat).__name__} did not return 'price' in entry_signal: {entry_signal}"
                    # Use SL/TP from entry_signal if present, otherwise compute
                    order_details = entry_signal.copy()
                    if 'stop_loss' not in order_details or 'take_profit' not in order_details:
                        if hasattr(strat, 'get_risk_parameters') and strat.__class__.__name__ == 'StrategyDoubleEMAStochOsc':
                            risk_params = strat.get_risk_parameters(current_price=entry_signal['price'], side=entry_signal['side'])
                        else:
                            risk_params = strat.get_risk_parameters()
                        order_details['stop_loss'] = risk_params.get('stop_loss')
                        order_details['take_profit'] = risk_params.get('take_profit')
                    order_manager.place_order_with_risk(
                        symbol=symbol,
                        side=order_details['side'],
                        order_type='market',
                        size=order_details['size'],
                        price=order_details.get('price'),
                        stop_loss=order_details.get('stop_loss'),
                        take_profit=order_details.get('take_profit'),
                        params=None,
                        reduce_only=False,
                        time_in_force='GoodTillCancel',
                        stop_price=None
                    )
                    strat.on_order_update(order_details)
                # Check for open position and exit
                if strat.position and strat.check_exit(strat.position):
                    trade_details = {
                        'strategy': type(strat).__name__,
                        'symbol': symbol,
                        'exit': True,
                        'timestamp': datetime.utcnow().isoformat(),
                    }
                    strat.on_trade_update(trade_details)
                    perf_tracker.record_trade(trade_details)
            logger.debug(f"Rolling drawdown: {perf_tracker.rolling_drawdown_curve()[-1] if perf_tracker.trades else 0}")
            logger.debug(f"Rolling Sharpe: {perf_tracker.rolling_sharpe()[-1] if perf_tracker.trades else 0}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info('Bot shutting down (KeyboardInterrupt).')
    except Exception as exc:
        logger.error(f'Bot crashed: {exc}')
        PerformanceTracker.persist_on_exception(perf_tracker)
        raise
    finally:
        data_fetcher.stop_websocket()
        perf_tracker.close_session()
        logger.info('Bot session closed.')

if __name__ == '__main__':
    main() 