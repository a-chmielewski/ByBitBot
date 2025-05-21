import os
import importlib
import json
import logging # Import logging for the helper function
from modules.logger import get_logger
from modules.exchange import ExchangeConnector
from modules.data_fetcher import LiveDataFetcher
from modules.order_manager import OrderManager, OrderExecutionError
from modules.performance_tracker import PerformanceTracker
from datetime import datetime, timezone
import time
import warnings

# Import StrategyTemplate for type checking in dynamic_import_strategy
from strategies.strategy_template import StrategyTemplate

warnings.simplefilter(action='ignore', category=FutureWarning)

CONFIG_PATH = 'config.json'
STRATEGY_DIR = 'strategies'


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def list_strategies():
    files = [f for f in os.listdir(STRATEGY_DIR) if f.endswith('.py') and not f.startswith('__') and 'template' not in f]
    return [os.path.splitext(f)[0] for f in files]

def select_strategies(available: list[str], logger_instance: logging.Logger): # Added logger
    logger_instance.info('Available strategies for selection:')
    for i, name in enumerate(available):
        print(f"  {i+1}. {name}")
        logger_instance.info(f"  {i+1}. {name}")
    
    selected_input = input('Select strategies (comma-separated indices, e.g. 1,2): ')
    logger_instance.info(f"User input for strategy selection: '{selected_input}'")
    
    indices = []
    if selected_input.strip(): # Check if input is not empty
        try:
            indices = [int(i.strip())-1 for i in selected_input.split(',') if i.strip().isdigit() and 0 <= int(i.strip())-1 < len(available)]
        except ValueError:
            logger_instance.error("Invalid input for strategy selection (non-integer value). No strategies selected.")
            return [] # Return empty if there's a non-integer value that's not filtered by isdigit
            
    selected_names = [available[i] for i in indices]
    logger_instance.info(f"Parsed selected indices: {indices}, Corresponding names: {selected_names}")
    return selected_names

def dynamic_import_strategy(name: str, base_class_to_check: type, logger_instance: logging.Logger) -> type:
    module_name = f"strategies.{name}"
    logger_instance.debug(f"Attempting to import module: {module_name}")
    try:
        module = importlib.import_module(module_name)
        logger_instance.debug(f"Successfully imported module: {module_name}. Inspected attributes: {dir(module)}")
    except ImportError as e:
        logger_instance.error(f"Failed to import module {module_name}: {e}", exc_info=True)
        raise  # Re-raise to be caught by the main loading loop

    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        # Ensure it's a class, a subclass of base_class_to_check, and not base_class_to_check itself
        if isinstance(attribute, type) and attribute is not base_class_to_check and issubclass(attribute, base_class_to_check):
            logger_instance.debug(f"Found valid strategy class '{attribute.__name__}' in {module_name}.")
            return attribute
            
    logger_instance.error(f"No valid strategy class (subclass of {base_class_to_check.__name__}) found in {module_name}.")
    raise ImportError(f"No valid strategy class (subclass of {base_class_to_check.__name__}) found in {module_name}")

def main():
    config = load_config()
    # Initialize the main bot logger
    bot_logger = get_logger('bot') # Renamed to bot_logger for clarity
    bot_logger.info('Bot starting up.')
    
    # Exchange
    ex_cfg = config['bybit']
    exchange = ExchangeConnector(api_key=ex_cfg['api_key'], api_secret=ex_cfg['api_secret'], testnet=False, logger=bot_logger)
    # Data fetcher
    default_cfg = config['default']
    symbol = default_cfg['coin_pair'].replace('/', '').upper()
    timeframe = default_cfg['timeframe']
    category = default_cfg.get('category', 'linear') # Get category from config, default to linear
    data_fetcher = LiveDataFetcher(exchange, symbol, timeframe, logger=bot_logger)
    data = data_fetcher.fetch_initial_data()
    # Start WebSocket for live data
    data_fetcher.start_websocket()
    bot_logger.info(f"Fetched initial OHLCV data: {len(data)} rows for {symbol} {timeframe}")
    # Order manager
    order_manager = OrderManager(exchange, logger=bot_logger)
    # Performance tracker
    perf_tracker = PerformanceTracker(logger=bot_logger)
    
    # Strategy selection
    available_strategy_files = list_strategies()
    bot_logger.info(f"Found strategy files: {available_strategy_files}")
    
    selected_strategy_names = select_strategies(available_strategy_files, bot_logger) # Pass bot_logger
    bot_logger.info(f"Strategies selected by user: {selected_strategy_names}")

    strategies = []
    if not selected_strategy_names:
        bot_logger.warning("No strategies were selected by the user, or selection failed.")
    else:
        for strat_name in selected_strategy_names:
            bot_logger.info(f"Attempting to load strategy: {strat_name}")
            try:
                # Pass bot_logger to dynamic_import_strategy
                StratClass = dynamic_import_strategy(strat_name, StrategyTemplate, bot_logger)
                # Each strategy instance gets its own logger
                strategy_specific_logger = get_logger(strat_name) 
                strategy_instance = StratClass(data.copy(), config, logger=strategy_specific_logger)
                strategies.append(strategy_instance)
                bot_logger.info(f"Successfully loaded and initialized strategy: {type(strategy_instance).__name__}")
            except ImportError as e:
                bot_logger.error(f"ImportError loading strategy module {strat_name}: {e}", exc_info=True)
            except Exception as e:
                bot_logger.error(f"Failed to load or initialize strategy class {strat_name}: {e}", exc_info=True)
    
    # Log the final list of loaded strategies
    loaded_strategy_class_names = [type(s).__name__ for s in strategies]
    bot_logger.info(f"Loaded strategies: {loaded_strategy_class_names if loaded_strategy_class_names else '[]'}")

    if not strategies:
        bot_logger.error("No strategies were successfully loaded. The bot will now exit.")
        if 'data_fetcher' in locals() and data_fetcher is not None:
            data_fetcher.stop_websocket()
        # Perf tracker might not be used if no trades happened.
        bot_logger.info('Bot session closed due to no strategies loaded.')
        return # Exit main() if no strategies are loaded

    # Initial state logging for each strategy
    for strat_instance in strategies:
        # Assuming strategies primarily operate on the main `symbol` defined in config for now.
        # If strategies can handle multiple symbols, this logging might need adjustment or be symbol-specific within the strategy.
        strat_instance.log_state_change(symbol, "awaiting_entry", f"Strategy {type(strat_instance).__name__} for {symbol}: Initialized. Looking for new entry conditions...")

    # Main trading loop
    try:
        bot_logger.info("Entering main trading loop.")
        while True:
            bot_logger.debug("Main loop iteration started.")
            
            bot_logger.debug("Calling data_fetcher.update_data()")
            data = data_fetcher.update_data()
            bot_logger.debug("data_fetcher.update_data() returned.")
            
            # Sync active orders with exchange and process adopted orders
            bot_logger.debug(f"Calling order_manager.sync_active_orders_with_exchange for {symbol}")
            adopted_orders = order_manager.sync_active_orders_with_exchange(symbol, category=category)
            bot_logger.debug("order_manager.sync_active_orders_with_exchange() returned.")

            # Check for and cancel orphaned conditional orders
            bot_logger.debug(f"Calling order_manager.check_and_cancel_orphaned_conditional_orders for {symbol} ({category})")
            try:
                order_manager.check_and_cancel_orphaned_conditional_orders(symbol, category=category)
            except Exception as e_orphan_check:
                bot_logger.error(f"Error during check_and_cancel_orphaned_conditional_orders: {e_orphan_check}", exc_info=True)
            bot_logger.debug("order_manager.check_and_cancel_orphaned_conditional_orders() returned.")

            if adopted_orders:
                for strat in strategies: # Notify all strategies (can be refined if strategies manage specific symbols)
                    for adopted_order in adopted_orders:
                        if adopted_order.get('symbol') == symbol: # Basic check
                            try:
                                bot_logger.info(f"Notifying strategy {type(strat).__name__} of adopted order {adopted_order.get('orderId')}")
                                strat.on_externally_synced_order(adopted_order, symbol)
                            except Exception as e_strat_notify:
                                bot_logger.error(f"Error notifying strategy {type(strat).__name__} of adopted order: {e_strat_notify}", exc_info=True)

            for strat in strategies:
                bot_logger.debug(f"Processing strategy {type(strat).__name__}")
                strat.data = data.copy()  # Ensure strategy uses a mutable copy of the latest data
                strat.init_indicators() # Initialize/recalculate all indicators on the (potentially new) data

                # Debug: log the latest row's indicator values
                # latest_row = strat.data.iloc[-1].to_dict()
                entry_signal = strat.check_entry(symbol=symbol)
                if entry_signal:
                    # Validate required fields are present in entry_signal
                    required_fields = ['side', 'size', 'sl_pct', 'tp_pct']
                    missing_fields = [field for field in required_fields if field not in entry_signal]
                    
                    if missing_fields:
                        bot_logger.error(f"Strategy {type(strat).__name__} produced invalid entry_signal missing required fields {missing_fields}: {entry_signal}")
                        continue  # Skip this signal and move to next strategy

                    # Validate price field based on order type
                    order_type = entry_signal.get('order_type', 'market')  # Default to market if not specified
                    if order_type != 'market' and 'price' not in entry_signal:
                        bot_logger.error(f"Strategy {type(strat).__name__} produced non-market order signal without 'price': {entry_signal}")
                        continue  # Skip this signal and move to next strategy

                    order_details = entry_signal.copy()

                    # The strategy should now always provide sl_pct and tp_pct.
                    # The OrderManager will calculate absolute SL/TP prices based on actual fill price.
                    # The old block for recalculating SL/TP if not in order_details is removed.

                    bot_logger.info(f"Order signal: {order_details}") # Log details before sending to OrderManager

                    try:
                        order_responses = order_manager.place_order_with_risk(
                         symbol=symbol,
                         side=order_details['side'],
                         order_type=order_details.get('order_type', 'market'),
                         size=order_details['size'],
                         signal_price=order_details.get('price'), # Price at the time of signal generation
                         sl_pct=order_details['sl_pct'],
                         tp_pct=order_details['tp_pct'],
                         params=order_details.get('params'), # Pass any extra params from strategy
                         reduce_only=order_details.get('reduce_only', False),
                         time_in_force=order_details.get('time_in_force', 'GoodTillCancel')
                        )
                    except OrderExecutionError as oe:
                        bot_logger.error(f"Order placement failed for {type(strat).__name__}: {oe}")
                        # Notify strategy of order failure with error response
                        error_response = {
                            'main_order': {
                                'result': {
                                    'orderId': None,
                                    'orderStatus': 'rejected',
                                    'error': str(oe)
                                }
                            }
                        }
                        try:
                            strat.on_order_update(error_response, symbol=symbol)
                        except Exception as callback_error:
                            bot_logger.error(f"Failed to notify strategy of error: {callback_error}", exc_info=True)
                            continue  # move on to next strategy without crashing the bot
                    except Exception as e:
                        bot_logger.error(f"Unexpected error during order placement for {type(strat).__name__}: {e}", exc_info=True)
                        error_response = {
                            'main_order': {
                                'result': {
                                    'orderId': None,
                                    'orderStatus': 'rejected',
                                    'error': f"Unexpected error: {str(e)}",
                                    'category': 'linear',
                                    'symbol': symbol,
                                    'side': order_details.get('side', 'N/A'),
                                }
                            }
                        }
                        try:
                            strat.on_order_update(error_response, symbol=symbol)
                        except Exception as callback_error:
                            bot_logger.error(f"Failed to notify strategy of error: {callback_error}", exc_info=True)
                            continue
                    # Now call on_order_update with the actual responses from OrderManager.
                    try:
                        strat.on_order_update(order_responses, symbol=symbol)
                    except Exception as callback_error:
                        bot_logger.error(f"Strategy callback error in {type(strat).__name__}.on_order_update: {callback_error}", exc_info=True)
                        continue  # move on to next strategy without crashing the bot

                # Check for open position and exit
                # Ensure strat_instance.position is a dict, as expected
                if not isinstance(strat_instance.position, dict):
                    strat_instance.position = {} # Initialize if not a dict to prevent errors

                current_position_details = strat_instance.position.get(symbol)
                if current_position_details and float(current_position_details.get('size', 0)) != 0:
                    exit_signal = strat.check_exit(symbol=symbol)
                    if exit_signal:
                        bot_logger.info(f"Exit signal received from {type(strat).__name__} for {symbol}: {exit_signal}")
                        try:
                            # Pass category to execute_strategy_exit
                            exit_order_response = order_manager.execute_strategy_exit(symbol, current_position_details, category=category)
                            bot_logger.info(f"Exit order response for {type(strat).__name__}: {exit_order_response}")
                            # Notify strategy of exit order update
                            try:
                                strat.on_order_update(exit_order_response, symbol=symbol)
                            except Exception as callback_error:
                                bot_logger.error(f"Strategy callback error in {type(strat).__name__}.on_order_update (exit): {callback_error}", exc_info=True)
                            
                            # Update performance tracker after successful exit
                            if exit_order_response and exit_order_response.get('exit_order', {}).get('result', {}).get('orderStatus', '').lower() == 'filled':
                                trade_summary = {
                                    'strategy': type(strat).__name__,
                                    'symbol': symbol,
                                    'entry_price': float(current_position_details.get('entry_price', 0)),
                                    'exit_price': float(exit_order_response['exit_order']['result'].get('avgPrice', 0)),
                                    'size': float(current_position_details.get('size', 0)),
                                    'side': current_position_details.get('side'),
                                    'pnl': float(exit_order_response.get('pnl', 0)), # Assuming OrderManager calculates this
                                    'timestamp': datetime.now(timezone.utc).isoformat()
                                }
                                perf_tracker.record_trade(trade_summary)
                                bot_logger.info(f"Trade recorded for {type(strat).__name__}: {trade_summary}")
                            
                            # Clear position from strategy after successful exit and recording
                            strat_instance.clear_position(symbol)
                            bot_logger.info(f"Position for {symbol} cleared from strategy {type(strat).__name__}.")

                        except OrderExecutionError as oe:
                            bot_logger.error(f"Exit order placement failed for {type(strat).__name__}: {oe}")
                            # Notify strategy of order failure with error response
                            error_response_exit = {
                                'exit_order': {
                                    'result': {
                                        'orderId': None,
                                        'orderStatus': 'rejected',
                                        'error': str(oe)
                                    }
                                }
                            }
                            try:
                                strat.on_order_update(error_response_exit, symbol=symbol)
                            except Exception as callback_error:
                                bot_logger.error(f"Failed to notify strategy of exit order error: {callback_error}", exc_info=True)
                        except Exception as e_exit:
                            bot_logger.error(f"Unexpected error during strategy exit for {type(strat).__name__}: {e_exit}", exc_info=True)
                            error_response_exit = {
                                'exit_order': {
                                    'result': {
                                        'orderId': None,
                                        'orderStatus': 'rejected',
                                        'error': f"Unexpected error: {str(e_exit)}"
                                    }
                                }
                            }
                            try:
                                strat.on_order_update(error_response_exit, symbol=symbol)
                            except Exception as callback_error:
                                bot_logger.error(f"Failed to notify strategy of unexpected exit order error: {callback_error}", exc_info=True)
            
            # Brief pause to prevent excessive CPU usage and API rate limit issues
            bot_logger.debug("Main loop iteration ended. Pausing...")
            time.sleep(0.1)
    except KeyboardInterrupt:
        bot_logger.info('Bot shutting down (KeyboardInterrupt).')
    except Exception as exc:
        bot_logger.error(f'Bot crashed: {exc}', exc_info=True) # Added exc_info=True
        PerformanceTracker.persist_on_exception(perf_tracker)
        raise 
    finally:
        if 'data_fetcher' in locals() and data_fetcher is not None:
            data_fetcher.stop_websocket()
        if 'perf_tracker' in locals() and perf_tracker is not None:
            perf_tracker.close_session()
        bot_logger.info('Bot session closed.')

if __name__ == '__main__':
    main() 