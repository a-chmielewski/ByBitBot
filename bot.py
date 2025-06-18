import os
import importlib
import json
import logging # Import logging for the helper function
from modules.logger import get_logger
from modules.exchange import ExchangeConnector
from modules.data_fetcher import LiveDataFetcher
from modules.order_manager import OrderManager, OrderExecutionError
from modules.performance_tracker import PerformanceTracker
from modules.market_analyzer import MarketAnalyzer, MarketAnalysisError
from modules.strategy_matrix import StrategyMatrix
from datetime import datetime, timezone
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re

# Import StrategyTemplate for type checking in dynamic_import_strategy
from strategies.strategy_template import StrategyTemplate

warnings.simplefilter(action='ignore', category=FutureWarning)

CONFIG_PATH = 'config.json'
STRATEGY_DIR = 'strategies'


def convert_strategy_class_to_module_name(strategy_class_name: str) -> str:
    """
    Convert a strategy class name to its corresponding module name.
    
    Args:
        strategy_class_name: Class name like 'StrategyAdaptiveTransitionalMomentum'
        
    Returns:
        Module name like 'adaptive_transitional_momentum_strategy'
    """
    # Special case mappings for strategies that don't follow the standard naming convention
    special_mappings = {
        'StrategyDoubleEMAStochOsc': 'double_EMA_StochOsc',  # Non-standard file name
        'StrategyEMAAdx': 'ema_adx_strategy',  # Handle EMA + ADX case properly
        'StrategyEMATrendRider': 'ema_adx_strategy',  # Class name doesn't match file name
        'StrategyRSIRangeScalping': 'rsi_range_scalping_strategy',  # rsirange_scalping_strategy -> rsi_range_scalping_strategy
        'StrategyATRMomentumBreakout': 'atr_momentum_breakout_strategy',  # atrmomentum_breakout_strategy -> atr_momentum_breakout_strategy
    }
    
    # Check for special mappings first
    if strategy_class_name in special_mappings:
        return special_mappings[strategy_class_name]
    
    # Standard conversion logic
    # Remove 'Strategy' prefix
    name_without_prefix = strategy_class_name.replace('Strategy', '')
    
    # Add underscores before capitals (while capitals still exist)
    snake_case_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name_without_prefix)
    
    # Convert to lowercase
    snake_case_name = snake_case_name.lower()
    
    # Add '_strategy' suffix  
    module_name = snake_case_name + '_strategy'
    
    return module_name


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def list_strategies():
    """
    List available strategies, filtering out template files and strategies marked as hidden.
    Returns a list of tuples: (strategy_module_name, strategy_class, market_type_tags)
    """
    files = [f for f in os.listdir(STRATEGY_DIR) if f.endswith('.py') and not f.startswith('__') and 'template' not in f]
    strategy_info = []
    
    for filename in files:
        module_name = os.path.splitext(filename)[0]
        try:
            # Dynamically import to check visibility and get market type tags
            strategy_class = dynamic_import_strategy(module_name, StrategyTemplate, get_logger('strategy_discovery'))
            
            # Check if strategy should be shown in selection
            if getattr(strategy_class, 'SHOW_IN_SELECTION', True):
                market_tags = getattr(strategy_class, 'MARKET_TYPE_TAGS', [])
                strategy_info.append((module_name, strategy_class, market_tags))
                
        except Exception as e:
            # If strategy fails to import, log but don't include it
            get_logger('strategy_discovery').warning(f"Failed to import strategy {module_name}: {e}")
            continue
    
    return strategy_info

def get_strategy_parameters(strategy_class_name: str) -> dict:
    """
    Map strategy class names to their trading parameters (category only).
    Symbol, timeframe, and leverage are now selected dynamically by the user.
    """
    strategy_params = {
        'StrategyDoubleEMAStochOsc': {
            'category': 'linear'
        },
        'StrategyBreakoutAndRetest': {
            'category': 'linear'
        },
        'StrategyEMAAdx': {
            'category': 'linear'
        }
    }
    
    return strategy_params.get(strategy_class_name, {
        'category': 'linear'
    })

def select_strategies(available: list[tuple], logger_instance: logging.Logger): # Updated to handle list of tuples
    """
    Display available strategies with market type tags and prompt user to select.
    Args:
        available: List of tuples (strategy_module_name, strategy_class, market_type_tags)
        logger_instance: Logger instance
    Returns:
        List of selected strategy module names
    """
    logger_instance.info('Available strategies for selection:')
    print("\nAvailable strategies:")
    print("=" * 60)
    
    for i, (module_name, strategy_class, market_tags) in enumerate(available):
        strategy_name = strategy_class.__name__
        tags_display = f"[{', '.join(market_tags)}]" if market_tags else "[NO TAGS]"
        display_line = f"  {i+1}. {strategy_name} {tags_display}"
        print(display_line)
        logger_instance.info(display_line)
    
    print("=" * 60)
    selected_input = input('Select strategies (comma-separated indices, e.g. 1,2): ')
    logger_instance.info(f"User input for strategy selection: '{selected_input}'")
    
    indices = []
    if selected_input.strip(): # Check if input is not empty
        try:
            indices = [int(i.strip())-1 for i in selected_input.split(',') if i.strip().isdigit() and 0 <= int(i.strip())-1 < len(available)]
        except ValueError:
            logger_instance.error("Invalid input for strategy selection (non-integer value). No strategies selected.")
            return [] # Return empty if there's a non-integer value that's not filtered by isdigit
            
    selected_names = [available[i][0] for i in indices]  # Extract module names from tuples
    logger_instance.info(f"Parsed selected indices: {indices}, Corresponding names: {selected_names}")
    return selected_names

def select_symbol(analysis_results: dict, logger_instance: logging.Logger) -> str:
    """
    Let user select a symbol from the analyzed symbols.
    
    Args:
        analysis_results: Market analysis results dictionary
        logger_instance: Logger instance
        
    Returns:
        Selected symbol string
    """
    symbols = list(analysis_results.keys())
    
    logger_instance.info('Available symbols from market analysis:')
    print("\nAvailable symbols:")
    print("=" * 70)
    
    for i, symbol in enumerate(symbols):
        # Get market types for this symbol
        symbol_data = analysis_results[symbol]
        market_types = []
        for timeframe, data in symbol_data.items():
            market_type = data.get('market_type', 'UNKNOWN')
            market_types.append(f"{timeframe}:{market_type}")
        
        market_summary = " | ".join(market_types)
        display_line = f"  {i+1}. {symbol:<15} [{market_summary}]"
        print(display_line)
        logger_instance.info(display_line)
    
    print("=" * 70)
    
    while True:
        selected_input = input('Select symbol (enter number): ').strip()
        logger_instance.info(f"User input for symbol selection: '{selected_input}'")
        
        try:
            index = int(selected_input) - 1
            if 0 <= index < len(symbols):
                selected_symbol = symbols[index]
                logger_instance.info(f"Selected symbol: {selected_symbol}")
                return selected_symbol
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(symbols)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

def select_timeframe(analysis_results: dict, selected_symbol: str, logger_instance: logging.Logger) -> str:
    """
    Let user select a timeframe for the chosen symbol.
    
    Args:
        analysis_results: Market analysis results dictionary
        selected_symbol: Previously selected symbol
        logger_instance: Logger instance
        
    Returns:
        Selected timeframe string
    """
    symbol_data = analysis_results[selected_symbol]
    timeframes = list(symbol_data.keys())
    
    logger_instance.info(f'Available timeframes for {selected_symbol}:')
    print(f"\nAvailable timeframes for {selected_symbol}:")
    print("=" * 50)
    
    for i, timeframe in enumerate(timeframes):
        data = symbol_data[timeframe]
        market_type = data.get('market_type', 'UNKNOWN')
        current_price = data.get('price_range', {}).get('current', 'N/A')
        display_line = f"  {i+1}. {timeframe:<5} [Market: {market_type}, Price: ${current_price}]"
        print(display_line)
        logger_instance.info(display_line)
    
    print("=" * 50)
    
    while True:
        selected_input = input('Select timeframe (enter number): ').strip()
        logger_instance.info(f"User input for timeframe selection: '{selected_input}'")
        
        try:
            index = int(selected_input) - 1
            if 0 <= index < len(timeframes):
                selected_timeframe = timeframes[index]
                logger_instance.info(f"Selected timeframe: {selected_timeframe}")
                return selected_timeframe
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(timeframes)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

def select_leverage(logger_instance: logging.Logger) -> int:
    """
    Let user select leverage for trading.
    
    Args:
        logger_instance: Logger instance
        
    Returns:
        Selected leverage as integer (1-50)
    """
    logger_instance.info('Leverage selection:')
    print("\n" + "="*60)
    print("LEVERAGE SELECTION")
    print("="*60)
    print("Choose your leverage multiplier (1-50):")
    print("  • Lower leverage (1-5): Conservative trading, lower risk")
    print("  • Medium leverage (10-25): Moderate risk/reward")
    print("  • Higher leverage (30-50): Aggressive trading, higher risk")
    print("  • Note: Higher leverage increases both potential gains and losses")
    print("="*60)
    
    while True:
        selected_input = input('Enter leverage (1-50): ').strip()
        logger_instance.info(f"User input for leverage selection: '{selected_input}'")
        
        try:
            leverage = int(selected_input)
            if 1 <= leverage <= 50:
                logger_instance.info(f"Selected leverage: {leverage}x")
                print(f"✅ Leverage set to {leverage}x")
                return leverage
            else:
                print("Invalid selection. Please enter a number between 1 and 50")
        except ValueError:
            print("Invalid input. Please enter a whole number.")

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

def run_market_analysis(exchange, config, logger):
    """
    Run market analysis for all configured symbols and timeframes.
    
    Args:
        exchange: ExchangeConnector instance
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Analysis results dictionary or None if analysis fails
    """
    try:
        logger.info("Starting market analysis...")
        
        # Initialize market analyzer
        market_analyzer = MarketAnalyzer(exchange, config, logger)
        
        # Run analysis for all symbols and timeframes
        analysis_results = market_analyzer.analyze_all_markets()
        
        # Get summary statistics
        summary = market_analyzer.get_market_summary(analysis_results)
        logger.info(f"Market analysis completed. Summary: {summary}")
        
        return analysis_results
        
    except MarketAnalysisError as e:
        logger.error(f"Market analysis failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during market analysis: {e}", exc_info=True)
        return None

def run_silent_market_analysis(exchange, config, symbol, timeframe, logger):
    """
    Run market analysis silently for a specific symbol and timeframe.
    Used for periodic checks without verbose output.
    
    Args:
        exchange: ExchangeConnector instance
        config: Configuration dictionary
        symbol: Symbol to analyze
        timeframe: Timeframe to analyze
        logger: Logger instance
        
    Returns:
        Market analysis results dictionary with structure: {symbol: {timeframe: {analysis_data}}}
        or None if analysis fails
    """
    try:
        logger.debug(f"Running silent market analysis for {symbol} {timeframe}")
        
        # Create a MarketAnalyzer with minimal initialization to avoid expensive symbol fetching
        # but still set up essential attributes needed for analysis
        market_analyzer = MarketAnalyzer.__new__(MarketAnalyzer)
        market_analyzer.exchange = exchange
        market_analyzer.config = config
        market_analyzer.logger = logger
        
        # Set up essential attributes without expensive operations
        market_config = config.get('market_analysis', {})
        market_analyzer.timeframes = market_config.get('timeframes', ['1m', '5m'])
        market_analyzer.use_dynamic_symbols = market_config.get('use_dynamic_symbols', False)
        market_analyzer.top_volume_count = market_config.get('top_volume_count', 10)
        market_analyzer.min_volume_usdt = market_config.get('min_volume_usdt', 1000000)
        
        # Skip the expensive symbol fetching/validation - we only need to analyze one symbol
        market_analyzer.symbols = [symbol]  # Set just the symbol we need
        
        # Analyze just the specific symbol/timeframe directly
        try:
            result = market_analyzer._analyze_symbol_timeframe(symbol, timeframe)
            market_type = result.get('market_type', 'UNKNOWN')
            
            # Log more details if analysis fails or returns UNKNOWN
            if market_type in ['UNKNOWN', 'INSUFFICIENT_DATA', 'ANALYSIS_FAILED']:
                analysis_details = result.get('analysis_details', {})
                data_points = result.get('data_points', 0)
                logger.warning(f"Silent analysis for {symbol} {timeframe} returned {market_type}. "
                             f"Data points: {data_points}, Details: {analysis_details}")
            else:
                logger.debug(f"Silent analysis result for {symbol} {timeframe}: {market_type}")
            
            # Return in the expected dictionary structure
            return {
                symbol: {
                    timeframe: result
                }
            }
        except Exception as e:
            logger.warning(f"Silent market analysis failed for {symbol} {timeframe}: {e}", exc_info=True)
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error during silent market analysis: {e}")
        return None

def check_strategy_market_compatibility(strategy_tags, current_market_type, symbol, timeframe, logger):
    """
    Check if current market type is compatible with strategy tags.
    
    Args:
        strategy_tags: List of market type tags from strategy
        current_market_type: Current market type from analysis
        symbol: Trading symbol
        timeframe: Trading timeframe
        logger: Logger instance
        
    Returns:
        bool: True if compatible, False if mismatch
    """
    if not strategy_tags or not current_market_type:
        # If no tags defined or analysis failed, assume compatible
        return True
    
    # Special handling for TEST tag - always compatible
    if 'TEST' in strategy_tags:
        return True
    
    # Check if current market type matches any strategy tag
    is_compatible = current_market_type in strategy_tags
    
    if is_compatible:
        logger.debug(f"Market compatibility check: {symbol} {timeframe} {current_market_type} matches strategy tags {strategy_tags}")
    else:
        logger.warning(f"Market compatibility mismatch: {symbol} {timeframe} is {current_market_type} but strategy expects {strategy_tags}")
    
    return is_compatible

def prompt_strategy_reselection(analysis_results, current_symbol, current_timeframe, current_market_type, available_strategies, logger):
    """
    Prompt user about market change and ask for strategy reselection.
    
    Args:
        analysis_results: Full market analysis results
        current_symbol: Currently trading symbol
        current_timeframe: Currently trading timeframe
        current_market_type: New market type detected
        available_strategies: Available strategy list
        logger: Logger instance
        
    Returns:
        tuple: (new_strategy_name, should_restart) or (None, False) to continue
    """
    print("\n" + "="*80)
    print("🚨 MARKET CONDITION CHANGE DETECTED 🚨")
    print("="*80)
    print(f"Trading pair: {current_symbol} {current_timeframe}")
    print(f"New market type: {current_market_type}")
    print(f"The market conditions have changed and may no longer match your current strategy.")
    print("\nFull market analysis:")
    
    # Print the analysis summary (reuse existing code from market_analyzer)
    try:
        if analysis_results:
            # Create temporary analyzer just to use the print function
            from modules.market_analyzer import MarketAnalyzer
            temp_analyzer = MarketAnalyzer.__new__(MarketAnalyzer)  # Create without calling __init__
            temp_analyzer.logger = logger
            temp_analyzer._print_analysis_summary(analysis_results)
    except Exception as e:
        logger.error(f"Error printing analysis summary: {e}")
        print("(Could not display full analysis)")
    
    print("\n" + "="*80)
    print("STRATEGY RESELECTION OPTIONS")
    print("="*80)
    print("1. Continue with current strategy (ignore market change)")
    print("2. Select a new strategy based on current market conditions")
    print("="*80)
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        logger.info(f"User choice for market change response: '{choice}'")
        
        if choice == "1":
            logger.info("User chose to continue with current strategy despite market change")
            return None, False
        elif choice == "2":
            logger.info("User chose to reselect strategy due to market change")
            # Let user select new strategy
            selected_strategy_names = select_strategies(available_strategies, logger)
            if selected_strategy_names:
                return selected_strategy_names[0], True  # Return first selected strategy
            else:
                print("No strategy selected. Continuing with current strategy.")
                return None, False
        else:
            print("Invalid choice. Please enter 1 or 2.")

def restart_configuration_with_new_strategy(new_strategy_name, available_strategies, analysis_results, exchange, config, bot_logger):
    """
    Restart the bot configuration process with a new strategy.
    This function handles the complete reconfiguration flow after strategy reselection.
    """
    bot_logger.info("="*60)
    bot_logger.info("RESTARTING CONFIGURATION WITH NEW STRATEGY")
    bot_logger.info("="*60)
    
    try:
        # Load the new strategy class
        bot_logger.info(f"Loading new strategy: {new_strategy_name}")
        try:
            StratClass = dynamic_import_strategy(new_strategy_name, StrategyTemplate, bot_logger)
            bot_logger.info(f"Successfully loaded strategy class: {StratClass.__name__}")
        except Exception as e:
            bot_logger.error(f"Failed to load new strategy {new_strategy_name}: {e}", exc_info=True)
            bot_logger.error("Aborting strategy change - bot will shut down")
            return
        
        # Let user select symbol and timeframe from analyzed markets
        bot_logger.info("="*60)
        bot_logger.info("SYMBOL AND TIMEFRAME SELECTION")
        bot_logger.info("="*60)
        
        selected_symbol = select_symbol(analysis_results, bot_logger)
        selected_timeframe = select_timeframe(analysis_results, selected_symbol, bot_logger)
        selected_leverage = select_leverage(bot_logger)
        
        # Get strategy parameters
        strategy_params = get_strategy_parameters(StratClass.__name__)
        
        # Set up trading parameters
        symbol = selected_symbol
        timeframe = selected_timeframe
        leverage = selected_leverage
        category = strategy_params['category']
        coin_pair = symbol.replace('USDT', '/USDT')
        
        bot_logger.info(f"New configuration: {coin_pair} ({symbol}), {timeframe}, {leverage}x leverage")
        
        # Set leverage on exchange
        try:
            bot_logger.info(f"Setting leverage to {leverage}x for {symbol}")
            exchange.set_leverage(symbol, leverage, category)
            bot_logger.info(f"✅ Successfully set leverage to {leverage}x for {symbol}")
        except Exception as e:
            bot_logger.error(f"Failed to set leverage to {leverage}x for {symbol}: {e}")
            bot_logger.error("Bot will continue, but orders may fail if current leverage is insufficient")
        
        # Initialize new data fetcher
        data_fetcher = LiveDataFetcher(exchange, symbol, timeframe, logger=bot_logger)
        data = data_fetcher.fetch_initial_data()
        data_fetcher.start_websocket()
        bot_logger.info(f"Fetched initial OHLCV data: {len(data)} rows for {symbol} {timeframe}")
        
        # Initialize new order manager and performance tracker
        order_manager = OrderManager(exchange, logger=bot_logger)
        perf_tracker = PerformanceTracker(logger=bot_logger)
        
        # Initialize the new strategy instance
        try:
            strategy_specific_logger = get_logger(new_strategy_name)
            strategy_instance = StratClass(data.copy(), config, logger=strategy_specific_logger)
            bot_logger.info(f"Successfully initialized new strategy: {type(strategy_instance).__name__}")
        except Exception as e:
            bot_logger.error(f"Failed to initialize new strategy {StratClass.__name__}: {e}", exc_info=True)
            if 'data_fetcher' in locals() and data_fetcher is not None:
                data_fetcher.stop_websocket()
            return
        
        # Log initial state
        strategy_instance.log_state_change(symbol, "awaiting_entry", f"Strategy {type(strategy_instance).__name__} for {symbol}: Initialized after strategy change. Looking for new entry conditions...")
        
        bot_logger.info("="*60)
        bot_logger.info("RESUMING TRADING WITH NEW CONFIGURATION")
        bot_logger.info("="*60)
        
        # Start new trading loop with the reconfigured parameters
        run_trading_loop(
            strategy_instance, symbol, timeframe, leverage, category,
            data_fetcher, order_manager, perf_tracker, exchange, config, bot_logger
        )
        
    except Exception as e:
        bot_logger.error(f"Error during strategy reconfiguration: {e}", exc_info=True)
        bot_logger.error("Strategy change failed - bot will shut down")

def run_trading_loop(strategy_instance, symbol, timeframe, leverage, category, data_fetcher, order_manager, perf_tracker, exchange, config, bot_logger):
    """
    Main trading loop that can be reused for strategy changes.
    """
    bot_logger.info("Entering main trading loop.")
    
    # Get available strategies for potential reselection
    available_strategies = list_strategies()
    
    # Get strategy tags for compatibility checking
    primary_strategy_tags = getattr(type(strategy_instance), 'MARKET_TYPE_TAGS', [])
    
    # Initialize timing for periodic market analysis
    last_market_check = datetime.now()
    market_check_interval = timedelta(minutes=60)  # Check every 60 minutes
    
    bot_logger.info(f"Periodic market analysis will run every {market_check_interval.total_seconds()/60:.0f} minutes")
    bot_logger.info(f"Current strategy tags: {primary_strategy_tags}")
    
    # Wrap strategy in list for compatibility with existing loop logic
    strategies = [strategy_instance]
    
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
            strat_instance = strat  # For consistency with existing variable names
            bot_logger.debug(f"Processing strategy {type(strat).__name__}")
            
            # Update strategy data efficiently - preserve indicators while adding new OHLCV rows
            if strat.data is not None and not strat.data.empty:
                # Strategy already has data - check if there are new rows to add
                if len(data) > len(strat.data):
                    # Get new rows that need to be added
                    new_rows = data.iloc[len(strat.data):]
                    
                    # Append new OHLCV rows to existing strategy data (preserving indicators)
                    if not new_rows.empty:
                        # Create empty indicator columns for new rows to match existing structure
                        new_rows_with_indicators = new_rows.copy()
                        indicator_cols = [col for col in strat.data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
                        for col in indicator_cols:
                            new_rows_with_indicators[col] = np.nan
                        
                        # Append the new rows
                        strat.data = pd.concat([strat.data, new_rows_with_indicators], ignore_index=False)
                        bot_logger.debug(f"Added {len(new_rows)} new rows to {type(strat).__name__} data, now has {len(strat.data)} rows")
                else:
                    # No new data - keep existing strategy data with indicators intact
                    bot_logger.debug(f"{type(strat).__name__} data unchanged, {len(strat.data)} rows with indicators preserved")
            else:
                # First time initialization - use fresh copy
                strat.data = data.copy()
                bot_logger.debug(f"Initialized {type(strat).__name__} data with {len(strat.data)} rows")
            
            # Use efficient indicator update if available, otherwise fall back to full init
            if hasattr(strat, 'update_indicators_for_new_row') and len(strat.data) > 1:
                strat.update_indicators_for_new_row()
            else:
                strat.init_indicators()

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
        
        # Periodic market analysis check (every 60 minutes)
        current_time = datetime.now()
        if current_time - last_market_check >= market_check_interval:
            bot_logger.info("="*60)
            bot_logger.info("RUNNING PERIODIC MARKET ANALYSIS CHECK")
            bot_logger.info("="*60)
            
            # Run silent market analysis for current symbol/timeframe
            current_market_type = run_silent_market_analysis(exchange, config, symbol, timeframe, bot_logger)
            
            if current_market_type:
                bot_logger.info(f"Current market type for {symbol} {timeframe}: {current_market_type}")
                
                # Check if market type still matches strategy
                is_compatible = check_strategy_market_compatibility(
                    primary_strategy_tags, current_market_type, symbol, timeframe, bot_logger
                )
                
                if not is_compatible:
                    bot_logger.warning(f"Market type mismatch detected! Strategy expects {primary_strategy_tags}, but market is {current_market_type}")
                    
                    # Run full market analysis for user display
                    full_analysis = run_market_analysis(exchange, config, bot_logger)
                    
                    # Prompt user for strategy reselection
                    new_strategy_name, should_restart = prompt_strategy_reselection(
                        full_analysis, symbol, timeframe, current_market_type, available_strategies, bot_logger
                    )
                    
                    if should_restart and new_strategy_name:
                        bot_logger.info(f"User selected new strategy: {new_strategy_name}. Restarting configuration...")
                        # Clean up current resources
                        if 'data_fetcher' in locals() and data_fetcher is not None:
                            data_fetcher.stop_websocket()
                        if 'perf_tracker' in locals() and perf_tracker is not None:
                            perf_tracker.close_session()
                        
                        # Instead of shutting down, restart the configuration process
                        # This will trigger a new configuration flow with the selected strategy
                        restart_configuration_with_new_strategy(
                            new_strategy_name, available_strategies, full_analysis, 
                            exchange, config, bot_logger
                        )
                        return  # Exit current main loop, new one will start
                    else:
                        bot_logger.info("Continuing with current strategy despite market change")
                else:
                    bot_logger.info(f"Market type compatibility confirmed: {current_market_type} matches strategy tags {primary_strategy_tags}")
            else:
                bot_logger.warning("Silent market analysis failed, skipping compatibility check")
            
            # Update last check time
            last_market_check = current_time
            bot_logger.info("="*60)
            bot_logger.info("PERIODIC MARKET ANALYSIS CHECK COMPLETED")
            bot_logger.info("="*60)
        
        # Brief pause to prevent excessive CPU usage and API rate limit issues
        bot_logger.debug("Main loop iteration ended. Pausing...")
        time.sleep(0.1)

def automatic_strategy_and_timeframe_selection(analysis_results: dict, selected_symbol: str, logger_instance: logging.Logger) -> tuple:
    """
    Automatically select the optimal strategy and execution timeframe based on market conditions using the Strategy Matrix.
    
    Args:
        analysis_results: Market analysis results dictionary
        selected_symbol: The selected trading symbol
        logger_instance: Logger instance
        
    Returns:
        tuple: (strategy_class_name, execution_timeframe, strategy_description, selection_reason)
    """
    logger_instance.info("="*60)
    logger_instance.info("AUTOMATIC STRATEGY AND TIMEFRAME SELECTION")
    logger_instance.info("="*60)
    
    # Initialize Strategy Matrix
    strategy_matrix = StrategyMatrix(logger_instance)
    
    # Get market conditions for the selected symbol
    symbol_analysis = analysis_results.get(selected_symbol, {})
    
    market_5min = symbol_analysis.get('5m', {}).get('market_type', 'UNKNOWN')
    market_1min = symbol_analysis.get('1m', {}).get('market_type', 'UNKNOWN')
    
    logger_instance.info(f"Market conditions for {selected_symbol}:")
    logger_instance.info(f"  5-minute timeframe: {market_5min}")
    logger_instance.info(f"  1-minute timeframe: {market_1min}")
    
    # Validate market conditions
    if not strategy_matrix.validate_market_conditions(market_5min, market_1min):
        logger_instance.error("Invalid market conditions detected. Cannot select strategy automatically.")
        return None, None, None, "Invalid market conditions"
    
    # Select strategy and timeframe using the matrix
    selected_strategy_class, execution_timeframe, selection_reason = strategy_matrix.select_strategy_and_timeframe(market_5min, market_1min)
    strategy_description = strategy_matrix.get_strategy_description(selected_strategy_class)
    
    logger_instance.info(f"Strategy Matrix Selection:")
    logger_instance.info(f"  Selected Strategy: {selected_strategy_class}")
    logger_instance.info(f"  Execution Timeframe: {execution_timeframe}")
    logger_instance.info(f"  Description: {strategy_description}")
    logger_instance.info(f"  Reason: {selection_reason}")
    
    # Display matrix summary for reference
    matrix_summary = strategy_matrix.get_matrix_summary()
    logger_instance.debug(f"Strategy Matrix:\n{matrix_summary}")
    
    logger_instance.info("="*60)
    
    return selected_strategy_class, execution_timeframe, strategy_description, selection_reason

def check_strategy_needs_change(analysis_results: dict, selected_symbol: str, current_strategy_class: str, current_timeframe: str, logger_instance: logging.Logger) -> tuple:
    """
    Check if the current strategy and timeframe are still optimal for current market conditions.
    
    Args:
        analysis_results: Current market analysis results
        selected_symbol: The trading symbol
        current_strategy_class: Current strategy class name
        current_timeframe: Current execution timeframe
        logger_instance: Logger instance
        
    Returns:
        tuple: (needs_change: bool, new_strategy_class: str, new_timeframe: str, reason: str)
    """
    logger_instance.info("Checking if strategy/timeframe change is needed based on current market conditions...")
    
    # Initialize Strategy Matrix
    strategy_matrix = StrategyMatrix(logger_instance)
    
    # Get current market conditions
    symbol_analysis = analysis_results.get(selected_symbol, {})
    market_5min = symbol_analysis.get('5m', {}).get('market_type', 'UNKNOWN')
    market_1min = symbol_analysis.get('1m', {}).get('market_type', 'UNKNOWN')
    
    # Get optimal strategy and timeframe for current conditions
    optimal_strategy_class, optimal_timeframe, selection_reason = strategy_matrix.select_strategy_and_timeframe(market_5min, market_1min)
    
    if optimal_strategy_class != current_strategy_class or optimal_timeframe != current_timeframe:
        reason = f"Market conditions changed. Optimal setup is now {optimal_strategy_class} on {optimal_timeframe} instead of {current_strategy_class} on {current_timeframe}. {selection_reason}"
        logger_instance.warning(reason)
        return True, optimal_strategy_class, optimal_timeframe, reason
    else:
        reason = f"Current setup {current_strategy_class} on {current_timeframe} is still optimal for market conditions {market_5min}(5m) + {market_1min}(1m)"
        logger_instance.info(reason)
        return False, current_strategy_class, current_timeframe, reason

def run_trading_loop_with_auto_strategy(strategy_instance, current_strategy_class, symbol, timeframe, leverage, category, data_fetcher, order_manager, perf_tracker, exchange, config, analysis_results, bot_logger):
    """
    Main trading loop with automatic strategy switching based on market conditions.
    
    Args:
        strategy_instance: Current strategy instance
        current_strategy_class: Current strategy class name
        symbol: Trading symbol
        timeframe: Trading timeframe
        leverage: Trading leverage
        category: Trading category
        data_fetcher: Data fetcher instance
        order_manager: Order manager instance
        perf_tracker: Performance tracker instance
        exchange: Exchange connector instance
        config: Configuration dictionary
        analysis_results: Initial market analysis results
        bot_logger: Logger instance
    """
    bot_logger.info("="*60)
    bot_logger.info("STARTING TRADING LOOP WITH AUTOMATIC STRATEGY MANAGEMENT")
    bot_logger.info("="*60)
    
    # Initialize timing for strategy checks
    last_strategy_check = time.time()
    strategy_check_interval = 15 * 60  # 15 minutes in seconds
    
    current_strategy = strategy_instance
    current_strategy_name = current_strategy_class
    
    bot_logger.info(f"Initial strategy: {current_strategy_name}")
    bot_logger.info(f"Strategy check interval: {strategy_check_interval / 60:.0f} minutes")
    
    while True:
        try:
            current_time = time.time()
            
            # Update data using the LiveDataFetcher interface
            try:
                latest_data = data_fetcher.update_data()
                
                # Check if strategy data has indicators (avoid overwriting them)
                if current_strategy.data is not None and not current_strategy.data.empty:
                    # Get indicator columns that exist in strategy data but not in raw OHLCV
                    base_ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    indicator_cols = [col for col in current_strategy.data.columns if col not in base_ohlcv_cols]
                    
                    if indicator_cols:
                        # Strategy data has indicators - merge carefully to preserve them
                        bot_logger.debug(f"Preserving {len(indicator_cols)} indicator columns during data update")
                        
                        # Update OHLCV columns with latest data, keeping indicators
                        for col in base_ohlcv_cols:
                            if col in latest_data.columns:
                                current_strategy.data[col] = latest_data[col].copy()
                        
                        # Ensure data length matches (trim if necessary)
                        if len(current_strategy.data) != len(latest_data):
                            current_strategy.data = current_strategy.data.iloc[-len(latest_data):].reset_index(drop=True)
                        
                        bot_logger.debug(f"Updated strategy OHLCV data while preserving indicators. Rows: {len(current_strategy.data)}")
                    else:
                        # No indicators yet - safe to replace entirely
                        current_strategy.data = latest_data.copy()
                        bot_logger.debug(f"Updated strategy data (no indicators to preserve). Rows: {len(latest_data)}")
                else:
                    # No existing data - safe to replace entirely
                    current_strategy.data = latest_data.copy()
                    bot_logger.debug(f"Initialized strategy data. Rows: {len(latest_data)}")
                
                # Now update indicators for the new row
                current_strategy.update_indicators_for_new_row()
                
            except Exception as e:
                bot_logger.debug(f"Data update failed: {e}")
                # Continue with existing data
            
            # Check for strategy change every 15 minutes (only when no active orders)
            if current_time - last_strategy_check >= strategy_check_interval:
                bot_logger.info("="*60)
                bot_logger.info("PERIODIC STRATEGY EVALUATION CHECK")
                bot_logger.info("="*60)
                
                # Check if there are any active orders
                has_active_orders = False
                if hasattr(current_strategy, 'position') and current_strategy.position.get(symbol):
                    has_active_orders = True
                    bot_logger.info(f"Active position exists for {symbol}. Skipping strategy evaluation until position closes.")
                elif hasattr(current_strategy, 'order_pending') and current_strategy.order_pending.get(symbol, False):
                    has_active_orders = True
                    bot_logger.info(f"Order pending for {symbol}. Skipping strategy evaluation until order completes.")
                
                if not has_active_orders:
                    bot_logger.info("No active orders detected. Proceeding with strategy evaluation...")
                    
                    # Run silent market analysis to get current conditions for both timeframes
                    try:
                        # Strategy matrix needs both 1m and 5m analysis, so fetch both
                        analysis_1m = run_silent_market_analysis(exchange, config, symbol, '1m', bot_logger)
                        analysis_5m = run_silent_market_analysis(exchange, config, symbol, '5m', bot_logger)
                        
                        # Combine the results
                        current_analysis = {}
                        if analysis_1m and symbol in analysis_1m:
                            current_analysis[symbol] = analysis_1m[symbol]
                        if analysis_5m and symbol in analysis_5m:
                            if symbol not in current_analysis:
                                current_analysis[symbol] = {}
                            current_analysis[symbol].update(analysis_5m[symbol])
                        
                        if current_analysis and symbol in current_analysis:
                            # Check if strategy/timeframe needs to change
                            needs_change, new_strategy_class, new_timeframe, reason = check_strategy_needs_change(
                                current_analysis, symbol, current_strategy_name, timeframe, bot_logger
                            )
                            
                            if needs_change:
                                bot_logger.warning(f"Strategy/timeframe change needed: {reason}")
                                
                                try:
                                    # Check if timeframe changed - if so, need to restart data fetcher
                                    timeframe_changed = new_timeframe != timeframe
                                    
                                    if timeframe_changed:
                                        bot_logger.info(f"Timeframe changing from {timeframe} to {new_timeframe}. Restarting data fetcher...")
                                        # Stop current data fetcher
                                        data_fetcher.stop_websocket()
                                        # Create new data fetcher with new timeframe
                                        data_fetcher = LiveDataFetcher(exchange, symbol, new_timeframe, logger=bot_logger)
                                        latest_data = data_fetcher.fetch_initial_data()
                                        data_fetcher.start_websocket()
                                        bot_logger.info(f"Data fetcher restarted with {new_timeframe} timeframe")
                                        # Update timeframe variable
                                        timeframe = new_timeframe
                                    else:
                                        latest_data = data_fetcher.get_data()
                                    
                                    # Load new strategy
                                    new_strategy_module = convert_strategy_class_to_module_name(new_strategy_class)
                                    
                                    bot_logger.info(f"Loading new strategy module: {new_strategy_module}")
                                    NewStratClass = dynamic_import_strategy(new_strategy_module, StrategyTemplate, bot_logger)
                                    
                                    # Create new strategy instance
                                    new_strategy_logger = get_logger(new_strategy_class.lower())
                                    new_strategy_instance = NewStratClass(latest_data.copy(), config, logger=new_strategy_logger)
                                    
                                    bot_logger.info(f"Successfully switched from {current_strategy_name} to {new_strategy_class} on {new_timeframe}")
                                    
                                    # Update current strategy references
                                    current_strategy = new_strategy_instance
                                    current_strategy_name = new_strategy_class
                                    
                                    # Log strategy change
                                    current_strategy.log_state_change(symbol, "awaiting_entry", 
                                        f"Strategy {new_strategy_class} on {new_timeframe} for {symbol}: Switched due to market condition change. Looking for entry conditions...")
                                    
                                except Exception as e:
                                    bot_logger.error(f"Failed to switch to new strategy {new_strategy_class} on {new_timeframe}: {e}")
                                    bot_logger.info("Continuing with current strategy and timeframe")
                            else:
                                bot_logger.info("Current strategy and timeframe remain optimal for market conditions")
                        else:
                            bot_logger.warning("Failed to get current market analysis for strategy evaluation")
                    except Exception as e:
                        bot_logger.error(f"Error during strategy evaluation: {e}")
                
                # Update last check time regardless of whether we could evaluate
                last_strategy_check = current_time
                bot_logger.info("="*60)
                bot_logger.info("STRATEGY EVALUATION CHECK COMPLETED")
                bot_logger.info("="*60)
            
            # Regular trading logic - check for entry
            if not hasattr(current_strategy, 'position') or not current_strategy.position.get(symbol):
                entry_signal = current_strategy.check_entry(symbol)
                if entry_signal:
                    bot_logger.info(f"Entry signal detected by {current_strategy_name}: {entry_signal}")
                    try:
                        # Extract parameters from entry_signal for place_order_with_risk method
                        side = entry_signal.get('side')
                        order_type = entry_signal.get('type', 'market')  # Default to market if not specified
                        size = entry_signal.get('size')
                        signal_price = entry_signal.get('price')
                        sl_pct = entry_signal.get('sl_pct')
                        tp_pct = entry_signal.get('tp_pct')
                        
                        # Create params dict for additional parameters
                        params = {}
                        for key, value in entry_signal.items():
                            if key not in ['side', 'type', 'size', 'price', 'sl_pct', 'tp_pct']:
                                params[key] = value
                        
                        order_responses = order_manager.place_order_with_risk(
                            symbol=symbol,
                            side=side,
                            order_type=order_type,
                            size=size,
                            signal_price=signal_price,
                            sl_pct=sl_pct,
                            tp_pct=tp_pct,
                            params=params if params else None
                        )
                        current_strategy.on_order_update(order_responses, symbol)
                        bot_logger.info(f"Orders placed successfully for {symbol}")
                    except OrderExecutionError as e:
                        bot_logger.error(f"Order execution failed: {e}")
                        current_strategy.order_pending[symbol] = False  # Reset pending state
                        current_strategy.active_order_id[symbol] = None
            
            # Check for exit if position exists
            if hasattr(current_strategy, 'position') and current_strategy.position.get(symbol):
                exit_signal = current_strategy.check_exit(symbol)
                if exit_signal:
                    bot_logger.info(f"Exit signal detected by {current_strategy_name}: {exit_signal}")
                    try:
                        # Get the position details from the strategy
                        position_to_close = current_strategy.position.get(symbol)
                        if position_to_close:
                            exit_order_responses = order_manager.execute_strategy_exit(symbol, position_to_close)
                            # Update strategy on successful exit
                            if exit_order_responses:
                                current_strategy.clear_position(symbol)
                                bot_logger.info(f"Position closed for {symbol}")
                        else:
                            bot_logger.error(f"Position data not found for {symbol} during exit")
                    except OrderExecutionError as e:
                        bot_logger.error(f"Exit order execution failed: {e}")
            
            # Brief pause to prevent excessive CPU usage
            time.sleep(0.1)
            
        except KeyboardInterrupt:
            bot_logger.info("Trading loop interrupted by user")
            break
        except Exception as e:
            bot_logger.error(f"Error in trading loop: {e}", exc_info=True)
            time.sleep(1)  # Pause before retrying

def main():
    config = load_config()
    # Initialize the main bot logger
    bot_logger = get_logger('bot') # Renamed to bot_logger for clarity
    bot_logger.info('Bot starting up.')
    
    # Exchange
    ex_cfg = config['bybit']
    exchange = ExchangeConnector(api_key=ex_cfg['api_key'], api_secret=ex_cfg['api_secret'], testnet=False, logger=bot_logger)
    
    # RUN MARKET ANALYSIS FIRST - before strategy selection
    bot_logger.info("="*60)
    bot_logger.info("RUNNING STARTUP MARKET ANALYSIS")
    bot_logger.info("="*60)
    
    analysis_results = run_market_analysis(exchange, config, bot_logger)
    
    if analysis_results:
        bot_logger.info("Market analysis completed successfully")
    else:
        bot_logger.warning("Market analysis failed or returned no results")
    
    bot_logger.info("="*60)
    bot_logger.info("CONTINUING WITH STRATEGY SETUP")
    bot_logger.info("="*60)
    
    # User selects symbol and timeframe from analyzed markets (NO MANUAL STRATEGY SELECTION)
    if not analysis_results:
        bot_logger.error("No market analysis results available for symbol selection. Bot will exit.")
        return
    
    bot_logger.info("="*60)
    bot_logger.info("SYMBOL AND LEVERAGE SELECTION")
    bot_logger.info("="*60)
    
    # Let user select symbol from analyzed markets
    selected_symbol = select_symbol(analysis_results, bot_logger)
    
    # Let user select leverage
    selected_leverage = select_leverage(bot_logger)
    
    # AUTOMATIC STRATEGY AND TIMEFRAME SELECTION based on market conditions
    selected_strategy_class, selected_timeframe, strategy_description, selection_reason = automatic_strategy_and_timeframe_selection(
        analysis_results, selected_symbol, bot_logger
    )
    
    if not selected_strategy_class or not selected_timeframe:
        bot_logger.error("Automatic strategy and timeframe selection failed. Bot will exit.")
        return
    
    bot_logger.info(f"Automatically selected strategy: {selected_strategy_class}")
    bot_logger.info(f"Automatically selected timeframe: {selected_timeframe}")
    bot_logger.info(f"Strategy description: {strategy_description}")
    
    # Load the selected strategy class
    try:
        # Convert class name to module name using helper function
        strategy_module_name = convert_strategy_class_to_module_name(selected_strategy_class)
        
        bot_logger.info(f"Attempting to load strategy module: {strategy_module_name}")
        StratClass = dynamic_import_strategy(strategy_module_name, StrategyTemplate, bot_logger)
        bot_logger.info(f"Successfully loaded strategy class: {StratClass.__name__}")
    except Exception as e:
        bot_logger.error(f"Failed to load automatically selected strategy {selected_strategy_class}: {e}")
        bot_logger.error("Falling back to StrategyBreakoutAndRetest with 1m timeframe as a working strategy")
        try:
            StratClass = dynamic_import_strategy('breakout_and_retest_strategy', StrategyTemplate, bot_logger)
            selected_strategy_class = 'StrategyBreakoutAndRetest'
            selected_timeframe = '1m'  # Use 1m as fallback timeframe
            bot_logger.info(f"Successfully loaded fallback strategy: {StratClass.__name__} on {selected_timeframe}")
        except Exception as fallback_error:
            bot_logger.error(f"Even fallback strategy failed to load: {fallback_error}")
            return
    
    # Get strategy-specific parameters (category only, leverage now user-selected)
    strategy_params = get_strategy_parameters(StratClass.__name__)
    
    # Use user selections and automatically selected strategy/timeframe
    symbol = selected_symbol
    timeframe = selected_timeframe  # Automatically selected timeframe
    leverage = selected_leverage  # Use user-selected leverage
    category = strategy_params['category']
    coin_pair = symbol.replace('USDT', '/USDT')  # Convert format for display
    
    bot_logger.info(f"Final trading parameters: {coin_pair} ({symbol}), {timeframe}, {leverage}x leverage")
    bot_logger.info(f"Selected strategy: {selected_strategy_class}")
    bot_logger.info(f"Selected timeframe: {timeframe}")
    
    # Set leverage on the exchange for the selected symbol
    try:
        bot_logger.info(f"Setting leverage to {leverage}x for {symbol}")
        exchange.set_leverage(symbol, leverage, category)
        bot_logger.info(f"✅ Successfully set leverage to {leverage}x for {symbol}")
    except Exception as e:
        bot_logger.error(f"Failed to set leverage to {leverage}x for {symbol}: {e}")
        bot_logger.error("Bot will continue, but orders may fail if current leverage is insufficient")
    
    bot_logger.info("="*60)
    
    # Data fetcher with strategy-determined parameters
    data_fetcher = LiveDataFetcher(exchange, symbol, timeframe, logger=bot_logger)
    data = data_fetcher.fetch_initial_data()
    # Start WebSocket for live data
    data_fetcher.start_websocket()
    bot_logger.info(f"Fetched initial OHLCV data: {len(data)} rows for {symbol} {timeframe}")
    
    # Order manager
    order_manager = OrderManager(exchange, logger=bot_logger)
    # Performance tracker
    perf_tracker = PerformanceTracker(logger=bot_logger)

    # Initialize the selected strategy instance
    try:
        # Create strategy-specific logger
        strategy_logger = get_logger(selected_strategy_class.lower())
        strategy_instance = StratClass(data.copy(), config, logger=strategy_logger)
        bot_logger.info(f"Successfully initialized strategy: {type(strategy_instance).__name__}")
    except Exception as e:
        bot_logger.error(f"Failed to initialize strategy class {StratClass.__name__}: {e}", exc_info=True)
        if 'data_fetcher' in locals() and data_fetcher is not None:
            data_fetcher.stop_websocket()
        bot_logger.info('Bot session closed due to strategy initialization failure.')
        return

    # Initial state logging
    strategy_instance.log_state_change(symbol, "awaiting_entry", f"Strategy {type(strategy_instance).__name__} for {symbol}: Initialized. Looking for new entry conditions...")

    # Main trading loop with automatic strategy switching
    try:
        # Start trading loop with automatic strategy management
        run_trading_loop_with_auto_strategy(
            strategy_instance, selected_strategy_class, symbol, timeframe, leverage, category,
            data_fetcher, order_manager, perf_tracker, exchange, config, analysis_results, bot_logger
        )
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