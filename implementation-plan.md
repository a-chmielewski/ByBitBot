### 5. Exchange Module

Done
- Fixed HTTP client instantiation in exchange.py to use correct keyword arguments for pybit.unified_trading.HTTP. This resolves the authentication TypeError and enables proper ByBit API connection. 

## Logging State Visibility
Done
Added stateful, non-flooding logging to all strategies and the template. Logs now reflect only meaningful state changes (waiting, entry, in position, exit) without flooding, improving clarity and operational transparency. 

### 3. Strategy Management

Done
- Ensured all strategies return required keys ('side', 'size', 'price') in entry signals.
- Added robust assertion checks and docstring clarifications to prevent missing key errors. 

### 2. Configuration Management

Done
- Created config.json for API keys and default parameters.
- Created strategies/ and modules/ directories with __init__.py files.
- Added strategy_template.py and example_strategy.py as the base and example for all strategies.
- Symbol is now normalized to ByBit format (no slash, uppercase) immediately after loading from config.json, ensuring compatibility across all modules. 

### 6. Order Management

Done
- OrderManager now enforces Bybit minimum order size for all trades, using get_min_order_amount from ExchangeConnector.
- Strategies can specify a larger size, but never below the exchange minimum. 