import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timezone, timedelta
from .data_fetcher import LiveDataFetcher, DataFetchError

# Import risk utilities for advanced calculations
try:
    from .risk_utilities import compute_atr, volatility_regime
except ImportError:
    # Fallback if risk_utilities not available
    def compute_atr(df, period=14):
        # Simple ATR fallback calculation
        high_low = df['high'] - df['low']
        high_close_prev = abs(df['high'] - df['close'].shift(1))
        low_close_prev = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=period).mean()
    
    def volatility_regime(atr_pct, thresholds=(0.5, 1.5)):
        if atr_pct < thresholds[0]:
            return "low"
        elif atr_pct > thresholds[1]:
            return "high"
        else:
            return "normal"

class MarketAnalysisError(Exception):
    """Custom exception for market analysis errors."""
    pass

class MarketAnalyzer:
    """
    Analyzes market conditions for multiple symbols and timeframes.
    Fetches OHLCV data and determines market type for each symbol/timeframe combination.
    """
    
    def __init__(self, exchange, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize the market analyzer.
        
        Args:
            exchange: ExchangeConnector instance
            config: Configuration dictionary containing market_analysis section
            logger: Logger instance
        """
        self.exchange = exchange
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Get market analysis configuration
        market_config = config.get('market_analysis', {})
        self.timeframes = market_config.get('timeframes', ['1m', '5m'])
        
        # Determine symbol source (static list or dynamic top volume)
        self.use_dynamic_symbols = market_config.get('use_dynamic_symbols', False)
        self.top_volume_count = market_config.get('top_volume_count', 10)
        self.min_volume_usdt = market_config.get('min_volume_usdt', 1000000)  # 1M USDT minimum
        
        # Enhanced caching and performance settings
        self.cache_validity_seconds = market_config.get('analysis_cache_seconds', 300)  # 5 minutes default
        self.max_computation_time_seconds = market_config.get('max_computation_time_seconds', 30)  # 30 seconds max
        self.atr_period = market_config.get('atr_period', 14)  # ATR calculation period
        self.vol_regime_thresholds = market_config.get('volatility_thresholds', (0.5, 1.5))  # Low/high volatility thresholds
        
        # Initialize caching system
        self._cache_lock = threading.Lock()
        self._atr_cache: Dict[str, Dict] = {}  # symbol -> {value, timestamp, timeframe}
        self._vol_regime_cache: Dict[str, Dict] = {}  # symbol -> {regime, timestamp, atr_pct}
        self._market_regime_cache: Dict[str, Dict] = {}  # symbol -> {regime, timestamp, components}
        self._data_cache: Dict[str, Dict] = {}  # symbol -> {data, timestamp, timeframe}
        
        if self.use_dynamic_symbols:
            self.logger.info(f"Using dynamic symbol selection: top {self.top_volume_count} by volume")
            self.symbols = self._fetch_top_volume_symbols()
        else:
            self.symbols = market_config.get('symbols', [])
            self.logger.info(f"Using static symbol list from config")
        
        if not self.symbols:
            raise MarketAnalysisError("No symbols available for market analysis")
            
        self.logger.info(f"Initialized MarketAnalyzer for {len(self.symbols)} symbols and {len(self.timeframes)} timeframes")
        self.logger.info(f"Enhanced features: ATR period={self.atr_period}, Cache validity={self.cache_validity_seconds}s, Max computation time={self.max_computation_time_seconds}s")

    def _fetch_top_volume_symbols(self) -> List[str]:
        """
        Fetch the top volume symbols from the exchange.
        
        Returns:
            List of symbol names
        """
        try:
            self.logger.info(f"Fetching top {self.top_volume_count} symbols by 24h volume (min ${self.min_volume_usdt:,.0f})")
            
            top_symbols = self.exchange.get_top_volume_symbols(
                category='linear',
                count=self.top_volume_count,
                min_volume_usdt=self.min_volume_usdt
            )
            
            if not top_symbols:
                self.logger.warning("No symbols returned from dynamic volume search, falling back to default list")
                # Fallback to a basic list if dynamic fetch fails
                return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']
            
            self.logger.info(f"Dynamic symbol selection successful: {top_symbols}")
            return top_symbols
            
        except Exception as e:
            self.logger.error(f"Failed to fetch top volume symbols: {e}")
            self.logger.warning("Falling back to default symbol list")
            return ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', 'XRPUSDT']

    def _validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and is tradeable on ByBit.
        
        Args:
            symbol: Trading symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            # Try to get instrument info for the symbol
            response = self.exchange._api_call_with_backoff(
                self.exchange.client.get_instruments_info,
                category='linear',
                symbol=symbol
            )
            
            if response and 'result' in response and 'list' in response['result']:
                instruments = response['result']['list']
                if instruments and len(instruments) > 0:
                    # Check if symbol is active
                    instrument = instruments[0]
                    status = instrument.get('status', '')
                    if status.lower() == 'trading':
                        self.logger.debug(f"Symbol {symbol} validated successfully")
                        return True
                    else:
                        self.logger.warning(f"Symbol {symbol} exists but status is: {status}")
                        return False
                        
            self.logger.warning(f"Symbol {symbol} not found in instruments list")
            return False
            
        except Exception as e:
            self.logger.warning(f"Could not validate symbol {symbol}: {e}")
            return False

    def analyze_all_markets(self) -> Dict[str, Dict[str, Dict]]:
        """
        Analyze all configured symbols and timeframes.
        
        Returns:
            Dict with structure: {symbol: {timeframe: {data, market_type, analysis_time}}}
        """
        results = {}
        
        self.logger.info("Starting market analysis for all configured symbols and timeframes...")
        
        for symbol in self.symbols:
            results[symbol] = {}
            self.logger.info(f"Analyzing {symbol}...")
            
            # Validate symbol first
            if not self._validate_symbol(symbol):
                self.logger.error(f"Symbol {symbol} is invalid or not tradeable - skipping analysis")
                for timeframe in self.timeframes:
                    results[symbol][timeframe] = {
                        'data': None,
                        'market_type': 'INVALID_SYMBOL',
                        'analysis_time': pd.Timestamp.now(),
                        'error': f'Symbol {symbol} is invalid or not tradeable on ByBit'
                    }
                continue
            
            for timeframe in self.timeframes:
                try:
                    result = self._analyze_symbol_timeframe(symbol, timeframe)
                    results[symbol][timeframe] = result
                    self.logger.info(f"  {timeframe}: Market type = {result['market_type']}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to analyze {symbol} {timeframe}: {e}")
                    results[symbol][timeframe] = {
                        'data': None,
                        'market_type': 'ANALYSIS_FAILED',
                        'analysis_time': pd.Timestamp.now(),
                        'error': str(e)
                    }
                    
                # Small delay between requests to avoid rate limiting
                time.sleep(0.2)  # Increased delay to avoid rate limits
        
        self._print_analysis_summary(results)
        return results

    def _analyze_symbol_timeframe(self, symbol: str, timeframe: str) -> Dict:
        """
        Analyze a specific symbol and timeframe combination.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe string (e.g., '1m', '5m')
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Create a temporary data fetcher for this symbol/timeframe
            # Need enough data for longest period indicators (atr_240, bb_width_avg_200)
            # Plus buffer for reliable calculation
            window_size = 350 if timeframe == '5m' else 400  # More data for 1m due to higher noise
            data_fetcher = LiveDataFetcher(
                exchange=self.exchange,
                symbol=symbol,
                timeframe=timeframe,
                window_size=window_size,
                logger=self.logger
            )
            
            # Fetch initial OHLCV data
            data = data_fetcher.fetch_initial_data()
            
            if data.empty:
                raise MarketAnalysisError(f"No data returned for {symbol} {timeframe}")
            
            # Calculate technical indicators
            data_with_indicators = self._calculate_indicators(data.copy())
            
            # Determine market type using comprehensive algorithms
            market_type, analysis_details = self._determine_market_type(data_with_indicators, symbol, timeframe)
            
            return {
                'data': data_with_indicators,
                'market_type': market_type,
                'analysis_time': pd.Timestamp.now(),
                'data_points': len(data_with_indicators),
                'price_range': {
                    'high': float(data['high'].max()),
                    'low': float(data['low'].min()),
                    'current': float(data['close'].iloc[-1])
                },
                'analysis_details': analysis_details
            }
            
        except DataFetchError as e:
            raise MarketAnalysisError(f"Data fetch failed for {symbol} {timeframe}: {e}")
        except Exception as e:
            raise MarketAnalysisError(f"Analysis failed for {symbol} {timeframe}: {e}")

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all required technical indicators for market type classification.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with added indicator columns
        """
        try:
            # Calculate ATR (Average True Range)
            data['tr'] = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    abs(data['high'] - data['close'].shift(1)),
                    abs(data['low'] - data['close'].shift(1))
                )
            )
            
            # ATR with different periods
            data['atr_14'] = data['tr'].rolling(window=14, min_periods=14).mean()
            data['atr_12'] = data['tr'].rolling(window=12, min_periods=12).mean()
            data['atr_48'] = data['tr'].rolling(window=48, min_periods=48).mean()
            data['atr_50'] = data['tr'].rolling(window=50, min_periods=50).mean()
            data['atr_60'] = data['tr'].rolling(window=60, min_periods=60).mean()
            # Use adaptive min_periods for long-term ATR to handle insufficient data
            atr_240_min_periods = min(240, len(data) // 2) if len(data) >= 120 else 60
            data['atr_240'] = data['tr'].rolling(window=240, min_periods=atr_240_min_periods).mean()
            
            # ADX calculation
            data = self._calculate_adx(data, period=14)
            
            # Moving averages
            data['ema_9'] = data['close'].ewm(span=9, adjust=False).mean()
            data['ema_10'] = data['close'].ewm(span=10, adjust=False).mean()
            data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
            data['ema_30'] = data['close'].ewm(span=30, adjust=False).mean()
            data['ema_34'] = data['close'].ewm(span=34, adjust=False).mean()
            data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
            data['sma_50'] = data['close'].rolling(window=50, min_periods=50).mean()
            data['sma_100'] = data['close'].rolling(window=100, min_periods=100).mean()
            
            # Bollinger Bands
            data = self._calculate_bollinger_bands(data, period=20, std_dev=2)
            
            # Price structure indicators
            data['highest_high_20'] = data['high'].rolling(window=20, min_periods=20).max()
            data['lowest_low_20'] = data['low'].rolling(window=20, min_periods=20).min()
            data['highest_high_50'] = data['high'].rolling(window=50, min_periods=50).max()
            data['lowest_low_50'] = data['low'].rolling(window=50, min_periods=50).min()
            data['highest_high_60'] = data['high'].rolling(window=60, min_periods=60).max()
            data['lowest_low_60'] = data['low'].rolling(window=60, min_periods=60).min()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            raise

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate ADX, +DI, and -DI indicators.
        
        Args:
            data: DataFrame with OHLC data
            period: Period for ADX calculation
            
        Returns:
            DataFrame with ADX, +DI, -DI columns added
        """
        try:
            # Calculate directional movement
            data['dm_plus'] = np.where(
                (data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low']),
                np.maximum(data['high'] - data['high'].shift(1), 0),
                0
            )
            
            data['dm_minus'] = np.where(
                (data['low'].shift(1) - data['low']) > (data['high'] - data['high'].shift(1)),
                np.maximum(data['low'].shift(1) - data['low'], 0),
                0
            )
            
            # Smooth the directional movements and true range
            data['dm_plus_smooth'] = data['dm_plus'].rolling(window=period, min_periods=period).mean()
            data['dm_minus_smooth'] = data['dm_minus'].rolling(window=period, min_periods=period).mean()
            data['tr_smooth'] = data['tr'].rolling(window=period, min_periods=period).mean()
            
            # Calculate +DI and -DI
            data['plus_di'] = 100 * (data['dm_plus_smooth'] / data['tr_smooth'])
            data['minus_di'] = 100 * (data['dm_minus_smooth'] / data['tr_smooth'])
            
            # Calculate DX
            data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
            
            # Calculate ADX as smoothed DX
            data['adx'] = data['dx'].rolling(window=period, min_periods=period).mean()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            raise

    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and Band Width.
        
        Args:
            data: DataFrame with close prices
            period: Period for moving average
            std_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with Bollinger Band columns added
        """
        try:
            data['bb_middle'] = data['close'].rolling(window=period, min_periods=period).mean()
            data['bb_std'] = data['close'].rolling(window=period, min_periods=period).std()
            data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * std_dev)
            data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * std_dev)
            
            # Calculate Bollinger Band Width
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            
            # Calculate average BB width for comparison with adaptive min_periods
            bb_100_min_periods = min(100, len(data) // 3) if len(data) >= 60 else 30
            bb_200_min_periods = min(200, len(data) // 2) if len(data) >= 100 else 50
            data['bb_width_avg_100'] = data['bb_width'].rolling(window=100, min_periods=bb_100_min_periods).mean()
            data['bb_width_avg_200'] = data['bb_width'].rolling(window=200, min_periods=bb_200_min_periods).mean()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            raise

    def _determine_market_type(self, data: pd.DataFrame, symbol: str, timeframe: str) -> Tuple[str, Dict]:
        """
        Determine the market type using comprehensive algorithmic classification.
        
        Args:
            data: OHLCV DataFrame with indicators
            symbol: Trading symbol
            timeframe: Timeframe string ('1m', '5m')
            
        Returns:
            Tuple of (market_type, analysis_details)
        """
        if len(data) < 50:
            return "INSUFFICIENT_DATA", {"reason": "Not enough data for analysis"}
        
        try:
            # Get the latest values
            latest = data.iloc[-1]
            analysis_details = {}
            
            # Determine timeframe-specific parameters
            if timeframe == '5m':
                # 5-minute parameters
                adx_trend_threshold = 25
                adx_range_threshold = 20
                atr_short_col = 'atr_12'  # ~1 hour
                atr_long_col = 'atr_48'   # ~4 hours
                bb_width_avg_col = 'bb_width_avg_100'
                fast_ma_col = 'ema_20'
                slow_ma_col = 'ema_50'
                range_lookback = 50
                volatility_high_atr_ratio = 1.2
                volatility_high_bb_ratio = 1.5
                volatility_low_atr_ratio = 0.9
                volatility_low_bb_ratio = 0.8
            else:  # 1m
                # 1-minute parameters
                adx_trend_threshold = 22
                adx_range_threshold = 18
                atr_short_col = 'atr_60'   # 1 hour
                atr_long_col = 'atr_240'   # 4 hours
                bb_width_avg_col = 'bb_width_avg_200'
                fast_ma_col = 'ema_10'
                slow_ma_col = 'ema_30'
                range_lookback = 60
                volatility_high_atr_ratio = 1.2
                volatility_high_bb_ratio = 1.5
                volatility_low_atr_ratio = 0.9
                volatility_low_bb_ratio = 0.8
            
            # Extract current values
            adx = latest['adx'] if pd.notna(latest['adx']) else 0
            atr_short = latest[atr_short_col] if pd.notna(latest[atr_short_col]) else 0
            atr_long = latest[atr_long_col] if pd.notna(latest[atr_long_col]) else 0
            bb_width = latest['bb_width'] if pd.notna(latest['bb_width']) else 0
            bb_width_avg = latest[bb_width_avg_col] if pd.notna(latest[bb_width_avg_col]) else 0
            fast_ma = latest[fast_ma_col] if pd.notna(latest[fast_ma_col]) else latest['close']
            slow_ma = latest[slow_ma_col] if pd.notna(latest[slow_ma_col]) else latest['close']
            plus_di = latest['plus_di'] if pd.notna(latest['plus_di']) else 0
            minus_di = latest['minus_di'] if pd.notna(latest['minus_di']) else 0
            
            # Store analysis details
            analysis_details.update({
                'adx': float(adx),
                'atr_short': float(atr_short),
                'atr_long': float(atr_long),
                'bb_width': float(bb_width),
                'bb_width_avg': float(bb_width_avg),
                'plus_di': float(plus_di),
                'minus_di': float(minus_di)
            })
            
            # HIERARCHY: Check conditions in order of priority
            
            # 1. HIGH-VOLATILITY CHECK (highest priority)
            if atr_long > 0 and bb_width_avg > 0:
                atr_ratio = atr_short / atr_long
                bb_ratio = bb_width / bb_width_avg
                
                if atr_ratio > volatility_high_atr_ratio and bb_ratio > volatility_high_bb_ratio:
                    analysis_details.update({
                        'atr_ratio': float(atr_ratio),
                        'bb_ratio': float(bb_ratio),
                        'reason': 'ATR and BB width significantly elevated'
                    })
                    return "HIGH_VOLATILITY", analysis_details
            
            # 2. LOW-VOLATILITY CHECK
            if atr_long > 0 and bb_width_avg > 0:
                atr_ratio = atr_short / atr_long
                bb_ratio = bb_width / bb_width_avg
                
                if atr_ratio < volatility_low_atr_ratio and bb_ratio < volatility_low_bb_ratio:
                    analysis_details.update({
                        'atr_ratio': float(atr_ratio),
                        'bb_ratio': float(bb_ratio),
                        'reason': 'ATR and BB width significantly contracted'
                    })
                    return "LOW_VOLATILITY", analysis_details
            
            # 3. TRENDING CHECK
            if adx > adx_trend_threshold:
                # Check directional movement and MA separation
                ma_diff = abs(fast_ma - slow_ma)
                atr_threshold = 0.3 * atr_short if atr_short > 0 else 0
                
                # Confirm direction with +DI/-DI or MA alignment
                has_direction = (plus_di > minus_di and latest['close'] > slow_ma) or \
                               (minus_di > plus_di and latest['close'] < slow_ma)
                
                if ma_diff > atr_threshold and has_direction:
                    trend_direction = "BULLISH" if plus_di > minus_di else "BEARISH"
                    analysis_details.update({
                        'trend_direction': trend_direction,
                        'ma_separation': float(ma_diff),
                        'atr_threshold': float(atr_threshold),
                        'reason': f'Strong ADX ({adx:.1f}) with directional bias'
                    })
                    return "TRENDING", analysis_details
            
            # 4. RANGING CHECK
            if adx < adx_range_threshold:
                # Check price range boundaries
                range_data = data.tail(range_lookback)
                if len(range_data) >= range_lookback:
                    highest_high = range_data['high'].max()
                    lowest_low = range_data['low'].min()
                    current_price = latest['close']
                    range_pct = (highest_high - lowest_low) / current_price
                    
                    # Check for flat MA (low slope)
                    ma_slope_threshold = 0.001 * current_price  # 0.1% of price
                    ma_slope = abs(slow_ma - data[slow_ma_col].iloc[-10]) if len(data) >= 10 else 0
                    
                    if timeframe == '5m':
                        range_threshold = 0.03  # 3%
                    else:  # 1m
                        range_threshold = 0.01  # 1%
                    
                    if range_pct < range_threshold and ma_slope < ma_slope_threshold:
                        analysis_details.update({
                            'range_pct': float(range_pct),
                            'ma_slope': float(ma_slope),
                            'reason': f'Low ADX ({adx:.1f}) with confined price range'
                        })
                        return "RANGING", analysis_details
            
            # 5. TRANSITIONAL (default for unclear conditions)
            # This covers ADX in the 20-25 zone or mixed signals
            reason = "Mixed signals or ADX in transition zone"
            if adx_range_threshold <= adx <= adx_trend_threshold:
                reason = f"ADX in transition zone ({adx:.1f})"
            
            analysis_details.update({
                'reason': reason,
                'adx_zone': 'transition' if adx_range_threshold <= adx <= adx_trend_threshold else 'unclear'
            })
            return "TRANSITIONAL", analysis_details
            
        except Exception as e:
            self.logger.error(f"Error in market type determination: {e}")
            return "ANALYSIS_FAILED", {"error": str(e)}
    
    def _print_analysis_summary(self, results: Dict[str, Dict[str, Dict]]):
        """
        Print a formatted summary of the market analysis results to console.
        
        Args:
            results: Analysis results dictionary
        """
        print("\n" + "="*90)
        print("COMPREHENSIVE MARKET ANALYSIS SUMMARY")
        print("="*90)
        
        for symbol in results:
            print(f"\n{symbol}:")
            print("-" * (len(symbol) + 1))
            
            for timeframe in results[symbol]:
                result = results[symbol][timeframe]
                market_type = result.get('market_type', 'UNKNOWN')
                data_points = result.get('data_points', 0)
                
                if 'price_range' in result:
                    current_price = result['price_range']['current']
                    
                    # Get additional analysis details
                    details = result.get('analysis_details', {})
                    adx = details.get('adx', 0)
                    
                    # Format with ADX value
                    print(f"  {timeframe:>3}: {market_type:<20} | Price: ${current_price:>10.4f} | ADX: {adx:>5.1f} | Data: {data_points:>3} bars")
                    
                    # Print reason if available
                    if 'reason' in details:
                        print(f"       Reason: {details['reason']}")
                else:
                    error = result.get('error', 'Unknown error')
                    print(f"  {timeframe:>3}: {market_type:<20} | Error: {error}")
        
        print("\n" + "="*90)
        print("MARKET TYPE LEGEND:")
        print("  TRENDING          - Strong directional movement (ADX > threshold)")
        print("  RANGING           - Sideways/oscillating within bounds (ADX < threshold)")
        print("  HIGH_VOLATILITY   - Unusually large price swings (ATR/BB expansion)")
        print("  LOW_VOLATILITY    - Unusually quiet market (ATR/BB contraction)")
        print("  TRANSITIONAL      - Mixed signals or regime change in progress")
        print("  INVALID_SYMBOL    - Symbol not found or not tradeable on exchange")
        print("  ANALYSIS_FAILED   - Could not analyze due to technical error")
        print("  INSUFFICIENT_DATA - Not enough data points for reliable analysis")
        print("="*90)

    def get_market_summary(self, results: Dict[str, Dict[str, Dict]]) -> Dict[str, int]:
        """
        Get a summary count of market types across all symbols and timeframes.
        
        Args:
            results: Analysis results dictionary
            
        Returns:
            Dict with market type counts
        """
        market_type_counts = {}
        
        for symbol in results:
            for timeframe in results[symbol]:
                market_type = results[symbol][timeframe].get('market_type', 'UNKNOWN')
                market_type_counts[market_type] = market_type_counts.get(market_type, 0) + 1
        
        return market_type_counts

    # ==================== ENHANCED VOLATILITY & REGIME DETECTION ====================
    
    def get_atr_pct(self, symbol: str, timeframe: str = '1m') -> float:
        """
        Calculate ATR as percentage of current price.
        
        This provides a normalized volatility measure that can be compared across 
        different price levels and assets.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe for analysis (default: '1m')
            
        Returns:
            ATR as percentage of current price (e.g., 2.5 = 2.5%)
            
        Raises:
            MarketAnalysisError: If calculation fails or insufficient data
        """
        try:
            cache_key = f"{symbol}_{timeframe}"
            current_time = datetime.now(timezone.utc)
            
            # Check cache first
            with self._cache_lock:
                if cache_key in self._atr_cache:
                    cache_entry = self._atr_cache[cache_key]
                    cache_age = (current_time - cache_entry['timestamp']).total_seconds()
                    
                    if cache_age < self.cache_validity_seconds:
                        self.logger.debug(f"ATR cache hit for {symbol} {timeframe} (age: {cache_age:.1f}s)")
                        return cache_entry['value']
            
            # Start computation timer
            computation_start = time.time()
            
            # Get market data with timeout protection
            data = self._get_cached_data(symbol, timeframe)
            if data is None or len(data) < self.atr_period + 5:
                raise MarketAnalysisError(f"Insufficient data for ATR calculation: {len(data) if data is not None else 0} bars")
            
            # Check computation time
            if time.time() - computation_start > self.max_computation_time_seconds:
                raise MarketAnalysisError(f"ATR computation timeout for {symbol}")
            
            # Calculate ATR using risk utilities
            try:
                atr_series = compute_atr(data, period=self.atr_period)
                if atr_series is None or atr_series.empty or atr_series.iloc[-1] is None:
                    raise ValueError("ATR calculation returned invalid result")
                
                current_atr = float(atr_series.iloc[-1])
                current_price = float(data['close'].iloc[-1])
                
                if current_price <= 0:
                    raise ValueError(f"Invalid current price: {current_price}")
                
                atr_pct = (current_atr / current_price) * 100
                
            except Exception as e:
                # Fallback to simple ATR calculation
                self.logger.warning(f"Risk utilities ATR failed for {symbol}, using fallback: {e}")
                tr = self._calculate_true_range(data)
                atr_simple = tr.rolling(window=self.atr_period, min_periods=self.atr_period).mean()
                current_atr = float(atr_simple.iloc[-1])
                current_price = float(data['close'].iloc[-1])
                atr_pct = (current_atr / current_price) * 100
            
            # Validate result
            if not (0 <= atr_pct <= 50):  # Sanity check: ATR should be 0-50% of price
                self.logger.warning(f"ATR percentage seems unusual for {symbol}: {atr_pct:.2f}%")
            
            # Cache the result
            with self._cache_lock:
                self._atr_cache[cache_key] = {
                    'value': atr_pct,
                    'timestamp': current_time,
                    'timeframe': timeframe,
                    'computation_time_ms': (time.time() - computation_start) * 1000
                }
            
            self.logger.debug(f"Calculated ATR for {symbol} {timeframe}: {atr_pct:.3f}%")
            return atr_pct
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR percentage for {symbol} {timeframe}: {e}")
            raise MarketAnalysisError(f"ATR calculation failed for {symbol}: {str(e)}")

    def get_vol_regime(self, symbol: str, timeframe: str = '1m') -> str:
        """
        Determine volatility regime using ATR percentage analysis.
        
        Uses RiskUtilities.volatility_regime to classify current market volatility
        into low, normal, or high regimes based on ATR percentage thresholds.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe for analysis (default: '1m')
            
        Returns:
            Volatility regime: 'low', 'normal', or 'high'
            
        Raises:
            MarketAnalysisError: If analysis fails
        """
        try:
            cache_key = f"{symbol}_{timeframe}"
            current_time = datetime.now(timezone.utc)
            
            # Check cache first
            with self._cache_lock:
                if cache_key in self._vol_regime_cache:
                    cache_entry = self._vol_regime_cache[cache_key]
                    cache_age = (current_time - cache_entry['timestamp']).total_seconds()
                    
                    if cache_age < self.cache_validity_seconds:
                        self.logger.debug(f"Volatility regime cache hit for {symbol} {timeframe} (age: {cache_age:.1f}s)")
                        return cache_entry['regime']
            
            # Start computation timer
            computation_start = time.time()
            
            # Get ATR percentage
            atr_pct = self.get_atr_pct(symbol, timeframe)
            
            # Check computation time
            if time.time() - computation_start > self.max_computation_time_seconds:
                raise MarketAnalysisError(f"Volatility regime computation timeout for {symbol}")
            
            # Determine regime using risk utilities
            try:
                regime = volatility_regime(atr_pct, thresholds=self.vol_regime_thresholds)
            except Exception as e:
                # Fallback regime determination
                self.logger.warning(f"Risk utilities volatility_regime failed for {symbol}, using fallback: {e}")
                if atr_pct < self.vol_regime_thresholds[0]:
                    regime = "low"
                elif atr_pct > self.vol_regime_thresholds[1]:
                    regime = "high"
                else:
                    regime = "normal"
            
            # Validate regime
            if regime not in ['low', 'normal', 'high']:
                self.logger.warning(f"Invalid volatility regime '{regime}' for {symbol}, defaulting to 'normal'")
                regime = "normal"
            
            # Cache the result
            with self._cache_lock:
                self._vol_regime_cache[cache_key] = {
                    'regime': regime,
                    'timestamp': current_time,
                    'atr_pct': atr_pct,
                    'thresholds': self.vol_regime_thresholds,
                    'computation_time_ms': (time.time() - computation_start) * 1000
                }
            
            self.logger.debug(f"Volatility regime for {symbol} {timeframe}: {regime} (ATR: {atr_pct:.3f}%)")
            return regime
            
        except Exception as e:
            self.logger.error(f"Error determining volatility regime for {symbol} {timeframe}: {e}")
            raise MarketAnalysisError(f"Volatility regime analysis failed for {symbol}: {str(e)}")

    def get_market_regime(self, symbol: str, timeframe: str = '1m') -> str:
        """
        Determine comprehensive market regime using composite analysis.
        
        Combines trend strength (ADX/slope), realized volatility (ATR), and other
        market structure indicators to classify the overall market regime.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe for analysis (default: '1m')
            
        Returns:
            Market regime: 'trending_low_vol', 'trending_high_vol', 'ranging_low_vol', 
                          'ranging_high_vol', 'transitional', or 'breakout'
            
        Raises:
            MarketAnalysisError: If analysis fails
        """
        try:
            cache_key = f"{symbol}_{timeframe}"
            current_time = datetime.now(timezone.utc)
            
            # Check cache first
            with self._cache_lock:
                if cache_key in self._market_regime_cache:
                    cache_entry = self._market_regime_cache[cache_key]
                    cache_age = (current_time - cache_entry['timestamp']).total_seconds()
                    
                    if cache_age < self.cache_validity_seconds:
                        self.logger.debug(f"Market regime cache hit for {symbol} {timeframe} (age: {cache_age:.1f}s)")
                        return cache_entry['regime']
            
            # Start computation timer
            computation_start = time.time()
            
            # Get basic components
            data = self._get_cached_data(symbol, timeframe)
            if data is None or len(data) < 50:  # Need sufficient data for regime analysis
                raise MarketAnalysisError(f"Insufficient data for market regime analysis: {len(data) if data is not None else 0} bars")
            
            # Check computation time
            if time.time() - computation_start > self.max_computation_time_seconds:
                raise MarketAnalysisError(f"Market regime computation timeout for {symbol}")
            
            # Calculate indicators with timeout protection
            try:
                data_with_indicators = self._calculate_indicators(data)
            except Exception as e:
                raise MarketAnalysisError(f"Indicator calculation failed for {symbol}: {e}")
            
            # Get volatility regime
            vol_regime = self.get_vol_regime(symbol, timeframe)
            
            # Get trend strength using ADX
            current_adx = float(data_with_indicators['adx'].iloc[-1]) if not pd.isna(data_with_indicators['adx'].iloc[-1]) else 25.0
            
            # Calculate trend slope using EMA
            ema_20 = data_with_indicators['ema_20'].iloc[-20:] if len(data_with_indicators) >= 20 else data_with_indicators['ema_20']
            if len(ema_20) >= 10:
                # Simple slope calculation over last 10 periods
                x = np.arange(len(ema_20[-10:]))
                y = ema_20[-10:].values
                slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0
                slope_pct = (slope / ema_20.iloc[-1]) * 100 if ema_20.iloc[-1] != 0 else 0
            else:
                slope_pct = 0
            
            # Calculate Bollinger Band position for ranging detection
            current_price = float(data_with_indicators['close'].iloc[-1])
            bb_upper = float(data_with_indicators['bb_upper'].iloc[-1]) if not pd.isna(data_with_indicators['bb_upper'].iloc[-1]) else current_price * 1.02
            bb_lower = float(data_with_indicators['bb_lower'].iloc[-1]) if not pd.isna(data_with_indicators['bb_lower'].iloc[-1]) else current_price * 0.98
            bb_width = float(data_with_indicators['bb_width'].iloc[-1]) if not pd.isna(data_with_indicators['bb_width'].iloc[-1]) else 0.04
            
            # Determine regime based on composite analysis
            regime_components = {
                'adx': current_adx,
                'slope_pct': slope_pct,
                'vol_regime': vol_regime,
                'bb_width': bb_width,
                'bb_position': (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
            }
            
            # Regime determination logic
            if current_adx > 30 and abs(slope_pct) > 0.05:  # Strong trend
                if vol_regime == 'high':
                    regime = 'trending_high_vol'
                else:
                    regime = 'trending_low_vol'
            elif current_adx < 20 and abs(slope_pct) < 0.02:  # Ranging market
                if vol_regime == 'high':
                    regime = 'ranging_high_vol'
                else:
                    regime = 'ranging_low_vol'
            elif vol_regime == 'high' and bb_width > 0.06:  # High volatility breakout conditions
                regime = 'breakout'
            else:  # Mixed signals or transitional phase
                regime = 'transitional'
            
            # Check computation time again
            computation_time = time.time() - computation_start
            if computation_time > self.max_computation_time_seconds:
                self.logger.warning(f"Market regime computation exceeded time limit: {computation_time:.2f}s")
            
            # Cache the result
            with self._cache_lock:
                self._market_regime_cache[cache_key] = {
                    'regime': regime,
                    'timestamp': current_time,
                    'components': regime_components,
                    'timeframe': timeframe,
                    'computation_time_ms': computation_time * 1000
                }
            
            self.logger.debug(f"Market regime for {symbol} {timeframe}: {regime} (ADX: {current_adx:.1f}, Slope: {slope_pct:.3f}%, Vol: {vol_regime})")
            return regime
            
        except Exception as e:
            self.logger.error(f"Error determining market regime for {symbol} {timeframe}: {e}")
            raise MarketAnalysisError(f"Market regime analysis failed for {symbol}: {str(e)}")

    def _get_cached_data(self, symbol: str, timeframe: str, min_periods: int = 100) -> Optional[pd.DataFrame]:
        """
        Get cached OHLCV data or fetch fresh data if cache miss/expired.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for data
            min_periods: Minimum number of periods required
            
        Returns:
            DataFrame with OHLCV data or None if unavailable
        """
        try:
            cache_key = f"{symbol}_{timeframe}"
            current_time = datetime.now(timezone.utc)
            
            # Check cache first
            with self._cache_lock:
                if cache_key in self._data_cache:
                    cache_entry = self._data_cache[cache_key]
                    cache_age = (current_time - cache_entry['timestamp']).total_seconds()
                    
                    if cache_age < self.cache_validity_seconds and len(cache_entry['data']) >= min_periods:
                        return cache_entry['data']
            
            # Fetch fresh data
            try:
                data_fetcher = LiveDataFetcher(self.exchange, symbol, timeframe, window_size=max(min_periods, 200))
                # First fetch initial data to populate the data fetcher
                data = data_fetcher.fetch_initial_data()
                
                if data is None or len(data) < min_periods:
                    self.logger.warning(f"Insufficient data fetched for {symbol} {timeframe}: {len(data) if data is not None else 0} bars")
                    return None
                
                # Cache the fresh data
                with self._cache_lock:
                    self._data_cache[cache_key] = {
                        'data': data.copy(),
                        'timestamp': current_time,
                        'timeframe': timeframe
                    }
                
                return data
                
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol} {timeframe}: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in data caching for {symbol} {timeframe}: {e}")
            return None

    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate True Range for fallback ATR calculation.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            True Range series
        """
        try:
            high_low = data['high'] - data['low']
            high_close_prev = abs(data['high'] - data['close'].shift(1))
            low_close_prev = abs(data['low'] - data['close'].shift(1))
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            return true_range
            
        except Exception as e:
            self.logger.error(f"Error calculating true range: {e}")
            # Return fallback series
            return data['high'] - data['low']

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear analysis cache for specified symbol or all symbols.
        
        Args:
            symbol: Symbol to clear cache for, or None to clear all
        """
        try:
            with self._cache_lock:
                if symbol is None:
                    # Clear all caches
                    self._atr_cache.clear()
                    self._vol_regime_cache.clear()
                    self._market_regime_cache.clear()
                    self._data_cache.clear()
                    self.logger.info("Cleared all analysis caches")
                else:
                    # Clear caches for specific symbol
                    symbol_keys = [key for key in self._atr_cache.keys() if key.startswith(f"{symbol}_")]
                    for key in symbol_keys:
                        self._atr_cache.pop(key, None)
                        self._vol_regime_cache.pop(key, None)
                        self._market_regime_cache.pop(key, None)
                        self._data_cache.pop(key, None)
                    self.logger.info(f"Cleared analysis cache for {symbol}")
                    
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def get_cache_stats(self) -> Dict[str, Dict]:
        """
        Get cache statistics for monitoring and debugging.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            with self._cache_lock:
                current_time = datetime.now(timezone.utc)
                
                def analyze_cache(cache_dict: Dict, cache_name: str) -> Dict:
                    stats = {
                        'total_entries': len(cache_dict),
                        'symbols': set(),
                        'timeframes': set(),
                        'oldest_entry_age_seconds': 0,
                        'newest_entry_age_seconds': float('inf'),
                        'avg_computation_time_ms': 0
                    }
                    
                    computation_times = []
                    for key, entry in cache_dict.items():
                        if '_' in key:
                            symbol_part = key.split('_')[0]
                            stats['symbols'].add(symbol_part)
                        
                        if 'timeframe' in entry:
                            stats['timeframes'].add(entry['timeframe'])
                        
                        if 'timestamp' in entry:
                            age = (current_time - entry['timestamp']).total_seconds()
                            stats['oldest_entry_age_seconds'] = max(stats['oldest_entry_age_seconds'], age)
                            stats['newest_entry_age_seconds'] = min(stats['newest_entry_age_seconds'], age)
                        
                        if 'computation_time_ms' in entry:
                            computation_times.append(entry['computation_time_ms'])
                    
                    if computation_times:
                        stats['avg_computation_time_ms'] = sum(computation_times) / len(computation_times)
                    
                    stats['symbols'] = list(stats['symbols'])
                    stats['timeframes'] = list(stats['timeframes'])
                    
                    if stats['newest_entry_age_seconds'] == float('inf'):
                        stats['newest_entry_age_seconds'] = 0
                    
                    return stats
                
                return {
                    'atr_cache': analyze_cache(self._atr_cache, 'ATR'),
                    'vol_regime_cache': analyze_cache(self._vol_regime_cache, 'Volatility Regime'),
                    'market_regime_cache': analyze_cache(self._market_regime_cache, 'Market Regime'),
                    'data_cache': analyze_cache(self._data_cache, 'Data Cache'),
                    'cache_validity_seconds': self.cache_validity_seconds,
                    'max_computation_time_seconds': self.max_computation_time_seconds
                }
                
        except Exception as e:
            self.logger.error(f"Error getting cache statistics: {e}")
            return {'error': str(e)} 