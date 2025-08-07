#!/usr/bin/env python3
"""
Unit tests for enhanced MarketAnalyzer volatility and regime detection
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.market_analyzer import MarketAnalyzer, MarketAnalysisError
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
from unittest.mock import Mock, patch

class TestEnhancedMarketAnalyzer(unittest.TestCase):
    """Test cases for enhanced MarketAnalyzer functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_exchange = Mock()
        self.config = {
            'market_analysis': {
                'timeframes': ['1m', '5m'],
                'use_dynamic_symbols': False,
                'symbols': ['BTCUSDT', 'ETHUSDT'],
                'analysis_cache_seconds': 300,
                'max_computation_time_seconds': 30,
                'atr_period': 14,
                'volatility_thresholds': (0.5, 1.5)
            }
        }
        
        # Create sample OHLCV data
        self.sample_data = self._create_sample_data()
        
    def _create_sample_data(self, periods=200):
        """Create sample OHLCV data for testing"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=periods, freq='1min')
        base_price = 50000.0
        returns = np.random.normal(0, 0.01, periods)
        prices = base_price * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.002,
            'low': prices * 0.998,
            'close': prices,
            'volume': np.random.uniform(100, 1000, periods)
        })
    
    @patch('modules.market_analyzer.LiveDataFetcher')
    def test_get_atr_pct_calculation(self, mock_data_fetcher):
        """Test ATR percentage calculation"""
        # Mock data fetcher
        mock_fetcher_instance = Mock()
        mock_fetcher_instance.fetch_initial_data.return_value = self.sample_data
        mock_data_fetcher.return_value = mock_fetcher_instance
        
        analyzer = MarketAnalyzer(self.mock_exchange, self.config)
        
        # Test ATR calculation
        atr_pct = analyzer.get_atr_pct('BTCUSDT', '1m')
        
        self.assertIsInstance(atr_pct, float)
        self.assertGreater(atr_pct, 0)
        self.assertLess(atr_pct, 50)  # Sanity check
        
    @patch('modules.market_analyzer.LiveDataFetcher')
    def test_get_vol_regime_thresholds(self, mock_data_fetcher):
        """Test volatility regime detection with different thresholds"""
        mock_fetcher_instance = Mock()
        mock_fetcher_instance.fetch_initial_data.return_value = self.sample_data
        mock_data_fetcher.return_value = mock_fetcher_instance
        
        # Test with default thresholds
        analyzer1 = MarketAnalyzer(self.mock_exchange, self.config)
        regime1 = analyzer1.get_vol_regime('BTCUSDT', '1m')
        
        # Test with wider thresholds (should tend toward 'normal')
        config2 = self.config.copy()
        config2['market_analysis']['volatility_thresholds'] = (0.1, 5.0)
        analyzer2 = MarketAnalyzer(self.mock_exchange, config2)
        regime2 = analyzer2.get_vol_regime('BTCUSDT', '1m')
        
        # Test with narrow thresholds (should tend toward 'high' or 'low')
        config3 = self.config.copy()
        config3['market_analysis']['volatility_thresholds'] = (1.5, 2.0)
        analyzer3 = MarketAnalyzer(self.mock_exchange, config3)
        regime3 = analyzer3.get_vol_regime('BTCUSDT', '1m')
        
        # Validate regimes
        valid_regimes = ['low', 'normal', 'high']
        self.assertIn(regime1, valid_regimes)
        self.assertIn(regime2, valid_regimes)
        self.assertIn(regime3, valid_regimes)
        
        # Wide thresholds should tend toward 'normal'
        self.assertEqual(regime2, 'normal')
        
    @patch('modules.market_analyzer.LiveDataFetcher')
    def test_caching_behavior(self, mock_data_fetcher):
        """Test caching functionality and performance"""
        mock_fetcher_instance = Mock()
        mock_fetcher_instance.fetch_initial_data.return_value = self.sample_data
        mock_data_fetcher.return_value = mock_fetcher_instance
        
        analyzer = MarketAnalyzer(self.mock_exchange, self.config)
        
        # First call - should compute and cache
        start_time = time.time()
        atr1 = analyzer.get_atr_pct('BTCUSDT', '1m')
        first_call_time = time.time() - start_time
        
        # Second call - should use cache
        start_time = time.time()
        atr2 = analyzer.get_atr_pct('BTCUSDT', '1m')
        second_call_time = time.time() - start_time
        
        # Results should be identical
        self.assertEqual(atr1, atr2)
        
        # Second call should be faster (cached)
        self.assertLess(second_call_time, first_call_time)
        
        # Test cache statistics
        cache_stats = analyzer.get_cache_stats()
        self.assertIn('atr_cache', cache_stats)
        self.assertEqual(cache_stats['atr_cache']['total_entries'], 1)
        
    @patch('modules.market_analyzer.LiveDataFetcher')
    def test_get_market_regime_classification(self, mock_data_fetcher):
        """Test market regime classification"""
        mock_fetcher_instance = Mock()
        mock_fetcher_instance.fetch_initial_data.return_value = self.sample_data
        mock_data_fetcher.return_value = mock_fetcher_instance
        
        analyzer = MarketAnalyzer(self.mock_exchange, self.config)
        
        market_regime = analyzer.get_market_regime('BTCUSDT', '1m')
        
        valid_regimes = ['trending_low_vol', 'trending_high_vol', 'ranging_low_vol', 
                        'ranging_high_vol', 'transitional', 'breakout']
        self.assertIn(market_regime, valid_regimes)
        
    @patch('modules.market_analyzer.LiveDataFetcher')
    def test_cache_clearing(self, mock_data_fetcher):
        """Test cache clearing functionality"""
        mock_fetcher_instance = Mock()
        mock_fetcher_instance.fetch_initial_data.return_value = self.sample_data
        mock_data_fetcher.return_value = mock_fetcher_instance
        
        analyzer = MarketAnalyzer(self.mock_exchange, self.config)
        
        # Populate cache
        analyzer.get_atr_pct('BTCUSDT', '1m')
        analyzer.get_vol_regime('BTCUSDT', '1m')
        
        # Check cache has entries
        cache_stats = analyzer.get_cache_stats()
        self.assertGreater(cache_stats['atr_cache']['total_entries'], 0)
        
        # Clear specific symbol cache
        analyzer.clear_cache('BTCUSDT')
        
        # Clear all cache
        analyzer.clear_cache()
        cache_stats_after = analyzer.get_cache_stats()
        self.assertEqual(cache_stats_after['atr_cache']['total_entries'], 0)
        
    @patch('modules.market_analyzer.LiveDataFetcher')
    def test_error_handling(self, mock_data_fetcher):
        """Test error handling for insufficient data"""
        # Mock insufficient data
        insufficient_data = self.sample_data.head(5)  # Only 5 rows
        mock_fetcher_instance = Mock()
        mock_fetcher_instance.fetch_initial_data.return_value = insufficient_data
        mock_data_fetcher.return_value = mock_fetcher_instance
        
        analyzer = MarketAnalyzer(self.mock_exchange, self.config)
        
        # Should raise MarketAnalysisError for insufficient data
        with self.assertRaises(MarketAnalysisError):
            analyzer.get_atr_pct('BTCUSDT', '1m')
            
    @patch('modules.market_analyzer.LiveDataFetcher')
    def test_stable_values(self, mock_data_fetcher):
        """Test that functions return stable values for same input"""
        mock_fetcher_instance = Mock()
        mock_fetcher_instance.fetch_initial_data.return_value = self.sample_data
        mock_data_fetcher.return_value = mock_fetcher_instance
        
        analyzer = MarketAnalyzer(self.mock_exchange, self.config)
        
        # Multiple calls should return same values
        results = []
        for _ in range(3):
            atr_pct = analyzer.get_atr_pct('BTCUSDT', '1m')
            vol_regime = analyzer.get_vol_regime('BTCUSDT', '1m')
            market_regime = analyzer.get_market_regime('BTCUSDT', '1m')
            results.append((atr_pct, vol_regime, market_regime))
        
        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            self.assertEqual(result, first_result)

if __name__ == '__main__':
    unittest.main()