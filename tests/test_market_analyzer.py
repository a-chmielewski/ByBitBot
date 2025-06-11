#!/usr/bin/env python3
"""
Test script for the Market Analyzer module.
This script tests the market analysis functionality independently.
"""

import json
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.logger import get_logger
from modules.exchange import ExchangeConnector
from modules.market_analyzer import MarketAnalyzer, MarketAnalysisError

def load_config():
    """Load configuration from config.json"""
    with open('config.json', 'r') as f:
        return json.load(f)

def test_market_analyzer():
    """Test the market analyzer functionality"""
    
    # Initialize logger
    logger = get_logger('market_analyzer_test')
    logger.info("Starting Market Analyzer Test")
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Initialize exchange connection
        ex_cfg = config['bybit']
        exchange = ExchangeConnector(
            api_key=ex_cfg['api_key'], 
            api_secret=ex_cfg['api_secret'], 
            testnet=False, 
            logger=logger
        )
        logger.info("Exchange connection initialized")
        
        # Initialize market analyzer
        market_analyzer = MarketAnalyzer(exchange, config, logger)
        logger.info("Market analyzer initialized")
        
        # Run market analysis
        logger.info("Starting market analysis for all configured symbols...")
        analysis_results = market_analyzer.analyze_all_markets()
        
        # Display results summary
        if analysis_results:
            summary = market_analyzer.get_market_summary(analysis_results)
            logger.info(f"Analysis completed. Market type summary: {summary}")
            
            # Count successful analyses
            successful_analyses = 0
            total_analyses = 0
            
            for symbol in analysis_results:
                for timeframe in analysis_results[symbol]:
                    total_analyses += 1
                    if analysis_results[symbol][timeframe]['market_type'] != 'ANALYSIS_FAILED':
                        successful_analyses += 1
            
            logger.info(f"Success rate: {successful_analyses}/{total_analyses} ({successful_analyses/total_analyses*100:.1f}%)")
            
            print(f"\n✅ Test completed successfully!")
            print(f"📊 Analyzed {total_analyses} symbol/timeframe combinations")
            print(f"✔️  {successful_analyses} successful analyses ({successful_analyses/total_analyses*100:.1f}%)")
            print(f"📈 Market type distribution: {summary}")
            
        else:
            logger.error("Market analysis returned no results")
            print("❌ Test failed - no analysis results returned")
            
    except MarketAnalysisError as e:
        logger.error(f"Market analysis error: {e}")
        print(f"❌ Market analysis error: {e}")
        
    except Exception as e:
        logger.error(f"Unexpected error during test: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    print("🚀 Testing Market Analyzer...")
    print("=" * 50)
    test_market_analyzer()
    print("=" * 50)
    print("🏁 Test completed.") 