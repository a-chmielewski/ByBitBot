import unittest
import tempfile
import shutil
import os
from datetime import datetime, timezone
from modules.performance_tracker import (
    PerformanceTracker, 
    MarketContext, 
    OrderDetails, 
    RiskMetrics, 
    TradeStatus,
    AlertType
)

class DummyLogger:
    def info(self, msg): pass
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")
    def debug(self, msg): pass

class TestEnhancedPerformanceTracker(unittest.TestCase):
    """Test the enhanced functionality of the refactored PerformanceTracker"""
    
    def setUp(self):
        """Set up test environment"""
        self.tmp_dir = tempfile.mkdtemp()
        self.logger = DummyLogger()
        
        # Custom alert thresholds for testing
        alert_thresholds = {
            'max_drawdown_pct': 3.0,  # Lower threshold for testing
            'consecutive_losses': 2,   # Lower threshold for testing
            'win_rate_threshold': 50.0,
            'risk_limit_pct': 8.0
        }
        
        self.tracker = PerformanceTracker(
            log_dir=self.tmp_dir, 
            logger=self.logger,
            alert_thresholds=alert_thresholds
        )
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.tmp_dir)
    
    def test_enhanced_trade_recording(self):
        """Test recording trades with enhanced data model"""
        print("\n=== Testing Enhanced Trade Recording ===")
        
        # Create enhanced trade data
        market_context = {
            'market_5m': 'TRENDING',
            'market_1m': 'HIGH_VOLATILITY', 
            'strategy_selection_reason': 'Matrix selection: TRENDING+HIGH_VOLATILITY optimal for ATR Momentum',
            'execution_timeframe': '1m',
            'volatility_regime': 'high',
            'trend_strength': 0.75,
            'market_session': 'Europe'
        }
        
        order_details = {
            'main_order_id': 'order_12345',
            'sl_order_id': 'sl_12345',
            'tp_order_id': 'tp_12345',
            'retry_attempts': 1,
            'slippage_pct': 0.02,
            'spread_at_entry': 0.05,
            'spread_at_exit': 0.03,
            'signal_to_execution_delay_ms': 250,
            'execution_quality_score': 85.5,
            'order_type': 'market',
            'time_in_force': 'GoodTillCancel',
            'reduce_only': False
        }
        
        risk_metrics = {
            'planned_sl_pct': 1.5,
            'actual_sl_pct': 1.4,
            'planned_tp_pct': 3.0,
            'actual_tp_pct': 2.8,
            'risk_reward_ratio': 2.0,
            'position_size_pct': 5.0,
            'leverage_used': 50.0,
            'max_adverse_excursion': -0.5,
            'max_favorable_excursion': 3.2
        }
        
        enhanced_trade = {
            'strategy': 'StrategyATRMomentumBreakout',
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'entry_price': 45000.0,
            'exit_price': 46260.0,
            'size': 0.1,
            'pnl': 126.0,
            'entry_timestamp': '2024-01-15T10:30:00Z',
            'exit_timestamp': '2024-01-15T11:15:00Z',
            'market_context': market_context,
            'order_details': order_details,
            'risk_metrics': risk_metrics,
            'status': 'filled',
            'exit_reason': 'take_profit_hit',
            'trade_duration_seconds': 2700,
            'return_pct': 2.8,
            'notes': 'Clean breakout trade in trending market'
        }
        
        # Record the enhanced trade
        trade_id = self.tracker.record_trade(enhanced_trade)
        
        # Verify the trade was recorded correctly
        self.assertIsNotNone(trade_id)
        self.assertEqual(len(self.tracker.trades), 1)
        
        recorded_trade = self.tracker.get_trade_by_id(trade_id)
        self.assertIsNotNone(recorded_trade)
        self.assertEqual(recorded_trade.strategy, 'StrategyATRMomentumBreakout')
        self.assertEqual(recorded_trade.market_context.market_5m, 'TRENDING')
        self.assertEqual(recorded_trade.order_details.retry_attempts, 1)
        self.assertEqual(recorded_trade.risk_metrics.leverage_used, 50.0)
        
        print(f"✓ Enhanced trade recorded successfully: {trade_id}")
        print(f"✓ Market context: {recorded_trade.market_context.market_5m} + {recorded_trade.market_context.market_1m}")
        print(f"✓ Order quality score: {recorded_trade.order_details.execution_quality_score}")
        print(f"✓ Risk-reward ratio: {recorded_trade.risk_metrics.risk_reward_ratio}")
    
    def test_backward_compatibility(self):
        """Test that legacy trade format still works"""
        print("\n=== Testing Backward Compatibility ===")
        
        # Legacy trade format (like current bot.py uses)
        legacy_trade = {
            'strategy': 'StrategyBreakoutAndRetest',
            'symbol': 'ETHUSDT',
            'entry_price': 2500.0,
            'exit_price': 2475.0,
            'size': 0.5,
            'side': 'sell',
            'pnl': -12.5,
            'timestamp': '2024-01-15T12:00:00Z'
        }
        
        trade_id = self.tracker.record_trade(legacy_trade)
        
        # Verify it was converted correctly
        recorded_trade = self.tracker.get_trade_by_id(trade_id)
        self.assertIsNotNone(recorded_trade)
        self.assertEqual(recorded_trade.strategy, 'StrategyBreakoutAndRetest')
        self.assertEqual(recorded_trade.pnl, -12.5)
        self.assertEqual(recorded_trade.status, TradeStatus.FILLED)
        
        print(f"✓ Legacy trade format converted successfully: {trade_id}")
        print(f"✓ Trade status: {recorded_trade.status}")
    
    def test_alert_system(self):
        """Test the alert system functionality"""
        print("\n=== Testing Alert System ===")
        
        # Record trades that trigger alerts
        losing_trades = [
            {'strategy': 'TestStrategy', 'symbol': 'BTCUSDT', 'side': 'buy', 
             'entry_price': 45000, 'exit_price': 44000, 'size': 0.1, 'pnl': -100,
             'entry_timestamp': '2024-01-15T10:00:00Z'},
            {'strategy': 'TestStrategy', 'symbol': 'BTCUSDT', 'side': 'buy',
             'entry_price': 44000, 'exit_price': 43000, 'size': 0.1, 'pnl': -100,
             'entry_timestamp': '2024-01-15T11:00:00Z'},
            {'strategy': 'TestStrategy', 'symbol': 'BTCUSDT', 'side': 'buy',
             'entry_price': 43000, 'exit_price': 42000, 'size': 0.1, 'pnl': -100,
             'entry_timestamp': '2024-01-15T12:00:00Z'}
        ]
        
        for trade in losing_trades:
            self.tracker.record_trade(trade)
        
        # Check alerts were generated
        self.assertGreater(len(self.tracker.recent_alerts), 0)
        
        # Check consecutive losses alert
        consecutive_loss_alerts = [alert for alert in self.tracker.recent_alerts 
                                 if alert['type'] == AlertType.CONSECUTIVE_LOSSES]
        self.assertGreater(len(consecutive_loss_alerts), 0)
        
        print(f"✓ Generated {len(self.tracker.recent_alerts)} alerts")
        print(f"✓ Consecutive losses: {self.tracker.consecutive_losses}")
        print(f"✓ Alert types: {[alert['type'].value for alert in self.tracker.recent_alerts]}")
    
    def test_comprehensive_statistics(self):
        """Test comprehensive statistics functionality"""
        print("\n=== Testing Comprehensive Statistics ===")
        
        # Add multiple trades with different contexts
        trades = [
            {
                'strategy': 'StrategyATRMomentumBreakout', 'symbol': 'BTCUSDT', 'side': 'buy',
                'entry_price': 45000, 'exit_price': 46350, 'size': 0.1, 'pnl': 135,
                'entry_timestamp': '2024-01-15T10:00:00Z',
                'market_context': {'market_5m': 'TRENDING', 'market_1m': 'HIGH_VOLATILITY', 
                                 'strategy_selection_reason': 'Optimal for trending+high vol',
                                 'execution_timeframe': '1m'},
                'risk_metrics': {'leverage_used': 50, 'position_size_pct': 5.0, 'risk_reward_ratio': 2.5}
            },
            {
                'strategy': 'StrategyRSIRangeScalping', 'symbol': 'ETHUSDT', 'side': 'sell',
                'entry_price': 2500, 'exit_price': 2475, 'size': 0.5, 'pnl': -12.5,
                'entry_timestamp': '2024-01-15T11:00:00Z',
                'market_context': {'market_5m': 'RANGING', 'market_1m': 'RANGING',
                                 'strategy_selection_reason': 'Perfect for ranging conditions',
                                 'execution_timeframe': '1m'},
                'risk_metrics': {'leverage_used': 25, 'position_size_pct': 3.0, 'risk_reward_ratio': 1.5}
            },
            {
                'strategy': 'StrategyATRMomentumBreakout', 'symbol': 'BTCUSDT', 'side': 'buy',
                'entry_price': 46000, 'exit_price': 46920, 'size': 0.1, 'pnl': 92,
                'entry_timestamp': '2024-01-15T12:00:00Z',
                'market_context': {'market_5m': 'TRENDING', 'market_1m': 'HIGH_VOLATILITY',
                                 'strategy_selection_reason': 'Continuation of trend',
                                 'execution_timeframe': '1m'},
                'risk_metrics': {'leverage_used': 50, 'position_size_pct': 5.0, 'risk_reward_ratio': 2.0}
            }
        ]
        
        for trade in trades:
            self.tracker.record_trade(trade)
        
        # Get comprehensive statistics
        stats = self.tracker.get_comprehensive_statistics()
        
        # Verify statistics
        self.assertEqual(stats['total_trades'], 3)
        self.assertAlmostEqual(stats['win_rate'], 66.67, places=1)  # 2 wins out of 3
        self.assertEqual(stats['cumulative_pnl'], 214.5)  # 135 - 12.5 + 92
        
        # Check strategy performance breakdown
        strategy_perf = stats['strategy_performance']
        self.assertIn('StrategyATRMomentumBreakout', strategy_perf)
        self.assertIn('StrategyRSIRangeScalping', strategy_perf)
        
        # Check market context performance
        market_perf = stats['market_context_performance']
        self.assertIn('TRENDING_HIGH_VOLATILITY', market_perf)
        self.assertIn('RANGING_RANGING', market_perf)
        
        # Check risk metrics
        risk_summary = stats['risk_metrics']
        self.assertGreater(risk_summary['avg_leverage'], 0)
        
        print(f"✓ Total trades: {stats['total_trades']}")
        print(f"✓ Win rate: {stats['win_rate']:.2f}%")
        print(f"✓ Cumulative PnL: {stats['cumulative_pnl']}")
        print(f"✓ Strategies tracked: {list(strategy_perf.keys())}")
        print(f"✓ Market contexts: {list(market_perf.keys())}")
        print(f"✓ Average leverage: {risk_summary['avg_leverage']:.1f}x")
    
    def test_market_context_analysis(self):
        """Test market context performance analysis"""
        print("\n=== Testing Market Context Analysis ===")
        
        # Add trades with different market contexts
        contexts = [
            ('TRENDING', 'TRENDING', 150),     # Should be profitable
            ('TRENDING', 'HIGH_VOLATILITY', 75),
            ('RANGING', 'RANGING', -25),       # Might be less profitable
            ('RANGING', 'LOW_VOLATILITY', 30),
            ('HIGH_VOLATILITY', 'TRENDING', 100)
        ]
        
        for i, (market_5m, market_1m, pnl) in enumerate(contexts):
            trade = {
                'strategy': 'TestStrategy',
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'entry_price': 45000,
                'exit_price': 45000 + (pnl * 10),  # Adjust exit price based on pnl
                'size': 0.1,
                'pnl': pnl,
                'entry_timestamp': f'2024-01-15T{10+i}:00:00Z',
                'market_context': {
                    'market_5m': market_5m,
                    'market_1m': market_1m,
                    'strategy_selection_reason': f'Selected for {market_5m}+{market_1m}',
                    'execution_timeframe': '1m'
                }
            }
            self.tracker.record_trade(trade)
        
        # Analyze market context performance
        context_performance = self.tracker.get_market_context_performance()
        
        # Verify analysis
        self.assertIn('TRENDING_TRENDING', context_performance)
        self.assertIn('RANGING_RANGING', context_performance)
        
        # Check that TRENDING_TRENDING performed better than RANGING_RANGING
        trending_perf = context_performance['TRENDING_TRENDING']
        ranging_perf = context_performance['RANGING_RANGING']
        
        self.assertGreater(trending_perf['avg_pnl'], ranging_perf['avg_pnl'])
        
        print("✓ Market context analysis completed:")
        for context, perf in context_performance.items():
            print(f"  {context}: {perf['trade_count']} trades, avg PnL: {perf['avg_pnl']:.1f}")
    
    def test_session_management(self):
        """Test session management features"""
        print("\n=== Testing Session Management ===")
        
        # Check session ID was generated
        self.assertIsNotNone(self.tracker.session_id)
        self.assertTrue(self.tracker.session_id.startswith('session_'))
        
        # Add some trades
        trade = {
            'strategy': 'TestStrategy', 'symbol': 'BTCUSDT', 'side': 'buy',
            'entry_price': 45000, 'exit_price': 45500, 'size': 0.1, 'pnl': 50,
            'entry_timestamp': '2024-01-15T10:00:00Z'
        }
        self.tracker.record_trade(trade)
        
        # Close session and check files are created
        self.tracker.close_session()
        
        # Check that session files were created
        files = os.listdir(self.tmp_dir)
        csv_files = [f for f in files if f.endswith('.csv')]
        json_files = [f for f in files if f.endswith('.json')]
        summary_files = [f for f in files if f.startswith('session_summary_')]
        
        self.assertGreater(len(csv_files), 0)
        self.assertGreater(len(json_files), 0)
        self.assertGreater(len(summary_files), 0)
        
        print(f"✓ Session ID: {self.tracker.session_id}")
        print(f"✓ Files created: {len(files)} total")
        print(f"  - CSV files: {len(csv_files)}")
        print(f"  - JSON files: {len(json_files)}")
        print(f"  - Summary files: {len(summary_files)}")
    
    def test_dataframe_export(self):
        """Test enhanced DataFrame export functionality"""
        print("\n=== Testing DataFrame Export ===")
        
        # Add a trade with full enhanced data
        trade = {
            'strategy': 'StrategyATRMomentumBreakout',
            'symbol': 'BTCUSDT',
            'side': 'buy',
            'entry_price': 45000,
            'exit_price': 46350,
            'size': 0.1,
            'pnl': 135,
            'entry_timestamp': '2024-01-15T10:00:00Z',
            'exit_timestamp': '2024-01-15T11:30:00Z',
            'market_context': {
                'market_5m': 'TRENDING',
                'market_1m': 'HIGH_VOLATILITY',
                'strategy_selection_reason': 'Optimal for trending+high vol',
                'execution_timeframe': '1m'
            },
            'order_details': {
                'main_order_id': 'order_123',
                'retry_attempts': 0,
                'slippage_pct': 0.01,
                'execution_quality_score': 95.0
            },
            'risk_metrics': {
                'leverage_used': 50,
                'position_size_pct': 5.0,
                'risk_reward_ratio': 2.5
            }
        }
        
        self.tracker.record_trade(trade)
        
        # Export to DataFrame
        df = self.tracker.to_dataframe()
        
        # Verify DataFrame structure
        self.assertEqual(len(df), 1)
        self.assertIn('strategy', df.columns)
        self.assertIn('market_market_5m', df.columns)  # Flattened market context
        self.assertIn('order_main_order_id', df.columns)  # Flattened order details
        self.assertIn('risk_leverage_used', df.columns)  # Flattened risk metrics
        
        # Check timestamp conversion
        self.assertTrue(df['entry_timestamp'].dtype.name.startswith('datetime'))
        
        print(f"✓ DataFrame exported with {len(df.columns)} columns")
        print(f"✓ Market context columns: {[col for col in df.columns if col.startswith('market_')]}")
        print(f"✓ Order detail columns: {[col for col in df.columns if col.startswith('order_')]}")
        print(f"✓ Risk metric columns: {[col for col in df.columns if col.startswith('risk_')]}")

def run_all_tests():
    """Run all enhanced performance tracker tests"""
    print("=" * 80)
    print("ENHANCED PERFORMANCE TRACKER TESTS")
    print("=" * 80)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnhancedPerformanceTracker)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("✅ ALL ENHANCED TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)

if __name__ == '__main__':
    run_all_tests() 