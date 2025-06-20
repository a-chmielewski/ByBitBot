import unittest
import tempfile
import shutil
import os
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, MagicMock, patch
import sys
import numpy as np

# Add the parent directory to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.session_manager import (
    SessionManager, SessionMetadata, SessionSummary, SessionStatus
)
from modules.performance_tracker import PerformanceTracker
from modules.real_time_monitor import RealTimeMonitor
from modules.advanced_risk_manager import AdvancedRiskManager

class TestSessionManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock components
        self.mock_performance_tracker = Mock(spec=PerformanceTracker)
        self.mock_real_time_monitor = Mock(spec=RealTimeMonitor)
        self.mock_risk_manager = Mock(spec=AdvancedRiskManager)
        
        # Mock performance tracker methods
        self.mock_performance_tracker.get_comprehensive_statistics.return_value = {
            'total_trades': 10,
            'win_rate': 65.0,
            'cumulative_pnl': 150.0,
            'max_drawdown': 25.0,
            'profit_factor': 1.8,
            'expectancy': 15.0,
            'consecutive_wins': 3,
            'consecutive_losses': 1
        }
        
        # Mock real-time monitor
        self.mock_real_time_monitor.alert_history = []
        self.mock_real_time_monitor.active_alerts = []
        
        # Mock risk manager
        self.mock_risk_manager.get_risk_summary.return_value = {
            'portfolio_risk': {'total_exposure_pct': 15.0},
            'risk_level': 'moderate'
        }
    
    def create_fresh_session_manager(self):
        """Create a fresh session manager for each test"""
        temp_dir = tempfile.mkdtemp()
        return SessionManager(base_dir=temp_dir), temp_dir
    
    def tearDown(self):
        """Clean up test fixtures"""
        pass
    
    def test_session_creation(self):
        """Test session creation functionality"""
        print("\n=== Testing Session Creation ===")
        
        session_manager, temp_dir = self.create_fresh_session_manager()
        
        try:
            configuration = {
                'risk_pct': 2.0,
                'max_positions': 5,
                'alert_thresholds': {'max_drawdown_pct': 5.0}
            }
            
            market_conditions = {
                'market_5m': 'TRENDING',
                'market_1m': 'RANGING',
                'volatility': 'MEDIUM'
            }
            
            session_id = session_manager.create_session(
                strategy_name="EMA_Crossover",
                symbol="BTCUSDT",
                timeframe="1m",
                leverage=10.0,
                configuration=configuration,
                market_conditions=market_conditions,
                initial_balance=10000.0,
                notes="Test trading session",
                tags=["test", "ema_strategy"]
            )
            
            # Verify session was created
            self.assertIsNotNone(session_id)
            self.assertTrue(session_id.startswith("session_"))
            self.assertIn("EMA_Crossover", session_id)
            
            # Verify session is in active sessions
            active_sessions = session_manager.get_active_sessions()
            self.assertIn(session_id, active_sessions)
            
            # Verify session metadata
            metadata = active_sessions[session_id]
            self.assertEqual(metadata.strategy_name, "EMA_Crossover")
            self.assertEqual(metadata.symbol, "BTCUSDT")
            self.assertEqual(metadata.timeframe, "1m")
            self.assertEqual(metadata.leverage, 10.0)
            self.assertEqual(metadata.initial_balance, 10000.0)
            self.assertEqual(metadata.status, SessionStatus.ACTIVE)
            
            # Verify current session ID is set
            self.assertEqual(session_manager.get_current_session_id(), session_id)
            
            print(f"✓ Session created successfully: {session_id}")
            print(f"✓ Session metadata validated")
            print(f"✓ Current session ID set correctly")
        finally:
            shutil.rmtree(temp_dir)
    
    def test_session_lifecycle(self):
        """Test complete session lifecycle"""
        print("\n=== Testing Session Lifecycle ===")
        
        # Create session
        session_id = self.session_manager.create_session(
            strategy_name="Test_Strategy",
            symbol="ETHUSDT",
            timeframe="5m",
            leverage=5.0,
            configuration={'risk_pct': 1.5},
            initial_balance=5000.0
        )
        
        # Test pause/resume
        self.session_manager.pause_session(session_id)
        active_sessions = self.session_manager.get_active_sessions()
        self.assertEqual(active_sessions[session_id].status, SessionStatus.PAUSED)
        print("✓ Session paused successfully")
        
        self.session_manager.resume_session(session_id)
        active_sessions = self.session_manager.get_active_sessions()
        self.assertEqual(active_sessions[session_id].status, SessionStatus.ACTIVE)
        print("✓ Session resumed successfully")
        
        # End session
        session_summary = self.session_manager.end_session(
            session_id,
            final_balance=5150.0,
            performance_tracker=self.mock_performance_tracker,
            real_time_monitor=self.mock_real_time_monitor,
            risk_manager=self.mock_risk_manager
        )
        
        # Verify session ended
        self.assertNotIn(session_id, self.session_manager.get_active_sessions())
        self.assertIn(session_id, self.session_manager.get_session_history())
        self.assertEqual(session_summary.metadata.status, SessionStatus.COMPLETED)
        self.assertEqual(session_summary.metadata.final_balance, 5150.0)
        
        print("✓ Session ended successfully")
        print(f"✓ Session summary generated: {session_summary.total_trades} trades")
        
        # Verify session files were archived
        archive_dir = self.session_manager.archive_dir / session_id
        self.assertTrue(archive_dir.exists())
        self.assertTrue((archive_dir / "summary.json").exists())
        print("✓ Session archived successfully")
    
    def test_multi_session_management(self):
        """Test managing multiple sessions"""
        print("\n=== Testing Multi-Session Management ===")
        
        # Clear any existing history for this test
        self.session_manager.session_history.clear()
        
        session_ids = []
        
        # Create multiple sessions
        for i in range(3):
            session_id = self.session_manager.create_session(
                strategy_name=f"Strategy_{i+1}",
                symbol="BTCUSDT",
                timeframe="1m",
                leverage=float(10 + i * 5),
                configuration={'risk_pct': 2.0 + i * 0.5},
                initial_balance=10000.0 + i * 1000
            )
            session_ids.append(session_id)
        
        # Verify all sessions are active
        active_sessions = self.session_manager.get_active_sessions()
        self.assertEqual(len(active_sessions), 3)
        
        print(f"✓ Created {len(session_ids)} sessions")
        
        # End all sessions with different performance
        for i, session_id in enumerate(session_ids):
            # Mock different performance for each session
            self.mock_performance_tracker.get_comprehensive_statistics.return_value = {
                'total_trades': 5 + i * 3,
                'win_rate': 50.0 + i * 10,
                'cumulative_pnl': 100.0 + i * 50,
                'max_drawdown': 20.0 - i * 5,
                'profit_factor': 1.2 + i * 0.3,
                'expectancy': 10.0 + i * 5
            }
            
            self.session_manager.end_session(
                session_id,
                final_balance=10000.0 + i * 200,
                performance_tracker=self.mock_performance_tracker,
                real_time_monitor=self.mock_real_time_monitor,
                risk_manager=self.mock_risk_manager
            )
        
        # Verify all sessions are in history
        session_history = self.session_manager.get_session_history()
        self.assertEqual(len(session_history), 3)
        
        print(f"✓ All {len(session_ids)} sessions completed and archived")
        
        # Test session comparison
        comparison = self.session_manager.get_session_comparison(session_ids)
        
        self.assertIn('sessions', comparison)
        self.assertIn('comparative_metrics', comparison)
        self.assertIn('performance_ranking', comparison)
        self.assertIn('insights', comparison)
        
        # Verify comparative metrics
        self.assertIn('total_trades', comparison['comparative_metrics'])
        self.assertIn('win_rate', comparison['comparative_metrics'])
        self.assertIn('final_pnl', comparison['comparative_metrics'])
        
        print("✓ Session comparison generated successfully")
        print(f"✓ Comparative metrics: {list(comparison['comparative_metrics'].keys())}")
    
    def test_cross_session_analysis(self):
        """Test cross-session analysis functionality"""
        print("\n=== Testing Cross-Session Analysis ===")
        
        # Clear any existing history for this test
        self.session_manager.session_history.clear()
        
        # Create and end multiple sessions with varying performance
        session_data = [
            {'strategy': 'EMA_Cross', 'pnl': 150, 'win_rate': 65, 'trades': 12},
            {'strategy': 'RSI_Mean', 'pnl': -50, 'win_rate': 45, 'trades': 8},
            {'strategy': 'EMA_Cross', 'pnl': 200, 'win_rate': 70, 'trades': 15},
            {'strategy': 'Bollinger', 'pnl': 100, 'win_rate': 60, 'trades': 10}
        ]
        
        session_ids = []
        for i, data in enumerate(session_data):
            session_id = self.session_manager.create_session(
                strategy_name=data['strategy'],
                symbol="BTCUSDT",
                timeframe="1m",
                leverage=10.0,
                configuration={'risk_pct': 2.0}
            )
            session_ids.append(session_id)
            
            # Mock performance data
            self.mock_performance_tracker.get_comprehensive_statistics.return_value = {
                'total_trades': data['trades'],
                'win_rate': data['win_rate'],
                'cumulative_pnl': data['pnl'],
                'max_drawdown': 25.0,
                'profit_factor': 1.5,
                'expectancy': data['pnl'] / data['trades']
            }
            
            self.session_manager.end_session(
                session_id,
                final_balance=10000 + data['pnl'],
                performance_tracker=self.mock_performance_tracker,
                real_time_monitor=self.mock_real_time_monitor,
                risk_manager=self.mock_risk_manager
            )
        
        # Test cross-session analysis
        analysis = self.session_manager.get_cross_session_analysis()
        
        self.assertIn('summary', analysis)
        self.assertIn('strategy_performance', analysis)
        self.assertIn('configuration_impact', analysis)
        self.assertIn('learning_curve', analysis)
        self.assertIn('optimization_insights', analysis)
        
        # Verify summary
        summary = analysis['summary']
        self.assertEqual(summary['total_sessions'], 4)
        self.assertEqual(summary['total_trades'], sum(d['trades'] for d in session_data))
        
        # Verify strategy performance analysis
        strategy_perf = analysis['strategy_performance']
        self.assertIn('EMA_Cross', strategy_perf)
        self.assertEqual(strategy_perf['EMA_Cross']['session_count'], 2)  # EMA_Cross used twice
        
        print("✓ Cross-session analysis completed successfully")
        print(f"✓ Analyzed {len(session_data)} sessions")
        print(f"✓ Strategy performance breakdown: {list(strategy_perf.keys())}")
        
        # Test historical trends
        trends = self.session_manager.get_historical_trends(period_days=30)
        
        self.assertIn('performance_trend', trends)
        self.assertIn('win_rate_trend', trends)
        self.assertIn('strategy_usage_evolution', trends)
        
        print("✓ Historical trends analysis completed")
    
    def test_session_data_export(self):
        """Test session data export functionality"""
        print("\n=== Testing Session Data Export ===")
        
        # Create and end a session
        session_id = self.session_manager.create_session(
            strategy_name="Export_Test",
            symbol="BTCUSDT",
            timeframe="1m",
            leverage=10.0,
            configuration={'risk_pct': 2.0}
        )
        
        self.session_manager.end_session(
            session_id,
            final_balance=10100.0,
            performance_tracker=self.mock_performance_tracker,
            real_time_monitor=self.mock_real_time_monitor,
            risk_manager=self.mock_risk_manager
        )
        
        # Test JSON export
        json_file = self.session_manager.export_session_data(format='json')
        self.assertTrue(os.path.exists(json_file))
        
        # Verify JSON content
        with open(json_file, 'r') as f:
            export_data = json.load(f)
        
        self.assertIn('sessions', export_data)
        self.assertIn('summary', export_data)
        self.assertIn(session_id, export_data['sessions'])
        
        print(f"✓ JSON export successful: {os.path.basename(json_file)}")
        
        # Test CSV export
        csv_file = self.session_manager.export_session_data(format='csv')
        self.assertTrue(os.path.exists(csv_file))
        
        print(f"✓ CSV export successful: {os.path.basename(csv_file)}")
    
    def test_session_persistence(self):
        """Test session persistence and recovery"""
        print("\n=== Testing Session Persistence ===")
        
        # Create session
        session_id = self.session_manager.create_session(
            strategy_name="Persistence_Test",
            symbol="ETHUSDT",
            timeframe="5m",
            leverage=15.0,
            configuration={'risk_pct': 3.0},
            notes="Persistence test session"
        )
        
        # End session
        self.session_manager.end_session(
            session_id,
            final_balance=10200.0,
            performance_tracker=self.mock_performance_tracker,
            real_time_monitor=self.mock_real_time_monitor,
            risk_manager=self.mock_risk_manager
        )
        
        # Create new SessionManager instance to test loading
        new_session_manager = SessionManager(base_dir=self.temp_dir)
        
        # Verify session was loaded from archive
        session_history = new_session_manager.get_session_history()
        self.assertIn(session_id, session_history)
        
        loaded_session = session_history[session_id]
        self.assertEqual(loaded_session.metadata.strategy_name, "Persistence_Test")
        self.assertEqual(loaded_session.metadata.symbol, "ETHUSDT")
        self.assertEqual(loaded_session.metadata.final_balance, 10200.0)
        
        print("✓ Session persistence verified")
        print(f"✓ Loaded session: {loaded_session.metadata.session_id}")
    
    def test_error_handling(self):
        """Test error handling in session management"""
        print("\n=== Testing Error Handling ===")
        
        # Test ending non-existent session
        with self.assertRaises(ValueError):
            self.session_manager.end_session("non_existent_session")
        
        print("✓ Non-existent session error handling verified")
        
        # Test session comparison with invalid IDs
        comparison = self.session_manager.get_session_comparison(["invalid_id"])
        self.assertEqual(comparison['error'], 'No valid sessions found')
        
        print("✓ Invalid session comparison error handling verified")
        
        # Test cross-session analysis with no sessions
        empty_session_manager = SessionManager(base_dir=tempfile.mkdtemp())
        analysis = empty_session_manager.get_cross_session_analysis()
        self.assertIn('summary', analysis)
        
        print("✓ Empty session manager error handling verified")
    
    def test_performance_insights(self):
        """Test performance insights generation"""
        print("\n=== Testing Performance Insights ===")
        
        # Create sessions with different leverage values
        leverage_values = [5.0, 10.0, 15.0, 20.0]
        session_ids = []
        
        for i, leverage in enumerate(leverage_values):
            session_id = self.session_manager.create_session(
                strategy_name="Leverage_Test",
                symbol="BTCUSDT",
                timeframe="1m",
                leverage=leverage,
                configuration={'risk_pct': 2.0}
            )
            session_ids.append(session_id)
            
            # Mock performance decreasing with higher leverage
            pnl = 200 - (leverage * 5)  # Higher leverage = lower PnL
            
            self.mock_performance_tracker.get_comprehensive_statistics.return_value = {
                'total_trades': 10,
                'win_rate': 70 - leverage,  # Higher leverage = lower win rate
                'cumulative_pnl': pnl,
                'max_drawdown': leverage * 2,  # Higher leverage = higher drawdown
                'profit_factor': 2.0 - (leverage * 0.05),
                'expectancy': pnl / 10
            }
            
            self.session_manager.end_session(
                session_id,
                final_balance=10000 + pnl,
                performance_tracker=self.mock_performance_tracker,
                real_time_monitor=self.mock_real_time_monitor,
                risk_manager=self.mock_risk_manager
            )
        
        # Get cross-session analysis with insights
        analysis = self.session_manager.get_cross_session_analysis()
        
        # Verify optimization insights
        insights = analysis['optimization_insights']
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)
        
        # Verify configuration impact analysis
        config_impact = analysis['configuration_impact']
        self.assertIn('leverage_analysis', config_impact)
        
        leverage_analysis = config_impact['leverage_analysis']
        self.assertIn('5.0x', leverage_analysis)
        self.assertIn('20.0x', leverage_analysis)
        
        # Best leverage should be 5.0x (lowest, highest PnL)
        best_leverage_pnl = max(leverage_analysis.values(), key=lambda x: x['avg_pnl'])
        print(f"✓ Performance insights generated")
        print(f"✓ Configuration impact analysis completed")
        print(f"✓ Optimization insights: {len(insights)} insights generated")
    
    def run_all_tests(self):
        """Run all tests in sequence"""
        test_methods = [
            self.test_session_creation,
            self.test_session_lifecycle,
            self.test_multi_session_management,
            self.test_cross_session_analysis,
            self.test_session_data_export,
            self.test_session_persistence,
            self.test_error_handling,
            self.test_performance_insights
        ]
        
        print("🚀 Starting SessionManager Comprehensive Test Suite")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for test_method in test_methods:
            try:
                test_method()
                passed += 1
                print(f"✅ {test_method.__name__} PASSED")
            except Exception as e:
                failed += 1
                print(f"❌ {test_method.__name__} FAILED: {str(e)}")
        
        print("=" * 60)
        print(f"📊 Test Results: {passed} passed, {failed} failed")
        
        if failed == 0:
            print("🎉 All SessionManager tests passed successfully!")
        else:
            print(f"⚠️  {failed} test(s) failed. Please review the issues.")
        
        return failed == 0

if __name__ == '__main__':
    # Run comprehensive test suite
    test_suite = TestSessionManager()
    test_suite.setUp()
    
    try:
        success = test_suite.run_all_tests()
        if success:
            print("\n✅ SessionManager implementation is working correctly!")
        else:
            print("\n❌ SessionManager implementation has issues that need to be fixed.")
    finally:
        test_suite.tearDown() 