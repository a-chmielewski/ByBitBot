#!/usr/bin/env python3
"""
Test script to verify module integration and initialization.
This script tests that all enhanced modules can be imported and work together.
"""

import sys
import os
import logging
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_module_imports():
    """Test that all enhanced modules can be imported."""
    print("Testing module imports...")
    
    try:
        from modules.performance_tracker import PerformanceTracker
        print("✅ PerformanceTracker imported successfully")
        
        from modules.session_manager import SessionManager
        print("✅ SessionManager imported successfully")
        
        from modules.advanced_risk_manager import AdvancedRiskManager
        print("✅ AdvancedRiskManager imported successfully")
        
        from modules.real_time_monitor import RealTimeMonitor
        print("✅ RealTimeMonitor imported successfully")
        
        from modules.logger import get_logger
        print("✅ Logger imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_module_initialization():
    """Test that all modules can be initialized together."""
    print("\nTesting module initialization...")
    
    try:
        from modules.performance_tracker import PerformanceTracker
        from modules.session_manager import SessionManager
        from modules.advanced_risk_manager import AdvancedRiskManager
        from modules.real_time_monitor import RealTimeMonitor
        from modules.logger import get_logger
        
        # Create a test logger
        logger = get_logger('integration_test')
        print("✅ Logger initialized")
        
        # Initialize PerformanceTracker first (base dependency)
        perf_tracker = PerformanceTracker(logger=logger)
        print("✅ PerformanceTracker initialized")
        
        # Initialize SessionManager
        session_manager = SessionManager(
            base_dir="test_sessions",
            logger=logger
        )
        print("✅ SessionManager initialized")
        
        # Initialize AdvancedRiskManager (mock exchange for testing)
        class MockExchange:
            def get_account_balance(self):
                return {'result': {'USDT': {'walletBalance': '1000.0'}}}
            
            def fetch_positions(self, symbol, category):
                return {'result': {'list': []}}
                
        mock_exchange = MockExchange()
        risk_manager = AdvancedRiskManager(
            exchange=mock_exchange,
            performance_tracker=perf_tracker,
            logger=logger
        )
        print("✅ AdvancedRiskManager initialized")
        
        # Initialize RealTimeMonitor (depends on PerformanceTracker + SessionManager)
        real_time_monitor = RealTimeMonitor(
            performance_tracker=perf_tracker,
            session_manager=session_manager,
            logger=logger
        )
        print("✅ RealTimeMonitor initialized")
        
        return True, {
            'perf_tracker': perf_tracker,
            'session_manager': session_manager,
            'risk_manager': risk_manager,
            'real_time_monitor': real_time_monitor
        }
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return False, {}

def test_basic_functionality(modules):
    """Test basic functionality of integrated modules."""
    print("\nTesting basic functionality...")
    
    try:
        perf_tracker = modules['perf_tracker']
        session_manager = modules['session_manager']
        risk_manager = modules['risk_manager']
        real_time_monitor = modules['real_time_monitor']
        
        # Test session creation
        session_id = session_manager.create_session(
            strategy_name="TestStrategy",
            symbol="BTCUSDT",
            timeframe="1m",
            leverage=10,
            market_conditions={"1m_market_type": "TRENDING"},
            configuration={"test": True}
        )
        print(f"✅ Session created: {session_id}")
        
        # Test risk assessment
        risk_assessment = risk_manager.validate_trade_risk(
            symbol="BTCUSDT",
            side="buy",
            size=0.01,
            entry_price=50000,
            stop_loss_price=49000,
            take_profit_price=51500,
            leverage=10.0
        )
        print(f"✅ Risk assessment completed: {risk_assessment.get('approved', True)}")
        
        # Test performance tracking
        metrics = perf_tracker.get_comprehensive_statistics()
        print(f"✅ Performance metrics retrieved: {len(metrics)} metrics")
        
        # Test real-time monitor start/stop
        real_time_monitor.start_monitoring()
        print("✅ RealTimeMonitor started")
        
        real_time_monitor.stop_monitoring()
        print("✅ RealTimeMonitor stopped")
        
        # Test session closure
        session_manager.end_active_sessions("Integration test complete")
        print("✅ Sessions ended successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("MODULE INTEGRATION TEST")
    print("=" * 60)
    
    # Test imports
    if not test_module_imports():
        print("\n❌ Module import test failed")
        return False
    
    # Test initialization
    success, modules = test_module_initialization()
    if not success:
        print("\n❌ Module initialization test failed")
        return False
    
    # Test basic functionality
    if not test_basic_functionality(modules):
        print("\n❌ Basic functionality test failed")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL INTEGRATION TESTS PASSED")
    print("=" * 60)
    print("\nAll enhanced modules are properly integrated and functional!")
    print("The bot is ready to use the new features:")
    print("• Advanced Risk Management")
    print("• Real-Time Performance Monitoring")
    print("• Session Management & Analytics")
    print("• Enhanced Performance Tracking")
    
    return True

if __name__ == "__main__":
    main() 