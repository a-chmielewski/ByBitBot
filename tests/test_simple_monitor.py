"""
Simplified tests for RealTimeMonitor - avoiding threading issues
"""
import pytest
from unittest.mock import Mock
from datetime import datetime, timezone

from modules.real_time_monitor import (
    RealTimeMonitor, 
    Alert, 
    AlertSeverity, 
    AlertChannel,
    RealTimeMetrics,
    ConsoleFormatter
)

class TestSimpleRealTimeMonitor:
    """Simplified tests for RealTimeMonitor"""
    
    def test_basic_initialization(self):
        """Test basic initialization without threading"""
        monitor = RealTimeMonitor(
            performance_tracker=None,
            dashboard_update_interval=0.1
        )
        assert monitor.dashboard_enabled is True
        assert monitor.current_metrics.total_trades == 0
    
    def test_console_formatter(self):
        """Test console formatting utilities"""
        # Test currency formatting
        result = ConsoleFormatter.format_currency(100.50, colored=False)
        assert result == "$100.50"
        
        # Test percentage formatting
        result = ConsoleFormatter.format_percentage(75.5, colored=False)
        assert result == "75.50%"
    
    def test_alert_creation(self):
        """Test alert creation"""
        monitor = RealTimeMonitor(performance_tracker=None)
        
        alert = monitor._create_alert(
            alert_type="TEST",
            severity=AlertSeverity.WARNING,
            message="Test alert",
            value=10.0,
            threshold=5.0
        )
        
        assert alert.type == "TEST"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Test alert"
        assert alert.value == 10.0
        assert alert.threshold == 5.0
    
    def test_metrics_update_no_tracker(self):
        """Test metrics update when no performance tracker is provided"""
        monitor = RealTimeMonitor(performance_tracker=None)
        
        # Should not crash
        monitor.update_metrics()
        
        # Metrics should remain at defaults
        assert monitor.current_metrics.total_trades == 0
        assert monitor.current_metrics.cumulative_pnl == 0.0
    
    def test_alert_thresholds_update(self):
        """Test updating alert thresholds"""
        monitor = RealTimeMonitor(performance_tracker=None)
        
        new_thresholds = {'max_drawdown_pct': 10.0, 'consecutive_losses': 5}
        monitor.update_alert_thresholds(new_thresholds)
        
        assert monitor.alert_thresholds['max_drawdown_pct'] == 10.0
        assert monitor.alert_thresholds['consecutive_losses'] == 5
    
    def test_strategy_and_market_setting(self):
        """Test setting current strategy and market conditions"""
        monitor = RealTimeMonitor(performance_tracker=None)
        
        monitor.set_current_strategy("TestStrategy")
        assert monitor.current_metrics.current_strategy == "TestStrategy"
        
        monitor.set_current_market_conditions("TRENDING_HIGH_VOLATILITY")
        assert monitor.current_metrics.current_market_conditions == "TRENDING_HIGH_VOLATILITY"
    
    def test_dashboard_enable_disable(self):
        """Test dashboard enable/disable functionality"""
        monitor = RealTimeMonitor(performance_tracker=None)
        
        # Initially enabled
        assert monitor.dashboard_enabled is True
        
        # Disable dashboard
        monitor.disable_dashboard()
        assert monitor.dashboard_enabled is False
        
        # Re-enable dashboard
        monitor.enable_dashboard()
        assert monitor.dashboard_enabled is True
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        monitor = RealTimeMonitor(performance_tracker=None)
        summary = monitor.get_performance_summary()
        
        assert 'current_metrics' in summary
        assert 'active_alerts' in summary
        assert 'alert_thresholds' in summary
        assert 'monitoring_status' in summary
    
    def test_alert_acknowledgment(self):
        """Test alert acknowledgment"""
        monitor = RealTimeMonitor(performance_tracker=None)
        
        # Create an alert
        alert = monitor._create_alert(
            alert_type="TEST",
            severity=AlertSeverity.INFO,
            message="Test alert",
            value=1.0,
            threshold=0.5
        )
        
        monitor._add_alert(alert)
        
        # Initially not acknowledged
        assert not alert.acknowledged
        
        # Acknowledge the alert
        monitor.acknowledge_alert(alert.id)
        
        # Should be acknowledged now
        assert alert.acknowledged

if __name__ == "__main__":
    # Simple test runner
    import sys
    
    test_class = TestSimpleRealTimeMonitor()
    methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    passed = 0
    failed = 0
    
    for method_name in methods:
        try:
            print(f"Running {method_name}...")
            method = getattr(test_class, method_name)
            method()
            print(f"  ✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1) 