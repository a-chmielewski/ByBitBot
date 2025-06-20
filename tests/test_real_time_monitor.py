import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone

from modules.real_time_monitor import (
    RealTimeMonitor, 
    Alert, 
    AlertSeverity, 
    AlertChannel,
    RealTimeMetrics,
    ConsoleFormatter
)
from modules.performance_tracker import PerformanceTracker

class TestConsoleFormatter:
    """Test console formatting utilities"""
    
    def test_format_currency_positive(self):
        """Test currency formatting for positive values"""
        result = ConsoleFormatter.format_currency(100.50, colored=False)
        assert result == "$100.50"
    
    def test_format_currency_negative(self):
        """Test currency formatting for negative values"""
        result = ConsoleFormatter.format_currency(-50.25, colored=False)
        assert result == "$-50.25"
    
    def test_format_currency_zero(self):
        """Test currency formatting for zero value"""
        result = ConsoleFormatter.format_currency(0.0, colored=False)
        assert result == "$0.00"
    
    def test_format_percentage(self):
        """Test percentage formatting"""
        result = ConsoleFormatter.format_percentage(75.5, colored=False)
        assert result == "75.50%"
    
    def test_format_alert(self):
        """Test alert formatting"""
        alert = Alert(
            id="test_alert",
            type="TEST",
            severity=AlertSeverity.WARNING,
            message="Test alert message",
            value=10.0,
            threshold=5.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id="test_session"
        )
        
        result = ConsoleFormatter.format_alert(alert, colored=False)
        assert "[WARNING] Test alert message" in result

class TestRealTimeMonitor:
    """Test RealTimeMonitor functionality"""
    
    @pytest.fixture
    def mock_performance_tracker(self):
        """Create a mock PerformanceTracker"""
        tracker = Mock(spec=PerformanceTracker)
        tracker.get_comprehensive_statistics.return_value = {
            'session_id': 'test_session_123',
            'total_trades': 10,
            'win_rate': 60.0,
            'cumulative_pnl': 150.0,
            'max_drawdown': 25.0,
            'consecutive_wins': 2,
            'consecutive_losses': 0,
            'profit_factor': 1.5,
            'expectancy': 15.0,
            'avg_trade_duration': 3600.0,  # 1 hour in seconds
            'session_duration_hours': 2.0
        }
        tracker.high_watermark = 200.0
        tracker.cumulative_pnl = 150.0
        return tracker
    
    @pytest.fixture
    def real_time_monitor(self, mock_performance_tracker):
        """Create a RealTimeMonitor instance"""
        monitor = RealTimeMonitor(
            performance_tracker=mock_performance_tracker,
            alert_thresholds={
                'max_drawdown_pct': 5.0,
                'consecutive_losses': 3,
                'win_rate_threshold': 40.0,
                'risk_limit_pct': 10.0,
                'profit_factor_min': 1.2,
                'expectancy_min': 0.0,
                'max_trade_duration_hours': 24.0,
                'min_trades_for_analysis': 5
            },
            alert_channels=[AlertChannel.CONSOLE, AlertChannel.LOG],
            dashboard_update_interval=0.1  # Fast updates for testing
        )
        yield monitor
        # Cleanup: ensure monitoring is stopped
        if monitor.dashboard_thread:
            monitor.stop_monitoring()
    
    def test_initialization(self, real_time_monitor):
        """Test RealTimeMonitor initialization"""
        assert real_time_monitor.dashboard_enabled is True
        assert real_time_monitor.alert_thresholds['max_drawdown_pct'] == 5.0
        assert len(real_time_monitor.alert_channels) == 2
        assert real_time_monitor.current_metrics.total_trades == 0
    
    def test_update_metrics(self, real_time_monitor):
        """Test metrics update from PerformanceTracker"""
        real_time_monitor.update_metrics()
        
        # Verify metrics were updated
        assert real_time_monitor.current_metrics.session_id == 'test_session_123'
        assert real_time_monitor.current_metrics.total_trades == 10
        assert real_time_monitor.current_metrics.win_rate == 60.0
        assert real_time_monitor.current_metrics.cumulative_pnl == 150.0
        
        # Verify metrics are stored in history
        assert len(real_time_monitor.metrics_history) == 1
    
    def test_calculate_current_drawdown_pct(self, real_time_monitor):
        """Test drawdown percentage calculation"""
        real_time_monitor.update_metrics()
        
        # High watermark: 200.0, Current PnL: 150.0
        # Expected drawdown: (200.0 - 150.0) / 200.0 * 100 = 25%
        expected_drawdown = 25.0
        assert abs(real_time_monitor.current_metrics.current_drawdown - expected_drawdown) < 0.1
    
    def test_calculate_trades_per_hour(self, real_time_monitor):
        """Test trades per hour calculation"""
        real_time_monitor.update_metrics()
        
        # 10 trades in 2 hours = 5 trades/hour
        expected_rate = 5.0
        assert abs(real_time_monitor.current_metrics.trades_per_hour - expected_rate) < 0.1
    
    def test_drawdown_alert_generation(self, real_time_monitor):
        """Test that drawdown alerts are generated correctly"""
        # Set high drawdown to trigger alert
        real_time_monitor.performance_tracker.high_watermark = 100.0
        real_time_monitor.performance_tracker.cumulative_pnl = 85.0  # 15% drawdown
        
        real_time_monitor.update_metrics()
        
        # Should generate drawdown alert (15% > 5% threshold)
        drawdown_alerts = [a for a in real_time_monitor.active_alerts if a.type == "DRAWDOWN_CRITICAL"]
        assert len(drawdown_alerts) > 0
        assert drawdown_alerts[0].severity == AlertSeverity.CRITICAL
    
    def test_consecutive_losses_alert(self, real_time_monitor):
        """Test consecutive losses alert generation"""
        # Update tracker stats to show consecutive losses
        real_time_monitor.performance_tracker.get_comprehensive_statistics.return_value.update({
            'consecutive_losses': 4  # Exceeds threshold of 3
        })
        
        real_time_monitor.update_metrics()
        
        # Should generate consecutive losses alert
        loss_alerts = [a for a in real_time_monitor.active_alerts if a.type == "CONSECUTIVE_LOSSES"]
        assert len(loss_alerts) > 0
        assert loss_alerts[0].severity == AlertSeverity.WARNING
    
    def test_win_rate_alert(self, real_time_monitor):
        """Test win rate alert generation"""
        # Update tracker stats to show low win rate
        real_time_monitor.performance_tracker.get_comprehensive_statistics.return_value.update({
            'win_rate': 30.0,  # Below 40% threshold
            'total_trades': 10  # Above minimum for analysis
        })
        
        real_time_monitor.update_metrics()
        
        # Should generate win rate alert
        win_rate_alerts = [a for a in real_time_monitor.active_alerts if a.type == "WIN_RATE_LOW"]
        assert len(win_rate_alerts) > 0
        assert win_rate_alerts[0].severity == AlertSeverity.WARNING
    
    def test_profit_factor_alert(self, real_time_monitor):
        """Test profit factor alert generation"""
        # Update tracker stats to show low profit factor
        real_time_monitor.performance_tracker.get_comprehensive_statistics.return_value.update({
            'profit_factor': 0.8,  # Below 1.2 threshold
            'total_trades': 10  # Above minimum for analysis
        })
        
        real_time_monitor.update_metrics()
        
        # Should generate profit factor alert
        pf_alerts = [a for a in real_time_monitor.active_alerts if a.type == "PROFIT_FACTOR_LOW"]
        assert len(pf_alerts) > 0
        assert pf_alerts[0].severity == AlertSeverity.WARNING
    
    def test_expectancy_alert(self, real_time_monitor):
        """Test expectancy alert generation"""
        # Update tracker stats to show negative expectancy
        real_time_monitor.performance_tracker.get_comprehensive_statistics.return_value.update({
            'expectancy': -10.0,  # Below 0.0 threshold
            'total_trades': 10  # Above minimum for analysis
        })
        
        real_time_monitor.update_metrics()
        
        # Should generate expectancy alert
        exp_alerts = [a for a in real_time_monitor.active_alerts if a.type == "EXPECTANCY_NEGATIVE"]
        assert len(exp_alerts) > 0
        assert exp_alerts[0].severity == AlertSeverity.CRITICAL
    
    def test_alert_duplicate_prevention(self, real_time_monitor):
        """Test that duplicate alerts are not generated within 5 minutes"""
        # Set conditions for drawdown alert
        real_time_monitor.performance_tracker.high_watermark = 100.0
        real_time_monitor.performance_tracker.cumulative_pnl = 85.0
        
        # Generate first alert
        real_time_monitor.update_metrics()
        initial_alert_count = len(real_time_monitor.active_alerts)
        
        # Try to generate same alert immediately
        real_time_monitor.update_metrics()
        
        # Should not create duplicate alert
        assert len(real_time_monitor.active_alerts) == initial_alert_count
    
    def test_alert_callback_execution(self, real_time_monitor):
        """Test that alert callbacks are executed"""
        callback_executed = []
        
        def test_callback(alert: Alert):
            callback_executed.append(alert.type)
        
        real_time_monitor.add_alert_callback(test_callback)
        
        # Trigger alert
        real_time_monitor.performance_tracker.high_watermark = 100.0
        real_time_monitor.performance_tracker.cumulative_pnl = 85.0
        real_time_monitor.update_metrics()
        
        # Verify callback was executed
        assert "DRAWDOWN_CRITICAL" in callback_executed
    
    def test_alert_acknowledgment(self, real_time_monitor):
        """Test alert acknowledgment functionality"""
        # Generate an alert
        real_time_monitor.performance_tracker.high_watermark = 100.0
        real_time_monitor.performance_tracker.cumulative_pnl = 85.0
        real_time_monitor.update_metrics()
        
        # Get the alert ID
        alert = real_time_monitor.active_alerts[0]
        assert not alert.acknowledged
        
        # Acknowledge the alert
        real_time_monitor.acknowledge_alert(alert.id)
        
        # Verify acknowledgment
        assert alert.acknowledged
    
    def test_set_current_strategy(self, real_time_monitor):
        """Test setting current strategy for dashboard display"""
        strategy_name = "TestStrategy"
        real_time_monitor.set_current_strategy(strategy_name)
        
        assert real_time_monitor.current_metrics.current_strategy == strategy_name
    
    def test_set_current_market_conditions(self, real_time_monitor):
        """Test setting current market conditions for dashboard display"""
        conditions = "TRENDING_HIGH_VOLATILITY"
        real_time_monitor.set_current_market_conditions(conditions)
        
        assert real_time_monitor.current_metrics.current_market_conditions == conditions
    
    def test_alert_threshold_updates(self, real_time_monitor):
        """Test updating alert thresholds"""
        new_thresholds = {'max_drawdown_pct': 10.0, 'consecutive_losses': 5}
        real_time_monitor.update_alert_thresholds(new_thresholds)
        
        assert real_time_monitor.alert_thresholds['max_drawdown_pct'] == 10.0
        assert real_time_monitor.alert_thresholds['consecutive_losses'] == 5
    
    def test_dashboard_enable_disable(self, real_time_monitor):
        """Test dashboard enable/disable functionality"""
        # Initially enabled
        assert real_time_monitor.dashboard_enabled is True
        
        # Disable dashboard
        real_time_monitor.disable_dashboard()
        assert real_time_monitor.dashboard_enabled is False
        
        # Re-enable dashboard
        real_time_monitor.enable_dashboard()
        assert real_time_monitor.dashboard_enabled is True
    
    def test_performance_summary(self, real_time_monitor):
        """Test performance summary generation"""
        real_time_monitor.update_metrics()
        summary = real_time_monitor.get_performance_summary()
        
        assert 'current_metrics' in summary
        assert 'active_alerts' in summary
        assert 'alert_thresholds' in summary
        assert 'monitoring_status' in summary
        
        # Verify metrics are included
        assert summary['current_metrics']['total_trades'] == 10
        assert summary['current_metrics']['win_rate'] == 60.0
    
    def test_performance_trend_calculation(self, real_time_monitor):
        """Test performance trend calculation"""
        # Add multiple metrics snapshots to history
        for i in range(15):
            real_time_monitor.performance_tracker.get_comprehensive_statistics.return_value.update({
                'cumulative_pnl': 100.0 + (i * 10.0)  # Increasing trend
            })
            real_time_monitor.update_metrics()
        
        # Calculate trend
        trend = real_time_monitor._calculate_performance_trend()
        assert trend > 0  # Should be positive trend
    
    def test_metrics_history_size_limit(self, real_time_monitor):
        """Test that metrics history is limited to prevent memory issues"""
        # Set a small history limit for testing
        real_time_monitor.max_history_size = 5
        
        # Add more metrics than the limit
        for i in range(10):
            real_time_monitor.update_metrics()
        
        # Should not exceed the limit
        assert len(real_time_monitor.metrics_history) <= 5
    
    def test_alert_history_size_limit(self, real_time_monitor):
        """Test that alert history is limited"""
        # Generate many alerts by changing thresholds
        for i in range(150):  # More than the 100 limit
            real_time_monitor.performance_tracker.high_watermark = 100.0
            real_time_monitor.performance_tracker.cumulative_pnl = 85.0 - i  # Vary to avoid duplicates
            alert = real_time_monitor._create_alert(
                alert_type=f"TEST_ALERT_{i}",
                severity=AlertSeverity.INFO,
                message=f"Test alert {i}",
                value=float(i),
                threshold=10.0
            )
            real_time_monitor._add_alert(alert)
        
        # Should not exceed the limit
        assert len(real_time_monitor.alert_history) <= 100
    
    @patch('modules.real_time_monitor.os.path.exists')
    @patch('builtins.open', create=True)
    def test_file_alert_delivery(self, mock_open, mock_exists, real_time_monitor):
        """Test alert delivery to file"""
        mock_exists.return_value = False
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Create and deliver an alert
        alert = Alert(
            id="test_alert_file",
            type="TEST",
            severity=AlertSeverity.WARNING,
            message="Test file alert",
            value=10.0,
            threshold=5.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id="test_session"
        )
        
        # Test file delivery directly
        real_time_monitor._deliver_file_alert(alert)
        
        # Verify file operations were called
        mock_open.assert_called()
    
    def test_monitoring_thread_lifecycle(self, real_time_monitor):
        """Test starting and stopping monitoring thread"""
        try:
            # Start monitoring
            real_time_monitor.start_monitoring()
            assert real_time_monitor.dashboard_thread is not None
            assert real_time_monitor.dashboard_thread.is_alive()
            
            # Stop monitoring
            real_time_monitor.stop_monitoring()
            assert real_time_monitor.dashboard_thread is None
        finally:
            # Ensure cleanup even if test fails
            if real_time_monitor.dashboard_thread:
                real_time_monitor.stop_monitoring()
    
    def test_no_performance_tracker_handling(self):
        """Test behavior when no PerformanceTracker is provided"""
        monitor = RealTimeMonitor(performance_tracker=None)
        
        # Should not crash when updating metrics
        monitor.update_metrics()
        
        # Metrics should remain at defaults
        assert monitor.current_metrics.total_trades == 0
        assert monitor.current_metrics.cumulative_pnl == 0.0
    
    def test_error_handling_in_update_metrics(self, real_time_monitor):
        """Test error handling during metrics update"""
        # Make the performance tracker raise an exception
        real_time_monitor.performance_tracker.get_comprehensive_statistics.side_effect = Exception("Test error")
        
        # Should not crash
        real_time_monitor.update_metrics()
        
        # Metrics should remain unchanged
        assert real_time_monitor.current_metrics.total_trades == 0

if __name__ == "__main__":
    pytest.main([__file__]) 