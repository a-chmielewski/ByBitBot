import os
import sys
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
import numpy as np

try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init()
    COLORS_AVAILABLE = True
except ImportError:
    # Fallback if colorama not available
    COLORS_AVAILABLE = False
    class Fore:
        RED = GREEN = YELLOW = CYAN = MAGENTA = BLUE = WHITE = RESET = ""
    class Back:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertChannel(Enum):
    """Alert delivery channels"""
    CONSOLE = "console"
    LOG = "log"
    FILE = "file"
    WEBHOOK = "webhook"

@dataclass
class Alert:
    """Real-time alert data structure"""
    id: str
    type: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: str
    session_id: str
    acknowledged: bool = False
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RealTimeMetrics:
    """Real-time performance metrics snapshot"""
    timestamp: str
    session_id: str
    total_trades: int
    win_rate: float
    cumulative_pnl: float
    current_drawdown: float
    max_drawdown: float
    consecutive_wins: int
    consecutive_losses: int
    profit_factor: float
    expectancy: float
    avg_trade_duration: float
    active_positions: int
    session_duration_hours: float
    trades_per_hour: float
    current_strategy: Optional[str] = None
    current_market_conditions: Optional[str] = None
    current_symbol: Optional[str] = None

class ConsoleFormatter:
    """Console formatting utilities with color support"""
    
    @staticmethod
    def format_currency(value: float, colored: bool = True) -> str:
        """Format currency with colors"""
        if not colored or not COLORS_AVAILABLE:
            return f"${value:,.2f}"
        
        if value > 0:
            return f"{Fore.GREEN}${value:,.2f}{Style.RESET_ALL}"
        elif value < 0:
            return f"{Fore.RED}${value:,.2f}{Style.RESET_ALL}"
        else:
            return f"${value:,.2f}"
    
    @staticmethod
    def format_percentage(value: float, colored: bool = True) -> str:
        """Format percentage with colors"""
        if not colored or not COLORS_AVAILABLE:
            return f"{value:.2f}%"
        
        if value > 60:
            return f"{Fore.GREEN}{value:.2f}%{Style.RESET_ALL}"
        elif value > 40:
            return f"{Fore.YELLOW}{value:.2f}%{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}{value:.2f}%{Style.RESET_ALL}"
    
    @staticmethod
    def format_alert(alert: Alert, colored: bool = True) -> str:
        """Format alert with severity colors"""
        if not colored or not COLORS_AVAILABLE:
            return f"[{alert.severity.value.upper()}] {alert.message}"
        
        severity_colors = {
            AlertSeverity.INFO: Fore.CYAN,
            AlertSeverity.WARNING: Fore.YELLOW,
            AlertSeverity.CRITICAL: Fore.RED,
            AlertSeverity.EMERGENCY: Fore.MAGENTA + Style.BRIGHT
        }
        
        color = severity_colors.get(alert.severity, Fore.WHITE)
        return f"{color}[{alert.severity.value.upper()}]{Style.RESET_ALL} {alert.message}"
    
    @staticmethod
    def clear_screen():
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def move_cursor_up(lines: int):
        """Move cursor up N lines"""
        if os.name == 'nt':
            # Windows - use a different approach to avoid echo artifacts
            try:
                import subprocess
                for _ in range(lines):
                    subprocess.run(['powershell', '-Command', 'Write-Host "`e[1A" -NoNewline'], 
                                 capture_output=True, check=False)
            except:
                # Fallback - just clear screen on Windows
                ConsoleFormatter.clear_screen()
        else:
            # Unix/Linux/Mac
            sys.stdout.write(f'\033[{lines}A')
            sys.stdout.flush()

class RealTimeMonitor:
    """
    Real-time performance monitoring system with live console dashboard,
    advanced alerting, session management, and integration with PerformanceTracker.
    
    Features:
    - Live console dashboard with real-time updates
    - Multi-channel alert system (console, log, file, webhook)
    - Performance trend analysis and warnings
    - Strategy effectiveness monitoring
    - Risk limit monitoring
    - Market condition mismatch detection
    - Multi-session tracking and comparison
    - Cross-session performance analysis
    """
    
    def __init__(self, 
                 performance_tracker=None,
                 session_manager=None,
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 alert_channels: Optional[List[AlertChannel]] = None,
                 dashboard_update_interval: float = 1.0,
                 logger: Optional[logging.Logger] = None):
        
        self.performance_tracker = performance_tracker
        self.session_manager = session_manager
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Alert configuration
        self.alert_thresholds = alert_thresholds or {
            'max_drawdown_pct': 50.0,
            'consecutive_losses': 3,
            'win_rate_threshold': 40.0,
            'risk_limit_pct': 10.0,
            'profit_factor_min': 1.2,
            'expectancy_min': 0.0,
            'max_trade_duration_hours': 24.0,
            'min_trades_for_analysis': 5
        }
        
        self.alert_channels = alert_channels or [AlertChannel.CONSOLE, AlertChannel.LOG]
        
        # Active position tracking
        self.active_positions: Dict[str, Dict[str, Any]] = {}  # symbol -> position_info
        self.position_start_times: Dict[str, datetime] = {}  # symbol -> entry_time
        self.unrealized_pnl: Dict[str, float] = {}  # symbol -> unrealized_pnl
        
        # Strategy reference for position tracking
        self.tracked_strategies: List[Any] = []  # List of strategy instances to monitor
        
        # Real-time metrics
        self.current_metrics = RealTimeMetrics(
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id="",
            total_trades=0,
            win_rate=0.0,
            cumulative_pnl=0.0,
            current_drawdown=0.0,
            max_drawdown=0.0,
            consecutive_wins=0,
            consecutive_losses=0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_trade_duration=0.0,
            active_positions=0,
            session_duration_hours=0.0,
            trades_per_hour=0.0
        )
        
        # Alert management
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.last_alert_check = datetime.now(timezone.utc)
        
        # Dashboard configuration
        self.dashboard_enabled = True
        self.dashboard_update_interval = dashboard_update_interval
        self.adaptive_update_interval = True  # Adjust update frequency based on activity
        self.last_dashboard_update = None
        self.dashboard_thread = None
        self.stop_dashboard = threading.Event()
        self.last_trade_count = 0  # Track trade count changes
        
        # Performance tracking
        self.metrics_history: List[RealTimeMetrics] = []
        self.max_history_size = 1000  # Keep last 1000 metric snapshots
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Initialize
        self.session_start_time = datetime.now(timezone.utc)
        
        # Multi-session tracking
        self.session_metrics_history: Dict[str, List[RealTimeMetrics]] = {}
        self.cross_session_analysis_cache: Dict[str, Any] = {}
        self.session_comparison_data: Dict[str, Any] = {}
        
        self.logger.info("RealTimeMonitor initialized with multi-session support")
    
    def start_monitoring(self):
        """Start real-time monitoring and dashboard"""
        if self.dashboard_enabled and not self.dashboard_thread:
            self.dashboard_thread = threading.Thread(
                target=self._dashboard_loop,
                name="RealTimeMonitor-Dashboard",
                daemon=True
            )
            self.dashboard_thread.start()
            self.logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        if self.dashboard_thread:
            self.stop_dashboard.set()
            self.dashboard_thread.join(timeout=2.0)
            self.dashboard_thread = None
            self.logger.info("Real-time monitoring stopped")
    
    def update_metrics(self, force_update: bool = False):
        """Update real-time metrics from PerformanceTracker"""
        if not self.performance_tracker:
            return
        
        try:
            with self._lock:
                # Get comprehensive stats from PerformanceTracker
                stats = self.performance_tracker.get_comprehensive_statistics()
                
                # Handle case where there are no completed trades yet
                if 'error' in stats:
                    # Initialize default stats for when no trades exist yet
                    stats = {
                        'session_id': self.performance_tracker.session_id,
                        'total_trades': 0,
                        'win_rate': 0.0,
                        'cumulative_pnl': 0.0,
                        'max_drawdown': 0.0,
                        'consecutive_wins': 0,
                        'consecutive_losses': 0,
                        'profit_factor': 0.0,
                        'expectancy': 0.0,
                        'avg_trade_duration': 0.0,
                        'session_duration_hours': (datetime.now(timezone.utc) - self.performance_tracker.session_start_time).total_seconds() / 3600
                    }
                
                # Get active positions count and calculate total unrealized P&L
                active_positions_count = self._get_active_positions_count()
                total_unrealized_pnl = sum(self.unrealized_pnl.values())
                
                # Calculate total P&L including unrealized gains/losses
                realized_pnl = stats.get('cumulative_pnl', 0.0)
                total_pnl = realized_pnl + total_unrealized_pnl
                
                # Debug logging
                self.logger.debug(f"update_metrics: active_positions_count={active_positions_count}, "
                                f"total_unrealized_pnl={total_unrealized_pnl}, "
                                f"realized_pnl={realized_pnl}, total_pnl={total_pnl}")
                
                # Check if we have new trades for immediate dashboard update
                new_trades_detected = stats.get('total_trades', 0) > self.last_trade_count
                
                # Update current metrics
                self.current_metrics = RealTimeMetrics(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    session_id=stats.get('session_id', ''),
                    total_trades=stats.get('total_trades', 0),
                    win_rate=stats.get('win_rate', 0.0),
                    cumulative_pnl=total_pnl,  # Include unrealized P&L
                    current_drawdown=self._calculate_current_drawdown_pct(),
                    max_drawdown=stats.get('max_drawdown', 0.0),
                    consecutive_wins=stats.get('consecutive_wins', 0),
                    consecutive_losses=stats.get('consecutive_losses', 0),
                    profit_factor=stats.get('profit_factor', 0.0),
                    expectancy=stats.get('expectancy', 0.0),
                    avg_trade_duration=stats.get('avg_trade_duration', 0.0) / 3600.0,  # Convert to hours
                    active_positions=active_positions_count,
                    session_duration_hours=stats.get('session_duration_hours', 0.0),
                    trades_per_hour=self._calculate_trades_per_hour()
                )
                
                # Log when new trades are detected
                if new_trades_detected:
                    self.logger.info(f"📊 Dashboard detected new trade(s)! Total: {self.current_metrics.total_trades}, "
                                   f"New cumulative P&L: ${self.current_metrics.cumulative_pnl:.2f}")
                    self.last_trade_count = self.current_metrics.total_trades
                
                self.logger.debug(f"update_metrics: Updated current_metrics.active_positions={self.current_metrics.active_positions}, "
                                f"current_metrics.cumulative_pnl={self.current_metrics.cumulative_pnl}")
                
                # Store in history
                self.metrics_history.append(self.current_metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
                
                # Check alert conditions
                self._check_alert_conditions()
                
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}", exc_info=True)
    
    def _calculate_current_drawdown_pct(self) -> float:
        """Calculate current drawdown percentage"""
        if not self.performance_tracker:
            return 0.0
        
        high_watermark = self.performance_tracker.high_watermark
        current_pnl = self.performance_tracker.cumulative_pnl
        
        if high_watermark <= 0:
            return 0.0
        
        return ((high_watermark - current_pnl) / abs(high_watermark)) * 100
    
    def _get_active_positions_count(self) -> int:
        """Get count of active positions from tracked strategies"""
        try:
            active_count = 0
            current_positions = {}
            
            # Check all tracked strategies for active positions
            for strategy in self.tracked_strategies:
                if hasattr(strategy, 'position') and strategy.position:
                    for symbol, position_info in strategy.position.items():
                        if position_info and float(position_info.get('size', 0)) != 0:
                            active_count += 1
                            current_positions[symbol] = True  # Mark as currently active
                            
                            # Update our tracked positions
                            self.active_positions[symbol] = {
                                'strategy': type(strategy).__name__,
                                'symbol': symbol,
                                'side': position_info.get('side', 'unknown'),
                                'size': float(position_info.get('size', 0)),
                                'entry_price': float(position_info.get('entry_price', 0)),
                                'entry_time': position_info.get('timestamp', datetime.now(timezone.utc).isoformat()),
                                'unrealized_pnl': position_info.get('unrealized_pnl', 0),
                                'status': position_info.get('status', 'open')
                            }
                            
                            # Track position start time if not already tracked
                            if symbol not in self.position_start_times:
                                try:
                                    entry_time_str = position_info.get('timestamp', datetime.now(timezone.utc).isoformat())
                                    self.position_start_times[symbol] = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
                                except Exception:
                                    self.position_start_times[symbol] = datetime.now(timezone.utc)
            
            # Clean up positions that are no longer active
            symbols_to_remove = []
            for symbol in self.active_positions:
                if symbol not in current_positions:
                    # Position was closed
                    symbols_to_remove.append(symbol)
            
            for symbol in symbols_to_remove:
                self.active_positions.pop(symbol, None)
                self.position_start_times.pop(symbol, None)
                self.unrealized_pnl.pop(symbol, None)
            
            return active_count
            
        except Exception as e:
            self.logger.debug(f"Error getting active positions count: {e}")
            return len(self.active_positions)  # Fallback to cached count
    
    def _calculate_trades_per_hour(self) -> float:
        """Calculate trades per hour rate"""
        if not self.performance_tracker:
            return 0.0
        
        # Get session duration from comprehensive stats
        stats = self.performance_tracker.get_comprehensive_statistics()
        if 'error' in stats:
            return 0.0
            
        session_duration = stats.get('session_duration_hours', 0.0)
        total_trades = stats.get('total_trades', 0)
        
        if session_duration <= 0:
            return 0.0
        
        return total_trades / session_duration
    
    def _check_alert_conditions(self):
        """Check all alert conditions and generate alerts"""
        alerts_to_add = []
        
        # Drawdown alerts
        if self.current_metrics.current_drawdown > self.alert_thresholds['max_drawdown_pct']:
            alert = self._create_alert(
                alert_type="DRAWDOWN_CRITICAL",
                severity=AlertSeverity.CRITICAL,
                message=f"Critical drawdown reached: {self.current_metrics.current_drawdown:.2f}%",
                value=self.current_metrics.current_drawdown,
                threshold=self.alert_thresholds['max_drawdown_pct']
            )
            alerts_to_add.append(alert)
        
        # Consecutive losses
        if self.current_metrics.consecutive_losses >= self.alert_thresholds['consecutive_losses']:
            alert = self._create_alert(
                alert_type="CONSECUTIVE_LOSSES",
                severity=AlertSeverity.WARNING,
                message=f"Consecutive losses reached: {self.current_metrics.consecutive_losses}",
                value=self.current_metrics.consecutive_losses,
                threshold=self.alert_thresholds['consecutive_losses']
            )
            alerts_to_add.append(alert)
        
        # Win rate alerts (only if we have enough trades)
        if (self.current_metrics.total_trades >= self.alert_thresholds['min_trades_for_analysis'] and
            self.current_metrics.win_rate < self.alert_thresholds['win_rate_threshold']):
            alert = self._create_alert(
                alert_type="WIN_RATE_LOW",
                severity=AlertSeverity.WARNING,
                message=f"Win rate below threshold: {self.current_metrics.win_rate:.2f}%",
                value=self.current_metrics.win_rate,
                threshold=self.alert_thresholds['win_rate_threshold']
            )
            alerts_to_add.append(alert)
        
        # Profit factor alerts
        if (self.current_metrics.total_trades >= self.alert_thresholds['min_trades_for_analysis'] and
            self.current_metrics.profit_factor < self.alert_thresholds['profit_factor_min']):
            alert = self._create_alert(
                alert_type="PROFIT_FACTOR_LOW",
                severity=AlertSeverity.WARNING,
                message=f"Profit factor below minimum: {self.current_metrics.profit_factor:.2f}",
                value=self.current_metrics.profit_factor,
                threshold=self.alert_thresholds['profit_factor_min']
            )
            alerts_to_add.append(alert)
        
        # Expectancy alerts
        if (self.current_metrics.total_trades >= self.alert_thresholds['min_trades_for_analysis'] and
            self.current_metrics.expectancy < self.alert_thresholds['expectancy_min']):
            alert = self._create_alert(
                alert_type="EXPECTANCY_NEGATIVE",
                severity=AlertSeverity.CRITICAL,
                message=f"Negative expectancy detected: ${self.current_metrics.expectancy:.2f}",
                value=self.current_metrics.expectancy,
                threshold=self.alert_thresholds['expectancy_min']
            )
            alerts_to_add.append(alert)
        
        # Process new alerts
        for alert in alerts_to_add:
            self._add_alert(alert)
    
    def _create_alert(self, alert_type: str, severity: AlertSeverity, message: str, 
                     value: float, threshold: float, metadata: Optional[Dict] = None) -> Alert:
        """Create a new alert"""
        alert_id = f"{alert_type}_{int(datetime.now().timestamp())}"
        
        return Alert(
            id=alert_id,
            type=alert_type,
            severity=severity,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=datetime.now(timezone.utc).isoformat(),
            session_id=self.current_metrics.session_id,
            metadata=metadata or {}
        )
    
    def _add_alert(self, alert: Alert):
        """Add and process a new alert"""
        # Check if we already have a similar recent alert to avoid spam
        recent_similar = any(
            a.type == alert.type and 
            (datetime.fromisoformat(alert.timestamp.replace('Z', '+00:00')) - 
             datetime.fromisoformat(a.timestamp.replace('Z', '+00:00'))).total_seconds() < 300  # 5 minutes
            for a in self.active_alerts
        )
        
        if recent_similar:
            return
        
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Deliver alert through configured channels
        self._deliver_alert(alert)
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
        
        # Keep active alerts list manageable
        if len(self.active_alerts) > 20:
            self.active_alerts.pop(0)
        
        # Keep alert history manageable
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
    
    def _deliver_alert(self, alert: Alert):
        """Deliver alert through configured channels"""
        for channel in self.alert_channels:
            try:
                if channel == AlertChannel.CONSOLE:
                    self._deliver_console_alert(alert)
                elif channel == AlertChannel.LOG:
                    self._deliver_log_alert(alert)
                elif channel == AlertChannel.FILE:
                    self._deliver_file_alert(alert)
                # Webhook delivery would be implemented here
                
            except Exception as e:
                self.logger.error(f"Error delivering alert via {channel.value}: {e}")
    
    def _deliver_console_alert(self, alert: Alert):
        """Deliver alert to console"""
        formatted_alert = ConsoleFormatter.format_alert(alert)
        print(f"\n🚨 {formatted_alert}")
    
    def _deliver_log_alert(self, alert: Alert):
        """Deliver alert to logger"""
        log_methods = {
            AlertSeverity.INFO: self.logger.info,
            AlertSeverity.WARNING: self.logger.warning,
            AlertSeverity.CRITICAL: self.logger.error,
            AlertSeverity.EMERGENCY: self.logger.critical
        }
        
        log_method = log_methods.get(alert.severity, self.logger.info)
        log_method(f"ALERT [{alert.type}]: {alert.message}")
    
    def _deliver_file_alert(self, alert: Alert):
        """Deliver alert to file"""
        alert_file = "performance/alerts.json"
        os.makedirs(os.path.dirname(alert_file), exist_ok=True)
        
        try:
            # Load existing alerts
            alerts = []
            if os.path.exists(alert_file):
                with open(alert_file, 'r') as f:
                    alerts = json.load(f)
            
            # Add new alert
            alerts.append(asdict(alert))
            
            # Keep only last 500 alerts
            alerts = alerts[-500:]
            
            # Save back to file
            with open(alert_file, 'w') as f:
                json.dump(alerts, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving alert to file: {e}")
    
    def _dashboard_loop(self):
        """Main dashboard update loop with adaptive timing"""
        while not self.stop_dashboard.is_set():
            try:
                self.update_metrics()
                
                # Determine update interval based on activity
                current_interval = self._get_adaptive_update_interval()
                
                # Only update dashboard if there's meaningful activity or it's been a while
                should_update = self._should_update_dashboard()
                
                if self.dashboard_enabled and should_update:
                    self._update_console_dashboard()
                
                # Wait for next update
                self.stop_dashboard.wait(current_interval)
                
            except Exception as e:
                self.logger.error(f"Error in dashboard loop: {e}", exc_info=True)
                self.stop_dashboard.wait(5.0)  # Wait 5 seconds on error

    def _get_adaptive_update_interval(self) -> float:
        """Get adaptive update interval based on trading activity"""
        if not self.adaptive_update_interval:
            return self.dashboard_update_interval
        
        # Base intervals
        high_activity_interval = 1.0  # 1 second when trading
        medium_activity_interval = 3.0  # 3 seconds when positions are open
        low_activity_interval = 10.0  # 10 seconds when idle
        
        # Check if there's recent trading activity
        if self.current_metrics.total_trades > self.last_trade_count:
            # New trades - high frequency updates
            self.last_trade_count = self.current_metrics.total_trades
            return high_activity_interval
        elif self.current_metrics.active_positions > 0 or self.active_positions:
            # Active positions - medium frequency for real-time P&L
            return medium_activity_interval
        elif self.current_metrics.total_trades == 0:
            # No trades yet and no positions - moderate frequency
            return low_activity_interval  # Changed from 30.0 to 10.0
        else:
            # No new trades - moderate frequency
            return low_activity_interval

    def _should_update_dashboard(self) -> bool:
        """Determine if dashboard should be updated"""
        # Always update if there are new trades
        if self.current_metrics.total_trades > self.last_trade_count:
            return True
        
        # Update if there are active positions (to show real-time P&L)
        if self.current_metrics.active_positions > 0 or self.active_positions:
            return True
        
        # Update if there are active alerts
        if self.active_alerts:
            return True
        
        # Update every 30 seconds even if no activity (to show time progression)
        if not self.last_dashboard_update:
            return True
        
        time_since_last_update = (datetime.now() - self.last_dashboard_update).total_seconds()
        if time_since_last_update >= 30.0:
            return True
        
        return False
    
    def _update_console_dashboard(self):
        """Update the live console dashboard"""
        try:
            # Build dashboard content
            dashboard_lines = self._build_dashboard_content()
            
            # On Windows, just clear screen and redraw to avoid cursor issues
            if os.name == 'nt' and self.last_dashboard_update:
                ConsoleFormatter.clear_screen()
            elif self.last_dashboard_update and COLORS_AVAILABLE:
                # On Unix systems, try to move cursor up
                ConsoleFormatter.move_cursor_up(len(dashboard_lines))
            
            # Print new dashboard
            for line in dashboard_lines:
                print(line)
            
            self.last_dashboard_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating console dashboard: {e}")
    
    def _force_dashboard_update(self):
        """Force an immediate dashboard update (used when symbol/strategy changes)"""
        try:
            if self.dashboard_enabled:
                self.update_metrics(force_update=True)
                self._update_console_dashboard()
        except Exception as e:
            self.logger.error(f"Error in forced dashboard update: {e}")
    
    def _build_dashboard_content(self) -> List[str]:
        """Build the console dashboard content"""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"🚀 REAL-TIME TRADING PERFORMANCE MONITOR - {datetime.now().strftime('%H:%M:%S')}")
        lines.append("=" * 80)
        
        # Session info with symbol
        session_info = f"Session: {self.current_metrics.session_id[:16]}... | Duration: {self.current_metrics.session_duration_hours:.1f}h"
        if self.current_metrics.current_symbol:
            session_info += f" | Symbol: {self.current_metrics.current_symbol}"
        else:
            # Debug: Log when symbol is missing from display
            self.logger.debug(f"DEBUG: Symbol not displayed - current_symbol is: '{self.current_metrics.current_symbol}'")
        lines.append(session_info)
        lines.append("")
        
        # Key metrics row 1 - enhanced with active positions
        trades_info = f"Trades: {self.current_metrics.total_trades}"
        if self.current_metrics.active_positions > 0:
            trades_info += f" + {self.current_metrics.active_positions} open"
        trades_info += f" | Win Rate: {ConsoleFormatter.format_percentage(self.current_metrics.win_rate)}"
        pnl_info = f"P&L: {ConsoleFormatter.format_currency(self.current_metrics.cumulative_pnl)}"
        lines.append(f"{trades_info} | {pnl_info}")
        
        # Active positions details
        if self.active_positions:
            lines.append("")
            lines.append("📊 ACTIVE POSITIONS:")
            for symbol, pos in self.active_positions.items():
                side_emoji = "🟢" if pos['side'].lower() == 'buy' else "🔴"
                # Use the latest unrealized P&L from our tracking dict, fallback to position dict
                unrealized = self.unrealized_pnl.get(symbol, pos.get('unrealized_pnl', 0))
                duration = ""
                if symbol in self.position_start_times:
                    duration_secs = (datetime.now(timezone.utc) - self.position_start_times[symbol]).total_seconds()
                    hours = int(duration_secs // 3600)
                    minutes = int((duration_secs % 3600) // 60)
                    duration = f"{hours}h{minutes}m"
                
                pos_line = f"   {side_emoji} {symbol} {pos['side'].upper()} {pos['size']:.4f} @ ${pos['entry_price']:.2f}"
                if unrealized != 0:
                    pos_line += f" | P&L: {ConsoleFormatter.format_currency(unrealized)}"
                if duration:
                    pos_line += f" | {duration}"
                lines.append(pos_line)
        
        # Key metrics row 2
        drawdown_info = f"Current DD: {ConsoleFormatter.format_percentage(self.current_metrics.current_drawdown, colored=True)}"
        max_dd_info = f"Max DD: {ConsoleFormatter.format_currency(self.current_metrics.max_drawdown)}"
        lines.append(f"{drawdown_info} | {max_dd_info}")
        
        # Performance metrics
        pf_info = f"Profit Factor: {self.current_metrics.profit_factor:.2f}"
        exp_info = f"Expectancy: {ConsoleFormatter.format_currency(self.current_metrics.expectancy)}"
        lines.append(f"{pf_info} | {exp_info}")
        
        # Trading activity
        streak_info = f"Wins: {self.current_metrics.consecutive_wins} | Losses: {self.current_metrics.consecutive_losses}"
        rate_info = f"Rate: {self.current_metrics.trades_per_hour:.1f}/hr"
        lines.append(f"{streak_info} | {rate_info}")
        
        # Active positions and strategy info
        if self.current_metrics.current_strategy:
            strategy_info = f"Strategy: {self.current_metrics.current_strategy}"
            lines.append(strategy_info)
        
        if self.current_metrics.current_market_conditions:
            market_info = f"Market: {self.current_metrics.current_market_conditions}"
            lines.append(market_info)
        
        # Recent alerts
        if self.active_alerts:
            lines.append("")
            lines.append("🚨 ACTIVE ALERTS:")
            for alert in self.active_alerts[-3:]:  # Show last 3 alerts
                alert_line = f"   {ConsoleFormatter.format_alert(alert, colored=True)}"
                lines.append(alert_line)
        
        # Performance trend (simple)
        if len(self.metrics_history) >= 2:
            lines.append("")
            trend = self._calculate_performance_trend()
            trend_symbol = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
            lines.append(f"{trend_symbol} Trend: {trend:+.2f}% (last 10 trades)")
        
        lines.append("=" * 80)
        return lines
    
    def _calculate_performance_trend(self) -> float:
        """Calculate recent performance trend"""
        if len(self.metrics_history) < 10:
            return 0.0
        
        recent_metrics = self.metrics_history[-10:]
        if len(recent_metrics) < 2:
            return 0.0
        
        start_pnl = recent_metrics[0].cumulative_pnl
        end_pnl = recent_metrics[-1].cumulative_pnl
        
        if start_pnl == 0:
            return 0.0
        
        return ((end_pnl - start_pnl) / abs(start_pnl)) * 100
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function to be called when alerts are generated"""
        self.alert_callbacks.append(callback)
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                break
    
    def get_current_metrics(self) -> RealTimeMetrics:
        """Get current real-time metrics"""
        return self.current_metrics
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active alerts"""
        return self.active_alerts.copy()
    
    def get_alert_history(self) -> List[Alert]:
        """Get alert history"""
        return self.alert_history.copy()
    
    def update_alert_thresholds(self, new_thresholds: Dict[str, float]):
        """Update alert thresholds"""
        self.alert_thresholds.update(new_thresholds)
        self.logger.info(f"Alert thresholds updated: {new_thresholds}")

    def set_current_strategy(self, strategy_name: str):
        """Set current strategy name for dashboard display"""
        self.current_metrics.current_strategy = strategy_name

    def set_current_market_conditions(self, conditions: str):
        """Set current market conditions for dashboard display"""
        self.current_metrics.current_market_conditions = conditions

    def set_current_symbol(self, symbol: str):
        """Set current trading symbol for dashboard display"""
        self.current_metrics.current_symbol = symbol
        self.logger.info(f"✅ Symbol set for dashboard: {symbol}")
        # Log for debugging
        self.logger.debug(f"Dashboard symbol set to: '{self.current_metrics.current_symbol}'")
        # Force immediate dashboard update to ensure symbol is displayed
        if self.dashboard_enabled:
            self._force_dashboard_update()

    def disable_dashboard(self):
        """Disable the live dashboard"""
        self.dashboard_enabled = False
        self.logger.info("Live dashboard disabled")
    
    def enable_dashboard(self):
        """Enable the live dashboard"""
        self.dashboard_enabled = True
        self.logger.info("Live dashboard enabled")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'current_metrics': asdict(self.current_metrics),
            'active_alerts': [asdict(alert) for alert in self.active_alerts],
            'alert_thresholds': self.alert_thresholds,
            'monitoring_status': {
                'dashboard_enabled': self.dashboard_enabled,
                'monitoring_active': self.dashboard_thread is not None and self.dashboard_thread.is_alive(),
                'last_update': self.last_dashboard_update.isoformat() if self.last_dashboard_update else None
            },
            'session_info': self._get_session_info() if self.session_manager else None,
            'multi_session_summary': self._get_multi_session_summary() if self.session_manager else None
        }
    
    def track_session_metrics(self, session_id: str):
        """Track metrics for a specific session"""
        try:
            if session_id not in self.session_metrics_history:
                self.session_metrics_history[session_id] = []
            
            # Store current metrics with session context
            session_metrics = RealTimeMetrics(
                timestamp=self.current_metrics.timestamp,
                session_id=session_id,
                total_trades=self.current_metrics.total_trades,
                win_rate=self.current_metrics.win_rate,
                cumulative_pnl=self.current_metrics.cumulative_pnl,
                current_drawdown=self.current_metrics.current_drawdown,
                max_drawdown=self.current_metrics.max_drawdown,
                consecutive_wins=self.current_metrics.consecutive_wins,
                consecutive_losses=self.current_metrics.consecutive_losses,
                profit_factor=self.current_metrics.profit_factor,
                expectancy=self.current_metrics.expectancy,
                avg_trade_duration=self.current_metrics.avg_trade_duration,
                active_positions=self.current_metrics.active_positions,
                session_duration_hours=self.current_metrics.session_duration_hours,
                trades_per_hour=self.current_metrics.trades_per_hour,
                current_strategy=self.current_metrics.current_strategy,
                current_market_conditions=self.current_metrics.current_market_conditions
            )
            
            self.session_metrics_history[session_id].append(session_metrics)
            
            # Keep only recent metrics for memory management
            if len(self.session_metrics_history[session_id]) > 1000:
                self.session_metrics_history[session_id] = self.session_metrics_history[session_id][-1000:]
                
        except Exception as e:
            self.logger.error(f"Error tracking session metrics: {e}", exc_info=True)
    
    def get_session_comparison(self, session_ids: List[str]) -> Dict[str, Any]:
        """Get real-time comparison between sessions"""
        try:
            if not self.session_manager:
                return {'error': 'SessionManager not available'}
            
            comparison = self.session_manager.get_session_comparison(session_ids)
            
            # Add real-time metrics if available
            for session_id in session_ids:
                if session_id in self.session_metrics_history:
                    recent_metrics = self.session_metrics_history[session_id][-10:]  # Last 10 snapshots
                    comparison[f'{session_id}_real_time_trend'] = {
                        'pnl_trend': [m.cumulative_pnl for m in recent_metrics],
                        'win_rate_trend': [m.win_rate for m in recent_metrics],
                        'drawdown_trend': [m.current_drawdown for m in recent_metrics]
                    }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error generating session comparison: {e}", exc_info=True)
            return {'error': f'Session comparison failed: {str(e)}'}
    
    def get_cross_session_dashboard_data(self) -> Dict[str, Any]:
        """Get data for cross-session dashboard display"""
        try:
            if not self.session_manager:
                return {'error': 'SessionManager not available'}
            
            # Get cross-session analysis
            cross_analysis = self.session_manager.get_cross_session_analysis()
            
            # Get active sessions performance
            active_sessions = self.session_manager.get_active_sessions()
            active_performance = {}
            
            for session_id, metadata in active_sessions.items():
                if session_id in self.session_metrics_history:
                    recent_metrics = self.session_metrics_history[session_id][-1:]
                    if recent_metrics:
                        active_performance[session_id] = {
                            'strategy': metadata.strategy_name,
                            'symbol': metadata.symbol,
                            'current_pnl': recent_metrics[0].cumulative_pnl,
                            'win_rate': recent_metrics[0].win_rate,
                            'duration_hours': recent_metrics[0].session_duration_hours
                        }
            
            return {
                'cross_session_analysis': cross_analysis,
                'active_sessions_performance': active_performance,
                'session_count': len(self.session_manager.get_session_history()),
                'metrics_history_count': sum(len(metrics) for metrics in self.session_metrics_history.values())
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cross-session dashboard data: {e}", exc_info=True)
            return {'error': f'Dashboard data retrieval failed: {str(e)}'}
    
    def generate_session_alerts(self, session_id: str) -> List[Alert]:
        """Generate session-specific alerts"""
        try:
            session_alerts = []
            
            if not self.session_manager or session_id not in self.session_metrics_history:
                return session_alerts
            
            recent_metrics = self.session_metrics_history[session_id][-10:]  # Last 10 snapshots
            
            if len(recent_metrics) < 5:
                return session_alerts
            
            # Performance degradation alert
            recent_pnl_trend = [m.cumulative_pnl for m in recent_metrics[-5:]]
            if len(recent_pnl_trend) >= 3:
                slope = np.polyfit(range(len(recent_pnl_trend)), recent_pnl_trend, 1)[0]
                if slope < -10:  # Declining by more than $10 per snapshot
                    alert = self._create_alert(
                        alert_type="PERFORMANCE_DEGRADATION",
                        severity=AlertSeverity.WARNING,
                        message=f"Session {session_id}: Performance declining (slope: {slope:.2f})",
                        value=slope,
                        threshold=-10,
                        metadata={'session_id': session_id}
                    )
                    session_alerts.append(alert)
            
            # Win rate drop alert
            recent_win_rates = [m.win_rate for m in recent_metrics[-3:]]
            if len(recent_win_rates) >= 2:
                win_rate_drop = recent_win_rates[0] - recent_win_rates[-1]
                if win_rate_drop > 10:  # Win rate dropped by more than 10%
                    alert = self._create_alert(
                        alert_type="WIN_RATE_DROP",
                        severity=AlertSeverity.WARNING,
                        message=f"Session {session_id}: Win rate dropped by {win_rate_drop:.1f}%",
                        value=win_rate_drop,
                        threshold=10,
                        metadata={'session_id': session_id}
                    )
                    session_alerts.append(alert)
            
            return session_alerts
            
        except Exception as e:
            self.logger.error(f"Error generating session alerts: {e}", exc_info=True)
            return []
    
    def _get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        if not self.session_manager:
            return {}
        
        current_session_id = self.session_manager.get_current_session_id()
        if not current_session_id:
            return {}
        
        active_sessions = self.session_manager.get_active_sessions()
        if current_session_id in active_sessions:
            metadata = active_sessions[current_session_id]
            return {
                'current_session_id': current_session_id,
                'strategy': metadata.strategy_name,
                'symbol': metadata.symbol,
                'timeframe': metadata.timeframe,
                'leverage': metadata.leverage,
                'start_time': metadata.start_time
            }
        
        return {}
    
    def _get_multi_session_summary(self) -> Dict[str, Any]:
        """Get multi-session summary"""
        if not self.session_manager:
            return {}
        
        session_history = self.session_manager.get_session_history()
        if not session_history:
            return {'total_sessions': 0}
        
        # Calculate summary statistics
        total_sessions = len(session_history)
        total_pnl = sum(session.final_pnl for session in session_history.values())
        avg_win_rate = np.mean([session.win_rate for session in session_history.values()])
        avg_duration = np.mean([session.duration_hours for session in session_history.values()])
        
        return {
            'total_sessions': total_sessions,
            'total_pnl_all_sessions': total_pnl,
            'average_win_rate': avg_win_rate,
            'average_session_duration': avg_duration,
            'best_session_pnl': max(session.final_pnl for session in session_history.values()),
            'worst_session_pnl': min(session.final_pnl for session in session_history.values())
        }
    
    def add_strategy_for_tracking(self, strategy_instance):
        """Add a strategy instance to track for active positions"""
        if strategy_instance not in self.tracked_strategies:
            self.tracked_strategies.append(strategy_instance)
            self.logger.info(f"Added strategy {type(strategy_instance).__name__} for position tracking")
    
    def remove_strategy_from_tracking(self, strategy_instance):
        """Remove a strategy instance from tracking"""
        if strategy_instance in self.tracked_strategies:
            self.tracked_strategies.remove(strategy_instance)
            self.logger.info(f"Removed strategy {type(strategy_instance).__name__} from position tracking")
            
    def update_position_pnl(self, symbol: str, current_price: float):
        """Update unrealized P&L for an active position"""
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            entry_price = position.get('entry_price', 0)
            size = position.get('size', 0)
            side = position.get('side', 'buy')
            
            if entry_price > 0 and size > 0:
                if side.lower() == 'buy':
                    pnl = (current_price - entry_price) * size
                else:  # sell/short
                    pnl = (entry_price - current_price) * size
                
                self.unrealized_pnl[symbol] = pnl
                position['unrealized_pnl'] = pnl
                position['current_price'] = current_price 