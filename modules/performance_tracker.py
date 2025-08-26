import logging
import csv
import json
from typing import List, Dict, Optional, Any, Tuple
import os
import threading
import tempfile
import shutil
import pandas as pd
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum

class TradeStatus(Enum):
    """Trade status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

class AlertType(Enum):
    """Alert type enumeration"""
    DRAWDOWN_WARNING = "drawdown_warning"
    DRAWDOWN_CRITICAL = "drawdown_critical"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    WIN_RATE_DROP = "win_rate_drop"
    STRATEGY_UNDERPERFORMANCE = "strategy_underperformance"
    RISK_LIMIT_EXCEEDED = "risk_limit_exceeded"

@dataclass
class MarketContext:
    """Market context information for a trade"""
    market_5m: str  # 5-minute market condition
    market_1m: str  # 1-minute market condition
    strategy_selection_reason: str
    execution_timeframe: str
    volatility_regime: Optional[str] = None
    trend_strength: Optional[float] = None
    market_session: Optional[str] = None  # e.g., "Asia", "Europe", "US"

@dataclass
class OrderDetails:
    """Detailed order execution information"""
    main_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    retry_attempts: int = 0
    slippage_pct: Optional[float] = None
    spread_at_entry: Optional[float] = None
    spread_at_exit: Optional[float] = None
    signal_to_execution_delay_ms: Optional[int] = None
    execution_quality_score: Optional[float] = None  # 0-100 score
    order_type: str = "market"
    time_in_force: str = "GoodTillCancel"
    reduce_only: bool = False

@dataclass
class RiskMetrics:
    """Risk management metrics for a trade"""
    planned_sl_pct: Optional[float] = None
    actual_sl_pct: Optional[float] = None
    planned_tp_pct: Optional[float] = None
    actual_tp_pct: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    position_size_pct: Optional[float] = None  # Percentage of account used
    leverage_used: Optional[float] = None
    max_adverse_excursion: Optional[float] = None  # MAE
    max_favorable_excursion: Optional[float] = None  # MFE

@dataclass
class TradeRecord:
    """Enhanced trade record with comprehensive data"""
    # Basic trade info (backward compatibility)
    strategy: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    entry_timestamp: str
    exit_timestamp: Optional[str] = None
    
    # Enhanced data model
    market_context: Optional[MarketContext] = None
    order_details: Optional[OrderDetails] = None
    risk_metrics: Optional[RiskMetrics] = None
    
    # Trade lifecycle
    trade_id: Optional[str] = None
    status: TradeStatus = TradeStatus.PENDING
    exit_reason: Optional[str] = None
    
    # Performance metrics
    trade_duration_seconds: Optional[float] = None
    return_pct: Optional[float] = None
    
    # Additional context
    notes: Optional[str] = None
    session_id: Optional[str] = None

class PerformanceTracker:
    """
    Enhanced performance tracker with comprehensive trade data capture,
    real-time monitoring, and advanced analytics capabilities.
    
    Features:
    - Enhanced data model with market context and order details
    - Real-time performance monitoring with alerts
    - Advanced analytics and strategy effectiveness tracking
    - Session management and cross-session analysis
    - Backward compatibility with existing interface
    """
    
    def __init__(self, log_dir: str = "performance", logger: Optional[logging.Logger] = None,
                 alert_thresholds: Optional[Dict[str, float]] = None, 
                 real_time_monitor=None):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Real-time monitor integration
        self.real_time_monitor = real_time_monitor
        
        # Core data storage
        self.trades: List[TradeRecord] = []
        self.trade_lookup: Dict[str, TradeRecord] = {}  # trade_id -> TradeRecord
        
        # Performance metrics
        self.cumulative_pnl = 0.0
        self.max_drawdown = 0.0
        self.high_watermark = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Alert system
        self.alert_thresholds = alert_thresholds or {
            'max_drawdown_pct': 5.0,
            'consecutive_losses': 3,
            'win_rate_threshold': 40.0,
            'risk_limit_pct': 10.0
        }
        self.recent_alerts: List[Dict] = []
        
        # Session tracking - make unique per instance to avoid conflicts
        import os
        process_id = os.getpid()
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{process_id}"
        self.session_start_time = datetime.now(timezone.utc)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance statistics cache
        self._stats_cache: Dict[str, Any] = {}
        self._cache_timestamp = None
        self._cache_validity_seconds = 30  # Cache valid for 30 seconds
        
        self.logger.info(f"PerformanceTracker initialized with session ID: {self.session_id}")

    def record_trade(self, trade: Dict) -> str:
        """
        Record a completed trade with enhanced data capture.
        Maintains backward compatibility while supporting new enhanced format.
        
        Args:
            trade: Dict with trade details (supports both old and new format)
            
        Returns:
            str: Trade ID for tracking
        """
        try:
            with self._lock:
                # Generate trade ID if not provided
                trade_id = trade.get('trade_id', f"trade_{len(self.trades) + 1}_{int(datetime.now().timestamp())}")
                
                # Handle backward compatibility - convert old format to new
                if isinstance(trade, dict) and 'market_context' not in trade:
                    trade_record = self._convert_legacy_trade_format(trade, trade_id)
                else:
                    trade_record = self._create_trade_record(trade, trade_id)
                
                # Store trade
                self.trades.append(trade_record)
                self.trade_lookup[trade_id] = trade_record
                
                # Update performance metrics
                self._update_performance_metrics(trade_record)
                
                # Check for alerts
                self._check_alert_conditions(trade_record)
                
                # Invalidate cache
                self._invalidate_cache()
                
                self.logger.info(f"✅ TRADE RECORDED {trade_id}: {trade_record.strategy} {trade_record.side} "
                               f"{trade_record.symbol} PnL: ${trade_record.pnl:.2f} "
                               f"(Cumulative: ${self.cumulative_pnl:.2f}) [Total trades: {len(self.trades)}]")
                
                # Immediately notify RealTimeMonitor of new trade
                if self.real_time_monitor:
                    try:
                        self.real_time_monitor.update_metrics(force_update=True)
                        self.logger.debug(f"RealTimeMonitor notified of new trade {trade_id}")
                    except Exception as e:
                        self.logger.warning(f"Failed to notify RealTimeMonitor of new trade: {e}")
                
                return trade_id
                
        except Exception as exc:
            self.logger.error(f"Failed to record trade: {exc}", exc_info=True)
            return ""

    def _convert_legacy_trade_format(self, trade: Dict, trade_id: str) -> TradeRecord:
        """Convert legacy trade format to new TradeRecord format"""
        # Calculate duration if timestamps available
        duration = None
        entry_time = trade.get('entry_timestamp') or trade.get('entry_time') or trade.get('timestamp')
        exit_time = trade.get('exit_timestamp') or trade.get('exit_time') or trade.get('close_timestamp')
        
        if entry_time and exit_time:
            try:
                entry_dt = pd.to_datetime(entry_time)
                exit_dt = pd.to_datetime(exit_time)
                duration = (exit_dt - entry_dt).total_seconds()
            except Exception:
                pass
        
        # Calculate return percentage
        return_pct = None
        if trade.get('entry_price') and trade.get('entry_price') != 0:
            return_pct = (float(trade.get('pnl', 0)) / float(trade.get('entry_price'))) * 100
        
        return TradeRecord(
            trade_id=trade_id,
            strategy=trade.get('strategy', 'unknown'),
            symbol=trade.get('symbol', 'unknown'),
            side=trade.get('side', 'unknown'),
            entry_price=float(trade.get('entry_price', 0)),
            exit_price=float(trade.get('exit_price', 0)),
            size=float(trade.get('size', 0)),
            pnl=float(trade.get('pnl', 0)),
            entry_timestamp=entry_time or datetime.now(timezone.utc).isoformat(),
            exit_timestamp=exit_time,
            trade_duration_seconds=duration,
            return_pct=return_pct,
            status=TradeStatus.FILLED,
            session_id=self.session_id
        )

    def _create_trade_record(self, trade: Dict, trade_id: str) -> TradeRecord:
        """Create TradeRecord from enhanced trade data"""
        # Extract market context
        market_context = None
        if 'market_context' in trade:
            mc = trade['market_context']
            market_context = MarketContext(
                market_5m=mc.get('market_5m', 'UNKNOWN'),
                market_1m=mc.get('market_1m', 'UNKNOWN'),
                strategy_selection_reason=mc.get('strategy_selection_reason', ''),
                execution_timeframe=mc.get('execution_timeframe', '1m'),
                volatility_regime=mc.get('volatility_regime'),
                trend_strength=mc.get('trend_strength'),
                market_session=mc.get('market_session')
            )
        
        # Extract order details
        order_details = None
        if 'order_details' in trade:
            od = trade['order_details']
            order_details = OrderDetails(
                main_order_id=od.get('main_order_id'),
                sl_order_id=od.get('sl_order_id'),
                tp_order_id=od.get('tp_order_id'),
                retry_attempts=od.get('retry_attempts', 0),
                slippage_pct=od.get('slippage_pct'),
                spread_at_entry=od.get('spread_at_entry'),
                spread_at_exit=od.get('spread_at_exit'),
                signal_to_execution_delay_ms=od.get('signal_to_execution_delay_ms'),
                execution_quality_score=od.get('execution_quality_score'),
                order_type=od.get('order_type', 'market'),
                time_in_force=od.get('time_in_force', 'GoodTillCancel'),
                reduce_only=od.get('reduce_only', False)
            )
        
        # Extract risk metrics
        risk_metrics = None
        if 'risk_metrics' in trade:
            rm = trade['risk_metrics']
            risk_metrics = RiskMetrics(
                planned_sl_pct=rm.get('planned_sl_pct'),
                actual_sl_pct=rm.get('actual_sl_pct'),
                planned_tp_pct=rm.get('planned_tp_pct'),
                actual_tp_pct=rm.get('actual_tp_pct'),
                risk_reward_ratio=rm.get('risk_reward_ratio'),
                position_size_pct=rm.get('position_size_pct'),
                leverage_used=rm.get('leverage_used'),
                max_adverse_excursion=rm.get('max_adverse_excursion'),
                max_favorable_excursion=rm.get('max_favorable_excursion')
            )
        
        return TradeRecord(
            trade_id=trade_id,
            strategy=trade.get('strategy', 'unknown'),
            symbol=trade.get('symbol', 'unknown'),
            side=trade.get('side', 'unknown'),
            entry_price=float(trade.get('entry_price', 0)),
            exit_price=float(trade.get('exit_price', 0)),
            size=float(trade.get('size', 0)),
            pnl=float(trade.get('pnl', 0)),
            entry_timestamp=trade.get('entry_timestamp') or datetime.now(timezone.utc).isoformat(),
            exit_timestamp=trade.get('exit_timestamp'),
            market_context=market_context,
            order_details=order_details,
            risk_metrics=risk_metrics,
            status=TradeStatus(trade.get('status', 'filled')),
            exit_reason=trade.get('exit_reason'),
            trade_duration_seconds=trade.get('trade_duration_seconds'),
            return_pct=trade.get('return_pct'),
            notes=trade.get('notes'),
            session_id=self.session_id
        )

    def _update_performance_metrics(self, trade_record: TradeRecord):
        """Update cumulative performance metrics"""
        pnl = trade_record.pnl
        self.cumulative_pnl += pnl
        
        # Update high watermark and drawdown
        self.high_watermark = max(self.high_watermark, self.cumulative_pnl)
        drawdown = self.high_watermark - self.cumulative_pnl
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Update consecutive win/loss streaks
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def _check_alert_conditions(self, trade_record: TradeRecord):
        """Check for alert conditions and trigger alerts if necessary"""
        alerts = []
        
        # Drawdown alerts
        current_drawdown_pct = (self.max_drawdown / abs(self.high_watermark) * 100) if self.high_watermark != 0 else 0
        if current_drawdown_pct > self.alert_thresholds['max_drawdown_pct']:
            alerts.append({
                'type': AlertType.DRAWDOWN_CRITICAL,
                'message': f"Critical drawdown reached: {current_drawdown_pct:.2f}%",
                'value': current_drawdown_pct,
                'threshold': self.alert_thresholds['max_drawdown_pct'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        # Consecutive losses alert
        if self.consecutive_losses >= self.alert_thresholds['consecutive_losses']:
            alerts.append({
                'type': AlertType.CONSECUTIVE_LOSSES,
                'message': f"Consecutive losses reached: {self.consecutive_losses}",
                'value': self.consecutive_losses,
                'threshold': self.alert_thresholds['consecutive_losses'],
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        # Win rate alert (check if we have enough trades)
        if len(self.trades) >= 10:
            current_win_rate = self.win_rate()
            if current_win_rate < self.alert_thresholds['win_rate_threshold']:
                alerts.append({
                    'type': AlertType.WIN_RATE_DROP,
                    'message': f"Win rate below threshold: {current_win_rate:.2f}%",
                    'value': current_win_rate,
                    'threshold': self.alert_thresholds['win_rate_threshold'],
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
        
        # Log and store alerts
        for alert in alerts:
            self.recent_alerts.append(alert)
            self.logger.warning(f"ALERT: {alert['message']}")
            
        # Keep only recent alerts (last 50)
        self.recent_alerts = self.recent_alerts[-50:]

    def get_trade_by_id(self, trade_id: str) -> Optional[TradeRecord]:
        """Get a specific trade by ID"""
        return self.trade_lookup.get(trade_id)

    def update_trade_status(self, trade_id: str, status: TradeStatus, notes: Optional[str] = None):
        """Update the status of a trade"""
        with self._lock:
            if trade_id in self.trade_lookup:
                self.trade_lookup[trade_id].status = status
                if notes:
                    self.trade_lookup[trade_id].notes = notes
                self.logger.info(f"Updated trade {trade_id} status to {status.value}")
                self._invalidate_cache()

    def _invalidate_cache(self):
        """Invalidate the statistics cache"""
        self._stats_cache.clear()
        self._cache_timestamp = None

    def _get_cached_stats(self) -> Optional[Dict[str, Any]]:
        """Get cached statistics if still valid"""
        if (self._cache_timestamp and 
            (datetime.now().timestamp() - self._cache_timestamp) < self._cache_validity_seconds):
            return self._stats_cache
        return None

    def _cache_stats(self, stats: Dict[str, Any]):
        """Cache statistics"""
        self._stats_cache = stats
        self._cache_timestamp = datetime.now().timestamp()

    # Backward compatibility methods with enhanced functionality
    def win_rate(self, trades: Optional[List[TradeRecord]] = None) -> float:
        """Calculate win rate as a percentage"""
        trades = trades if trades is not None else self.trades
        if not trades:
            return 0.0
        wins = sum(1 for t in trades if t.pnl > 0)
        return (wins / len(trades) * 100)

    def cumulative_return(self, trades: Optional[List[TradeRecord]] = None) -> float:
        """Return cumulative PnL"""
        trades = trades if trades is not None else self.trades
        return sum(t.pnl for t in trades)

    def max_drawdown_value(self, trades: Optional[List[TradeRecord]] = None) -> float:
        """Return the maximum drawdown value"""
        trades = trades if trades is not None else self.trades
        if not trades:
            return 0.0
        
        high = 0.0
        max_dd = 0.0
        cum = 0.0
        
        for t in trades:
            cum += t.pnl
            high = max(high, cum)
            max_dd = max(max_dd, high - cum)
        
        return max_dd

    def average_trade_duration(self, trades: Optional[List[TradeRecord]] = None) -> float:
        """Compute average trade duration in seconds"""
        trades = trades if trades is not None else self.trades
        durations = [t.trade_duration_seconds for t in trades 
                    if t.trade_duration_seconds is not None]
        return sum(durations) / len(durations) if durations else 0.0

    def expectancy(self, trades: Optional[List[TradeRecord]] = None) -> float:
        """Compute expectancy: (win_rate × avg_win) – (loss_rate × avg_loss)"""
        trades = trades if trades is not None else self.trades
        if not trades:
            return 0.0
        
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl < 0]
        total = len(trades)
        
        win_rate = len(wins) / total
        loss_rate = len(losses) / total
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        
        return (win_rate * avg_win) - (loss_rate * avg_loss)

    def profit_factor(self, trades: Optional[List[TradeRecord]] = None) -> float:
        """Compute profit factor: sum(wins) / abs(sum(losses))"""
        trades = trades if trades is not None else self.trades
        if not trades:
            return 0.0
        
        wins = sum(t.pnl for t in trades if t.pnl > 0)
        losses = sum(t.pnl for t in trades if t.pnl < 0)
        
        return wins / abs(losses) if losses != 0 else float('inf')

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics with caching"""
        cached = self._get_cached_stats()
        if cached:
            return cached
        
        if not self.trades:
            return {'error': 'No trades available'}
        
        stats = {
            # Basic metrics
            'total_trades': len(self.trades),
            'win_rate': self.win_rate(),
            'cumulative_pnl': self.cumulative_pnl,
            'max_drawdown': self.max_drawdown,
            'avg_trade_duration': self.average_trade_duration(),
            'expectancy': self.expectancy(),
            'profit_factor': self.profit_factor(),
            
            # Enhanced metrics
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'session_id': self.session_id,
            'session_duration_hours': (datetime.now(timezone.utc) - self.session_start_time).total_seconds() / 3600,
            
            # Strategy breakdown
            'strategy_performance': self.get_strategy_performance(),
            
            # Risk metrics
            'risk_metrics': self.get_risk_metrics_summary(),
            
            # Recent alerts
            'recent_alerts': self.recent_alerts[-10:],  # Last 10 alerts
            
            # Market context analysis
            'market_context_performance': self.get_market_context_performance(),
            
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        
        self._cache_stats(stats)
        return stats

    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by strategy"""
        strategy_stats = {}
        
        for strategy in set(t.strategy for t in self.trades):
            strategy_trades = [t for t in self.trades if t.strategy == strategy]
            
            strategy_stats[strategy] = {
                'trade_count': len(strategy_trades),
                'win_rate': self.win_rate(strategy_trades),
                'cumulative_pnl': self.cumulative_return(strategy_trades),
                'avg_pnl': sum(t.pnl for t in strategy_trades) / len(strategy_trades),
                'max_drawdown': self.max_drawdown_value(strategy_trades),
                'profit_factor': self.profit_factor(strategy_trades),
                'avg_duration': self.average_trade_duration(strategy_trades)
            }
        
        return strategy_stats

    def get_risk_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of risk metrics"""
        trades_with_risk = [t for t in self.trades if t.risk_metrics is not None]
        
        if not trades_with_risk:
            return {'error': 'No risk metrics available'}
        
        # Calculate average risk metrics
        avg_leverage = []
        avg_position_size = []
        risk_reward_ratios = []
        
        for trade in trades_with_risk:
            rm = trade.risk_metrics
            if rm.leverage_used is not None:
                avg_leverage.append(rm.leverage_used)
            if rm.position_size_pct is not None:
                avg_position_size.append(rm.position_size_pct)
            if rm.risk_reward_ratio is not None:
                risk_reward_ratios.append(rm.risk_reward_ratio)
        
        return {
            'avg_leverage': sum(avg_leverage) / len(avg_leverage) if avg_leverage else 0,
            'avg_position_size_pct': sum(avg_position_size) / len(avg_position_size) if avg_position_size else 0,
            'avg_risk_reward_ratio': sum(risk_reward_ratios) / len(risk_reward_ratios) if risk_reward_ratios else 0,
            'trades_with_risk_data': len(trades_with_risk),
            'total_trades': len(self.trades)
        }

    def get_market_context_performance(self) -> Dict[str, Any]:
        """Get performance breakdown by market context"""
        trades_with_context = [t for t in self.trades if t.market_context is not None]
        
        if not trades_with_context:
            return {'error': 'No market context data available'}
        
        context_performance = {}
        
        # Group by market condition combinations
        for trade in trades_with_context:
            mc = trade.market_context
            key = f"{mc.market_5m}_{mc.market_1m}"
            
            if key not in context_performance:
                context_performance[key] = []
            context_performance[key].append(trade)
        
        # Calculate metrics for each context
        context_stats = {}
        for context, trades in context_performance.items():
            context_stats[context] = {
                'trade_count': len(trades),
                'win_rate': self.win_rate(trades),
                'cumulative_pnl': self.cumulative_return(trades),
                'avg_pnl': sum(t.pnl for t in trades) / len(trades),
                'execution_timeframes': list(set(t.market_context.execution_timeframe for t in trades))
            }
        
        return context_stats

    # Legacy method compatibility
    def group_metrics(self, by: str = 'strategy') -> Dict[str, Dict[str, Any]]:
        """Compute metrics grouped by specified field (backward compatibility)"""
        if by == 'strategy':
            return self.get_strategy_performance()
        elif by == 'symbol':
            return self._group_by_symbol()
        else:
            self.logger.warning(f"Unsupported grouping field: {by}")
            return {}

    def _group_by_symbol(self) -> Dict[str, Dict[str, Any]]:
        """Group metrics by symbol"""
        symbol_stats = {}
        
        for symbol in set(t.symbol for t in self.trades):
            symbol_trades = [t for t in self.trades if t.symbol == symbol]
            
            symbol_stats[symbol] = {
                'trade_count': len(symbol_trades),
                'win_rate': self.win_rate(symbol_trades),
                'cumulative_pnl': self.cumulative_return(symbol_trades),
                'avg_pnl': sum(t.pnl for t in symbol_trades) / len(symbol_trades),
                'strategies_used': list(set(t.strategy for t in symbol_trades))
            }
        
        return symbol_stats

    def rolling_drawdown_curve(self, window: int = 20) -> List[float]:
        """Compute rolling max drawdown curve over the last N trades"""
        if len(self.trades) < window:
            return [self.max_drawdown_value(self.trades[:i+1]) for i in range(len(self.trades))]
        
        curve = []
        for i in range(len(self.trades)):
            window_trades = self.trades[max(0, i - window + 1):i + 1]
            curve.append(self.max_drawdown_value(window_trades))
        
        return curve

    def rolling_sharpe(self, window: int = 20, risk_free_rate: float = 0.0) -> List[float]:
        """Compute rolling Sharpe ratio over the last N trades"""
        import numpy as np
        
        if len(self.trades) < 2:
            return [0.0] * len(self.trades)
        
        ratios = []
        for i in range(len(self.trades)):
            window_trades = self.trades[max(0, i - window + 1):i + 1]
            if len(window_trades) < 2:
                ratios.append(0.0)
                continue
                
            returns = [t.pnl for t in window_trades]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                ratios.append(0.0)
            else:
                sharpe = (mean_return - risk_free_rate) / std_return
                ratios.append(sharpe)
        
        return ratios

    def persist_to_csv(self, filename: str = "performance_log.csv"):
        """Persist all trades to a CSV file with enhanced data"""
        path = os.path.join(self.log_dir, filename)
        try:
            if not self.trades:
                self.logger.warning("No trades to persist to CSV.")
                return
            
            with self._lock:
                # Convert TradeRecord objects to dictionaries
                trade_dicts = []
                for trade in self.trades:
                    trade_dict = asdict(trade)
                    # Flatten nested structures for CSV compatibility
                    if trade_dict.get('market_context'):
                        mc = trade_dict.pop('market_context')
                        for key, value in mc.items():
                            trade_dict[f'market_{key}'] = value
                    
                    if trade_dict.get('order_details'):
                        od = trade_dict.pop('order_details')
                        for key, value in od.items():
                            trade_dict[f'order_{key}'] = value
                    
                    if trade_dict.get('risk_metrics'):
                        rm = trade_dict.pop('risk_metrics')
                        for key, value in rm.items():
                            trade_dict[f'risk_{key}'] = value
                    
                    # Convert enum to string
                    if 'status' in trade_dict:
                        trade_dict['status'] = trade_dict['status'].value if hasattr(trade_dict['status'], 'value') else str(trade_dict['status'])
                    
                    trade_dicts.append(trade_dict)
                
                # Write to temporary file first (atomic write)
                with tempfile.NamedTemporaryFile('w', delete=False, newline='') as tmpfile:
                    if trade_dicts:
                        writer = csv.DictWriter(tmpfile, fieldnames=trade_dicts[0].keys())
                        writer.writeheader()
                        writer.writerows(trade_dicts)
                    tempname = tmpfile.name
                
                shutil.move(tempname, path)
                
            self.logger.info(f"Persisted {len(self.trades)} trades to CSV: {path}")
            
        except Exception as exc:
            self.logger.error(f"Failed to persist trades to CSV: {exc}", exc_info=True)

    def persist_to_json(self, filename: str = "performance_log.json"):
        """Persist all trades to a JSON file with enhanced data"""
        path = os.path.join(self.log_dir, filename)
        try:
            with self._lock:
                # Convert TradeRecord objects to dictionaries
                trade_dicts = []
                for trade in self.trades:
                    trade_dict = asdict(trade)
                    # Convert enum to string
                    if 'status' in trade_dict:
                        trade_dict['status'] = trade_dict['status'].value if hasattr(trade_dict['status'], 'value') else str(trade_dict['status'])
                    trade_dicts.append(trade_dict)
                
                # Include metadata
                export_data = {
                    'session_id': self.session_id,
                    'session_start_time': self.session_start_time.isoformat(),
                    'export_time': datetime.now(timezone.utc).isoformat(),
                    'trades': trade_dicts,
                    'summary_statistics': self.get_comprehensive_statistics()
                }
                
                # Write to temporary file first (atomic write)
                with tempfile.NamedTemporaryFile('w', delete=False) as tmpfile:
                    json.dump(export_data, tmpfile, indent=2, default=str)
                    tempname = tmpfile.name
                
                shutil.move(tempname, path)
                
            self.logger.info(f"Persisted {len(self.trades)} trades to JSON: {path}")
            
        except Exception as exc:
            self.logger.error(f"Failed to persist trades to JSON: {exc}", exc_info=True)

    def close_session(self):
        """Persist trades to disk on shutdown with enhanced data"""
        self.persist_to_csv()
        self.persist_to_json()
        
        # Save session summary
        session_summary = {
            'session_id': self.session_id,
            'start_time': self.session_start_time.isoformat(),
            'end_time': datetime.now(timezone.utc).isoformat(),
            'total_trades': len(self.trades),
            'final_statistics': self.get_comprehensive_statistics()
        }
        
        summary_path = os.path.join(self.log_dir, f"session_summary_{self.session_id}.json")
        try:
            with open(summary_path, 'w') as f:
                json.dump(session_summary, f, indent=2, default=str)
            self.logger.info(f"Session summary saved: {summary_path}")
        except Exception as exc:
            self.logger.error(f"Failed to save session summary: {exc}")
        
        self.logger.info(f"PerformanceTracker session {self.session_id} closed and data persisted.")

    def to_dataframe(self) -> pd.DataFrame:
        """Return all trades as a pandas DataFrame with enhanced data"""
        if not self.trades:
            return pd.DataFrame()
        
        # Convert to dictionaries and flatten
        records = []
        for trade in self.trades:
            record = asdict(trade)
            
            # Flatten nested structures
            if record.get('market_context'):
                mc = record.pop('market_context')
                for key, value in mc.items():
                    record[f'market_{key}'] = value
            
            if record.get('order_details'):
                od = record.pop('order_details')
                for key, value in od.items():
                    record[f'order_{key}'] = value
            
            if record.get('risk_metrics'):
                rm = record.pop('risk_metrics')
                for key, value in rm.items():
                    record[f'risk_{key}'] = value
            
            # Convert enum to string
            if 'status' in record:
                record['status'] = record['status'].value if hasattr(record['status'], 'value') else str(record['status'])
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Convert timestamp columns to datetime
        timestamp_cols = [col for col in df.columns if 'timestamp' in col.lower()]
        for col in timestamp_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        return df

    @staticmethod
    def persist_on_exception(tracker: 'PerformanceTracker'):
        """Enhanced exception handling with comprehensive data persistence"""
        try:
            tracker.logger.error("Exception occurred, persisting all data...")
            tracker.close_session()
            
            # Additional emergency backup
            emergency_backup = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'session_id': tracker.session_id,
                'trade_count': len(tracker.trades),
                'cumulative_pnl': tracker.cumulative_pnl,
                'max_drawdown': tracker.max_drawdown,
                'recent_alerts': tracker.recent_alerts
            }
            
            emergency_path = os.path.join(tracker.log_dir, f"emergency_backup_{tracker.session_id}.json")
            with open(emergency_path, 'w') as f:
                json.dump(emergency_backup, f, indent=2)
            
            tracker.logger.info(f"Emergency backup saved: {emergency_path}")
            
        except Exception as exc:
            if hasattr(tracker, 'logger'):
                tracker.logger.error(f"Failed to persist trades on exception: {exc}")
    
    def set_real_time_monitor(self, real_time_monitor):
        """Set the RealTimeMonitor instance for automatic notifications"""
        self.real_time_monitor = real_time_monitor
        self.logger.info("RealTimeMonitor integration enabled") 