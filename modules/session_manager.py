import os
import json
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path

class SessionStatus(Enum):
    """Session status enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    ERROR = "error"

@dataclass
class SessionMetadata:
    """Session metadata information"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    status: SessionStatus = SessionStatus.ACTIVE
    strategy_name: Optional[str] = None
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    leverage: Optional[float] = None
    initial_balance: Optional[float] = None
    final_balance: Optional[float] = None
    configuration: Optional[Dict[str, Any]] = None
    market_conditions: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None

@dataclass
class SessionSummary:
    """Comprehensive session summary"""
    metadata: SessionMetadata
    performance_metrics: Dict[str, Any]
    trade_statistics: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    alert_summary: Dict[str, Any]
    duration_hours: float
    total_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    final_pnl: float

class SessionManager:
    """
    Comprehensive Session Management System
    
    Features:
    - Session lifecycle management (start, pause, resume, end)
    - Multi-session tracking and comparison
    - Cross-session performance analysis
    - Historical data management and archiving
    - Session state persistence and recovery
    - Configuration impact analysis
    - Learning curve tracking
    - Strategy evolution analysis
    """
    
    def __init__(self, base_dir: str = "sessions", logger: Optional[logging.Logger] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Session tracking
        self.active_sessions: Dict[str, SessionMetadata] = {}
        self.session_history: Dict[str, SessionSummary] = {}
        self.current_session_id: Optional[str] = None
        
        # Directories
        self.sessions_dir = self.base_dir / "active"
        self.archive_dir = self.base_dir / "archive"
        self.analysis_dir = self.base_dir / "analysis"
        
        for dir_path in [self.sessions_dir, self.archive_dir, self.analysis_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Cache for analysis results
        self._analysis_cache: Dict[str, Any] = {}
        self._cache_timestamp: Optional[float] = None
        self._cache_validity_seconds = 300  # 5 minutes
        
        # Load existing session data
        self._load_session_history()
        
        self.logger.info(f"SessionManager initialized with base directory: {self.base_dir}")

    def create_session(self, strategy_name: str, symbol: str, timeframe: str, 
                      leverage: float, configuration: Dict[str, Any],
                      market_conditions: Optional[Dict[str, Any]] = None,
                      initial_balance: Optional[float] = None,
                      notes: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> str:
        """Create a new trading session"""
        try:
            with self._lock:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{strategy_name}"
                
                metadata = SessionMetadata(
                    session_id=session_id,
                    start_time=datetime.now(timezone.utc).isoformat(),
                    strategy_name=strategy_name,
                    symbol=symbol,
                    timeframe=timeframe,
                    leverage=leverage,
                    initial_balance=initial_balance,
                    configuration=configuration.copy(),
                    market_conditions=market_conditions.copy() if market_conditions else None,
                    notes=notes,
                    tags=tags.copy() if tags else []
                )
                
                self.active_sessions[session_id] = metadata
                self.current_session_id = session_id
                
                # Create session directory
                session_dir = self.sessions_dir / session_id
                session_dir.mkdir(exist_ok=True)
                
                # Save session metadata
                self._save_session_metadata(metadata)
                
                self.logger.info(f"Created new session: {session_id}")
                return session_id
                
        except Exception as e:
            self.logger.error(f"Error creating session: {e}", exc_info=True)
            raise

    def end_session(self, session_id: str, final_balance: Optional[float] = None,
                   performance_tracker=None, real_time_monitor=None, 
                   risk_manager=None) -> SessionSummary:
        """End a trading session and generate comprehensive summary"""
        try:
            with self._lock:
                if session_id not in self.active_sessions:
                    raise ValueError(f"Session {session_id} not found in active sessions")
                
                metadata = self.active_sessions[session_id]
                metadata.end_time = datetime.now(timezone.utc).isoformat()
                metadata.status = SessionStatus.COMPLETED
                metadata.final_balance = final_balance
                
                # Calculate session duration
                start_time = datetime.fromisoformat(metadata.start_time.replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(metadata.end_time.replace('Z', '+00:00'))
                duration_hours = (end_time - start_time).total_seconds() / 3600
                
                # Gather performance data
                performance_metrics = {}
                trade_statistics = {}
                risk_metrics = {}
                alert_summary = {}
                
                if performance_tracker:
                    perf_stats = performance_tracker.get_comprehensive_statistics()
                    if 'error' not in perf_stats:
                        performance_metrics = perf_stats
                        trade_statistics = {
                            'total_trades': perf_stats.get('total_trades', 0),
                            'win_rate': perf_stats.get('win_rate', 0),
                            'profit_factor': perf_stats.get('profit_factor', 0),
                            'expectancy': perf_stats.get('expectancy', 0),
                            'max_drawdown': perf_stats.get('max_drawdown', 0),
                            'cumulative_pnl': perf_stats.get('cumulative_pnl', 0)
                        }
                
                if real_time_monitor:
                    try:
                        alert_history = getattr(real_time_monitor, 'alert_history', [])
                        active_alerts = getattr(real_time_monitor, 'active_alerts', [])
                        
                        alert_summary = {
                            'total_alerts': len(alert_history) if alert_history else 0,
                            'active_alerts': len(active_alerts) if active_alerts else 0,
                            'critical_alerts': len([a for a in alert_history 
                                                  if hasattr(a, 'severity') and a.severity.value == 'critical']),
                            'alert_types': list(set(a.type for a in alert_history if hasattr(a, 'type')))
                        }
                    except Exception:
                        alert_summary = {
                            'total_alerts': 0,
                            'active_alerts': 0,
                            'critical_alerts': 0,
                            'alert_types': []
                        }
                
                if risk_manager:
                    try:
                        risk_summary = risk_manager.get_risk_summary()
                        if isinstance(risk_summary, dict) and 'error' not in risk_summary:
                            risk_metrics = risk_summary
                    except Exception:
                        risk_metrics = {}
                
                # Create session summary
                session_summary = SessionSummary(
                    metadata=metadata,
                    performance_metrics=performance_metrics,
                    trade_statistics=trade_statistics,
                    risk_metrics=risk_metrics,
                    alert_summary=alert_summary,
                    duration_hours=duration_hours,
                    total_trades=trade_statistics.get('total_trades', 0),
                    win_rate=trade_statistics.get('win_rate', 0),
                    profit_factor=trade_statistics.get('profit_factor', 0),
                    max_drawdown=trade_statistics.get('max_drawdown', 0),
                    final_pnl=trade_statistics.get('cumulative_pnl', 0)
                )
                
                # Move to archive and save summary
                self._archive_session(session_id, session_summary)
                
                # Remove from active sessions
                del self.active_sessions[session_id]
                if self.current_session_id == session_id:
                    self.current_session_id = None
                
                # Store in history
                self.session_history[session_id] = session_summary
                
                # Invalidate analysis cache
                self._invalidate_cache()
                
                self.logger.info(f"Session {session_id} ended successfully")
                return session_summary
                
        except Exception as e:
            self.logger.error(f"Error ending session {session_id}: {e}", exc_info=True)
            raise

    def pause_session(self, session_id: str):
        """Pause an active session"""
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id].status = SessionStatus.PAUSED
                self.logger.info(f"Session {session_id} paused")

    def resume_session(self, session_id: str):
        """Resume a paused session"""
        with self._lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id].status = SessionStatus.ACTIVE
                self.logger.info(f"Session {session_id} resumed")

    def get_session_comparison(self, session_ids: List[str]) -> Dict[str, Any]:
        """Get detailed comparison between multiple sessions"""
        try:
            if not session_ids:
                return {'error': 'No session IDs provided'}
            
            comparison_data = {
                'sessions': {},
                'comparative_metrics': {},
                'performance_ranking': {},
                'insights': []
            }
            
            sessions_data = []
            for session_id in session_ids:
                if session_id in self.session_history:
                    sessions_data.append(self.session_history[session_id])
                    comparison_data['sessions'][session_id] = asdict(self.session_history[session_id])
            
            if not sessions_data:
                return {'error': 'No valid sessions found'}
            
            # Comparative metrics
            metrics = ['total_trades', 'win_rate', 'profit_factor', 'max_drawdown', 'final_pnl']
            for metric in metrics:
                values = [getattr(session, metric, 0) for session in sessions_data]
                comparison_data['comparative_metrics'][metric] = {
                    'values': dict(zip(session_ids, values)),
                    'best': max(values) if metric != 'max_drawdown' else min(values),
                    'worst': min(values) if metric != 'max_drawdown' else max(values),
                    'average': np.mean(values),
                    'std_dev': np.std(values)
                }
            
            # Performance ranking
            pnl_values = [(session_id, session.final_pnl) for session_id, session in zip(session_ids, sessions_data)]
            pnl_values.sort(key=lambda x: x[1], reverse=True)
            comparison_data['performance_ranking'] = {
                'by_pnl': pnl_values,
                'by_win_rate': sorted([(sid, s.win_rate) for sid, s in zip(session_ids, sessions_data)], 
                                    key=lambda x: x[1], reverse=True),
                'by_profit_factor': sorted([(sid, s.profit_factor) for sid, s in zip(session_ids, sessions_data)], 
                                         key=lambda x: x[1], reverse=True)
            }
            
            # Generate insights
            comparison_data['insights'] = self._generate_comparison_insights(sessions_data, session_ids)
            
            return comparison_data
            
        except Exception as e:
            self.logger.error(f"Error generating session comparison: {e}", exc_info=True)
            return {'error': f'Comparison failed: {str(e)}'}

    def get_cross_session_analysis(self) -> Dict[str, Any]:
        """Get comprehensive cross-session analysis"""
        try:
            cached = self._get_cached_analysis()
            if cached:
                return cached
            
            analysis = {
                'summary': self._get_cross_session_summary(),
                'strategy_performance': self._analyze_strategy_performance(),
                'configuration_impact': self._analyze_configuration_impact(),
                'learning_curve': self._analyze_learning_curve(),
                'temporal_patterns': self._analyze_temporal_patterns(),
                'risk_evolution': self._analyze_risk_evolution(),
                'optimization_insights': self._generate_optimization_insights()
            }
            
            self._cache_analysis(analysis)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating cross-session analysis: {e}", exc_info=True)
            return {'error': f'Analysis failed: {str(e)}'}

    def get_historical_trends(self, period_days: int = 30) -> Dict[str, Any]:
        """Get historical performance trends over specified period"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=period_days)
            
            relevant_sessions = [
                session for session in self.session_history.values()
                if datetime.fromisoformat(session.metadata.start_time.replace('Z', '+00:00')) > cutoff_date
            ]
            
            if not relevant_sessions:
                return {'error': f'No sessions found in the last {period_days} days'}
            
            # Sort by start time
            relevant_sessions.sort(key=lambda x: x.metadata.start_time)
            
            trends = {
                'period_days': period_days,
                'total_sessions': len(relevant_sessions),
                'performance_trend': self._calculate_performance_trend(relevant_sessions),
                'win_rate_trend': self._calculate_win_rate_trend(relevant_sessions),
                'risk_trend': self._calculate_risk_trend(relevant_sessions),
                'activity_trend': self._calculate_activity_trend(relevant_sessions),
                'strategy_usage_evolution': self._analyze_strategy_usage_evolution(relevant_sessions)
            }
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error calculating historical trends: {e}", exc_info=True)
            return {'error': f'Trend analysis failed: {str(e)}'}

    def export_session_data(self, session_ids: Optional[List[str]] = None, 
                           format: str = 'json') -> str:
        """Export session data to file"""
        try:
            if session_ids is None:
                session_ids = list(self.session_history.keys())
            
            export_data = {
                'export_timestamp': datetime.now(timezone.utc).isoformat(),
                'sessions': {},
                'summary': {
                    'total_sessions': len(session_ids),
                    'date_range': self._get_date_range(session_ids)
                }
            }
            
            for session_id in session_ids:
                if session_id in self.session_history:
                    export_data['sessions'][session_id] = asdict(self.session_history[session_id])
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"session_export_{timestamp}.{format}"
            filepath = self.analysis_dir / filename
            
            if format.lower() == 'json':
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif format.lower() == 'csv':
                # Convert to DataFrame for CSV export
                df = self._sessions_to_dataframe(session_ids)
                df.to_csv(filepath, index=False)
            
            self.logger.info(f"Session data exported to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error exporting session data: {e}", exc_info=True)
            raise

    # Helper methods
    def _save_session_metadata(self, metadata: SessionMetadata):
        """Save session metadata to file"""
        session_dir = self.sessions_dir / metadata.session_id
        metadata_file = session_dir / "metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)

    def _archive_session(self, session_id: str, summary: SessionSummary):
        """Archive completed session"""
        # Create archive directory for this session
        archive_session_dir = self.archive_dir / session_id
        archive_session_dir.mkdir(exist_ok=True)
        
        # Save session summary
        summary_file = archive_session_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2, default=str)
        
        # Move session files from active to archive
        active_session_dir = self.sessions_dir / session_id
        if active_session_dir.exists():
            import shutil
            for item in active_session_dir.iterdir():
                dest_path = archive_session_dir / item.name
                if dest_path.exists():
                    dest_path.unlink()  # Remove existing file
                shutil.move(str(item), str(archive_session_dir))
            active_session_dir.rmdir()

    def _load_session_history(self):
        """Load session history from archive"""
        try:
            for session_dir in self.archive_dir.iterdir():
                if session_dir.is_dir():
                    summary_file = session_dir / "summary.json"
                    if summary_file.exists():
                        with open(summary_file, 'r') as f:
                            summary_data = json.load(f)
                        
                        # Reconstruct SessionSummary object
                        metadata = SessionMetadata(**summary_data['metadata'])
                        summary = SessionSummary(
                            metadata=metadata,
                            performance_metrics=summary_data['performance_metrics'],
                            trade_statistics=summary_data['trade_statistics'],
                            risk_metrics=summary_data['risk_metrics'],
                            alert_summary=summary_data['alert_summary'],
                            duration_hours=summary_data['duration_hours'],
                            total_trades=summary_data['total_trades'],
                            win_rate=summary_data['win_rate'],
                            profit_factor=summary_data['profit_factor'],
                            max_drawdown=summary_data['max_drawdown'],
                            final_pnl=summary_data['final_pnl']
                        )
                        
                        self.session_history[session_dir.name] = summary
            
            self.logger.info(f"Loaded {len(self.session_history)} sessions from history")
            
        except Exception as e:
            self.logger.error(f"Error loading session history: {e}", exc_info=True)

    def _get_cached_analysis(self) -> Optional[Dict[str, Any]]:
        """Get cached analysis if valid"""
        if (self._cache_timestamp and 
            (datetime.now().timestamp() - self._cache_timestamp) < self._cache_validity_seconds):
            return self._analysis_cache
        return None

    def _cache_analysis(self, analysis: Dict[str, Any]):
        """Cache analysis results"""
        self._analysis_cache = analysis
        self._cache_timestamp = datetime.now().timestamp()

    def _invalidate_cache(self):
        """Invalidate analysis cache"""
        self._analysis_cache.clear()
        self._cache_timestamp = None

    def _get_cross_session_summary(self) -> Dict[str, Any]:
        """Get cross-session summary statistics"""
        if not self.session_history:
            return {'error': 'No session history available'}
        
        sessions = list(self.session_history.values())
        
        return {
            'total_sessions': len(sessions),
            'total_trades': sum(s.total_trades for s in sessions),
            'average_win_rate': np.mean([s.win_rate for s in sessions]),
            'average_profit_factor': np.mean([s.profit_factor for s in sessions if s.profit_factor > 0]),
            'total_pnl': sum(s.final_pnl for s in sessions),
            'average_session_duration': np.mean([s.duration_hours for s in sessions]),
            'best_session': max(sessions, key=lambda x: x.final_pnl).metadata.session_id,
            'worst_session': min(sessions, key=lambda x: x.final_pnl).metadata.session_id,
            'most_active_strategy': self._get_most_active_strategy(sessions),
            'date_range': self._get_date_range(list(self.session_history.keys()))
        }

    def _analyze_strategy_performance(self) -> Dict[str, Any]:
        """Analyze performance by strategy"""
        strategy_stats = {}
        
        for session in self.session_history.values():
            strategy = session.metadata.strategy_name
            if not strategy:
                continue
                
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'sessions': [],
                    'total_trades': 0,
                    'total_pnl': 0,
                    'win_rates': [],
                    'profit_factors': [],
                    'max_drawdowns': []
                }
            
            stats = strategy_stats[strategy]
            stats['sessions'].append(session.metadata.session_id)
            stats['total_trades'] += session.total_trades
            stats['total_pnl'] += session.final_pnl
            stats['win_rates'].append(session.win_rate)
            stats['profit_factors'].append(session.profit_factor)
            stats['max_drawdowns'].append(session.max_drawdown)
        
        # Calculate averages and rankings
        for strategy, stats in strategy_stats.items():
            stats['session_count'] = len(stats['sessions'])
            stats['avg_win_rate'] = np.mean(stats['win_rates'])
            stats['avg_profit_factor'] = np.mean(stats['profit_factors'])
            stats['avg_max_drawdown'] = np.mean(stats['max_drawdowns'])
            stats['pnl_per_session'] = stats['total_pnl'] / stats['session_count']
        
        return strategy_stats

    def _analyze_configuration_impact(self) -> Dict[str, Any]:
        """Analyze impact of different configurations"""
        config_impact = {
            'leverage_analysis': {},
            'timeframe_analysis': {},
            'symbol_analysis': {}
        }
        
        # Group sessions by configuration parameters
        leverage_groups = {}
        timeframe_groups = {}
        symbol_groups = {}
        
        for session in self.session_history.values():
            metadata = session.metadata
            
            # Leverage analysis
            if metadata.leverage:
                lev_key = f"{metadata.leverage}x"
                if lev_key not in leverage_groups:
                    leverage_groups[lev_key] = []
                leverage_groups[lev_key].append(session)
            
            # Timeframe analysis
            if metadata.timeframe:
                if metadata.timeframe not in timeframe_groups:
                    timeframe_groups[metadata.timeframe] = []
                timeframe_groups[metadata.timeframe].append(session)
            
            # Symbol analysis
            if metadata.symbol:
                if metadata.symbol not in symbol_groups:
                    symbol_groups[metadata.symbol] = []
                symbol_groups[metadata.symbol].append(session)
        
        # Calculate statistics for each group
        config_impact['leverage_analysis'] = self._calculate_group_stats(leverage_groups)
        config_impact['timeframe_analysis'] = self._calculate_group_stats(timeframe_groups)
        config_impact['symbol_analysis'] = self._calculate_group_stats(symbol_groups)
        
        return config_impact

    def _analyze_learning_curve(self) -> Dict[str, Any]:
        """Analyze learning curve and improvement over time"""
        if len(self.session_history) < 3:
            return {'error': 'Insufficient sessions for learning curve analysis'}
        
        # Sort sessions by start time
        sessions = sorted(self.session_history.values(), 
                         key=lambda x: x.metadata.start_time)
        
        # Calculate rolling averages
        window_size = min(5, len(sessions) // 2)
        
        metrics = {
            'win_rate_trend': [],
            'pnl_trend': [],
            'profit_factor_trend': [],
            'trade_frequency_trend': []
        }
        
        for i in range(window_size, len(sessions) + 1):
            window_sessions = sessions[i-window_size:i]
            
            metrics['win_rate_trend'].append({
                'session_index': i,
                'value': np.mean([s.win_rate for s in window_sessions])
            })
            
            metrics['pnl_trend'].append({
                'session_index': i,
                'value': np.mean([s.final_pnl for s in window_sessions])
            })
            
            metrics['profit_factor_trend'].append({
                'session_index': i,
                'value': np.mean([s.profit_factor for s in window_sessions if s.profit_factor > 0])
            })
            
            avg_trades_per_hour = np.mean([s.total_trades / s.duration_hours 
                                         for s in window_sessions if s.duration_hours > 0])
            metrics['trade_frequency_trend'].append({
                'session_index': i,
                'value': avg_trades_per_hour
            })
        
        # Calculate improvement rates
        improvement_analysis = {}
        for metric_name, trend_data in metrics.items():
            if len(trend_data) >= 2:
                first_value = trend_data[0]['value']
                last_value = trend_data[-1]['value']
                improvement_rate = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0
                improvement_analysis[metric_name] = {
                    'improvement_rate_pct': improvement_rate,
                    'trend_direction': 'improving' if improvement_rate > 5 else 'declining' if improvement_rate < -5 else 'stable'
                }
        
        return {
            'trends': metrics,
            'improvement_analysis': improvement_analysis,
            'learning_insights': self._generate_learning_insights(improvement_analysis)
        }

    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in trading performance"""
        temporal_analysis = {
            'hourly_patterns': {},
            'daily_patterns': {},
            'weekly_patterns': {},
            'monthly_patterns': {}
        }
        
        for session in self.session_history.values():
            start_time = datetime.fromisoformat(session.metadata.start_time.replace('Z', '+00:00'))
            
            # Hourly patterns
            hour = start_time.hour
            if hour not in temporal_analysis['hourly_patterns']:
                temporal_analysis['hourly_patterns'][hour] = []
            temporal_analysis['hourly_patterns'][hour].append(session.final_pnl)
            
            # Daily patterns
            day = start_time.strftime('%A')
            if day not in temporal_analysis['daily_patterns']:
                temporal_analysis['daily_patterns'][day] = []
            temporal_analysis['daily_patterns'][day].append(session.final_pnl)
            
            # Weekly patterns
            week = start_time.isocalendar()[1]
            if week not in temporal_analysis['weekly_patterns']:
                temporal_analysis['weekly_patterns'][week] = []
            temporal_analysis['weekly_patterns'][week].append(session.final_pnl)
            
            # Monthly patterns
            month = start_time.strftime('%B')
            if month not in temporal_analysis['monthly_patterns']:
                temporal_analysis['monthly_patterns'][month] = []
            temporal_analysis['monthly_patterns'][month].append(session.final_pnl)
        
        # Calculate statistics for each pattern
        for pattern_type, patterns in temporal_analysis.items():
            for period, pnl_values in patterns.items():
                temporal_analysis[pattern_type][period] = {
                    'session_count': len(pnl_values),
                    'avg_pnl': np.mean(pnl_values),
                    'total_pnl': sum(pnl_values),
                    'win_rate': len([p for p in pnl_values if p > 0]) / len(pnl_values) * 100
                }
        
        return temporal_analysis

    def _analyze_risk_evolution(self) -> Dict[str, Any]:
        """Analyze risk management evolution over time"""
        # This would analyze how risk parameters and outcomes change over time
        risk_evolution = {
            'drawdown_improvement': [],
            'position_sizing_evolution': [],
            'leverage_usage_patterns': []
        }
        
        sessions = sorted(self.session_history.values(), 
                         key=lambda x: x.metadata.start_time)
        
        for i, session in enumerate(sessions):
            risk_evolution['drawdown_improvement'].append({
                'session_index': i + 1,
                'max_drawdown': session.max_drawdown,
                'session_id': session.metadata.session_id
            })
            
            if session.metadata.leverage:
                risk_evolution['leverage_usage_patterns'].append({
                    'session_index': i + 1,
                    'leverage': session.metadata.leverage,
                    'pnl': session.final_pnl
                })
        
        return risk_evolution

    def _generate_optimization_insights(self) -> List[str]:
        """Generate optimization insights based on historical data"""
        insights = []
        
        if len(self.session_history) < 3:
            return ["Insufficient data for optimization insights"]
        
        # Strategy performance insights
        strategy_perf = self._analyze_strategy_performance()
        if strategy_perf:
            best_strategy = max(strategy_perf.items(), key=lambda x: x[1]['total_pnl'])
            insights.append(f"Best performing strategy: {best_strategy[0]} with {best_strategy[1]['total_pnl']:.2f} total PnL")
        
        # Configuration insights
        config_impact = self._analyze_configuration_impact()
        leverage_analysis = config_impact.get('leverage_analysis', {})
        if leverage_analysis:
            best_leverage = max(leverage_analysis.items(), key=lambda x: x[1]['avg_pnl'])
            insights.append(f"Optimal leverage appears to be {best_leverage[0]} with {best_leverage[1]['avg_pnl']:.2f} avg PnL per session")
        
        # Temporal insights
        temporal = self._analyze_temporal_patterns()
        daily_patterns = temporal.get('daily_patterns', {})
        if daily_patterns:
            best_day = max(daily_patterns.items(), key=lambda x: x[1]['avg_pnl'])
            insights.append(f"Best trading day: {best_day[0]} with {best_day[1]['avg_pnl']:.2f} average PnL")
        
        return insights

    def _generate_comparison_insights(self, sessions: List[SessionSummary], session_ids: List[str]) -> List[str]:
        """Generate insights from session comparison"""
        insights = []
        
        if len(sessions) < 2:
            return ["Need at least 2 sessions for comparison insights"]
        
        # Performance insights
        pnl_values = [s.final_pnl for s in sessions]
        best_idx = pnl_values.index(max(pnl_values))
        worst_idx = pnl_values.index(min(pnl_values))
        
        insights.append(f"Best session: {session_ids[best_idx]} with {pnl_values[best_idx]:.2f} PnL")
        insights.append(f"Worst session: {session_ids[worst_idx]} with {pnl_values[worst_idx]:.2f} PnL")
        
        # Strategy insights
        strategies = [s.metadata.strategy_name for s in sessions if s.metadata.strategy_name]
        if len(set(strategies)) > 1:
            strategy_pnl = {}
            for session, strategy in zip(sessions, strategies):
                if strategy not in strategy_pnl:
                    strategy_pnl[strategy] = []
                strategy_pnl[strategy].append(session.final_pnl)
            
            for strategy, pnls in strategy_pnl.items():
                avg_pnl = np.mean(pnls)
                insights.append(f"Strategy {strategy}: {avg_pnl:.2f} average PnL across {len(pnls)} session(s)")
        
        return insights

    def _calculate_group_stats(self, groups: Dict[str, List[SessionSummary]]) -> Dict[str, Any]:
        """Calculate statistics for grouped sessions"""
        group_stats = {}
        
        for group_name, sessions in groups.items():
            if not sessions:
                continue
                
            group_stats[group_name] = {
                'session_count': len(sessions),
                'total_pnl': sum(s.final_pnl for s in sessions),
                'avg_pnl': np.mean([s.final_pnl for s in sessions]),
                'avg_win_rate': np.mean([s.win_rate for s in sessions]),
                'avg_profit_factor': np.mean([s.profit_factor for s in sessions if s.profit_factor > 0]),
                'total_trades': sum(s.total_trades for s in sessions)
            }
        
        return group_stats

    def _calculate_performance_trend(self, sessions: List[SessionSummary]) -> Dict[str, Any]:
        """Calculate performance trend over time"""
        pnl_values = [s.final_pnl for s in sessions]
        
        if len(pnl_values) < 2:
            return {'trend': 'insufficient_data'}
        
        # Simple linear trend
        x = range(len(pnl_values))
        slope = np.polyfit(x, pnl_values, 1)[0]
        
        return {
            'trend': 'improving' if slope > 0 else 'declining',
            'slope': slope,
            'recent_performance': pnl_values[-3:] if len(pnl_values) >= 3 else pnl_values
        }

    def _calculate_win_rate_trend(self, sessions: List[SessionSummary]) -> Dict[str, Any]:
        """Calculate win rate trend over time"""
        win_rates = [s.win_rate for s in sessions]
        
        if len(win_rates) < 2:
            return {'trend': 'insufficient_data'}
        
        x = range(len(win_rates))
        slope = np.polyfit(x, win_rates, 1)[0]
        
        return {
            'trend': 'improving' if slope > 0 else 'declining',
            'slope': slope,
            'current_avg': np.mean(win_rates[-3:]) if len(win_rates) >= 3 else np.mean(win_rates)
        }

    def _calculate_risk_trend(self, sessions: List[SessionSummary]) -> Dict[str, Any]:
        """Calculate risk trend over time"""
        drawdowns = [s.max_drawdown for s in sessions]
        
        if len(drawdowns) < 2:
            return {'trend': 'insufficient_data'}
        
        x = range(len(drawdowns))
        slope = np.polyfit(x, drawdowns, 1)[0]
        
        return {
            'trend': 'improving' if slope < 0 else 'worsening',  # Lower drawdown is better
            'slope': slope,
            'recent_avg_drawdown': np.mean(drawdowns[-3:]) if len(drawdowns) >= 3 else np.mean(drawdowns)
        }

    def _calculate_activity_trend(self, sessions: List[SessionSummary]) -> Dict[str, Any]:
        """Calculate trading activity trend"""
        activity_rates = [s.total_trades / s.duration_hours for s in sessions if s.duration_hours > 0]
        
        if len(activity_rates) < 2:
            return {'trend': 'insufficient_data'}
        
        x = range(len(activity_rates))
        slope = np.polyfit(x, activity_rates, 1)[0]
        
        return {
            'trend': 'increasing' if slope > 0 else 'decreasing',
            'slope': slope,
            'avg_trades_per_hour': np.mean(activity_rates)
        }

    def _analyze_strategy_usage_evolution(self, sessions: List[SessionSummary]) -> Dict[str, Any]:
        """Analyze how strategy usage changes over time"""
        strategy_usage = {}
        
        for i, session in enumerate(sessions):
            if session.metadata.strategy_name:
                strategy = session.metadata.strategy_name
                if strategy not in strategy_usage:
                    strategy_usage[strategy] = []
                strategy_usage[strategy].append(i)
        
        return {
            'strategy_adoption_timeline': strategy_usage,
            'strategy_diversity': len(strategy_usage),
            'most_recent_strategies': list(set(s.metadata.strategy_name for s in sessions[-5:] 
                                             if s.metadata.strategy_name))
        }

    def _get_most_active_strategy(self, sessions: List[SessionSummary]) -> str:
        """Get the most frequently used strategy"""
        strategy_counts = {}
        for session in sessions:
            if session.metadata.strategy_name:
                strategy = session.metadata.strategy_name
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else "Unknown"

    def _get_date_range(self, session_ids: List[str]) -> Dict[str, str]:
        """Get date range for given sessions"""
        dates = []
        for session_id in session_ids:
            if session_id in self.session_history:
                start_time = self.session_history[session_id].metadata.start_time
                dates.append(datetime.fromisoformat(start_time.replace('Z', '+00:00')))
        
        if not dates:
            return {'start': 'Unknown', 'end': 'Unknown'}
        
        return {
            'start': min(dates).strftime('%Y-%m-%d'),
            'end': max(dates).strftime('%Y-%m-%d')
        }

    def _sessions_to_dataframe(self, session_ids: List[str]) -> pd.DataFrame:
        """Convert sessions to pandas DataFrame for CSV export"""
        rows = []
        
        for session_id in session_ids:
            if session_id in self.session_history:
                session = self.session_history[session_id]
                metadata = session.metadata
                
                row = {
                    'session_id': metadata.session_id,
                    'start_time': metadata.start_time,
                    'end_time': metadata.end_time,
                    'strategy_name': metadata.strategy_name,
                    'symbol': metadata.symbol,
                    'timeframe': metadata.timeframe,
                    'leverage': metadata.leverage,
                    'duration_hours': session.duration_hours,
                    'total_trades': session.total_trades,
                    'win_rate': session.win_rate,
                    'profit_factor': session.profit_factor,
                    'max_drawdown': session.max_drawdown,
                    'final_pnl': session.final_pnl,
                    'initial_balance': metadata.initial_balance,
                    'final_balance': metadata.final_balance
                }
                rows.append(row)
        
        return pd.DataFrame(rows)

    def _generate_learning_insights(self, improvement_analysis: Dict[str, Any]) -> List[str]:
        """Generate learning insights from improvement analysis"""
        insights = []
        
        for metric, analysis in improvement_analysis.items():
            direction = analysis['trend_direction']
            rate = analysis['improvement_rate_pct']
            
            if direction == 'improving':
                insights.append(f"{metric.replace('_', ' ').title()}: Improving by {rate:.1f}%")
            elif direction == 'declining':
                insights.append(f"{metric.replace('_', ' ').title()}: Declining by {abs(rate):.1f}%")
            else:
                insights.append(f"{metric.replace('_', ' ').title()}: Stable performance")
        
        return insights

    def get_active_sessions(self) -> Dict[str, SessionMetadata]:
        """Get all active sessions"""
        return self.active_sessions.copy()

    def end_active_sessions(self, reason: str = "Manual session end") -> List[SessionSummary]:
        """End all active sessions and return their summaries"""
        summaries = []
        active_session_ids = list(self.active_sessions.keys())
        
        for session_id in active_session_ids:
            try:
                summary = self.end_session(session_id)
                summary.metadata.notes = reason
                summaries.append(summary)
                self.logger.info(f"Ended session {session_id}: {reason}")
            except Exception as e:
                self.logger.error(f"Failed to end session {session_id}: {e}")
                
        return summaries

    def get_session_history(self) -> Dict[str, SessionSummary]:
        """Get complete session history"""
        return self.session_history.copy()

    def get_current_session_id(self) -> Optional[str]:
        """Get current active session ID"""
        return self.current_session_id 