import logging
import csv
import json
from typing import List, Dict, Optional, Any
import os
import time
import threading
import tempfile
import shutil
import pandas as pd

class PerformanceTracker:
    """
    Tracks and persists trade performance metrics for the trading bot.
    Metrics include win rate, profit/loss, cumulative returns, max drawdown, expectancy, profit factor, average trade duration, and rolling metrics.
    Results are saved to CSV and JSON for cross-session tracking.
    Supports per-strategy and per-symbol breakdowns and pandas DataFrame export.
    """
    def __init__(self, log_dir: str = "performance", logger: Optional[logging.Logger] = None):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.trades: List[Dict] = []
        self.cumulative_pnl = 0.0
        self.max_drawdown = 0.0
        self.high_watermark = 0.0
        self._lock = threading.Lock()

    def record_trade(self, trade: Dict):
        """
        Record a completed trade and update performance metrics.
        Args:
            trade: Dict with trade details (must include 'pnl', 'side', 'entry_price', 'exit_price', 'timestamp', etc.)
        """
        try:
            with self._lock:
                self.trades.append(trade)
                pnl = float(trade.get('pnl', 0.0))
                self.cumulative_pnl += pnl
                self.high_watermark = max(self.high_watermark, self.cumulative_pnl)
                drawdown = self.high_watermark - self.cumulative_pnl
                self.max_drawdown = max(self.max_drawdown, drawdown)
                self.logger.info(f"Recorded trade: {trade}. Cumulative PnL: {self.cumulative_pnl}, Max Drawdown: {self.max_drawdown}")
        except Exception as exc:
            self.logger.error(f"Failed to record trade: {exc}")

    def win_rate(self, trades: Optional[List[Dict]] = None) -> float:
        """
        Calculate win rate as a percentage.
        Returns:
            Win rate (0-100)
        """
        trades = trades if trades is not None else self.trades
        wins = sum(1 for t in trades if float(t.get('pnl', 0.0)) > 0)
        total = len(trades)
        return (wins / total * 100) if total > 0 else 0.0

    def cumulative_return(self, trades: Optional[List[Dict]] = None) -> float:
        """
        Return cumulative PnL.
        """
        trades = trades if trades is not None else self.trades
        return sum(float(t.get('pnl', 0.0)) for t in trades)

    def max_drawdown_value(self, trades: Optional[List[Dict]] = None) -> float:
        """
        Return the maximum drawdown value.
        """
        trades = trades if trades is not None else self.trades
        high = 0.0
        max_dd = 0.0
        cum = 0.0
        for t in trades:
            cum += float(t.get('pnl', 0.0))
            high = max(high, cum)
            max_dd = max(max_dd, high - cum)
        return max_dd

    def average_trade_duration(self, trades: Optional[List[Dict]] = None) -> float:
        """
        Compute average trade duration in seconds.
        """
        trades = trades if trades is not None else self.trades
        durations = []
        for t in trades:
            entry = t.get('entry_timestamp') or t.get('entry_time') or t.get('timestamp')
            exit_ = t.get('exit_timestamp') or t.get('exit_time') or t.get('close_timestamp')
            if entry and exit_:
                try:
                    durations.append(float(exit_) - float(entry))
                except Exception:
                    continue
        return sum(durations) / len(durations) if durations else 0.0

    def expectancy(self, trades: Optional[List[Dict]] = None) -> float:
        """
        Compute expectancy: (win_rate × avg_win) – (loss_rate × avg_loss)
        """
        trades = trades if trades is not None else self.trades
        wins = [float(t.get('pnl', 0.0)) for t in trades if float(t.get('pnl', 0.0)) > 0]
        losses = [abs(float(t.get('pnl', 0.0))) for t in trades if float(t.get('pnl', 0.0)) < 0]
        total = len(trades)
        win_rate = len(wins) / total if total else 0.0
        loss_rate = len(losses) / total if total else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        return (win_rate * avg_win) - (loss_rate * avg_loss)

    def profit_factor(self, trades: Optional[List[Dict]] = None) -> float:
        """
        Compute profit factor: sum(wins) / abs(sum(losses))
        """
        trades = trades if trades is not None else self.trades
        wins = sum(float(t.get('pnl', 0.0)) for t in trades if float(t.get('pnl', 0.0)) > 0)
        losses = sum(float(t.get('pnl', 0.0)) for t in trades if float(t.get('pnl', 0.0)) < 0)
        return wins / abs(losses) if losses != 0 else float('inf')

    def group_metrics(self, by: str = 'strategy') -> Dict[str, Dict[str, Any]]:
        """
        Compute win rate, PnL, and trade count per group (strategy or symbol).
        Returns a dict: {group_value: {win_rate, cumulative_return, trade_count}}
        """
        groups = {}
        for t in self.trades:
            key = t.get(by, 'unknown')
            if key not in groups:
                groups[key] = []
            groups[key].append(t)
        metrics = {}
        for key, group_trades in groups.items():
            metrics[key] = {
                'win_rate': self.win_rate(group_trades),
                'cumulative_return': self.cumulative_return(group_trades),
                'trade_count': len(group_trades),
                'profit_factor': self.profit_factor(group_trades),
                'expectancy': self.expectancy(group_trades),
                'avg_duration': self.average_trade_duration(group_trades),
                'max_drawdown': self.max_drawdown_value(group_trades),
            }
        return metrics

    def rolling_drawdown_curve(self, window: int = 20) -> List[float]:
        """
        Compute rolling max drawdown curve over the last N trades.
        Returns a list of drawdown values.
        """
        curve = []
        for i in range(len(self.trades)):
            window_trades = self.trades[max(0, i - window + 1):i + 1]
            curve.append(self.max_drawdown_value(window_trades))
        return curve

    def rolling_sharpe(self, window: int = 20, risk_free_rate: float = 0.0) -> List[float]:
        """
        Compute rolling Sharpe ratio over the last N trades.
        Returns a list of Sharpe ratios.
        """
        import numpy as np
        ratios = []
        for i in range(len(self.trades)):
            window_trades = self.trades[max(0, i - window + 1):i + 1]
            returns = [float(t.get('pnl', 0.0)) for t in window_trades]
            if len(returns) > 1:
                mean = np.mean(returns)
                std = np.std(returns)
                sharpe = (mean - risk_free_rate) / std if std != 0 else 0.0
                ratios.append(sharpe)
            else:
                ratios.append(0.0)
        return ratios

    def persist_to_csv(self, filename: str = "performance_log.csv"):
        """
        Persist all trades to a CSV file in the log directory using atomic write.
        """
        path = os.path.join(self.log_dir, filename)
        try:
            if not self.trades:
                self.logger.warning("No trades to persist to CSV.")
                return
            with self._lock:
                with tempfile.NamedTemporaryFile('w', delete=False, newline='') as tmpfile:
                    writer = csv.DictWriter(tmpfile, fieldnames=self.trades[0].keys())
                    writer.writeheader()
                    writer.writerows(self.trades)
                    tempname = tmpfile.name
                shutil.move(tempname, path)
            self.logger.info(f"Persisted trades to CSV: {path}")
        except Exception as exc:
            self.logger.error(f"Failed to persist trades to CSV: {exc}")

    def persist_to_json(self, filename: str = "performance_log.json"):
        """
        Persist all trades to a JSON file in the log directory using atomic write.
        """
        path = os.path.join(self.log_dir, filename)
        try:
            with self._lock:
                with tempfile.NamedTemporaryFile('w', delete=False) as tmpfile:
                    json.dump(self.trades, tmpfile, indent=2)
                    tempname = tmpfile.name
                shutil.move(tempname, path)
            self.logger.info(f"Persisted trades to JSON: {path}")
        except Exception as exc:
            self.logger.error(f"Failed to persist trades to JSON: {exc}")

    def close_session(self):
        """
        Persist trades to disk on shutdown (CSV and JSON).
        """
        self.persist_to_csv()
        self.persist_to_json()
        self.logger.info("PerformanceTracker session closed and data persisted.")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return all trades as a pandas DataFrame for in-memory analysis.
        """
        return pd.DataFrame(self.trades) 