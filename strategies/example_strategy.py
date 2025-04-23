from .strategy_template import StrategyTemplate
from typing import Any, Dict, Optional
import pandas as pd

class StrategyExample(StrategyTemplate):
    """
    Example strategy implementation for demonstration purposes.
    Demonstrates position tracking, config usage, and all hooks.
    """
    def on_init(self) -> None:
        """
        Optional: Pre-start setup (e.g., warm up indicators on extra historical data).
        """
        self.logger.info("StrategyExample on_init called (no-op by default).")

    def init_indicators(self) -> None:
        # Example: Calculate a simple moving average
        if isinstance(self.data, pd.DataFrame):
            window = self.config.get('sma_window', 20)  # Use strategy-specific config if present
            self.data['sma'] = self.data['close'].rolling(window=window).mean()
        # Add other indicators as needed

    def check_entry(self) -> Optional[Dict[str, Any]]:
        # Example entry logic: Buy if price crosses above SMA
        if isinstance(self.data, pd.DataFrame):
            if self.data['close'].iloc[-1] > self.data['sma'].iloc[-1]:
                return {"side": "buy", "size": 1.0, "price": self.data['close'].iloc[-1]}
        return None

    def check_exit(self, position: Any) -> bool:
        # Example exit logic: Exit if price crosses below SMA
        if isinstance(self.data, pd.DataFrame):
            if self.data['close'].iloc[-1] < self.data['sma'].iloc[-1]:
                return True
        return False

    def get_risk_parameters(self) -> Dict[str, Any]:
        # Example risk parameters, using config if available
        return {
            "stop_loss": self.config.get('stop_loss', 0.01),
            "take_profit": self.config.get('take_profit', 0.02)
        }

    def on_order_update(self, order: Dict[str, Any]) -> None:
        """
        Update position state when an order is filled.
        """
        if order.get('main_order', {}).get('result', {}).get('order_status') in ('Filled', 'filled'):
            self.position = order  # Store full order details or extract as needed

    def on_trade_update(self, trade: Dict[str, Any]) -> None:
        """
        Clear position on trade exit.
        """
        if trade.get('exit'):
            self.position = None

    def on_error(self, exception: Exception) -> None:
        """
        Handle errors raised during strategy logic.
        """
        self.logger.error(f"StrategyExample encountered an error: {exception}") 