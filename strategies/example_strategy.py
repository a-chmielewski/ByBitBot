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
                self.log_state_change('entry_signal', f"{type(self).__name__}: Entry conditions met, placing order.")
                # Do not set a default order size here; OrderManager will enforce Bybit minimum
                order_details = {"side": "buy", "size": self.config.get("order_size"), "price": self.data['close'].iloc[-1]}
                risk_params = self.get_risk_parameters()
                if "stop_loss" in risk_params:
                    order_details["stop_loss"] = risk_params["stop_loss"]
                if "take_profit" in risk_params:
                    order_details["take_profit"] = risk_params["take_profit"]
                return order_details
            else:
                self.log_state_change('waiting_for_entry', f"{type(self).__name__}: Waiting for entry opportunity.")
        return None

    def check_exit(self, position: Any) -> bool:
        # Example exit logic: Exit if price crosses below SMA
        if isinstance(self.data, pd.DataFrame):
            if self.data['close'].iloc[-1] < self.data['sma'].iloc[-1]:
                self.log_state_change('exit_signal', f"{type(self).__name__}: Exit conditions met, closing position.")
                return True
            else:
                self.log_state_change('waiting_for_exit', f"{type(self).__name__}: In position, monitoring for exit.")
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