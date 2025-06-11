from .strategy_template import StrategyTemplate
from typing import Any, Dict, Optional
import pandas as pd


class StrategyExample(StrategyTemplate):
    """
    Example strategy implementation for demonstration purposes.
    Demonstrates position tracking, config usage, and all hooks.
    It now uses the base class for order pending/active state management.
    """
    
    # Hide this strategy from selection menu - it's for development/reference only
    SHOW_IN_SELECTION: bool = False

    def on_init(self) -> None:
        """
        Optional: Pre-start setup (e.g., warm up indicators on extra historical data).
        """
        super().on_init()  # Call base on_init if it does anything in the future
        self.logger.info(f"{type(self).__name__} on_init called.")
        # Example: Access a strategy-specific config value
        self.sma_window = (
            self.config.get("strategy_configs", {})
            .get(type(self).__name__, {})
            .get("sma_window", 20)
        )
        self.logger.info(f"{type(self).__name__} using SMA window: {self.sma_window}")

    def init_indicators(self) -> None:
        if isinstance(self.data, pd.DataFrame) and not self.data.empty:
            # Ensure 'close' column exists
            if "close" not in self.data.columns:
                self.logger.error(
                    f"'{type(self).__name__}': 'close' column not found in data. Available: {self.data.columns}"
                )
                self.data["sma"] = pd.Series(dtype="float64")  # empty series
                return
            if len(self.data) < self.sma_window:
                self.logger.warning(
                    f"'{type(self).__name__}': Data length ({len(self.data)}) is less than SMA window ({self.sma_window}). SMA will have NaNs."
                )
            self.data = self.data.assign(
                sma=self.data["close"].rolling(window=self.sma_window, min_periods=1).mean()
            )
            self.logger.info(
                f"'{type(self).__name__}': SMA calculated with window {self.sma_window}. Last SMA: {self.data['sma'].iloc[-1] if not self.data['sma'].empty else 'N/A'}"
            )
        else:
            self.logger.warning(
                f"'{type(self).__name__}': Data for init_indicators is not a DataFrame or is empty."
            )
            if isinstance(
                self.data, pd.DataFrame
            ):  # if it is a dataframe, add empty sma column
                self.data["sma"] = pd.Series(dtype="float64")

    def _check_entry_conditions(self) -> Optional[Dict[str, Any]]:
        if (
            not isinstance(self.data, pd.DataFrame)
            or self.data.empty
            or "close" not in self.data.columns
            or "sma" not in self.data.columns
        ):
            # self.logger.debug(f"{type(self).__name__}: Data not ready for entry check.")
            return None

        # Ensure sma is not all NaN, which can happen if window is too large for initial data
        if pd.isna(self.data['sma'].iloc[-1]):
            # self.logger.debug(f"{type(self).__name__}: SMA is NaN. Cannot check entry.")
            return None

        # Example entry logic: Buy if price crosses above SMA
        if self.data["close"].iloc[-1] > self.data["sma"].iloc[-1]:
            self.log_state_change(
                "entry_signal",
                f"{type(self).__name__}: Entry conditions met (Close {self.data['close'].iloc[-1]} > SMA {self.data['sma'].iloc[-1]}). Preparing order.",
            )

            order_size_config = (
                self.config.get("strategy_configs", {})
                .get(type(self).__name__, {})
                .get("order_size")
            )
            if order_size_config is None:
                self.logger.error(f"{type(self).__name__}: Missing 'order_size' in config. Cannot submit order.")
                return None

            order_details = {
                "side": "buy",
                "price": self.data["close"].iloc[-1],
                "size": order_size_config,  # Can be None, OrderManager will handle minimums
            }
            # Risk parameters (sl_pct, tp_pct) will be added by the base check_entry method
            return order_details
        else:
            self.log_state_change(
                "waiting_for_entry",
                f"{type(self).__name__}: Waiting for entry (Close {self.data['close'].iloc[-1]} <= SMA {self.data['sma'].iloc[-1]}).",
            )
        return None

    def check_exit(self, position: Any) -> bool:
        if (
            not isinstance(self.data, pd.DataFrame)
            or self.data.empty
            or "close" not in self.data.columns
            or "sma" not in self.data.columns
        ):
            # self.logger.debug(f"{type(self).__name__}: Data not ready for exit check.")
            return False

        # Ensure sma is not all NaN
        if self.data["sma"].iloc[-1] != self.data["sma"].iloc[-1]:  # Check for NaN
            # self.logger.debug(f"{type(self).__name__}: SMA is NaN. Cannot check exit.")
            return False

        # Example exit logic: Exit if price crosses below SMA
        # Assuming 'position' contains info about the entry side, e.g., position['result']['side']
        # For this example, we'll assume it's a 'buy' position we are checking to exit.
        if (
            self.position
            and self.position.get("main_order", {})
            .get("result", {})
            .get("side", "")
            .lower()
            == "buy"
        ):
            if self.data["close"].iloc[-1] < self.data["sma"].iloc[-1]:
                self.log_state_change(
                    "exit_signal",
                    f"{type(self).__name__}: Exit conditions met (Close {self.data['close'].iloc[-1]} < SMA {self.data['sma'].iloc[-1]}). Closing position.",
                )
                return True
            else:
                self.log_state_change(
                    "waiting_for_exit",
                    f"{type(self).__name__}: In BUY position, monitoring for exit (Close {self.data['close'].iloc[-1]} >= SMA {self.data['sma'].iloc[-1]}).",
                )
        # Add logic for exiting SELL positions if your strategy can also go short
        elif (
            self.position
            and self.position.get("main_order", {})
            .get("result", {})
            .get("side", "")
            .lower()
            == "sell"
        ):
            if (
                self.data["close"].iloc[-1] > self.data["sma"].iloc[-1]
            ):  # Exit sell if price goes above SMA
                self.log_state_change(
                    "exit_signal",
                    f"{type(self).__name__}: Exit conditions met for SELL (Close {self.data['close'].iloc[-1]} > SMA {self.data['sma'].iloc[-1]}). Closing position.",
                )
                return True
            else:
                self.log_state_change(
                    "waiting_for_exit",
                    f"{type(self).__name__}: In SELL position, monitoring for exit (Close {self.data['close'].iloc[-1]} <= SMA {self.data['sma'].iloc[-1]}).",
                )
        return False

    def get_risk_parameters(self) -> Dict[str, Any]:
        # Example risk parameters, using config if available
        # These are now percentages, e.g., 0.01 for 1%
        # Values can be None if SL or TP is not desired
        default_sl = self.config.get("default", {}).get("default_sl_pct", 0.01)
        default_tp = self.config.get("default", {}).get("default_tp_pct", 0.02)

        strat_specific_config = self.config.get("strategy_configs", {}).get(
            type(self).__name__, {}
        )
        sl_pct = strat_specific_config.get("sl_pct", default_sl)
        tp_pct = strat_specific_config.get("tp_pct", default_tp)

        return {"sl_pct": sl_pct, "tp_pct": tp_pct}

    def on_error(self, exception: Exception) -> None:
        """
        Handle errors raised during strategy logic.
        """
        self.logger.error(
            f"'{type(self).__name__}' encountered an error: {exception}", exc_info=True
        )
