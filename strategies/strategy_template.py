import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class StrategyTemplate(ABC):
    """
    Abstract base class for all trading strategies.
    All strategies must inherit from this class and implement required methods.
    """
    def __init__(self, data: Any, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Args:
            data: Market data (passed in by the bot, e.g., OHLCV DataFrame or dict)
            config: Strategy-specific and global config parameters (dict)
            logger: Logger instance for strategy-specific logging
        Usage:
            - self.position should be updated when an order is filled (see on_order_update).
            - self.position should be cleared/updated on trade exit (see on_trade_update).
        """
        self.data = data
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.position = None  # Track current position state
        self.active_orders = []  # Track open orders
        self.on_init()
        self.init_indicators()

    def on_init(self) -> None:
        """
        Optional: Pre-start setup (e.g., warm up indicators on extra historical data).
        Override in your strategy if needed.
        """
        pass

    @abstractmethod
    def init_indicators(self) -> None:
        """
        Initialize indicators required by the strategy.
        """
        pass

    @abstractmethod
    def check_entry(self) -> Optional[Dict[str, Any]]:
        """
        Check entry conditions. Return dict with order details if entry signal, else None.
        Returns:
            dict | None: {"side": "buy"/"sell", "size": float, "price": float, ...} or None
        """
        pass

    @abstractmethod
    def check_exit(self, position: Any) -> bool:
        """
        Check exit conditions for an open position. Return True if exit signal, else False.
        Args:
            position: Current open position details
        Returns:
            bool
        """
        pass

    @abstractmethod
    def get_risk_parameters(self) -> Dict[str, Any]:
        """
        Return risk parameters (stop-loss, take-profit, etc.) for the strategy.
        Returns:
            dict: {"stop_loss": float, "take_profit": float, ...}
        """
        pass

    def on_order_update(self, order: Dict[str, Any]) -> None:
        """
        Optional: Handle order status updates (filled, partially filled, canceled, etc.)
        Args:
            order: Order update details (dict)
        Example:
            if order.get('status') == 'Filled':
                self.position = order  # or extract relevant position info
        Warns if not overridden and a position is opened.
        """
        if order.get('main_order', {}).get('result', {}).get('order_status') in ('Filled', 'filled'):
            self.logger.warning(f"on_order_update not overridden in {self.__class__.__name__}, but position opened.")

    def on_trade_update(self, trade: Dict[str, Any]) -> None:
        """
        Optional: Handle trade updates (position opened/closed, PnL, etc.)
        Args:
            trade: Trade update details (dict)
        Example:
            if trade.get('exit'):
                self.position = None
        Warns if not overridden and a position is closed.
        """
        if trade.get('exit'):
            self.logger.warning(f"on_trade_update not overridden in {self.__class__.__name__}, but position closed.")

    def on_error(self, exception: Exception) -> None:
        """
        Optional: Handle errors raised during strategy logic.
        Args:
            exception: The exception instance.
        Default: logs a warning.
        """
        self.logger.warning(f"Strategy {self.__class__.__name__} encountered an error: {exception}") 