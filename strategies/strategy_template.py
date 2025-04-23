import logging
from abc import ABC, abstractmethod

class StrategyTemplate(ABC):
    """
    Abstract base class for all trading strategies.
    All strategies must inherit from this class and implement required methods.
    """
    def __init__(self, data, config, logger=None):
        """
        Args:
            data: Market data (passed in by the bot, e.g., OHLCV DataFrame or dict)
            config: Strategy-specific and global config parameters
            logger: Logger instance for strategy-specific logging
        """
        self.data = data
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.position = None  # Track current position state
        self.active_orders = []  # Track open orders
        self.init_indicators()

    @abstractmethod
    def init_indicators(self):
        """
        Initialize indicators required by the strategy.
        """
        pass

    @abstractmethod
    def check_entry(self):
        """
        Check entry conditions. Return dict with order details if entry signal, else None.
        Returns:
            dict | None: {"side": "buy"/"sell", "size": float, "price": float, ...} or None
        """
        pass

    @abstractmethod
    def check_exit(self, position):
        """
        Check exit conditions for an open position. Return True if exit signal, else False.
        Args:
            position: Current open position details
        Returns:
            bool
        """
        pass

    @abstractmethod
    def get_risk_parameters(self):
        """
        Return risk parameters (stop-loss, take-profit, etc.) for the strategy.
        Returns:
            dict: {"stop_loss": float, "take_profit": float, ...}
        """
        pass

    def on_order_update(self, order):
        """
        Optional: Handle order status updates (filled, partially filled, canceled, etc.)
        Args:
            order: Order update details
        """
        pass

    def on_trade_update(self, trade):
        """
        Optional: Handle trade updates (position opened/closed, PnL, etc.)
        Args:
            trade: Trade update details
        """
        pass 