from .strategy_template import StrategyTemplate
import pandas as pd

class StrategyExample(StrategyTemplate):
    """
    Example strategy implementation for demonstration purposes.
    """
    def init_indicators(self):
        # Example: Calculate a simple moving average
        if isinstance(self.data, pd.DataFrame):
            self.data['sma'] = self.data['close'].rolling(window=20).mean()
        # Add other indicators as needed

    def check_entry(self):
        # Example entry logic: Buy if price crosses above SMA
        if isinstance(self.data, pd.DataFrame):
            if self.data['close'].iloc[-1] > self.data['sma'].iloc[-1]:
                return {"side": "buy", "size": 1.0, "price": self.data['close'].iloc[-1]}
        return None

    def check_exit(self, position):
        # Example exit logic: Exit if price crosses below SMA
        if isinstance(self.data, pd.DataFrame):
            if self.data['close'].iloc[-1] < self.data['sma'].iloc[-1]:
                return True
        return False

    def get_risk_parameters(self):
        # Example risk parameters
        return {"stop_loss": 0.01, "take_profit": 0.02} 