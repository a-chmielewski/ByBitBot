"""
Strategy Matrix Module

This module implements the Strategy Matrix for automatic strategy selection
based on market conditions analysis of 1-minute and 5-minute timeframes.
"""

import logging
from typing import Dict, Any, Optional, Tuple

class StrategyMatrix:
    """
    Strategy Matrix for automatic strategy selection based on market conditions.
    
    Matrix Structure:
    - Rows: 5-minute market conditions
    - Columns: 1-minute market conditions
    - Cells: Optimal strategy and execution timeframe for each combination
    """
    
    # Strategy Matrix mapping (5min_condition, 1min_condition) -> (strategy_class_name, execution_timeframe)
    STRATEGY_MATRIX = {
        # (5-min condition, 1-min condition): (strategy_class_name, execution_timeframe)
        
        # TRENDING row (5-min trending)
        ('TRENDING', 'TRENDING'): ('StrategyEMATrendRider', '5m'),  # Only combination using 5-min execution
        ('TRENDING', 'RANGING'): ('StrategyBreakoutAndRetest', '1m'),
        ('TRENDING', 'HIGH_VOLATILITY'): ('StrategyHighVolatilityTrendRider', '1m'),
        ('TRENDING', 'LOW_VOLATILITY'): ('StrategyLowVolatilityTrendPullback', '1m'),
        ('TRENDING', 'TRANSITIONAL'): ('StrategyBreakoutAndRetest', '1m'),
        
        # RANGING row (5-min ranging)
        ('RANGING', 'TRENDING'): ('StrategyRangeBreakoutMomentum', '1m'),
        ('RANGING', 'RANGING'): ('StrategyRSIRangeScalping', '1m'),  # Best fit for ranging conditions
        ('RANGING', 'HIGH_VOLATILITY'): ('StrategyVolatilityReversalScalping', '1m'),
        ('RANGING', 'LOW_VOLATILITY'): ('StrategyMicroRangeScalping', '1m'),
        ('RANGING', 'TRANSITIONAL'): ('StrategyRangeBreakoutMomentum', '1m'),
        
        # HIGH_VOLATILITY row (5-min high volatility)
        ('HIGH_VOLATILITY', 'TRENDING'): ('StrategyHighVolatilityTrendRider', '1m'),
        ('HIGH_VOLATILITY', 'RANGING'): ('StrategyVolatilityReversalScalping', '1m'),
        ('HIGH_VOLATILITY', 'HIGH_VOLATILITY'): ('StrategyATRMomentumBreakout', '1m'),
        ('HIGH_VOLATILITY', 'LOW_VOLATILITY'): ('StrategyVolatilityReversalScalping', '1m'),
        ('HIGH_VOLATILITY', 'TRANSITIONAL'): ('StrategyVolatilityReversalScalping', '1m'),
        
        # LOW_VOLATILITY row (5-min low volatility)
        ('LOW_VOLATILITY', 'TRENDING'): ('StrategyVolatilitySqueezeBreakout', '1m'),
        ('LOW_VOLATILITY', 'RANGING'): ('StrategyMicroRangeScalping', '1m'),
        ('LOW_VOLATILITY', 'HIGH_VOLATILITY'): ('StrategyVolatilityReversalScalping', '1m'),
        ('LOW_VOLATILITY', 'LOW_VOLATILITY'): ('StrategyMicroRangeScalping', '1m'),
        ('LOW_VOLATILITY', 'TRANSITIONAL'): ('StrategyVolatilitySqueezeBreakout', '1m'),
        
        # TRANSITIONAL row (5-min transitional)
        ('TRANSITIONAL', 'TRENDING'): ('StrategyAdaptiveTransitionalMomentum', '1m'),
        ('TRANSITIONAL', 'RANGING'): ('StrategyVolatilitySqueezeBreakout', '1m'),
        ('TRANSITIONAL', 'HIGH_VOLATILITY'): ('StrategyAdaptiveTransitionalMomentum', '1m'),
        ('TRANSITIONAL', 'LOW_VOLATILITY'): ('StrategyVolatilitySqueezeBreakout', '1m'),
        ('TRANSITIONAL', 'TRANSITIONAL'): ('StrategyAdaptiveTransitionalMomentum', '1m'),
    }
    
    # Alternative strategies for specific combinations (currently none - all combinations have definitive choices)
    ALTERNATIVE_STRATEGIES = {
        # No alternative strategies - each combination has a single optimal choice
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Strategy Matrix.
        
        Args:
            logger: Logger instance for strategy matrix operations
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    def select_strategy_and_timeframe(self, market_5min: str, market_1min: str) -> Tuple[str, str, str]:
        """
        Select optimal strategy and execution timeframe based on market conditions.
        
        Args:
            market_5min: 5-minute market condition (TRENDING, RANGING, HIGH_VOLATILITY, LOW_VOLATILITY, TRANSITIONAL)
            market_1min: 1-minute market condition (same options as above)
            
        Returns:
            Tuple[str, str, str]: (strategy_class_name, execution_timeframe, selection_reason)
        """
        matrix_key = (market_5min, market_1min)
        
        if matrix_key in self.STRATEGY_MATRIX:
            selected_strategy, execution_timeframe = self.STRATEGY_MATRIX[matrix_key]
            
            # Build selection reason
            timeframe_reason = "5-min execution for stable trend-following" if execution_timeframe == '5m' else "1-min execution for precision and rapid response"
            
            reason = f"Selected {selected_strategy} on {execution_timeframe} timeframe for {market_5min}(5m) + {market_1min}(1m). {timeframe_reason}"
            
            self.logger.info(f"Strategy Matrix: {reason}")
            return selected_strategy, execution_timeframe, reason
        else:
            # Fallback strategy if combination not found
            fallback_strategy = 'StrategyBreakoutAndRetest'
            fallback_timeframe = '1m'
            reason = f"Unknown market combination {market_5min}(5m) + {market_1min}(1m). Using fallback: {fallback_strategy} on {fallback_timeframe}"
            self.logger.warning(f"Strategy Matrix: {reason}")
            return fallback_strategy, fallback_timeframe, reason
    
    def select_strategy(self, market_5min: str, market_1min: str) -> Tuple[str, str]:
        """
        Legacy method for backward compatibility - returns only strategy and reason.
        
        Args:
            market_5min: 5-minute market condition
            market_1min: 1-minute market condition
            
        Returns:
            Tuple[str, str]: (strategy_class_name, selection_reason)
        """
        strategy, timeframe, reason = self.select_strategy_and_timeframe(market_5min, market_1min)
        return strategy, reason
    
    def get_strategy_description(self, strategy_class_name: str) -> str:
        """
        Get human-readable description of a strategy.
        
        Args:
            strategy_class_name: Name of the strategy class
            
        Returns:
            str: Human-readable strategy description
        """
        descriptions = {
            'StrategyEMATrendRider': 'EMA Trend Rider with ADX Filter',
            'StrategyMACDMomentumScalper': 'MACD Momentum Scalper',
            'StrategyBollingerMeanReversion': 'Bollinger Band Mean Reversion',
            'StrategyRSIRangeScalping': 'RSI Range Scalping with Candlestick Confirmation',
            'StrategyATRMomentumBreakout': 'ATR Momentum Breakout Scalper',
            'StrategyVolatilityReversalScalping': 'Volatility Reversal Scalping',
            'StrategyMicroRangeScalping': 'Micro Range Scalping',
            'StrategyVolatilitySqueezeBreakout': 'Volatility Squeeze Breakout Anticipation',
            'StrategyBollingerSqueezeBreakout': 'Bollinger Squeeze Breakout Strategy',
            'StrategyBreakoutAndRetest': 'Breakout and Retest',
            'StrategyHighVolatilityTrendRider': 'High-Volatility Trend Rider',
            'StrategyLowVolatilityTrendPullback': 'Low-Volatility Trend Pullback Scalper',
            'StrategyRangeBreakoutMomentum': 'Range Breakout Momentum Strategy',
            'StrategyAdaptiveTransitionalMomentum': 'Adaptive Transitional Momentum Breakout'
        }
        
        return descriptions.get(strategy_class_name, f"Unknown Strategy: {strategy_class_name}")
    
    def validate_market_conditions(self, market_5min: str, market_1min: str) -> bool:
        """
        Validate that market conditions are recognized by the matrix.
        
        Args:
            market_5min: 5-minute market condition
            market_1min: 1-minute market condition
            
        Returns:
            bool: True if conditions are valid, False otherwise
        """
        valid_conditions = {'TRENDING', 'RANGING', 'HIGH_VOLATILITY', 'LOW_VOLATILITY', 'TRANSITIONAL'}
        
        if market_5min not in valid_conditions:
            self.logger.error(f"Invalid 5-minute market condition: {market_5min}")
            return False
        
        if market_1min not in valid_conditions:
            self.logger.error(f"Invalid 1-minute market condition: {market_1min}")
            return False
        
        return True
    
    def get_matrix_summary(self) -> str:
        """
        Get a summary of the entire strategy matrix for logging/debugging.
        
        Returns:
            str: Formatted matrix summary with timeframes
        """
        summary = "Strategy Matrix Summary (with Execution Timeframes):\n"
        summary += "=" * 80 + "\n"
        
        conditions = ['TRENDING', 'RANGING', 'HIGH_VOLATILITY', 'LOW_VOLATILITY', 'TRANSITIONAL']
        
        # Header
        summary += f"{'5min \\ 1min':<15}"
        for condition in conditions:
            summary += f"{condition[:8]:<12}"
        summary += "\n" + "-" * 80 + "\n"
        
        # Matrix rows
        for row_condition in conditions:
            summary += f"{row_condition[:12]:<15}"
            for col_condition in conditions:
                matrix_entry = self.STRATEGY_MATRIX.get((row_condition, col_condition), ('N/A', 'N/A'))
                strategy, timeframe = matrix_entry
                strategy_short = strategy.replace('Strategy', '').replace('VolatilityReversalScalping', 'VolRevScalp')[:8]
                entry_display = f"{strategy_short}({timeframe})"
                summary += f"{entry_display:<12}"
            summary += "\n"
        
        summary += "\nTimeframe Legend:\n"
        summary += "5m = 5-minute execution (stable trend-following)\n"
        summary += "1m = 1-minute execution (precision scalping/breakouts)\n"
        
        return summary 