"""
Risk Utilities Module

Centralized risk management utility functions for the ByBit trading bot.
Provides production-ready functions for position sizing, stop-loss/take-profit calculations,
volatility analysis, and risk metrics computation.

All functions include robust input validation and comprehensive error handling.
No external network calls are made by any function in this module.
"""

import logging
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
import numpy as np
from enum import Enum


class VolatilityRegime(Enum):
    """Volatility regime classification"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class RiskUtilitiesError(Exception):
    """Custom exception for risk utilities errors"""
    pass


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for a given DataFrame.
    
    ATR measures volatility by calculating the average of true ranges over a specified period.
    True Range is the maximum of:
    - Current High - Current Low
    - Current High - Previous Close (absolute value)
    - Current Low - Previous Close (absolute value)
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Number of periods for ATR calculation (default: 14)
    
    Returns:
        pd.Series: ATR values indexed by DataFrame index
    
    Raises:
        RiskUtilitiesError: If input validation fails or calculation errors occur
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise RiskUtilitiesError(f"Expected pandas DataFrame, got {type(df)}")
    
    if df.empty:
        raise RiskUtilitiesError("DataFrame cannot be empty")
    
    required_columns = ['high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise RiskUtilitiesError(f"Missing required columns: {missing_columns}")
    
    if not isinstance(period, int) or period <= 0:
        raise RiskUtilitiesError(f"Period must be a positive integer, got {period}")
    
    if len(df) < period:
        raise RiskUtilitiesError(f"DataFrame has {len(df)} rows, need at least {period} for ATR calculation")
    
    try:
        # Calculate True Range components
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        prev_close = close.shift(1)
        
        # True Range calculation
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        # True Range is the maximum of the three components
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the exponential moving average of True Range
        # Use pandas ewm with alpha = 2/(period+1) for consistency with TA libraries
        alpha = 2.0 / (period + 1)
        atr = true_range.ewm(alpha=alpha, min_periods=period).mean()
        
        return atr
        
    except Exception as e:
        raise RiskUtilitiesError(f"ATR calculation failed: {str(e)}")


def atr_stop_levels(entry_price: float, side: str, atr: float, 
                   atr_mult_sl: float, atr_mult_tp: float) -> Dict[str, float]:
    """
    Calculate stop-loss and take-profit levels based on ATR.
    
    Uses ATR as a volatility-adjusted measure to set appropriate stop-loss and take-profit
    levels relative to the entry price and position side.
    
    Args:
        entry_price: Entry price for the position
        side: Position side ('long' or 'short')
        atr: Average True Range value
        atr_mult_sl: ATR multiplier for stop-loss (e.g., 2.0 = 2x ATR)
        atr_mult_tp: ATR multiplier for take-profit (e.g., 3.0 = 3x ATR)
    
    Returns:
        Dict with 'stop_loss' and 'take_profit' price levels
    
    Raises:
        RiskUtilitiesError: If input validation fails
    """
    # Input validation
    if not isinstance(entry_price, (int, float)) or entry_price <= 0:
        raise RiskUtilitiesError(f"Entry price must be positive number, got {entry_price}")
    
    if not isinstance(side, str) or side.lower() not in ['long', 'short']:
        raise RiskUtilitiesError(f"Side must be 'long' or 'short', got '{side}'")
    
    if not isinstance(atr, (int, float)) or atr <= 0:
        raise RiskUtilitiesError(f"ATR must be positive number, got {atr}")
    
    if not isinstance(atr_mult_sl, (int, float)) or atr_mult_sl <= 0:
        raise RiskUtilitiesError(f"ATR multiplier for SL must be positive, got {atr_mult_sl}")
    
    if not isinstance(atr_mult_tp, (int, float)) or atr_mult_tp <= 0:
        raise RiskUtilitiesError(f"ATR multiplier for TP must be positive, got {atr_mult_tp}")
    
    try:
        side_normalized = side.lower()
        
        if side_normalized == 'long':
            stop_loss = entry_price - (atr * atr_mult_sl)
            take_profit = entry_price + (atr * atr_mult_tp)
        else:  # short
            stop_loss = entry_price + (atr * atr_mult_sl)
            take_profit = entry_price - (atr * atr_mult_tp)
        
        # Ensure levels are positive
        if stop_loss <= 0:
            raise RiskUtilitiesError(f"Calculated stop-loss ({stop_loss:.6f}) is non-positive")
        if take_profit <= 0:
            raise RiskUtilitiesError(f"Calculated take-profit ({take_profit:.6f}) is non-positive")
        
        return {
            'stop_loss': round(stop_loss, 8),
            'take_profit': round(take_profit, 8)
        }
        
    except Exception as e:
        if isinstance(e, RiskUtilitiesError):
            raise
        raise RiskUtilitiesError(f"ATR stop levels calculation failed: {str(e)}")


def progressive_take_profit_levels(entry_price: float, side: str, 
                                 levels: List[float]) -> List[float]:
    """
    Calculate progressive take-profit levels based on percentage distances from entry.
    
    Allows for scaling out of positions at multiple profit levels.
    
    Args:
        entry_price: Entry price for the position
        side: Position side ('long' or 'short')
        levels: List of percentage distances from entry (e.g., [0.01, 0.02, 0.03] for 1%, 2%, 3%)
    
    Returns:
        List of take-profit price levels in ascending order
    
    Raises:
        RiskUtilitiesError: If input validation fails
    """
    # Input validation
    if not isinstance(entry_price, (int, float)) or entry_price <= 0:
        raise RiskUtilitiesError(f"Entry price must be positive number, got {entry_price}")
    
    if not isinstance(side, str) or side.lower() not in ['long', 'short']:
        raise RiskUtilitiesError(f"Side must be 'long' or 'short', got '{side}'")
    
    if not isinstance(levels, list) or not levels:
        raise RiskUtilitiesError(f"Levels must be non-empty list, got {type(levels)}")
    
    # Validate each level
    for i, level in enumerate(levels):
        if not isinstance(level, (int, float)) or level <= 0:
            raise RiskUtilitiesError(f"Level {i} must be positive number, got {level}")
    
    try:
        side_normalized = side.lower()
        tp_levels = []
        
        for level_pct in levels:
            if side_normalized == 'long':
                tp_price = entry_price * (1 + level_pct)
            else:  # short
                tp_price = entry_price * (1 - level_pct)
            
            if tp_price <= 0:
                raise RiskUtilitiesError(f"Calculated TP level ({tp_price:.6f}) is non-positive for level {level_pct}")
            
            tp_levels.append(round(tp_price, 8))
        
        # Sort levels appropriately (ascending for long, descending for short)
        if side_normalized == 'long':
            tp_levels.sort()
        else:
            tp_levels.sort(reverse=True)
        
        return tp_levels
        
    except Exception as e:
        if isinstance(e, RiskUtilitiesError):
            raise
        raise RiskUtilitiesError(f"Progressive TP levels calculation failed: {str(e)}")


def update_trailing_stop(side: str, current_price: float, 
                        trail_price: Optional[float], trail_offset: float) -> float:
    """
    Update trailing stop-loss price based on current market conditions.
    
    Maintains a stop-loss that follows favorable price movement while protecting
    against adverse moves.
    
    Args:
        side: Position side ('long' or 'short')
        current_price: Current market price
        trail_price: Current trailing stop price (None for first calculation)
        trail_offset: Trailing offset distance from current price
    
    Returns:
        Updated trailing stop price
    
    Raises:
        RiskUtilitiesError: If input validation fails
    """
    # Input validation
    if not isinstance(side, str) or side.lower() not in ['long', 'short']:
        raise RiskUtilitiesError(f"Side must be 'long' or 'short', got '{side}'")
    
    if not isinstance(current_price, (int, float)) or current_price <= 0:
        raise RiskUtilitiesError(f"Current price must be positive number, got {current_price}")
    
    if trail_price is not None and (not isinstance(trail_price, (int, float)) or trail_price <= 0):
        raise RiskUtilitiesError(f"Trail price must be positive number or None, got {trail_price}")
    
    if not isinstance(trail_offset, (int, float)) or trail_offset <= 0:
        raise RiskUtilitiesError(f"Trail offset must be positive number, got {trail_offset}")
    
    try:
        side_normalized = side.lower()
        
        if side_normalized == 'long':
            # For long positions, trail stop moves up with price
            new_trail_stop = current_price - trail_offset
            
            # Only update if new stop is higher than current (or first calculation)
            if trail_price is None or new_trail_stop > trail_price:
                updated_stop = new_trail_stop
            else:
                updated_stop = trail_price
                
        else:  # short
            # For short positions, trail stop moves down with price
            new_trail_stop = current_price + trail_offset
            
            # Only update if new stop is lower than current (or first calculation)
            if trail_price is None or new_trail_stop < trail_price:
                updated_stop = new_trail_stop
            else:
                updated_stop = trail_price
        
        # Ensure result is positive
        if updated_stop <= 0:
            raise RiskUtilitiesError(f"Calculated trailing stop ({updated_stop:.6f}) is non-positive")
        
        return round(updated_stop, 8)
        
    except Exception as e:
        if isinstance(e, RiskUtilitiesError):
            raise
        raise RiskUtilitiesError(f"Trailing stop update failed: {str(e)}")


def position_size_vol_normalized(account_equity: float, risk_per_trade: float, 
                                atr: float, tick_value: float) -> float:
    """
    Calculate position size normalized by volatility using ATR.
    
    Adjusts position size based on market volatility to maintain consistent risk
    across different market conditions.
    
    Args:
        account_equity: Total account equity
        risk_per_trade: Maximum risk per trade as decimal (e.g., 0.02 for 2%)
        atr: Average True Range value
        tick_value: Value of one tick/pip in account currency
    
    Returns:
        Position size (number of contracts/shares)
    
    Raises:
        RiskUtilitiesError: If input validation fails
    """
    # Input validation
    if not isinstance(account_equity, (int, float)) or account_equity <= 0:
        raise RiskUtilitiesError(f"Account equity must be positive number, got {account_equity}")
    
    if not isinstance(risk_per_trade, (int, float)) or not (0 < risk_per_trade <= 1):
        raise RiskUtilitiesError(f"Risk per trade must be between 0 and 1, got {risk_per_trade}")
    
    if not isinstance(atr, (int, float)) or atr <= 0:
        raise RiskUtilitiesError(f"ATR must be positive number, got {atr}")
    
    if not isinstance(tick_value, (int, float)) or tick_value <= 0:
        raise RiskUtilitiesError(f"Tick value must be positive number, got {tick_value}")
    
    try:
        # Maximum dollar risk for this trade
        max_risk_dollars = account_equity * risk_per_trade
        
        # Risk per contract/share based on ATR
        risk_per_unit = atr * tick_value
        
        # Position size calculation
        position_size = max_risk_dollars / risk_per_unit
        
        # Ensure reasonable result
        if position_size <= 0:
            raise RiskUtilitiesError(f"Calculated position size ({position_size:.6f}) is non-positive")
        
        return round(position_size, 6)
        
    except Exception as e:
        if isinstance(e, RiskUtilitiesError):
            raise
        raise RiskUtilitiesError(f"Volatility-normalized position sizing failed: {str(e)}")


def kelly_fraction_capped(edge: float, win_prob: float, cap: float = 0.1) -> float:
    """
    Calculate Kelly Criterion fraction with maximum cap for risk management.
    
    The Kelly Criterion determines optimal position sizing based on win probability
    and expected edge. A cap is applied to prevent excessive position sizes.
    
    Args:
        edge: Expected edge/advantage (e.g., 0.5 for 50% expected return)
        win_prob: Probability of winning (between 0 and 1)
        cap: Maximum fraction to risk (default: 0.1 = 10%)
    
    Returns:
        Kelly fraction capped at maximum value
    
    Raises:
        RiskUtilitiesError: If input validation fails
    """
    # Input validation
    if not isinstance(edge, (int, float)):
        raise RiskUtilitiesError(f"Edge must be a number, got {type(edge)}")
    
    if not isinstance(win_prob, (int, float)) or not (0 < win_prob < 1):
        raise RiskUtilitiesError(f"Win probability must be between 0 and 1, got {win_prob}")
    
    if not isinstance(cap, (int, float)) or not (0 < cap <= 1):
        raise RiskUtilitiesError(f"Cap must be between 0 and 1, got {cap}")
    
    try:
        # Kelly formula: f = (bp - q) / b
        # where b = odds received on the wager (edge), p = probability of winning, q = probability of losing
        lose_prob = 1 - win_prob
        
        # Avoid division by zero
        if edge == 0:
            return 0.0
        
        # Kelly fraction calculation
        kelly_fraction = (edge * win_prob - lose_prob) / edge
        
        # Apply constraints
        kelly_fraction = max(0.0, kelly_fraction)  # Never negative
        kelly_fraction = min(cap, kelly_fraction)   # Apply cap
        
        return round(kelly_fraction, 6)
        
    except Exception as e:
        if isinstance(e, RiskUtilitiesError):
            raise
        raise RiskUtilitiesError(f"Kelly fraction calculation failed: {str(e)}")


def volatility_regime(atr_pct: float, thresholds: Tuple[float, float] = (0.5, 1.5)) -> str:
    """
    Classify current volatility regime based on ATR percentage.
    
    Categorizes market volatility into low, normal, or high regimes to inform
    strategy selection and risk management decisions.
    
    Args:
        atr_pct: ATR as percentage of price (e.g., 0.02 for 2%)
        thresholds: Tuple of (low_threshold, high_threshold) for classification
    
    Returns:
        Volatility regime: 'low', 'normal', or 'high'
    
    Raises:
        RiskUtilitiesError: If input validation fails
    """
    # Input validation
    if not isinstance(atr_pct, (int, float)) or atr_pct < 0:
        raise RiskUtilitiesError(f"ATR percentage must be non-negative number, got {atr_pct}")
    
    if not isinstance(thresholds, tuple) or len(thresholds) != 2:
        raise RiskUtilitiesError(f"Thresholds must be tuple of 2 values, got {thresholds}")
    
    low_threshold, high_threshold = thresholds
    
    if not isinstance(low_threshold, (int, float)) or not isinstance(high_threshold, (int, float)):
        raise RiskUtilitiesError(f"Threshold values must be numbers, got {thresholds}")
    
    if low_threshold >= high_threshold:
        raise RiskUtilitiesError(f"Low threshold must be less than high threshold, got {thresholds}")
    
    if low_threshold < 0 or high_threshold < 0:
        raise RiskUtilitiesError(f"Thresholds must be non-negative, got {thresholds}")
    
    try:
        if atr_pct < low_threshold:
            return VolatilityRegime.LOW.value
        elif atr_pct > high_threshold:
            return VolatilityRegime.HIGH.value
        else:
            return VolatilityRegime.NORMAL.value
            
    except Exception as e:
        raise RiskUtilitiesError(f"Volatility regime classification failed: {str(e)}")


def zscore(series: pd.Series, lookback: int = 100) -> pd.Series:
    """
    Calculate rolling z-score for a pandas Series.
    
    Z-score measures how many standard deviations a value is from the mean,
    useful for identifying statistical outliers and mean reversion opportunities.
    
    Args:
        series: Pandas Series of values
        lookback: Number of periods for rolling calculation (default: 100)
    
    Returns:
        pd.Series: Z-score values indexed by original series index
    
    Raises:
        RiskUtilitiesError: If input validation fails
    """
    # Input validation
    if not isinstance(series, pd.Series):
        raise RiskUtilitiesError(f"Expected pandas Series, got {type(series)}")
    
    if series.empty:
        raise RiskUtilitiesError("Series cannot be empty")
    
    if not isinstance(lookback, int) or lookback <= 0:
        raise RiskUtilitiesError(f"Lookback must be positive integer, got {lookback}")
    
    if len(series) < lookback:
        raise RiskUtilitiesError(f"Series has {len(series)} values, need at least {lookback} for z-score calculation")
    
    try:
        # Convert to numeric, handling any string/object types
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        # Check for too many NaN values
        if numeric_series.isna().sum() > len(numeric_series) * 0.5:
            raise RiskUtilitiesError("Series contains too many non-numeric values")
        
        # Calculate rolling mean and standard deviation
        rolling_mean = numeric_series.rolling(window=lookback, min_periods=lookback).mean()
        rolling_std = numeric_series.rolling(window=lookback, min_periods=lookback).std()
        
        # Calculate z-score, handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            z_scores = (numeric_series - rolling_mean) / rolling_std
        
        # Replace inf and -inf with NaN
        z_scores = z_scores.replace([np.inf, -np.inf], np.nan)
        
        return z_scores
        
    except Exception as e:
        if isinstance(e, RiskUtilitiesError):
            raise
        raise RiskUtilitiesError(f"Z-score calculation failed: {str(e)}")


# Module-level logger for internal use
_logger = logging.getLogger(__name__)

# Export main functions for external use
__all__ = [
    'compute_atr',
    'atr_stop_levels', 
    'progressive_take_profit_levels',
    'update_trailing_stop',
    'position_size_vol_normalized',
    'kelly_fraction_capped',
    'volatility_regime',
    'zscore',
    'VolatilityRegime',
    'RiskUtilitiesError'
]