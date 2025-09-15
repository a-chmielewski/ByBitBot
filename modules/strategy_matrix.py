"""
Enhanced Strategy Matrix Module

This module implements the comprehensive Strategy Matrix for automatic strategy selection
based on market conditions analysis, including detailed risk management parameters,
portfolio constraints, and correlation grouping for each strategy.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict

@dataclass
class StopLossConfig:
    """Stop loss configuration for strategies"""
    mode: str = 'atr_mult'  # 'fixed_pct' | 'atr_mult'
    fixed_pct: float = 0.025  # Fixed percentage (2.5%) - widened to reduce premature stops
    atr_multiplier: float = 3.5  # ATR multiplier for dynamic stops - widened for market noise
    max_loss_pct: float = 0.16  # Maximum loss percentage (16%) - widened for volatile markets

@dataclass
class TakeProfitConfig:
    """Take profit configuration for strategies"""
    mode: str = 'progressive_levels'  # 'fixed_pct' | 'progressive_levels'
    fixed_pct: float = 0.06  # Fixed percentage (6%) - increased from 4%
    progressive_levels: List[float] = None  # Progressive levels - widened for better R:R
    partial_exit_sizes: List[float] = None  # Partial exit sizes [0.3, 0.3, 0.4] - let winners run longer
    
    def __post_init__(self):
        if self.progressive_levels is None:
            self.progressive_levels = [0.015, 0.035, 0.08]  # Quicker: 1.5%, 3.5%, 8%
        if self.partial_exit_sizes is None:
            self.partial_exit_sizes = [0.5, 0.25, 0.25]  # Take 50% at first target, 25% at second

@dataclass
class TrailingStopConfig:
    """Trailing stop configuration for strategies"""
    enabled: bool = True  # Enable by default to let winners run
    mode: str = 'atr_mult'  # 'price_pct' | 'atr_mult'
    offset_pct: float = 0.02  # 2% trailing offset - widened
    atr_multiplier: float = 2.0  # ATR multiplier for trailing offset - widened
    activation_pct: float = 0.035  # Activate after 3.5% profit - after second TP target hit

@dataclass
class PositionSizingConfig:
    """Position sizing configuration for strategies"""
    mode: str = 'fixed_notional'  # 'fixed_notional' | 'vol_normalized' | 'kelly_capped'
    fixed_notional: float = 1000.0  # Fixed dollar amount
    risk_per_trade: float = 0.01  # Risk per trade (1% of equity)
    kelly_cap: float = 0.1  # Kelly criterion cap (10%)
    tick_value: float = 0.01  # Tick value for calculations
    max_position_pct: float = 5.0  # Max position as % of equity

@dataclass
class LeverageByRegimeConfig:
    """Leverage multipliers by volatility regime"""
    low: float = 1.2    # 20% higher leverage in low volatility
    normal: float = 1.0  # Base leverage in normal volatility
    high: float = 0.8   # 20% lower leverage in high volatility

@dataclass
class PortfolioTagsConfig:
    """Portfolio tags for correlation grouping"""
    sector: str = 'crypto'  # Sector classification (crypto, forex, commodities)
    factor: str = 'momentum'  # Factor exposure (momentum, mean_reversion, volatility)
    correlation_group: str = 'btc_correlated'  # Correlation group identifier
    market_cap_tier: str = 'large'  # Market cap tier (large, mid, small)

@dataclass
class TradingLimitsConfig:
    """Trading limits and constraints"""
    max_concurrent_trades: int = 3  # Max concurrent trades for this strategy
    max_per_symbol: int = 1  # Max trades per symbol
    min_time_between_trades: int = 300  # Min seconds between trades (5 min)
    daily_trade_limit: int = 20  # Max trades per day
    max_daily_drawdown_pct: float = 0.05  # Max daily drawdown (5%)

@dataclass
class StrategyRiskProfile:
    """Complete risk profile for a strategy"""
    strategy_name: str
    description: str
    market_type_tags: List[str]
    execution_timeframe: str
    
    # Risk management components
    stop_loss: StopLossConfig
    take_profit: TakeProfitConfig
    trailing_stop: TrailingStopConfig
    position_sizing: PositionSizingConfig
    leverage_by_regime: LeverageByRegimeConfig
    portfolio_tags: PortfolioTagsConfig
    trading_limits: TradingLimitsConfig
    
    # Additional metadata
    min_volatility_pct: float = 0.5  # Minimum volatility to trade
    max_volatility_pct: float = 10.0  # Maximum volatility to trade
    preferred_sessions: List[str] = None  # Preferred trading sessions
    
    def __post_init__(self):
        if self.preferred_sessions is None:
            self.preferred_sessions = ['london', 'new_york', 'tokyo']

class StrategyMatrix:
    """
    Strategy Matrix for automatic strategy selection based on market conditions.
    
    Matrix Structure:
    - Rows: 5-minute market conditions
    - Columns: 1-minute market conditions
    - Cells: Optimal strategy and execution timeframe for each combination
    """
    
    # Comprehensive Strategy Risk Profiles
    STRATEGY_RISK_PROFILES = {
        'StrategyEMATrendRider': StrategyRiskProfile(
            strategy_name='StrategyEMATrendRider',
            description='EMA Trend Rider with Enhanced ADX Filter (25+) & Volume Confirmation (1.5x) - Stable trend-following on 5m timeframe',
            market_type_tags=['TRENDING'],
            execution_timeframe='5m',
            stop_loss=StopLossConfig(mode='atr_mult', atr_multiplier=4.5, max_loss_pct=0.08),
            take_profit=TakeProfitConfig(mode='progressive_levels', progressive_levels=[0.008, 0.018, 0.035]),
            trailing_stop=TrailingStopConfig(enabled=True, mode='atr_mult', atr_multiplier=3.0, activation_pct=0.005),
            position_sizing=PositionSizingConfig(mode='vol_normalized', risk_per_trade=0.008, max_position_pct=4.0),
            leverage_by_regime=LeverageByRegimeConfig(low=1.3, normal=1.0, high=0.7),
            portfolio_tags=PortfolioTagsConfig(factor='trend_following', correlation_group='trend_momentum'),
            trading_limits=TradingLimitsConfig(max_concurrent_trades=2, min_time_between_trades=900),
            min_volatility_pct=0.3, max_volatility_pct=3.0
        ),
        
        'StrategyATRMomentumBreakout': StrategyRiskProfile(
            strategy_name='StrategyATRMomentumBreakout',
            description='ATR Momentum Breakout for high-volatility momentum trading',
            market_type_tags=['HIGH_VOLATILITY'],
            execution_timeframe='1m',
            stop_loss=StopLossConfig(mode='atr_mult', atr_multiplier=4.0, max_loss_pct=0.08),
            take_profit=TakeProfitConfig(mode='progressive_levels', progressive_levels=[0.006, 0.015, 0.030]),
            trailing_stop=TrailingStopConfig(enabled=True, mode='atr_mult', atr_multiplier=3.0, activation_pct=0.005),
            position_sizing=PositionSizingConfig(mode='vol_normalized', risk_per_trade=0.012, max_position_pct=3.0),
            leverage_by_regime=LeverageByRegimeConfig(low=1.0, normal=0.9, high=0.6),
            portfolio_tags=PortfolioTagsConfig(factor='momentum', correlation_group='high_vol_breakout'),
            trading_limits=TradingLimitsConfig(max_concurrent_trades=4, min_time_between_trades=180),
            min_volatility_pct=1.5, max_volatility_pct=15.0
        ),
        
        'StrategyBreakoutAndRetest': StrategyRiskProfile(
            strategy_name='StrategyBreakoutAndRetest',
            description='Breakout and Retest with Enhanced Volume Confirmation (2.0x) for trend continuation trades',
            market_type_tags=['TRENDING', 'TRANSITIONAL'],
            execution_timeframe='1m',
            stop_loss=StopLossConfig(mode='atr_mult', atr_multiplier=3.2, max_loss_pct=0.08),
            take_profit=TakeProfitConfig(mode='progressive_levels', progressive_levels=[0.04, 0.08, 0.16]),
            trailing_stop=TrailingStopConfig(enabled=True, mode='atr_mult', atr_multiplier=3.0, activation_pct=0.005),
            position_sizing=PositionSizingConfig(mode='vol_normalized', risk_per_trade=0.01, max_position_pct=3.5),
            leverage_by_regime=LeverageByRegimeConfig(low=1.2, normal=1.0, high=0.8),
            portfolio_tags=PortfolioTagsConfig(factor='momentum', correlation_group='breakout_momentum'),
            trading_limits=TradingLimitsConfig(max_concurrent_trades=3, min_time_between_trades=300),
            min_volatility_pct=0.8, max_volatility_pct=8.0
        ),
        
        'StrategyRSIRangeScalping': StrategyRiskProfile(
            strategy_name='StrategyRSIRangeScalping',
            description='RSI Range Scalping with candlestick confirmation for ranging markets',
            market_type_tags=['RANGING'],
            execution_timeframe='1m',
            stop_loss=StopLossConfig(mode='atr_mult', atr_multiplier=2.8, max_loss_pct=0.06),
            take_profit=TakeProfitConfig(mode='progressive_levels', progressive_levels=[0.008, 0.016, 0.032]),
            trailing_stop=TrailingStopConfig(enabled=True, mode='atr_mult', atr_multiplier=2.0, activation_pct=0.005),
            position_sizing=PositionSizingConfig(mode='fixed_notional', fixed_notional=800.0, max_position_pct=2.5),
            leverage_by_regime=LeverageByRegimeConfig(low=1.4, normal=1.2, high=0.8),
            portfolio_tags=PortfolioTagsConfig(factor='mean_reversion', correlation_group='range_scalping'),
            trading_limits=TradingLimitsConfig(max_concurrent_trades=5, min_time_between_trades=120, daily_trade_limit=30),
            min_volatility_pct=0.2, max_volatility_pct=2.5
        ),
        
        'StrategyVolatilityReversalScalping': StrategyRiskProfile(
            strategy_name='StrategyVolatilityReversalScalping',
            description='Volatility Reversal Scalping for high-vol mean reversion',
            market_type_tags=['HIGH_VOLATILITY', 'RANGING'],
            execution_timeframe='1m',
            stop_loss=StopLossConfig(mode='atr_mult', atr_multiplier=4.0, max_loss_pct=0.07),
            take_profit=TakeProfitConfig(mode='progressive_levels', progressive_levels=[0.006, 0.015, 0.030]),
            trailing_stop=TrailingStopConfig(enabled=True, mode='atr_mult', atr_multiplier=2.8, activation_pct=0.005),
            position_sizing=PositionSizingConfig(mode='vol_normalized', risk_per_trade=0.015, max_position_pct=3.0),
            leverage_by_regime=LeverageByRegimeConfig(low=1.1, normal=0.9, high=0.6),
            portfolio_tags=PortfolioTagsConfig(factor='mean_reversion', correlation_group='vol_reversal'),
            trading_limits=TradingLimitsConfig(max_concurrent_trades=4, min_time_between_trades=150),
            min_volatility_pct=1.0, max_volatility_pct=12.0
        ),
        
        'StrategyMicroRangeScalping': StrategyRiskProfile(
            strategy_name='StrategyMicroRangeScalping',
            description='Micro Range Scalping for low-volatility tight ranges',
            market_type_tags=['LOW_VOLATILITY', 'RANGING'],
            execution_timeframe='1m',
            stop_loss=StopLossConfig(mode='fixed_pct', fixed_pct=0.0005, max_loss_pct=0.001),
            take_profit=TakeProfitConfig(mode='progressive_levels', progressive_levels=[0.0015, 0.003, 0.005]),
            trailing_stop=TrailingStopConfig(enabled=True, mode='atr_mult', atr_multiplier=1.5, activation_pct=0.0008),
            position_sizing=PositionSizingConfig(mode='fixed_notional', fixed_notional=1200.0, max_position_pct=4.0),
            leverage_by_regime=LeverageByRegimeConfig(low=1.5, normal=1.3, high=1.0),
            portfolio_tags=PortfolioTagsConfig(factor='mean_reversion', correlation_group='micro_scalping'),
            trading_limits=TradingLimitsConfig(max_concurrent_trades=6, min_time_between_trades=60, daily_trade_limit=40),
            min_volatility_pct=0.1, max_volatility_pct=1.0
        ),
        
        'StrategyAdaptiveTransitionalMomentum': StrategyRiskProfile(
            strategy_name='StrategyAdaptiveTransitionalMomentum',
            description='Adaptive Transitional Momentum for regime change detection',
            market_type_tags=['TRANSITIONAL'],
            execution_timeframe='1m',
            stop_loss=StopLossConfig(mode='atr_mult', atr_multiplier=4.2, max_loss_pct=0.08),
            take_profit=TakeProfitConfig(mode='progressive_levels', progressive_levels=[0.04, 0.08, 0.16]),
            trailing_stop=TrailingStopConfig(enabled=True, mode='atr_mult', atr_multiplier=3.0, activation_pct=0.005),
            position_sizing=PositionSizingConfig(mode='kelly_capped', kelly_cap=0.08, risk_per_trade=0.012),
            leverage_by_regime=LeverageByRegimeConfig(low=1.1, normal=1.0, high=0.7),
            portfolio_tags=PortfolioTagsConfig(factor='adaptive', correlation_group='transitional_momentum'),
            trading_limits=TradingLimitsConfig(max_concurrent_trades=3, min_time_between_trades=240),
            min_volatility_pct=0.5, max_volatility_pct=8.0
        ),
        
        'StrategyHighVolatilityTrendRider': StrategyRiskProfile(
            strategy_name='StrategyHighVolatilityTrendRider',
            description='High-Volatility Trend Rider with Enhanced ADX Filter (25+) & Volume Confirmation (1.5x) for volatile trending markets',
            market_type_tags=['HIGH_VOLATILITY', 'TRENDING'],
            execution_timeframe='1m',
            stop_loss=StopLossConfig(mode='atr_mult', atr_multiplier=3.5, max_loss_pct=0.10),
            take_profit=TakeProfitConfig(mode='progressive_levels', progressive_levels=[0.05, 0.10, 0.20]),
            trailing_stop=TrailingStopConfig(enabled=True, mode='atr_mult', atr_multiplier=3.5, activation_pct=0.005),
            position_sizing=PositionSizingConfig(mode='vol_normalized', risk_per_trade=0.01, max_position_pct=3.0),
            leverage_by_regime=LeverageByRegimeConfig(low=1.0, normal=0.8, high=0.5),
            portfolio_tags=PortfolioTagsConfig(factor='trend_following', correlation_group='high_vol_trend'),
            trading_limits=TradingLimitsConfig(max_concurrent_trades=2, min_time_between_trades=600),
            min_volatility_pct=2.0, max_volatility_pct=20.0
        ),
        
        'StrategyLowVolatilityTrendPullback': StrategyRiskProfile(
            strategy_name='StrategyLowVolatilityTrendPullback',
            description='Low-Volatility Trend Pullback for quiet trending markets',
            market_type_tags=['LOW_VOLATILITY', 'TRENDING'],
            execution_timeframe='1m',
            stop_loss=StopLossConfig(mode='atr_mult', atr_multiplier=4.0, max_loss_pct=0.06),
            take_profit=TakeProfitConfig(mode='progressive_levels', progressive_levels=[0.006, 0.015, 0.030]),
            trailing_stop=TrailingStopConfig(enabled=True, mode='atr_mult', atr_multiplier=2.8, activation_pct=0.005),
            position_sizing=PositionSizingConfig(mode='vol_normalized', risk_per_trade=0.008, max_position_pct=3.5),
            leverage_by_regime=LeverageByRegimeConfig(low=1.4, normal=1.2, high=0.9),
            portfolio_tags=PortfolioTagsConfig(factor='trend_following', correlation_group='low_vol_trend'),
            trading_limits=TradingLimitsConfig(max_concurrent_trades=4, min_time_between_trades=180),
            min_volatility_pct=0.2, max_volatility_pct=1.5
        ),
        
        'StrategyRangeBreakoutMomentum': StrategyRiskProfile(
            strategy_name='StrategyRangeBreakoutMomentum',
            description='Range Breakout Momentum for range-to-trend transitions',
            market_type_tags=['RANGING', 'TRANSITIONAL'],
            execution_timeframe='1m',
            stop_loss=StopLossConfig(mode='atr_mult', atr_multiplier=3.0, max_loss_pct=0.08),
            take_profit=TakeProfitConfig(mode='progressive_levels', progressive_levels=[0.04, 0.08, 0.16]),
            trailing_stop=TrailingStopConfig(enabled=True, mode='atr_mult', atr_multiplier=3.0, activation_pct=0.005),
            position_sizing=PositionSizingConfig(mode='vol_normalized', risk_per_trade=0.011, max_position_pct=3.5),
            leverage_by_regime=LeverageByRegimeConfig(low=1.2, normal=1.0, high=0.8),
            portfolio_tags=PortfolioTagsConfig(factor='momentum', correlation_group='range_breakout'),
            trading_limits=TradingLimitsConfig(max_concurrent_trades=3, min_time_between_trades=300),
            min_volatility_pct=0.6, max_volatility_pct=6.0
        ),
        
        'StrategyVolatilitySqueezeBreakout': StrategyRiskProfile(
            strategy_name='StrategyVolatilitySqueezeBreakout',
            description='Volatility Squeeze Breakout for low-to-high vol transitions',
            market_type_tags=['LOW_VOLATILITY', 'TRANSITIONAL'],
            execution_timeframe='1m',
            stop_loss=StopLossConfig(mode='atr_mult', atr_multiplier=4.2, max_loss_pct=0.08),
            take_profit=TakeProfitConfig(mode='progressive_levels', progressive_levels=[0.04, 0.08, 0.16]),
            trailing_stop=TrailingStopConfig(enabled=True, mode='atr_mult', atr_multiplier=3.0, activation_pct=0.005),
            position_sizing=PositionSizingConfig(mode='vol_normalized', risk_per_trade=0.01, max_position_pct=3.5),
            leverage_by_regime=LeverageByRegimeConfig(low=1.3, normal=1.1, high=0.8),
            portfolio_tags=PortfolioTagsConfig(factor='volatility', correlation_group='squeeze_breakout'),
            trading_limits=TradingLimitsConfig(max_concurrent_trades=3, min_time_between_trades=420),
            min_volatility_pct=0.3, max_volatility_pct=4.0
        )
    }
    
    # Strategy Matrix mapping (5min_condition, 1min_condition) -> strategy_name
    STRATEGY_MATRIX = {
        # TRENDING row (5-min trending)
        ('TRENDING', 'TRENDING'): 'StrategyEMATrendRider',
        ('TRENDING', 'RANGING'): 'StrategyBreakoutAndRetest',
        ('TRENDING', 'HIGH_VOLATILITY'): 'StrategyHighVolatilityTrendRider',
        ('TRENDING', 'LOW_VOLATILITY'): 'StrategyLowVolatilityTrendPullback',
        ('TRENDING', 'TRANSITIONAL'): 'StrategyBreakoutAndRetest',
        
        # RANGING row (5-min ranging)
        ('RANGING', 'TRENDING'): 'StrategyRangeBreakoutMomentum',
        ('RANGING', 'RANGING'): 'StrategyRSIRangeScalping',
        ('RANGING', 'HIGH_VOLATILITY'): 'StrategyVolatilityReversalScalping',
        ('RANGING', 'LOW_VOLATILITY'): 'StrategyMicroRangeScalping',
        ('RANGING', 'TRANSITIONAL'): 'StrategyRangeBreakoutMomentum',
        
        # HIGH_VOLATILITY row (5-min high volatility)
        ('HIGH_VOLATILITY', 'TRENDING'): 'StrategyHighVolatilityTrendRider',
        ('HIGH_VOLATILITY', 'RANGING'): 'StrategyVolatilityReversalScalping',
        ('HIGH_VOLATILITY', 'HIGH_VOLATILITY'): 'StrategyATRMomentumBreakout',
        ('HIGH_VOLATILITY', 'LOW_VOLATILITY'): 'StrategyVolatilityReversalScalping',
        ('HIGH_VOLATILITY', 'TRANSITIONAL'): 'StrategyVolatilityReversalScalping',
        
        # LOW_VOLATILITY row (5-min low volatility)
        ('LOW_VOLATILITY', 'TRENDING'): 'StrategyVolatilitySqueezeBreakout',
        ('LOW_VOLATILITY', 'RANGING'): 'StrategyMicroRangeScalping',
        ('LOW_VOLATILITY', 'HIGH_VOLATILITY'): 'StrategyVolatilityReversalScalping',
        ('LOW_VOLATILITY', 'LOW_VOLATILITY'): 'StrategyMicroRangeScalping',
        ('LOW_VOLATILITY', 'TRANSITIONAL'): 'StrategyVolatilitySqueezeBreakout',
        
        # TRANSITIONAL row (5-min transitional)
        ('TRANSITIONAL', 'TRENDING'): 'StrategyAdaptiveTransitionalMomentum',
        ('TRANSITIONAL', 'RANGING'): 'StrategyVolatilitySqueezeBreakout',
        ('TRANSITIONAL', 'HIGH_VOLATILITY'): 'StrategyAdaptiveTransitionalMomentum',
        ('TRANSITIONAL', 'LOW_VOLATILITY'): 'StrategyVolatilitySqueezeBreakout',
        ('TRANSITIONAL', 'TRANSITIONAL'): 'StrategyAdaptiveTransitionalMomentum',
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
    
    def select_strategy_and_timeframe(self, market_5min: str, market_1min: str, analysis_5min: dict = None, analysis_1min: dict = None, directional_bias: str = 'NEUTRAL') -> Tuple[str, str, str]:
        """
        Select optimal strategy and execution timeframe based on market conditions with context-aware logic.
        
        Args:
            market_5min: 5-minute market condition (TRENDING, RANGING, HIGH_VOLATILITY, LOW_VOLATILITY, TRANSITIONAL)
            market_1min: 1-minute market condition (same options as above)
            analysis_5min: Optional detailed analysis for 5m timeframe
            analysis_1min: Optional detailed analysis for 1m timeframe
            directional_bias: Higher timeframe directional bias (BEARISH, BULLISH, NEUTRAL, etc.)
            
        Returns:
            Tuple[str, str, str]: (strategy_class_name, execution_timeframe, selection_reason)
        """
        # Extract combined regime information if available
        combined_5m = analysis_5min.get('analysis_details', {}).get('combined_regime', '') if analysis_5min else ''
        combined_1m = analysis_1min.get('analysis_details', {}).get('combined_regime', '') if analysis_1min else ''
        
        # CONTEXT-AWARE SELECTION LOGIC
        
        # 1. Handle trending markets with volatility context
        if market_5min == "TRENDING":
            if 'trending_high_vol' in combined_5m or market_1min == "HIGH_VOLATILITY":
                selected_strategy = 'StrategyHighVolatilityTrendRider'
                context_reason = "High volatility trend requires robust trend rider"
            elif 'trending_low_vol' in combined_5m or market_1min == "LOW_VOLATILITY":
                selected_strategy = 'StrategyLowVolatilityTrendPullback'
                context_reason = "Low volatility trend suits pullback strategy"
            elif market_1min == "TRENDING":
                selected_strategy = 'StrategyEMATrendRider'  # Both timeframes trending
                context_reason = "Stable multi-timeframe trend"
            else:
                selected_strategy = 'StrategyBreakoutAndRetest'
                context_reason = "Trend with mixed 1m conditions"
        
        # 2. Handle ranging markets with volatility context
        elif market_5min == "RANGING":
            if 'micro_range_low_vol' in combined_5m or (market_1min == "LOW_VOLATILITY" and 'micro_range' in combined_1m):
                selected_strategy = 'StrategyMicroRangeScalping'
                context_reason = "Micro range with low volatility"
            elif market_1min == "HIGH_VOLATILITY":
                selected_strategy = 'StrategyVolatilityReversalScalping'
                context_reason = "Range with high volatility spikes"
            elif market_1min == "TRENDING":
                selected_strategy = 'StrategyRangeBreakoutMomentum'
                context_reason = "Range with 1m trend breakout potential"
            elif market_1min == "TRANSITIONAL":
                selected_strategy = 'StrategyRangeBreakoutMomentum'
                context_reason = "Range with 1m transitional breakout potential"
            else:
                selected_strategy = 'StrategyRSIRangeScalping'
                context_reason = "Standard range conditions"
        
        # 3. Handle high volatility with confirmation
        elif market_5min == "HIGH_VOLATILITY":
            if 'high_vol_no_trend' in combined_5m:
                selected_strategy = 'StrategyATRMomentumBreakout'
                context_reason = "Pure high volatility momentum"
            elif market_1min == "TRENDING" or 'trending_high_vol' in combined_1m:
                selected_strategy = 'StrategyHighVolatilityTrendRider'
                context_reason = "High volatility with trend component"
            else:
                selected_strategy = 'StrategyVolatilityReversalScalping'
                context_reason = "High volatility mean reversion setup"
        
        # 4. Handle low volatility with squeeze detection
        elif market_5min == "LOW_VOLATILITY":
            if 'micro_range_low_vol' in combined_5m:
                selected_strategy = 'StrategyMicroRangeScalping'
                context_reason = "Confirmed micro range environment"
            elif market_1min == "TRANSITIONAL" and 'squeeze_breakout_setup' in combined_1m:
                selected_strategy = 'StrategyVolatilitySqueezeBreakout'
                context_reason = "Volatility squeeze with breakout momentum detected"
            elif market_1min == "TRENDING":
                selected_strategy = 'StrategyLowVolatilityTrendPullback'
                context_reason = "Low volatility with trending component"
            else:
                # Avoid squeeze breakout without confirmation
                selected_strategy = 'StrategyMicroRangeScalping'
                context_reason = "Low volatility without breakout signals"
        
        # 5. Handle transitional states
        elif market_5min == "TRANSITIONAL":
            if 'squeeze_breakout_setup' in combined_5m:
                selected_strategy = 'StrategyVolatilitySqueezeBreakout'
                context_reason = "Confirmed volatility squeeze setup"
            elif market_1min == "HIGH_VOLATILITY":
                selected_strategy = 'StrategyAdaptiveTransitionalMomentum'
                context_reason = "Transitional with high volatility momentum"
            elif market_1min == "TRENDING":
                selected_strategy = 'StrategyBreakoutAndRetest'
                context_reason = "Transitional with trend emergence"
            else:
                selected_strategy = 'StrategyAdaptiveTransitionalMomentum'
                context_reason = "General transitional conditions"
        
        # 6. Fallback to matrix for standard combinations
        else:
            matrix_key = (market_5min, market_1min)
            if matrix_key in self.STRATEGY_MATRIX:
                selected_strategy = self.STRATEGY_MATRIX[matrix_key]
                context_reason = f"Standard matrix selection for {matrix_key}"
            else:
                selected_strategy = 'StrategyBreakoutAndRetest'
                context_reason = f"Fallback for unknown combination {matrix_key}"
        
        # Get execution timeframe from risk profile
        if selected_strategy in self.STRATEGY_RISK_PROFILES:
            risk_profile = self.STRATEGY_RISK_PROFILES[selected_strategy]
            execution_timeframe = risk_profile.execution_timeframe
            description = risk_profile.description
        else:
            execution_timeframe = '1m'  # Default fallback
            description = "Strategy description not available"
        
        # Apply directional bias preference (favor short-oriented strategies in bearish conditions)
        bias_adjustment = ""
        if directional_bias in ['BEARISH', 'BEARISH_BIASED']:
            # Prefer strategies that work well with short trades
            short_friendly_strategies = {
                'StrategyBreakoutAndRetest': 'StrategyVolatilityReversalScalping',
                'StrategyEMATrendRider': 'StrategyHighVolatilityTrendRider',
                'StrategyRSIRangeScalping': 'StrategyVolatilityReversalScalping'
            }
            
            if selected_strategy in short_friendly_strategies:
                alternative = short_friendly_strategies[selected_strategy]
                if alternative in self.STRATEGY_RISK_PROFILES:
                    selected_strategy = alternative
                    bias_adjustment = f" (switched to {alternative} for bearish bias)"
        
        # Build comprehensive selection reason
        timeframe_reason = "5-min execution for stable trend-following" if execution_timeframe == '5m' else "1-min execution for precision and rapid response"
        
        reason = f"Context-aware selection: {selected_strategy} on {execution_timeframe} for {market_5min}(5m) + {market_1min}(1m). {context_reason}. {timeframe_reason}.{bias_adjustment}"
        
        self.logger.info(f"Strategy Matrix: {reason}")
        return selected_strategy, execution_timeframe, reason

    def get_strategy_risk_profile(self, strategy_name: str) -> Optional[StrategyRiskProfile]:
        """
        Get comprehensive risk profile for a strategy.
        
        Args:
            strategy_name: Name of the strategy class
            
        Returns:
            StrategyRiskProfile or None if not found
        """
        return self.STRATEGY_RISK_PROFILES.get(strategy_name)
    
    def get_all_strategy_profiles(self) -> Dict[str, StrategyRiskProfile]:
        """
        Get all strategy risk profiles.
        
        Returns:
            Dictionary of all strategy risk profiles
        """
        return self.STRATEGY_RISK_PROFILES.copy()
    
    def get_strategies_by_factor(self, factor: str) -> List[str]:
        """
        Get strategies by factor exposure.
        
        Args:
            factor: Factor type (momentum, mean_reversion, trend_following, etc.)
            
        Returns:
            List of strategy names with that factor exposure
        """
        strategies = []
        for name, profile in self.STRATEGY_RISK_PROFILES.items():
            if profile.portfolio_tags.factor == factor:
                strategies.append(name)
        return strategies
    
    def get_strategies_by_correlation_group(self, correlation_group: str) -> List[str]:
        """
        Get strategies by correlation group.
        
        Args:
            correlation_group: Correlation group identifier
            
        Returns:
            List of strategy names in that correlation group
        """
        strategies = []
        for name, profile in self.STRATEGY_RISK_PROFILES.items():
            if profile.portfolio_tags.correlation_group == correlation_group:
                strategies.append(name)
        return strategies
    
    def get_max_concurrent_trades_for_strategy(self, strategy_name: str) -> int:
        """
        Get maximum concurrent trades allowed for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Maximum concurrent trades (default: 3)
        """
        profile = self.get_strategy_risk_profile(strategy_name)
        return profile.trading_limits.max_concurrent_trades if profile else 3
    
    def validate_volatility_regime_for_strategy(self, strategy_name: str, current_volatility_pct: float) -> bool:
        """
        Validate if current volatility is suitable for the strategy.
        
        Args:
            strategy_name: Name of the strategy
            current_volatility_pct: Current market volatility percentage
            
        Returns:
            True if volatility is within strategy's acceptable range
        """
        profile = self.get_strategy_risk_profile(strategy_name)
        if not profile:
            return True  # Allow if profile not found
            
        return profile.min_volatility_pct <= current_volatility_pct <= profile.max_volatility_pct
    
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
            str: Formatted matrix summary with timeframes and risk info
        """
        summary = "Enhanced Strategy Matrix Summary (with Risk Profiles):\n"
        summary += "=" * 100 + "\n"
        
        conditions = ['TRENDING', 'RANGING', 'HIGH_VOLATILITY', 'LOW_VOLATILITY', 'TRANSITIONAL']
        
        # Header
        summary += f"{'5min \\ 1min':<15}"
        for condition in conditions:
            summary += f"{condition[:8]:<14}"
        summary += "\n" + "-" * 100 + "\n"
        
        # Matrix rows
        for row_condition in conditions:
            summary += f"{row_condition[:12]:<15}"
            for col_condition in conditions:
                strategy_name = self.STRATEGY_MATRIX.get((row_condition, col_condition), 'N/A')
                if strategy_name != 'N/A' and strategy_name in self.STRATEGY_RISK_PROFILES:
                    profile = self.STRATEGY_RISK_PROFILES[strategy_name]
                    strategy_short = strategy_name.replace('Strategy', '')[:8]
                    timeframe = profile.execution_timeframe
                    entry_display = f"{strategy_short}({timeframe})"
                else:
                    entry_display = "N/A"
                summary += f"{entry_display:<14}"
            summary += "\n"
        
        summary += "\n" + "=" * 100 + "\n"
        summary += "STRATEGY RISK PROFILE SUMMARY:\n"
        summary += "=" * 100 + "\n"
        
        for strategy_name, profile in self.STRATEGY_RISK_PROFILES.items():
            summary += f"\n{strategy_name}:\n"
            summary += f"  Description: {profile.description}\n"
            summary += f"  Timeframe: {profile.execution_timeframe}\n"
            summary += f"  Stop Loss: {profile.stop_loss.mode} ({profile.stop_loss.atr_multiplier}x ATR | {profile.stop_loss.fixed_pct:.1%})\n"
            summary += f"  Take Profit: {profile.take_profit.mode} {profile.take_profit.progressive_levels if profile.take_profit.mode == 'progressive_levels' else f'{profile.take_profit.fixed_pct:.1%}'}\n"
            summary += f"  Position Sizing: {profile.position_sizing.mode} (Risk: {profile.position_sizing.risk_per_trade:.1%})\n"
            summary += f"  Leverage Regime: Low={profile.leverage_by_regime.low}x, Normal={profile.leverage_by_regime.normal}x, High={profile.leverage_by_regime.high}x\n"
            summary += f"  Max Trades: {profile.trading_limits.max_concurrent_trades} concurrent, {profile.trading_limits.max_per_symbol} per symbol\n"
            summary += f"  Factor: {profile.portfolio_tags.factor}, Correlation Group: {profile.portfolio_tags.correlation_group}\n"
            summary += f"  Volatility Range: {profile.min_volatility_pct:.1f}% - {profile.max_volatility_pct:.1f}%\n"
        
        summary += "\nTimeframe Legend:\n"
        summary += "5m = 5-minute execution (stable trend-following)\n"
        summary += "1m = 1-minute execution (precision scalping/breakouts)\n"
        
        return summary
        
    def get_portfolio_correlation_matrix(self) -> Dict[str, List[str]]:
        """
        Get correlation groups for portfolio risk management.
        
        Returns:
            Dictionary mapping correlation groups to strategy lists
        """
        correlation_groups = {}
        for strategy_name, profile in self.STRATEGY_RISK_PROFILES.items():
            group = profile.portfolio_tags.correlation_group
            if group not in correlation_groups:
                correlation_groups[group] = []
            correlation_groups[group].append(strategy_name)
        
        return correlation_groups 