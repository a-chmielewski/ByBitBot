import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

# Import risk utilities for correlation and advanced calculations
try:
    from .risk_utilities import zscore, compute_atr
except ImportError:
    # Fallback if risk_utilities not available
    def zscore(series, lookback=100):
        return series
    def compute_atr(df, period=14):
        return df['close'] * 0.01

class RiskLevel(Enum):
    """Risk level enumeration"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"

class RiskViolationType(Enum):
    """Risk violation type enumeration"""
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    POSITION_SIZE_LIMIT = "position_size_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    CORRELATION_LIMIT = "correlation_limit"
    VOLATILITY_LIMIT = "volatility_limit"
    EXPOSURE_LIMIT = "exposure_limit"
    # New portfolio-level violations
    TOTAL_OPEN_RISK_LIMIT = "total_open_risk_limit"
    ASSET_EXPOSURE_LIMIT = "asset_exposure_limit"
    DIRECTIONAL_EXPOSURE_LIMIT = "directional_exposure_limit"
    DRAWDOWN_CIRCUIT_BREAKER = "drawdown_circuit_breaker"
    CORRELATION_SCALING = "correlation_scaling"


class EnforceAction(Enum):
    """Portfolio limit enforcement actions"""
    ALLOW = "allow"
    DENY = "deny"
    SCALE_DOWN = "scale_down"
    DEFER = "defer"


class TradingState(Enum):
    """Trading state enumeration"""
    ACTIVE = "active"
    PAUSED = "paused"
    EMERGENCY_STOP = "emergency_stop"
    CIRCUIT_BREAKER = "circuit_breaker"

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    # Daily/Session limits
    max_daily_loss_pct: float = 5.0  # Maximum daily loss percentage
    max_daily_loss_amount: Optional[float] = None  # Maximum daily loss amount
    max_drawdown_pct: float = 10.0  # Maximum drawdown percentage
    max_consecutive_losses: int = 5  # Maximum consecutive losses
    
    # Position limits
    max_position_size_pct: float = 10.0  # Maximum position size as % of account
    max_total_exposure_pct: float = 30.0  # Maximum total exposure
    max_leverage: float = 25.0  # Maximum leverage allowed
    
    # Risk management
    min_risk_reward_ratio: float = 1.5  # Minimum risk/reward ratio
    max_correlation_threshold: float = 0.7  # Maximum correlation between positions
    volatility_multiplier: float = 1.5  # Volatility-based position sizing multiplier
    
    # Time-based limits
    max_trades_per_hour: int = 10  # Maximum trades per hour
    cooldown_after_loss_minutes: int = 30  # Cooldown after significant loss
    
    # Enhanced portfolio-level limits
    auto_disable_on_daily_loss: bool = True  # Auto-disable new entries on daily loss limit
    max_total_open_risk_pct: float = 5.0  # Maximum sum of all position risks
    max_asset_exposure_pct: float = 25.0  # Maximum exposure per asset
    max_long_exposure_pct: float = 60.0  # Maximum net long exposure
    max_short_exposure_pct: float = 60.0  # Maximum net short exposure
    circuit_breaker_drawdown_pct: float = 6.0  # Drawdown level for circuit breaker
    circuit_breaker_lookback_hours: int = 24  # Lookback period for drawdown calculation
    circuit_breaker_pause_hours: int = 4  # Hours to pause trading after circuit breaker
    correlation_scaling_threshold: float = 0.7  # Correlation threshold for position scaling
    correlation_scaling_factor: float = 0.5  # Factor to scale position size by correlation


@dataclass
class EnforceResult:
    """Result of portfolio limit enforcement"""
    action: EnforceAction
    allowed: bool
    original_size: float
    recommended_size: float
    scaling_factor: float
    reason: str
    violations: List[Dict[str, Any]]
    warnings: List[str]


@dataclass 
class CircuitBreakerState:
    """Circuit breaker state tracking"""
    is_active: bool = False
    activated_at: Optional[datetime] = None
    reason: str = ""
    pause_until: Optional[datetime] = None
    trigger_drawdown_pct: float = 0.0
    recovery_threshold_pct: float = 2.0  # Drawdown must improve by this much to auto-recover


@dataclass
class AssetExposure:
    """Per-asset exposure tracking"""
    symbol: str
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    net_exposure: float = 0.0
    total_exposure: float = 0.0
    exposure_pct: float = 0.0
    position_count: int = 0
    
    
@dataclass
class CorrelationData:
    """Correlation analysis data"""
    symbol_pairs: Dict[Tuple[str, str], float]
    max_correlation: float = 0.0
    correlated_symbols: List[str] = None
    scaling_required: bool = False
    recommended_scaling_factor: float = 1.0

@dataclass
class PositionRisk:
    """Individual position risk metrics"""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    position_value: float
    leverage: float
    risk_amount: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    time_in_position: Optional[float] = None  # Hours
    volatility_score: Optional[float] = None

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_account_value: float
    total_exposure: float
    total_exposure_pct: float
    net_exposure: float
    net_exposure_pct: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    daily_pnl: float
    daily_pnl_pct: float
    current_drawdown_pct: float
    var_95: Optional[float] = None  # Value at Risk 95%
    correlation_risk: Optional[float] = None
    concentration_risk: Optional[float] = None

class AdvancedRiskManager:
    """
    Advanced Risk Management System
    
    Features:
    - Dynamic position sizing based on volatility and market conditions
    - Portfolio-level risk monitoring and limits
    - Real-time risk assessment and alerts
    - Correlation analysis between positions
    - Volatility-adjusted risk parameters
    - Emergency risk controls and position liquidation
    - Risk reporting and analytics
    """
    
    def __init__(self, exchange, performance_tracker, session_manager, strategy_matrix, config: Dict = None, logger: Optional[logging.Logger] = None):
        self.exchange = exchange
        self.performance_tracker = performance_tracker
        self.session_manager = session_manager
        self.strategy_matrix = strategy_matrix
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Load risk limits from config
        self.risk_limits = RiskLimits(**self.config.get('risk_limits', {}))
        
        # State tracking
        self.positions: Dict[str, PositionRisk] = {}
        self.risk_violations: List[Dict[str, Any]] = []
        self.trading_state = TradingState.ACTIVE
        self.circuit_breaker_state = CircuitBreakerState()
        self.drawdown_history: List[Dict[str, Any]] = []
        self.rolling_drawdown_pct = 0.0
        self.daily_loss_limit_hit = False
        self.total_open_risk = 0.0
        self.asset_exposures: Dict[str, AssetExposure] = {}
        self.emergency_stop_active = False
        
        # Thread safety and caching
        self._lock = threading.Lock()
        self._portfolio_risk_cache = None
        self._cache_timestamp = None
        self._cache_validity_seconds = self.config.get('cache_validity_seconds', 10)
        
        self.logger.info("AdvancedRiskManager initialized with enhanced portfolio-level controls")

    def calculate_trade_parameters(self, symbol: str, side: str, entry_price: float, 
                                   strategy_risk_profile, directional_bias: str, latest_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Calculate all trade parameters including position size, SL, and TP.
        Includes directional bias filter.
        """
        try:
            # 1. Apply Directional Bias Filter
            if not self._apply_directional_filter(side, directional_bias, latest_data):
                self.logger.info(f"Trade signal '{side}' for {symbol} filtered out by directional bias '{directional_bias}'.")
                return None

            # 2. Calculate Stop-Loss and Take-Profit
            sl_tp = self._calculate_sl_tp(symbol, side, entry_price, strategy_risk_profile, latest_data)
            if not sl_tp:
                return None
            
            stop_loss_price = sl_tp['stop_loss_price']
            
            # 3. Calculate Position Size
            position_size_info = self._calculate_position_size(
                symbol=symbol,
                entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                strategy_risk_profile=strategy_risk_profile
            )
            
            if not position_size_info or position_size_info['quantity'] <= 0:
                self.logger.warning("Position size calculation resulted in zero or negative quantity.")
                return None

            # 4. Final Validation (optional, can add portfolio checks here)

            return {**sl_tp, **position_size_info}

        except Exception as e:
            self.logger.error(f"Error calculating trade parameters for {symbol}: {e}", exc_info=True)
            return None

    def _apply_directional_filter(self, side: str, directional_bias: str, latest_data: pd.DataFrame) -> bool:
        """
        Apply higher-timeframe directional bias and stricter entry criteria.
        """
        if directional_bias == 'BEARISH' and side == 'BUY':
            self.logger.warning("FILTER: Blocking LONG trade due to strong BEARISH 1h bias.")
            return False
        
        if directional_bias == 'BULLISH' and side == 'BUY':
            # Apply stricter criteria for long trades even in a bullish market
            if 'adx' in latest_data.columns:
                adx = latest_data['adx'].iloc[-1]
                
                # Stricter ADX threshold for long entries
                if adx < self.config.get('long_entry_adx_threshold', 30):
                    self.logger.info(f"FILTER: Blocking LONG trade. ADX ({adx:.2f}) is below stricter threshold of 30.")
                    return False
            else:
                self.logger.warning("ADX column not found in latest_data, skipping ADX filter for long entry.")
        
        # Allow all short trades and long trades that pass the stricter criteria
        return True

    def _calculate_sl_tp(self, symbol: str, side: str, entry_price: float, 
                         strategy_risk_profile, latest_data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Calculates stop-loss and take-profit prices."""
        try:
            sl_config = strategy_risk_profile.stop_loss
            tp_config = strategy_risk_profile.take_profit
            
            # Stop-Loss Calculation
            stop_loss_price = 0.0
            if sl_config.mode == 'atr_mult':
                atr = latest_data['atr_14'].iloc[-1]
                offset = atr * sl_config.atr_multiplier
            else: # fixed_pct
                offset = entry_price * sl_config.fixed_pct

            # Max loss cap
            max_loss_offset = entry_price * sl_config.max_loss_pct
            offset = min(offset, max_loss_offset)

            if side == 'BUY':
                stop_loss_price = entry_price - offset
            else: # SELL
                stop_loss_price = entry_price + offset

            # Take-Profit Calculation (simplified for first target)
            take_profit_price = 0.0
            if tp_config.mode == 'progressive_levels' and tp_config.progressive_levels:
                tp_pct = tp_config.progressive_levels[0] # Use first TP level for initial order
                tp_offset = entry_price * tp_pct
            else: # fixed_pct
                tp_offset = entry_price * tp_config.fixed_pct
            
            if side == 'BUY':
                take_profit_price = entry_price + tp_offset
            else: # SELL
                take_profit_price = entry_price - tp_offset

            return {'stop_loss_price': stop_loss_price, 'take_profit_price': take_profit_price}
        
        except Exception as e:
            self.logger.error(f"Error calculating SL/TP for {symbol}: {e}")
            return None

    def _calculate_position_size(self, symbol: str, entry_price: float, stop_loss_price: float, 
                                 strategy_risk_profile) -> Optional[Dict[str, float]]:
        """Calculates position size based on risk parameters."""
        try:
            sizing_config = strategy_risk_profile.position_sizing
            account_balance = self.exchange.get_balance() # Simplified

            risk_per_trade_abs = abs(entry_price - stop_loss_price)
            if risk_per_trade_abs == 0:
                self.logger.warning("Risk per trade is zero, cannot calculate position size.")
                return None

            quantity = 0.0
            if sizing_config.mode == 'fixed_notional':
                quantity = sizing_config.fixed_notional / entry_price
            elif sizing_config.mode == 'vol_normalized' or sizing_config.mode == 'kelly_capped':
                risk_amount = account_balance * sizing_config.risk_per_trade
                quantity = risk_amount / risk_per_trade_abs

            # Apply max position constraint
            max_position_value = account_balance * (sizing_config.max_position_pct / 100)
            max_quantity = max_position_value / entry_price
            
            final_quantity = min(quantity, max_quantity)

            # TODO: Add portfolio-level risk checks from enforce_portfolio_limits
            
            return {'quantity': final_quantity}

        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return None

    def calculate_position_size(self, symbol: str, strategy_name: str, market_context: Dict[str, Any], 
                               risk_pct: float = 0.01, entry_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Public method for calculating position size used by bot.py
        Delegates to strategy matrix for proper risk profile-based calculation
        """
        try:
            # Get strategy risk profile from strategy matrix
            strategy_risk_profile = self.strategy_matrix.get_strategy_risk_profile(strategy_name)
            if not strategy_risk_profile:
                # Fallback calculation if no strategy profile
                account_balance = self.exchange.get_balance()
                if not account_balance or account_balance <= 0:
                    return {'size': 0.001, 'error': 'Invalid account balance'}
                
                # Simple risk-based sizing
                risk_amount = account_balance * risk_pct
                stop_distance_pct = 0.02  # Assume 2% stop loss
                position_size = risk_amount / ((entry_price or 1.0) * stop_distance_pct)
                position_size = max(position_size, 0.001)
                
                return {'size': position_size, 'risk_amount': risk_amount, 'account_balance': account_balance}
            
            # Use strategy's position sizing configuration
            account_balance = self.exchange.get_balance()
            if not account_balance or account_balance <= 0:
                return {'size': 0.001, 'error': 'Invalid account balance'}
            
            sizing_config = strategy_risk_profile.position_sizing
            
            if sizing_config.mode == 'fixed_notional':
                position_size = sizing_config.fixed_notional / (entry_price or 1.0)
            elif sizing_config.mode == 'vol_normalized':
                # Use risk percentage for volatility normalized sizing
                risk_amount = account_balance * risk_pct
                stop_distance_pct = 0.02  # Default stop distance
                position_size = risk_amount / ((entry_price or 1.0) * stop_distance_pct)
            elif sizing_config.mode == 'kelly_capped':
                # Conservative Kelly sizing
                kelly_fraction = min(risk_pct * 2, 0.05)  # Cap at 5%
                position_size = (account_balance * kelly_fraction) / (entry_price or 1.0)
            else:
                # Default to risk-based sizing
                risk_amount = account_balance * risk_pct
                stop_distance_pct = 0.02
                position_size = risk_amount / ((entry_price or 1.0) * stop_distance_pct)
            
            # Apply max position constraint
            max_position_pct = sizing_config.max_position_pct / 100
            max_position_value = account_balance * max_position_pct
            max_quantity = max_position_value / (entry_price or 1.0)
            
            final_size = min(max(position_size, 0.001), max_quantity)
            
            return {
                'size': final_size,
                'risk_amount': account_balance * risk_pct,
                'account_balance': account_balance,
                'sizing_mode': sizing_config.mode
            }
            
        except Exception as e:
            self.logger.error(f"Error in calculate_position_size: {e}")
            return {'size': 0.001, 'error': str(e)}

    def validate_trade_risk(self, symbol: str, side: str, size: float, entry_price: float,
                           stop_loss_price: Optional[float] = None, take_profit_price: Optional[float] = None,
                           leverage: float = 1.0) -> Dict[str, Any]:
        """
        Validate if a trade meets risk management criteria
        
        Returns:
            Dict with validation results and recommendations
        """
        try:
            validation_result = {
                'approved': True,
                'violations': [],
                'warnings': [],
                'recommendations': [],
                'risk_score': 0.0
            }
            
            # Validate input parameters
            if size is None or entry_price is None:
                validation_result['approved'] = False
                validation_result['violations'].append({
                    'type': 'INVALID_PARAMETERS',
                    'message': f'Invalid parameters: size={size}, entry_price={entry_price}'
                })
                return validation_result
            
            # Check if emergency stop is active
            if self.emergency_stop_active:
                validation_result['approved'] = False
                validation_result['violations'].append({
                    'type': 'EMERGENCY_STOP_ACTIVE',
                    'message': 'Emergency stop is active, no new trades allowed'
                })
                return validation_result
            
            # Get account info
            account_info = self._get_account_info()
            if not account_info:
                validation_result['approved'] = False
                validation_result['violations'].append({
                    'type': 'ACCOUNT_INFO_UNAVAILABLE',
                    'message': 'Cannot validate trade without account information'
                })
                return validation_result
            
            account_balance = account_info.get('total_balance', 0)
            position_value = float(size) * float(entry_price)
            
            # 1. Position size validation
            position_size_pct = (position_value / account_balance) * 100
            if position_size_pct > self.risk_limits.max_position_size_pct:
                validation_result['approved'] = False
                validation_result['violations'].append({
                    'type': RiskViolationType.POSITION_SIZE_LIMIT.value,
                    'message': f'Position size {position_size_pct:.2f}% exceeds limit {self.risk_limits.max_position_size_pct}%',
                    'current': position_size_pct,
                    'limit': self.risk_limits.max_position_size_pct
                })
            
            # 2. Leverage validation
            if leverage > self.risk_limits.max_leverage:
                validation_result['approved'] = False
                validation_result['violations'].append({
                    'type': RiskViolationType.LEVERAGE_LIMIT.value,
                    'message': f'Leverage {leverage}x exceeds limit {self.risk_limits.max_leverage}x',
                    'current': leverage,
                    'limit': self.risk_limits.max_leverage
                })
            
            # 3. Daily loss limit validation
            daily_pnl = self._get_daily_pnl()
            if self.risk_limits.max_daily_loss_amount:
                if daily_pnl < -self.risk_limits.max_daily_loss_amount:
                    validation_result['approved'] = False
                    validation_result['violations'].append({
                        'type': RiskViolationType.DAILY_LOSS_LIMIT.value,
                        'message': f'Daily loss ${abs(daily_pnl):.2f} exceeds limit ${self.risk_limits.max_daily_loss_amount}',
                        'current': daily_pnl,
                        'limit': -self.risk_limits.max_daily_loss_amount
                    })
            
            daily_loss_pct = (abs(daily_pnl) / account_balance) * 100 if daily_pnl < 0 else 0
            if daily_loss_pct > self.risk_limits.max_daily_loss_pct:
                validation_result['approved'] = False
                validation_result['violations'].append({
                    'type': RiskViolationType.DAILY_LOSS_LIMIT.value,
                    'message': f'Daily loss {daily_loss_pct:.2f}% exceeds limit {self.risk_limits.max_daily_loss_pct}%',
                    'current': daily_loss_pct,
                    'limit': self.risk_limits.max_daily_loss_pct
                })
            
            # 4. Consecutive losses validation
            consecutive_losses = self._get_consecutive_losses()
            if consecutive_losses >= self.risk_limits.max_consecutive_losses:
                validation_result['approved'] = False
                validation_result['violations'].append({
                    'type': RiskViolationType.CONSECUTIVE_LOSSES.value,
                    'message': f'Consecutive losses {consecutive_losses} reaches limit {self.risk_limits.max_consecutive_losses}',
                    'current': consecutive_losses,
                    'limit': self.risk_limits.max_consecutive_losses
                })
            
            # 5. Risk-reward ratio validation
            if stop_loss_price and take_profit_price:
                risk_reward_ratio = self._calculate_risk_reward_ratio(
                    entry_price, stop_loss_price, take_profit_price, side
                )
                if risk_reward_ratio < self.risk_limits.min_risk_reward_ratio:
                    validation_result['warnings'].append({
                        'type': 'LOW_RISK_REWARD_RATIO',
                        'message': f'Risk/reward ratio {risk_reward_ratio:.2f} below recommended {self.risk_limits.min_risk_reward_ratio}',
                        'current': risk_reward_ratio,
                        'recommended': self.risk_limits.min_risk_reward_ratio
                    })
            
            # 6. Total exposure validation
            current_exposure = self._get_total_exposure()
            new_exposure = current_exposure + position_value
            new_exposure_pct = (new_exposure / account_balance) * 100
            
            if new_exposure_pct > self.risk_limits.max_total_exposure_pct:
                validation_result['approved'] = False
                validation_result['violations'].append({
                    'type': RiskViolationType.EXPOSURE_LIMIT.value,
                    'message': f'Total exposure would be {new_exposure_pct:.2f}%, exceeding limit {self.risk_limits.max_total_exposure_pct}%',
                    'current': new_exposure_pct,
                    'limit': self.risk_limits.max_total_exposure_pct
                })
            
            # 7. Trade frequency validation
            if self._check_trade_frequency_violation():
                validation_result['warnings'].append({
                    'type': 'HIGH_TRADE_FREQUENCY',
                    'message': f'High trade frequency detected, consider reducing trading activity'
                })
            
            # Calculate overall risk score (0-100)
            risk_factors = [
                position_size_pct / self.risk_limits.max_position_size_pct,
                leverage / self.risk_limits.max_leverage,
                daily_loss_pct / self.risk_limits.max_daily_loss_pct,
                consecutive_losses / self.risk_limits.max_consecutive_losses,
                new_exposure_pct / self.risk_limits.max_total_exposure_pct
            ]
            validation_result['risk_score'] = min(100, max(risk_factors) * 100)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating trade risk: {e}", exc_info=True)
            return {
                'approved': False,
                'violations': [{'type': 'VALIDATION_ERROR', 'message': f'Risk validation failed: {str(e)}'}],
                'warnings': [],
                'recommendations': [],
                'risk_score': 100.0
            }

    def get_portfolio_risk_assessment(self) -> PortfolioRisk:
        """Get comprehensive portfolio risk assessment"""
        try:
            # Check cache first
            if self._portfolio_risk_cache and self._cache_timestamp:
                if (datetime.now().timestamp() - self._cache_timestamp) < self._cache_validity_seconds:
                    return self._portfolio_risk_cache
            
            with self._lock:
                account_info = self._get_account_info()
                if not account_info:
                    return self._get_empty_portfolio_risk()
                
                total_balance = account_info.get('total_balance', 0)
                
                # Calculate exposure metrics
                total_exposure = self._get_total_exposure()
                total_exposure_pct = (total_exposure / total_balance * 100) if total_balance > 0 else 0
                
                net_exposure = self._get_net_exposure()
                net_exposure_pct = (net_exposure / total_balance * 100) if total_balance > 0 else 0
                
                # Calculate P&L metrics
                unrealized_pnl = self._get_unrealized_pnl()
                unrealized_pnl_pct = (unrealized_pnl / total_balance * 100) if total_balance > 0 else 0
                
                daily_pnl = self._get_daily_pnl()
                daily_pnl_pct = (daily_pnl / total_balance * 100) if total_balance > 0 else 0
                
                # Calculate drawdown
                current_drawdown_pct = self._calculate_current_drawdown_pct()
                
                # Advanced risk metrics
                var_95 = self._calculate_var_95()
                correlation_risk = self._calculate_correlation_risk()
                concentration_risk = self._calculate_concentration_risk()
                
                portfolio_risk = PortfolioRisk(
                    total_account_value=total_balance,
                    total_exposure=total_exposure,
                    total_exposure_pct=total_exposure_pct,
                    net_exposure=net_exposure,
                    net_exposure_pct=net_exposure_pct,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_pnl_pct=unrealized_pnl_pct,
                    daily_pnl=daily_pnl,
                    daily_pnl_pct=daily_pnl_pct,
                    current_drawdown_pct=current_drawdown_pct,
                    var_95=var_95,
                    correlation_risk=correlation_risk,
                    concentration_risk=concentration_risk
                )
                
                # Cache the result
                self._portfolio_risk_cache = portfolio_risk
                self._cache_timestamp = datetime.now().timestamp()
                
                return portfolio_risk
                
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk: {e}", exc_info=True)
            return self._get_empty_portfolio_risk()

    def check_emergency_conditions(self) -> Dict[str, Any]:
        """Check for emergency conditions that require immediate action"""
        try:
            emergency_conditions = {
                'emergency_stop_required': False,
                'conditions': [],
                'recommended_actions': []
            }
            
            portfolio_risk = self.get_portfolio_risk_assessment()
            
            # Check drawdown emergency
            if portfolio_risk.current_drawdown_pct > self.risk_limits.max_drawdown_pct:
                emergency_conditions['emergency_stop_required'] = True
                emergency_conditions['conditions'].append({
                    'type': 'CRITICAL_DRAWDOWN',
                    'message': f'Drawdown {portfolio_risk.current_drawdown_pct:.2f}% exceeds emergency limit {self.risk_limits.max_drawdown_pct}%',
                    'severity': 'CRITICAL'
                })
                emergency_conditions['recommended_actions'].append('STOP_ALL_TRADING')
                emergency_conditions['recommended_actions'].append('CLOSE_ALL_POSITIONS')
            
            # Check daily loss emergency
            if portfolio_risk.daily_pnl_pct < -self.risk_limits.max_daily_loss_pct:
                emergency_conditions['emergency_stop_required'] = True
                emergency_conditions['conditions'].append({
                    'type': 'DAILY_LOSS_LIMIT_EXCEEDED',
                    'message': f'Daily loss {abs(portfolio_risk.daily_pnl_pct):.2f}% exceeds limit {self.risk_limits.max_daily_loss_pct}%',
                    'severity': 'CRITICAL'
                })
                emergency_conditions['recommended_actions'].append('STOP_ALL_TRADING')
            
            # Check exposure emergency
            if portfolio_risk.total_exposure_pct > self.risk_limits.max_total_exposure_pct * 1.2:  # 20% buffer
                emergency_conditions['emergency_stop_required'] = True
                emergency_conditions['conditions'].append({
                    'type': 'EXCESSIVE_EXPOSURE',
                    'message': f'Total exposure {portfolio_risk.total_exposure_pct:.2f}% critically high',
                    'severity': 'HIGH'
                })
                emergency_conditions['recommended_actions'].append('REDUCE_POSITIONS')
            
            return emergency_conditions
            
        except Exception as e:
            self.logger.error(f"Error checking emergency conditions: {e}", exc_info=True)
            return {
                'emergency_stop_required': True,
                'conditions': [{'type': 'SYSTEM_ERROR', 'message': f'Emergency check failed: {str(e)}', 'severity': 'CRITICAL'}],
                'recommended_actions': ['STOP_ALL_TRADING']
            }

    def activate_emergency_stop(self, reason: str):
        """Activate emergency stop"""
        with self._lock:
            self.emergency_stop_active = True
            self.logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")
            
            # Record the emergency stop without calling portfolio risk assessment
            # to avoid circular dependency
            self.risk_violations.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'type': 'EMERGENCY_STOP',
                'reason': reason,
                'emergency_stop_active': True
            })

    def deactivate_emergency_stop(self, reason: str):
        """Deactivate emergency stop"""
        with self._lock:
            self.emergency_stop_active = False
            self.logger.info(f"Emergency stop deactivated: {reason}")

    # Helper methods
    def _get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information from exchange"""
        try:
            # This would integrate with your exchange connector
            # For now, return mock data
            return {
                'total_balance': 10000.0,  # Mock balance
                'available_balance': 9000.0,
                'unrealized_pnl': 0.0
            }
        except Exception as e:
            self.logger.error(f"Error fetching account info: {e}")
            return None

    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            # This would integrate with your exchange connector
            # For now, return mock price
            return 50000.0  # Mock price
        except Exception as e:
            self.logger.error(f"Error fetching current price for {symbol}: {e}")
            return None

    def _get_market_condition_adjustment(self, market_context: Dict[str, Any]) -> float:
        """Get position size adjustment based on market conditions"""
        try:
            market_5m = market_context.get('market_5m', 'UNKNOWN')
            market_1m = market_context.get('market_1m', 'UNKNOWN')
            
            # Reduce size in high volatility conditions
            if 'HIGH_VOLATILITY' in [market_5m, market_1m]:
                return 0.7  # Reduce position size by 30%
            elif 'TRANSITIONAL' in [market_5m, market_1m]:
                return 0.8  # Reduce position size by 20%
            elif 'TRENDING' in [market_5m, market_1m]:
                return 1.0  # Normal position size
            elif 'RANGING' in [market_5m, market_1m]:
                return 0.9  # Slightly reduce position size
            else:
                return 0.8  # Conservative default
                
        except Exception:
            return 0.8  # Conservative fallback

    def _get_volatility_adjustment(self, volatility_data: Optional[Dict[str, float]]) -> float:
        """Get position size adjustment based on volatility"""
        try:
            if not volatility_data:
                return 1.0
            
            # Use ATR or volatility score if available
            volatility_score = volatility_data.get('volatility_score', 1.0)
            
            # Higher volatility = smaller position size
            if volatility_score > 2.0:
                return 0.6
            elif volatility_score > 1.5:
                return 0.8
            elif volatility_score < 0.5:
                return 1.2  # Can increase size in low volatility
            else:
                return 1.0
                
        except Exception:
            return 1.0

    def _get_portfolio_risk_adjustment(self) -> float:
        """Get position size adjustment based on current portfolio risk"""
        try:
            portfolio_risk = self.get_portfolio_risk_assessment()
            
            # Reduce size if already high exposure
            if portfolio_risk.total_exposure_pct > 20:
                return 0.7
            elif portfolio_risk.total_exposure_pct > 15:
                return 0.8
            elif portfolio_risk.current_drawdown_pct > 3:
                return 0.7
            else:
                return 1.0
                
        except Exception:
            return 0.8

    def _get_strategy_risk_adjustment(self, strategy_name: str) -> float:
        """Get position size adjustment based on strategy characteristics"""
        try:
            # Different strategies have different risk profiles
            high_risk_strategies = ['volatility_reversal', 'micro_range', 'breakout']
            medium_risk_strategies = ['momentum', 'trend_rider']
            
            strategy_lower = strategy_name.lower()
            
            if any(risk_term in strategy_lower for risk_term in high_risk_strategies):
                return 0.7  # Reduce size for high-risk strategies
            elif any(risk_term in strategy_lower for risk_term in medium_risk_strategies):
                return 0.9  # Slightly reduce size
            else:
                return 1.0  # Normal size
                
        except Exception:
            return 0.9

    def _get_fallback_position_size(self, risk_pct: float) -> Dict[str, Any]:
        """Get fallback position size when calculation fails"""
        return {
            'recommended_size': 0.01,  # Very small fallback size
            'risk_amount': 100.0,  # Fallback risk amount
            'risk_pct': risk_pct,
            'market_adjustment': 0.5,
            'volatility_adjustment': 0.5,
            'strategy_adjustment': 0.5,
            'current_price': 50000.0,
            'account_balance': 10000.0,
            'max_position_value': 1000.0,
            'error': 'Using fallback values due to calculation error'
        }

    def _get_daily_pnl(self) -> float:
        """Get daily P&L from performance tracker"""
        try:
            if self.performance_tracker and hasattr(self.performance_tracker, 'get_comprehensive_statistics'):
                stats = self.performance_tracker.get_comprehensive_statistics()
                if 'error' not in stats:
                    # Calculate daily P&L from recent trades
                    today = datetime.now(timezone.utc).date()
                    daily_trades = [
                        trade for trade in self.performance_tracker.trades
                        if trade.entry_timestamp and 
                        pd.to_datetime(trade.entry_timestamp).date() == today
                    ]
                    return sum(trade.pnl for trade in daily_trades)
            return 0.0
        except Exception:
            return 0.0

    def _get_consecutive_losses(self) -> int:
        """Get consecutive losses from performance tracker"""
        try:
            if self.performance_tracker:
                return getattr(self.performance_tracker, 'consecutive_losses', 0)
            return 0
        except Exception:
            return 0

    def _calculate_risk_reward_ratio(self, entry_price: float, stop_loss_price: float, 
                                   take_profit_price: float, side: str) -> float:
        """Calculate risk/reward ratio"""
        try:
            if side.lower() == 'long':
                risk = abs(entry_price - stop_loss_price)
                reward = abs(take_profit_price - entry_price)
            else:  # short
                risk = abs(stop_loss_price - entry_price)
                reward = abs(entry_price - take_profit_price)
            
            return reward / risk if risk > 0 else 0.0
        except Exception:
            return 0.0

    def _get_total_exposure(self) -> float:
        """Get total portfolio exposure"""
        try:
            # This would integrate with position tracking
            # For now, return mock data
            return sum(pos.position_value for pos in self.positions.values())
        except Exception:
            return 0.0

    def _get_net_exposure(self) -> float:
        """Get net portfolio exposure (long - short)"""
        try:
            long_exposure = sum(pos.position_value for pos in self.positions.values() if pos.side == 'long')
            short_exposure = sum(pos.position_value for pos in self.positions.values() if pos.side == 'short')
            return long_exposure - short_exposure
        except Exception:
            return 0.0

    def _get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L"""
        try:
            return sum(pos.unrealized_pnl for pos in self.positions.values())
        except Exception:
            return 0.0

    def _calculate_current_drawdown_pct(self) -> float:
        """Calculate current drawdown percentage"""
        try:
            if self.performance_tracker:
                high_watermark = getattr(self.performance_tracker, 'high_watermark', 0)
                current_pnl = getattr(self.performance_tracker, 'cumulative_pnl', 0)
                
                if high_watermark > 0:
                    return ((high_watermark - current_pnl) / high_watermark) * 100
            return 0.0
        except Exception:
            return 0.0

    def _calculate_var_95(self) -> Optional[float]:
        """Calculate Value at Risk at 95% confidence level"""
        try:
            if not self.performance_tracker or len(self.performance_tracker.trades) < 20:
                return None
            
            # Get recent trade returns
            recent_trades = self.performance_tracker.trades[-50:]  # Last 50 trades
            returns = [trade.return_pct for trade in recent_trades if trade.return_pct is not None]
            
            if len(returns) < 10:
                return None
            
            # Calculate 5th percentile (95% VaR)
            return float(np.percentile(returns, 5))
            
        except Exception:
            return None

    def _calculate_correlation_risk(self) -> Optional[float]:
        """Calculate correlation risk between positions"""
        try:
            if len(self.positions) < 2:
                return 0.0
            
            # This would require historical price data for correlation calculation
            # For now, return a placeholder
            return 0.3  # Mock correlation risk
            
        except Exception:
            return None

    def _calculate_concentration_risk(self) -> Optional[float]:
        """Calculate concentration risk (largest position as % of portfolio)"""
        try:
            if not self.positions:
                return 0.0
            
            total_exposure = self._get_total_exposure()
            if total_exposure == 0:
                return 0.0
            
            largest_position = max(pos.position_value for pos in self.positions.values())
            return (largest_position / total_exposure) * 100
            
        except Exception:
            return None

    def _check_trade_frequency_violation(self) -> bool:
        """Check if trade frequency exceeds limits"""
        try:
            if not self.performance_tracker:
                return False
            
            # Check trades in last hour
            one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            recent_trades = [
                trade for trade in self.performance_tracker.trades
                if trade.entry_timestamp and 
                pd.to_datetime(trade.entry_timestamp) > one_hour_ago
            ]
            
            return len(recent_trades) > self.risk_limits.max_trades_per_hour
            
        except Exception:
            return False

    def _get_empty_portfolio_risk(self) -> PortfolioRisk:
        """Get empty portfolio risk object"""
        return PortfolioRisk(
            total_account_value=0.0,
            total_exposure=0.0,
            total_exposure_pct=0.0,
            net_exposure=0.0,
            net_exposure_pct=0.0,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            current_drawdown_pct=0.0
        )

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary for reporting"""
        try:
            portfolio_risk = self.get_portfolio_risk_assessment()
            emergency_conditions = self.check_emergency_conditions()
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'emergency_stop_active': self.emergency_stop_active,
                'portfolio_risk': asdict(portfolio_risk),
                'emergency_conditions': emergency_conditions,
                'risk_limits': asdict(self.risk_limits),
                'active_positions': len(self.positions),
                'recent_violations': self.risk_violations[-10:],  # Last 10 violations
                'risk_level': self._assess_overall_risk_level(portfolio_risk)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating risk summary: {e}", exc_info=True)
            return {'error': f'Risk summary generation failed: {str(e)}'}

    def _assess_overall_risk_level(self, portfolio_risk: PortfolioRisk) -> str:
        """Assess overall risk level"""
        try:
            risk_factors = []
            
            # Drawdown risk
            if portfolio_risk.current_drawdown_pct > 7:
                risk_factors.append('HIGH')
            elif portfolio_risk.current_drawdown_pct > 3:
                risk_factors.append('MEDIUM')
            
            # Exposure risk
            if portfolio_risk.total_exposure_pct > 25:
                risk_factors.append('HIGH')
            elif portfolio_risk.total_exposure_pct > 15:
                risk_factors.append('MEDIUM')
            
            # Daily loss risk
            if portfolio_risk.daily_pnl_pct < -3:
                risk_factors.append('HIGH')
            elif portfolio_risk.daily_pnl_pct < -1:
                risk_factors.append('MEDIUM')
            
            # Determine overall risk level
            if 'HIGH' in risk_factors or len(risk_factors) >= 3:
                return RiskLevel.AGGRESSIVE.value
            elif 'MEDIUM' in risk_factors or len(risk_factors) >= 2:
                return RiskLevel.MODERATE.value
            else:
                return RiskLevel.CONSERVATIVE.value
                
        except Exception:
            return RiskLevel.MODERATE.value

    # ==================== ENHANCED PORTFOLIO-LEVEL CONTROLS ====================
    
    def enforce_portfolio_limits(self, account_state: Dict[str, Any], 
                               open_positions: List[Dict[str, Any]], 
                               candidate_order: Dict[str, Any]) -> EnforceResult:
        """
        Enforce comprehensive portfolio-level risk limits and controls.
        
        This is the main entry point for portfolio-level risk management that checks:
        - Daily loss limits with auto-disable functionality
        - Maximum open risk across all positions
        - Per-asset and directional exposure limits
        - Real-time drawdown circuit breakers
        - Correlation-aware position sizing
        
        Args:
            account_state: Current account balance, equity, and PnL information
            open_positions: List of current open positions with risk metrics
            candidate_order: Proposed order with symbol, side, size, entry_price, etc.
            
        Returns:
            EnforceResult: Detailed enforcement result with action, scaling, and reasons
        """
        try:
            with self._lock if not self.read_only_mode else threading.RLock():
                # Initialize result structure
                result = EnforceResult(
                    action=EnforceAction.ALLOW,
                    allowed=True,
                    original_size=candidate_order.get('size', 0.0),
                    recommended_size=candidate_order.get('size', 0.0),
                    scaling_factor=1.0,
                    reason="Portfolio limits check passed",
                    violations=[],
                    warnings=[]
                )
                
                # Update internal state with current data
                self._update_portfolio_state(account_state, open_positions)
                
                # 1. Check daily loss limit with auto-disable
                daily_loss_check = self._check_daily_loss_limit(account_state)
                if not daily_loss_check['allowed']:
                    result.action = EnforceAction.DENY
                    result.allowed = False
                    result.recommended_size = 0.0
                    result.scaling_factor = 0.0
                    result.reason = daily_loss_check['reason']
                    result.violations.append(daily_loss_check)
                    return result
                
                # 2. Check circuit breaker status
                circuit_breaker_check = self._check_circuit_breaker(account_state)
                if not circuit_breaker_check['allowed']:
                    result.action = EnforceAction.DEFER
                    result.allowed = False
                    result.recommended_size = 0.0
                    result.scaling_factor = 0.0
                    result.reason = circuit_breaker_check['reason']
                    result.violations.append(circuit_breaker_check)
                    return result
                
                # 3. Check maximum total open risk
                total_risk_check = self._check_total_open_risk(candidate_order, account_state)
                if not total_risk_check['allowed']:
                    if total_risk_check.get('can_scale', False):
                        scaling_factor = total_risk_check['max_scaling_factor']
                        result.action = EnforceAction.SCALE_DOWN
                        result.recommended_size = result.original_size * scaling_factor
                        result.scaling_factor = scaling_factor
                        result.reason = f"Scaled down due to total risk limit: {total_risk_check['reason']}"
                        result.warnings.append(total_risk_check['reason'])
                    else:
                        result.action = EnforceAction.DENY
                        result.allowed = False
                        result.recommended_size = 0.0
                        result.scaling_factor = 0.0
                        result.reason = total_risk_check['reason']
                        result.violations.append(total_risk_check)
                        return result
                
                # 4. Check per-asset exposure limits
                asset_exposure_check = self._check_asset_exposure_limits(candidate_order, account_state)
                if not asset_exposure_check['allowed']:
                    if asset_exposure_check.get('can_scale', False):
                        scaling_factor = min(result.scaling_factor, asset_exposure_check['max_scaling_factor'])
                        result.action = EnforceAction.SCALE_DOWN
                        result.recommended_size = result.original_size * scaling_factor
                        result.scaling_factor = scaling_factor
                        result.reason = f"Scaled down due to asset exposure: {asset_exposure_check['reason']}"
                        result.warnings.append(asset_exposure_check['reason'])
                    else:
                        result.action = EnforceAction.DENY
                        result.allowed = False
                        result.recommended_size = 0.0
                        result.scaling_factor = 0.0
                        result.reason = asset_exposure_check['reason']
                        result.violations.append(asset_exposure_check)
                        return result
                
                # 5. Check directional exposure limits
                directional_check = self._check_directional_exposure_limits(candidate_order, account_state)
                if not directional_check['allowed']:
                    if directional_check.get('can_scale', False):
                        scaling_factor = min(result.scaling_factor, directional_check['max_scaling_factor'])
                        result.action = EnforceAction.SCALE_DOWN
                        result.recommended_size = result.original_size * scaling_factor
                        result.scaling_factor = scaling_factor
                        result.reason = f"Scaled down due to directional exposure: {directional_check['reason']}"
                        result.warnings.append(directional_check['reason'])
                    else:
                        result.action = EnforceAction.DENY
                        result.allowed = False
                        result.recommended_size = 0.0
                        result.scaling_factor = 0.0
                        result.reason = directional_check['reason']
                        result.violations.append(directional_check)
                        return result
                
                # 6. Check correlation-aware scaling
                correlation_check = self._check_correlation_limits(candidate_order, open_positions)
                if correlation_check.get('scaling_required', False):
                    scaling_factor = min(result.scaling_factor, correlation_check['recommended_scaling_factor'])
                    result.action = EnforceAction.SCALE_DOWN
                    result.recommended_size = result.original_size * scaling_factor
                    result.scaling_factor = scaling_factor
                    result.reason = f"Scaled down due to correlation risk: {correlation_check['reason']}"
                    result.warnings.append(correlation_check['reason'])
                
                # Log the enforcement result
                if result.action != EnforceAction.ALLOW:
                    self.logger.info(f"Portfolio enforcement: {result.action.value} - {result.reason}")
                    if result.scaling_factor < 1.0:
                        self.logger.info(f"Position size scaled from {result.original_size:.6f} to {result.recommended_size:.6f} (factor: {result.scaling_factor:.3f})")
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error in portfolio limits enforcement: {e}", exc_info=True)
            # Return conservative result on error
            return EnforceResult(
                action=EnforceAction.DENY,
                allowed=False,
                original_size=candidate_order.get('size', 0.0),
                recommended_size=0.0,
                scaling_factor=0.0,
                reason=f"Risk management system error: {str(e)}",
                violations=[{
                    'type': 'SYSTEM_ERROR',
                    'message': f"Portfolio limits check failed: {str(e)}",
                    'severity': 'CRITICAL'
                }],
                warnings=[]
            )

    def _update_portfolio_state(self, account_state: Dict[str, Any], open_positions: List[Dict[str, Any]]) -> None:
        """Update internal portfolio state with current data"""
        try:
            # Update asset exposures
            self.asset_exposures.clear()
            total_long_exposure = 0.0
            total_short_exposure = 0.0
            total_risk = 0.0
            
            account_equity = account_state.get('equity', account_state.get('total_balance', 10000.0))
            
            for position in open_positions:
                symbol = position.get('symbol', '')
                side = position.get('side', '').lower()
                size = abs(position.get('size', 0.0))
                entry_price = position.get('entry_price', 0.0)
                current_price = position.get('current_price', entry_price)
                
                if not symbol or size <= 0:
                    continue
                
                # Calculate position value and exposure
                position_value = size * current_price
                exposure_pct = (position_value / account_equity) * 100 if account_equity > 0 else 0
                
                # Calculate position risk (distance to stop loss)
                stop_loss = position.get('stop_loss_price', 0.0)
                if stop_loss > 0:
                    if side == 'long':
                        risk_per_unit = max(0, entry_price - stop_loss)
                    else:
                        risk_per_unit = max(0, stop_loss - entry_price)
                    position_risk = (risk_per_unit * size / account_equity) * 100
                else:
                    # Fallback: assume 2% risk per position
                    position_risk = 2.0
                
                total_risk += position_risk
                
                # Update asset exposure
                if symbol not in self.asset_exposures:
                    self.asset_exposures[symbol] = AssetExposure(symbol=symbol)
                
                asset_exp = self.asset_exposures[symbol]
                if side == 'long':
                    asset_exp.long_exposure += position_value
                    total_long_exposure += position_value
                else:
                    asset_exp.short_exposure += position_value
                    total_short_exposure += position_value
                
                asset_exp.net_exposure = asset_exp.long_exposure - asset_exp.short_exposure
                asset_exp.total_exposure = asset_exp.long_exposure + asset_exp.short_exposure
                asset_exp.exposure_pct = (asset_exp.total_exposure / account_equity) * 100 if account_equity > 0 else 0
                asset_exp.position_count += 1
            
            self.total_open_risk = total_risk
            
            # Update drawdown tracking for circuit breaker
            self._update_drawdown_tracking(account_state)
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio state: {e}", exc_info=True)

    def _check_daily_loss_limit(self, account_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check daily loss limit with auto-disable functionality"""
        try:
            daily_pnl = account_state.get('daily_pnl', 0.0)
            account_equity = account_state.get('equity', account_state.get('total_balance', 10000.0))
            
            if account_equity <= 0:
                return {
                    'allowed': False,
                    'reason': "Invalid account equity for daily loss check",
                    'type': RiskViolationType.DAILY_LOSS_LIMIT.value
                }
            
            daily_loss_pct = abs(daily_pnl / account_equity) * 100 if daily_pnl < 0 else 0.0
            
            # Check if daily loss limit is exceeded
            if daily_loss_pct >= self.risk_limits.max_daily_loss_pct:
                if self.risk_limits.auto_disable_on_daily_loss and not self.daily_loss_limit_hit:
                    self.daily_loss_limit_hit = True
                    self.trading_state = TradingState.PAUSED
                    self.logger.critical(f"Daily loss limit hit: {daily_loss_pct:.2f}% >= {self.risk_limits.max_daily_loss_pct}%. Auto-disabling new entries.")
                
                return {
                    'allowed': False,
                    'reason': f"Daily loss limit exceeded: {daily_loss_pct:.2f}% >= {self.risk_limits.max_daily_loss_pct}%",
                    'type': RiskViolationType.DAILY_LOSS_LIMIT.value,
                    'daily_loss_pct': daily_loss_pct,
                    'limit': self.risk_limits.max_daily_loss_pct
                }
            
            return {
                'allowed': True,
                'reason': f"Daily loss check passed: {daily_loss_pct:.2f}% < {self.risk_limits.max_daily_loss_pct}%",
                'daily_loss_pct': daily_loss_pct
            }
            
        except Exception as e:
            self.logger.error(f"Error checking daily loss limit: {e}")
            return {
                'allowed': False,
                'reason': f"Daily loss check error: {str(e)}",
                'type': RiskViolationType.DAILY_LOSS_LIMIT.value
            }

    def _check_circuit_breaker(self, account_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check circuit breaker status based on rolling drawdown"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Check if circuit breaker is currently active
            if self.circuit_breaker_state.is_active:
                if self.circuit_breaker_state.pause_until and current_time < self.circuit_breaker_state.pause_until:
                    time_remaining = self.circuit_breaker_state.pause_until - current_time
                    hours_remaining = time_remaining.total_seconds() / 3600
                    
                    return {
                        'allowed': False,
                        'reason': f"Circuit breaker active: {self.circuit_breaker_state.reason}. Trading paused for {hours_remaining:.1f} more hours.",
                        'type': RiskViolationType.DRAWDOWN_CIRCUIT_BREAKER.value,
                        'pause_until': self.circuit_breaker_state.pause_until,
                        'trigger_reason': self.circuit_breaker_state.reason
                    }
                else:
                    # Circuit breaker period expired, check if we can reactivate
                    if self.rolling_drawdown_pct <= (self.circuit_breaker_state.trigger_drawdown_pct - self.circuit_breaker_state.recovery_threshold_pct):
                        self._deactivate_circuit_breaker("Automatic recovery: drawdown improved sufficiently")
                    else:
                        # Extend pause if drawdown hasn't improved enough
                        self.circuit_breaker_state.pause_until = current_time + timedelta(hours=self.risk_limits.circuit_breaker_pause_hours)
                        self.logger.warning(f"Circuit breaker extended: current drawdown {self.rolling_drawdown_pct:.2f}% still too high")
                        
                        return {
                            'allowed': False,
                            'reason': f"Circuit breaker extended: drawdown {self.rolling_drawdown_pct:.2f}% not sufficiently improved",
                            'type': RiskViolationType.DRAWDOWN_CIRCUIT_BREAKER.value
                        }
            
            # Check if circuit breaker should be activated
            if self.rolling_drawdown_pct >= self.risk_limits.circuit_breaker_drawdown_pct:
                self._activate_circuit_breaker(f"Rolling {self.risk_limits.circuit_breaker_lookback_hours}h drawdown {self.rolling_drawdown_pct:.2f}% exceeds {self.risk_limits.circuit_breaker_drawdown_pct}% threshold")
                
                return {
                    'allowed': False,
                    'reason': f"Circuit breaker activated: {self.rolling_drawdown_pct:.2f}% drawdown exceeds {self.risk_limits.circuit_breaker_drawdown_pct}% threshold",
                    'type': RiskViolationType.DRAWDOWN_CIRCUIT_BREAKER.value,
                    'drawdown_pct': self.rolling_drawdown_pct,
                    'threshold': self.risk_limits.circuit_breaker_drawdown_pct
                }
            
            return {
                'allowed': True,
                'reason': f"Circuit breaker check passed: {self.rolling_drawdown_pct:.2f}% < {self.risk_limits.circuit_breaker_drawdown_pct}%",
                'drawdown_pct': self.rolling_drawdown_pct
            }
            
        except Exception as e:
            self.logger.error(f"Error checking circuit breaker: {e}")
            return {
                'allowed': False,
                'reason': f"Circuit breaker check error: {str(e)}",
                'type': RiskViolationType.DRAWDOWN_CIRCUIT_BREAKER.value
            }

    def _check_total_open_risk(self, candidate_order: Dict[str, Any], account_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check maximum total open risk across all positions"""
        try:
            # Calculate risk for the candidate order
            size = candidate_order.get('size', 0.0)
            entry_price = candidate_order.get('entry_price', 0.0)
            stop_loss_pct = candidate_order.get('stop_loss_pct', 2.0)  # Default 2% if not specified
            
            account_equity = account_state.get('equity', account_state.get('total_balance', 10000.0))
            
            if account_equity <= 0 or size <= 0:
                return {
                    'allowed': False,
                    'reason': "Invalid account equity or position size for risk calculation",
                    'type': RiskViolationType.TOTAL_OPEN_RISK_LIMIT.value
                }
            
            # Calculate candidate order risk
            candidate_risk_pct = (stop_loss_pct / 100) * (size * entry_price / account_equity) * 100
            
            # Total risk would be current open risk + candidate risk
            projected_total_risk = self.total_open_risk + candidate_risk_pct
            
            if projected_total_risk <= self.risk_limits.max_total_open_risk_pct:
                return {
                    'allowed': True,
                    'reason': f"Total risk check passed: {projected_total_risk:.2f}% <= {self.risk_limits.max_total_open_risk_pct}%",
                    'current_risk_pct': self.total_open_risk,
                    'candidate_risk_pct': candidate_risk_pct,
                    'projected_risk_pct': projected_total_risk
                }
            
            # Check if we can scale down to fit within limits
            available_risk = self.risk_limits.max_total_open_risk_pct - self.total_open_risk
            if available_risk > 0:
                max_scaling_factor = available_risk / candidate_risk_pct
                max_scaling_factor = min(max_scaling_factor, 1.0)  # Never scale up
                
                if max_scaling_factor >= 0.1:  # Allow scaling if at least 10% of original size
                    return {
                        'allowed': False,
                        'can_scale': True,
                        'max_scaling_factor': max_scaling_factor,
                        'reason': f"Total risk would exceed limit: {projected_total_risk:.2f}% > {self.risk_limits.max_total_open_risk_pct}%. Scaling available.",
                        'type': RiskViolationType.TOTAL_OPEN_RISK_LIMIT.value,
                        'available_risk_pct': available_risk
                    }
            
            return {
                'allowed': False,
                'can_scale': False,
                'reason': f"Total risk limit exceeded: {projected_total_risk:.2f}% > {self.risk_limits.max_total_open_risk_pct}%. Current: {self.total_open_risk:.2f}%",
                'type': RiskViolationType.TOTAL_OPEN_RISK_LIMIT.value,
                'current_risk_pct': self.total_open_risk,
                'projected_risk_pct': projected_total_risk,
                'limit': self.risk_limits.max_total_open_risk_pct
            }
            
        except Exception as e:
            self.logger.error(f"Error checking total open risk: {e}")
            return {
                'allowed': False,
                'reason': f"Total risk check error: {str(e)}",
                'type': RiskViolationType.TOTAL_OPEN_RISK_LIMIT.value
            }

    def _check_asset_exposure_limits(self, candidate_order: Dict[str, Any], account_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check per-asset exposure limits"""
        try:
            symbol = candidate_order.get('symbol', '')
            size = candidate_order.get('size', 0.0)
            entry_price = candidate_order.get('entry_price', 0.0)
            
            account_equity = account_state.get('equity', account_state.get('total_balance', 10000.0))
            
            if not symbol or size <= 0 or account_equity <= 0:
                return {
                    'allowed': False,
                    'reason': "Invalid symbol, size, or account equity for asset exposure check",
                    'type': RiskViolationType.ASSET_EXPOSURE_LIMIT.value
                }
            
            # Get current asset exposure
            current_exposure = self.asset_exposures.get(symbol, AssetExposure(symbol=symbol))
            
            # Calculate candidate order exposure
            candidate_exposure = size * entry_price
            projected_exposure = current_exposure.total_exposure + candidate_exposure
            projected_exposure_pct = (projected_exposure / account_equity) * 100
            
            if projected_exposure_pct <= self.risk_limits.max_asset_exposure_pct:
                return {
                    'allowed': True,
                    'reason': f"Asset exposure check passed for {symbol}: {projected_exposure_pct:.2f}% <= {self.risk_limits.max_asset_exposure_pct}%",
                    'current_exposure_pct': current_exposure.exposure_pct,
                    'projected_exposure_pct': projected_exposure_pct
                }
            
            # Check if we can scale down to fit within limits
            available_exposure_pct = self.risk_limits.max_asset_exposure_pct - current_exposure.exposure_pct
            if available_exposure_pct > 0:
                candidate_exposure_pct = (candidate_exposure / account_equity) * 100
                max_scaling_factor = available_exposure_pct / candidate_exposure_pct
                max_scaling_factor = min(max_scaling_factor, 1.0)
                
                if max_scaling_factor >= 0.1:  # Allow scaling if at least 10% of original size
                    return {
                        'allowed': False,
                        'can_scale': True,
                        'max_scaling_factor': max_scaling_factor,
                        'reason': f"Asset exposure would exceed limit for {symbol}: {projected_exposure_pct:.2f}% > {self.risk_limits.max_asset_exposure_pct}%. Scaling available.",
                        'type': RiskViolationType.ASSET_EXPOSURE_LIMIT.value,
                        'available_exposure_pct': available_exposure_pct
                    }
            
            return {
                'allowed': False,
                'can_scale': False,
                'reason': f"Asset exposure limit exceeded for {symbol}: {projected_exposure_pct:.2f}% > {self.risk_limits.max_asset_exposure_pct}%",
                'type': RiskViolationType.ASSET_EXPOSURE_LIMIT.value,
                'current_exposure_pct': current_exposure.exposure_pct,
                'projected_exposure_pct': projected_exposure_pct,
                'limit': self.risk_limits.max_asset_exposure_pct
            }
            
        except Exception as e:
            self.logger.error(f"Error checking asset exposure limits: {e}")
            return {
                'allowed': False,
                'reason': f"Asset exposure check error: {str(e)}",
                'type': RiskViolationType.ASSET_EXPOSURE_LIMIT.value
            }

    def _check_directional_exposure_limits(self, candidate_order: Dict[str, Any], account_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check directional (long/short) exposure limits"""
        try:
            side = candidate_order.get('side', '').lower()
            size = candidate_order.get('size', 0.0)
            entry_price = candidate_order.get('entry_price', 0.0)
            
            account_equity = account_state.get('equity', account_state.get('total_balance', 10000.0))
            
            if side not in ['long', 'short'] or size <= 0 or account_equity <= 0:
                return {
                    'allowed': False,
                    'reason': "Invalid side, size, or account equity for directional exposure check",
                    'type': RiskViolationType.DIRECTIONAL_EXPOSURE_LIMIT.value
                }
            
            # Calculate current directional exposures
            total_long_exposure = sum(exp.long_exposure for exp in self.asset_exposures.values())
            total_short_exposure = sum(exp.short_exposure for exp in self.asset_exposures.values())
            
            candidate_exposure = size * entry_price
            
            if side == 'long':
                projected_long_exposure = total_long_exposure + candidate_exposure
                projected_long_pct = (projected_long_exposure / account_equity) * 100
                limit = self.risk_limits.max_long_exposure_pct
                
                if projected_long_pct <= limit:
                    return {
                        'allowed': True,
                        'reason': f"Long exposure check passed: {projected_long_pct:.2f}% <= {limit}%",
                        'current_long_exposure_pct': (total_long_exposure / account_equity) * 100,
                        'projected_long_exposure_pct': projected_long_pct
                    }
                
                # Check scaling
                current_long_pct = (total_long_exposure / account_equity) * 100
                available_long_pct = limit - current_long_pct
                if available_long_pct > 0:
                    candidate_exposure_pct = (candidate_exposure / account_equity) * 100
                    max_scaling_factor = available_long_pct / candidate_exposure_pct
                    max_scaling_factor = min(max_scaling_factor, 1.0)
                    
                    if max_scaling_factor >= 0.1:
                        return {
                            'allowed': False,
                            'can_scale': True,
                            'max_scaling_factor': max_scaling_factor,
                            'reason': f"Long exposure would exceed limit: {projected_long_pct:.2f}% > {limit}%. Scaling available.",
                            'type': RiskViolationType.DIRECTIONAL_EXPOSURE_LIMIT.value
                        }
                
                return {
                    'allowed': False,
                    'can_scale': False,
                    'reason': f"Long exposure limit exceeded: {projected_long_pct:.2f}% > {limit}%",
                    'type': RiskViolationType.DIRECTIONAL_EXPOSURE_LIMIT.value,
                    'projected_exposure_pct': projected_long_pct,
                    'limit': limit
                }
                
            else:  # short
                projected_short_exposure = total_short_exposure + candidate_exposure
                projected_short_pct = (projected_short_exposure / account_equity) * 100
                limit = self.risk_limits.max_short_exposure_pct
                
                if projected_short_pct <= limit:
                    return {
                        'allowed': True,
                        'reason': f"Short exposure check passed: {projected_short_pct:.2f}% <= {limit}%",
                        'current_short_exposure_pct': (total_short_exposure / account_equity) * 100,
                        'projected_short_exposure_pct': projected_short_pct
                    }
                
                # Check scaling
                current_short_pct = (total_short_exposure / account_equity) * 100
                available_short_pct = limit - current_short_pct
                if available_short_pct > 0:
                    candidate_exposure_pct = (candidate_exposure / account_equity) * 100
                    max_scaling_factor = available_short_pct / candidate_exposure_pct
                    max_scaling_factor = min(max_scaling_factor, 1.0)
                    
                    if max_scaling_factor >= 0.1:
                        return {
                            'allowed': False,
                            'can_scale': True,
                            'max_scaling_factor': max_scaling_factor,
                            'reason': f"Short exposure would exceed limit: {projected_short_pct:.2f}% > {limit}%. Scaling available.",
                            'type': RiskViolationType.DIRECTIONAL_EXPOSURE_LIMIT.value
                        }
                
                return {
                    'allowed': False,
                    'can_scale': False,
                    'reason': f"Short exposure limit exceeded: {projected_short_pct:.2f}% > {limit}%",
                    'type': RiskViolationType.DIRECTIONAL_EXPOSURE_LIMIT.value,
                    'projected_exposure_pct': projected_short_pct,
                    'limit': limit
                }
                
        except Exception as e:
            self.logger.error(f"Error checking directional exposure limits: {e}")
            return {
                'allowed': False,
                'reason': f"Directional exposure check error: {str(e)}",
                'type': RiskViolationType.DIRECTIONAL_EXPOSURE_LIMIT.value
            }

    def _check_correlation_limits(self, candidate_order: Dict[str, Any], open_positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check correlation-aware position sizing"""
        try:
            candidate_symbol = candidate_order.get('symbol', '')
            
            if not candidate_symbol or not open_positions:
                return {
                    'scaling_required': False,
                    'recommended_scaling_factor': 1.0,
                    'reason': "No correlation scaling needed - no open positions or invalid symbol",
                    'max_correlation': 0.0
                }
            
            # Get symbols from open positions
            open_symbols = [pos.get('symbol', '') for pos in open_positions if pos.get('symbol', '')]
            open_symbols = [s for s in open_symbols if s]  # Remove empty strings
            
            if not open_symbols:
                return {
                    'scaling_required': False,
                    'recommended_scaling_factor': 1.0,
                    'reason': "No valid open positions for correlation analysis",
                    'max_correlation': 0.0
                }
            
            # Calculate correlations (simplified - in production would use price data)
            # For now, use symbol similarity heuristics
            max_correlation = 0.0
            correlated_symbols = []
            
            for open_symbol in open_symbols:
                correlation = self._estimate_symbol_correlation(candidate_symbol, open_symbol)
                if correlation > self.risk_limits.correlation_scaling_threshold:
                    max_correlation = max(max_correlation, correlation)
                    correlated_symbols.append(open_symbol)
            
            if max_correlation <= self.risk_limits.correlation_scaling_threshold:
                return {
                    'scaling_required': False,
                    'recommended_scaling_factor': 1.0,
                    'reason': f"No high correlation found. Max correlation: {max_correlation:.2f}",
                    'max_correlation': max_correlation
                }
            
            # Apply correlation-based scaling
            # Higher correlation = more scaling down
            scaling_factor = 1.0 - ((max_correlation - self.risk_limits.correlation_scaling_threshold) * self.risk_limits.correlation_scaling_factor)
            scaling_factor = max(scaling_factor, 0.1)  # Never scale below 10%
            
            return {
                'scaling_required': True,
                'recommended_scaling_factor': scaling_factor,
                'reason': f"High correlation detected with {correlated_symbols}. Max correlation: {max_correlation:.2f} > {self.risk_limits.correlation_scaling_threshold}",
                'type': RiskViolationType.CORRELATION_SCALING.value,
                'max_correlation': max_correlation,
                'correlated_symbols': correlated_symbols,
                'threshold': self.risk_limits.correlation_scaling_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error checking correlation limits: {e}")
            return {
                'scaling_required': False,
                'recommended_scaling_factor': 0.8,  # Conservative fallback
                'reason': f"Correlation check error: {str(e)}. Applying conservative scaling.",
                'max_correlation': 0.0
            }

    def _estimate_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Estimate correlation between two symbols using heuristics.
        In production, this should use actual price correlation calculation.
        """
        if symbol1 == symbol2:
            return 1.0
        
        # Simple heuristics based on symbol similarity
        # In production, this would use historical price data
        
        # Same base currency (e.g., BTCUSDT vs BTCPERP)
        base1 = symbol1.replace('USDT', '').replace('PERP', '').replace('USD', '')
        base2 = symbol2.replace('USDT', '').replace('PERP', '').replace('USD', '')
        
        if base1 == base2:
            return 0.95  # Very high correlation for same asset
        
        # Major crypto pairs tend to be correlated
        major_cryptos = ['BTC', 'ETH', 'BNB', 'ADA', 'DOT', 'SOL']
        if base1 in major_cryptos and base2 in major_cryptos:
            return 0.75  # High correlation among major cryptos
        
        # Meme coins tend to be correlated 
        meme_coins = ['DOGE', 'SHIB', 'PEPE', 'FLOKI', 'BONK']
        if base1 in meme_coins and base2 in meme_coins:
            return 0.8  # High correlation among meme coins
        
        # Layer 1 tokens correlation
        layer1_tokens = ['ETH', 'BNB', 'ADA', 'DOT', 'SOL', 'AVAX', 'MATIC']
        if base1 in layer1_tokens and base2 in layer1_tokens:
            return 0.7  # Moderate-high correlation
        
        # Default low correlation
        return 0.3

    def _update_drawdown_tracking(self, account_state: Dict[str, Any]) -> None:
        """Update drawdown tracking for circuit breaker"""
        try:
            current_time = datetime.now(timezone.utc)
            current_equity = account_state.get('equity', account_state.get('total_balance', 10000.0))
            
            # Add current equity to history
            self.drawdown_history.append({
                'timestamp': current_time,
                'equity': current_equity
            })
            
            # Remove old data beyond lookback period
            cutoff_time = current_time - timedelta(hours=self.risk_limits.circuit_breaker_lookback_hours)
            self.drawdown_history = [
                entry for entry in self.drawdown_history 
                if entry['timestamp'] > cutoff_time
            ]
            
            # Calculate rolling maximum drawdown
            if len(self.drawdown_history) < 2:
                self.rolling_drawdown_pct = 0.0
                return
            
            # Find peak equity in the lookback period
            peak_equity = max(entry['equity'] for entry in self.drawdown_history)
            
            # Calculate current drawdown from peak
            if peak_equity > 0:
                self.rolling_drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100
            else:
                self.rolling_drawdown_pct = 0.0
                
        except Exception as e:
            self.logger.error(f"Error updating drawdown tracking: {e}")
            self.rolling_drawdown_pct = 0.0

    def _activate_circuit_breaker(self, reason: str) -> None:
        """Activate the circuit breaker"""
        current_time = datetime.now(timezone.utc)
        
        self.circuit_breaker_state.is_active = True
        self.circuit_breaker_state.activated_at = current_time
        self.circuit_breaker_state.reason = reason
        self.circuit_breaker_state.pause_until = current_time + timedelta(hours=self.risk_limits.circuit_breaker_pause_hours)
        self.circuit_breaker_state.trigger_drawdown_pct = self.rolling_drawdown_pct
        
        self.trading_state = TradingState.CIRCUIT_BREAKER
        
        self.logger.critical(f"CIRCUIT BREAKER ACTIVATED: {reason}. Trading paused until {self.circuit_breaker_state.pause_until}")

    def _deactivate_circuit_breaker(self, reason: str) -> None:
        """Deactivate the circuit breaker"""
        self.circuit_breaker_state.is_active = False
        self.circuit_breaker_state.activated_at = None
        self.circuit_breaker_state.pause_until = None
        self.circuit_breaker_state.reason = ""
        
        # Return to active trading if no other issues
        if self.trading_state == TradingState.CIRCUIT_BREAKER and not self.emergency_stop_active and not self.daily_loss_limit_hit:
            self.trading_state = TradingState.ACTIVE
        
        self.logger.info(f"Circuit breaker deactivated: {reason}")

    def set_read_only_mode(self, read_only: bool) -> None:
        """Set read-only mode for non-blocking operations"""
        self.read_only_mode = read_only
        if read_only:
            self.logger.info("Risk manager set to read-only mode")
        else:
            self.logger.info("Risk manager returned to normal mode")

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status for monitoring"""
        try:
            return {
                'trading_state': self.trading_state.value,
                'emergency_stop_active': self.emergency_stop_active,
                'daily_loss_limit_hit': self.daily_loss_limit_hit,
                'circuit_breaker': {
                    'is_active': self.circuit_breaker_state.is_active,
                    'activated_at': self.circuit_breaker_state.activated_at.isoformat() if self.circuit_breaker_state.activated_at else None,
                    'pause_until': self.circuit_breaker_state.pause_until.isoformat() if self.circuit_breaker_state.pause_until else None,
                    'reason': self.circuit_breaker_state.reason,
                    'trigger_drawdown_pct': self.circuit_breaker_state.trigger_drawdown_pct
                },
                'total_open_risk_pct': self.total_open_risk,
                'rolling_drawdown_pct': self.rolling_drawdown_pct,
                'asset_exposures': {
                    symbol: {
                        'long_exposure': exp.long_exposure,
                        'short_exposure': exp.short_exposure,
                        'net_exposure': exp.net_exposure,
                        'total_exposure': exp.total_exposure,
                        'exposure_pct': exp.exposure_pct,
                        'position_count': exp.position_count
                    }
                    for symbol, exp in self.asset_exposures.items()
                },
                'risk_limits': {
                    'max_daily_loss_pct': self.risk_limits.max_daily_loss_pct,
                    'max_total_open_risk_pct': self.risk_limits.max_total_open_risk_pct,
                    'max_asset_exposure_pct': self.risk_limits.max_asset_exposure_pct,
                    'circuit_breaker_drawdown_pct': self.risk_limits.circuit_breaker_drawdown_pct,
                    'correlation_scaling_threshold': self.risk_limits.correlation_scaling_threshold
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting portfolio status: {e}")
            return {
                'trading_state': 'error',
                'error': str(e)
            } 