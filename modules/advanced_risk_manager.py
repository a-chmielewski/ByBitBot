import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

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
    
    def __init__(self, exchange, performance_tracker, logger: Optional[logging.Logger] = None,
                 risk_limits: Optional[RiskLimits] = None):
        self.exchange = exchange
        self.performance_tracker = performance_tracker
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.risk_limits = risk_limits or RiskLimits()
        
        # Risk state tracking
        self.positions: Dict[str, PositionRisk] = {}
        self.risk_violations: List[Dict[str, Any]] = []
        self.emergency_stop_active = False
        self.last_trade_time = None
        self.daily_start_balance = None
        self.session_start_time = datetime.now(timezone.utc)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Risk calculation cache
        self._portfolio_risk_cache = None
        self._cache_timestamp = None
        self._cache_validity_seconds = 10  # Cache valid for 10 seconds
        
        self.logger.info("AdvancedRiskManager initialized with comprehensive risk controls")

    def calculate_position_size(self, symbol: str, strategy_name: str, market_context: Dict[str, Any],
                              risk_pct: float = 2.0, volatility_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate optimal position size using advanced risk management
        
        Args:
            symbol: Trading symbol
            strategy_name: Name of the strategy
            market_context: Market condition context
            risk_pct: Risk percentage of account
            volatility_data: Volatility metrics (ATR, volatility regime, etc.)
            
        Returns:
            Dict with position sizing recommendations
        """
        try:
            with self._lock:
                # Get current account information
                account_info = self._get_account_info()
                if not account_info:
                    return self._get_fallback_position_size(risk_pct)
                
                account_balance = account_info.get('total_balance', 0)
                if account_balance <= 0:
                    self.logger.error("Invalid account balance for position sizing")
                    return self._get_fallback_position_size(risk_pct)
                
                # Base risk amount
                base_risk_amount = account_balance * (risk_pct / 100)
                
                # Adjust for market conditions
                market_adjustment = self._get_market_condition_adjustment(market_context)
                
                # Adjust for volatility
                volatility_adjustment = self._get_volatility_adjustment(volatility_data)
                
                # Adjust for strategy-specific factors
                strategy_adjustment = self._get_strategy_risk_adjustment(strategy_name)
                
                # Calculate final risk amount (removed portfolio_adjustment to prevent circular dependency)
                adjusted_risk_amount = base_risk_amount * market_adjustment * volatility_adjustment * strategy_adjustment
                
                # Apply risk limits
                max_risk_amount = account_balance * (self.risk_limits.max_position_size_pct / 100)
                final_risk_amount = min(adjusted_risk_amount, max_risk_amount)
                
                # Get current price for position size calculation
                current_price = self._get_current_price(symbol)
                if not current_price:
                    return self._get_fallback_position_size(risk_pct)
                
                # Calculate position size based on risk amount and stop loss
                # This will be refined when we have actual stop loss price
                estimated_position_size = final_risk_amount / current_price
                
                return {
                    'recommended_size': estimated_position_size,
                    'risk_amount': final_risk_amount,
                    'risk_pct': (final_risk_amount / account_balance) * 100,
                    'market_adjustment': market_adjustment,
                    'volatility_adjustment': volatility_adjustment,
                    'strategy_adjustment': strategy_adjustment,
                    'current_price': current_price,
                    'account_balance': account_balance,
                    'max_position_value': max_risk_amount
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}", exc_info=True)
            return self._get_fallback_position_size(risk_pct)

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