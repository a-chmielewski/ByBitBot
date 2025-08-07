"""
Configuration Loader Module

This module provides validation and typed access to strategy configurations,
integrating with the enhanced Strategy Matrix for comprehensive risk management.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import asdict
import json
from pathlib import Path

from .strategy_matrix import (
    StrategyMatrix, StrategyRiskProfile, StopLossConfig, TakeProfitConfig,
    TrailingStopConfig, PositionSizingConfig, LeverageByRegimeConfig,
    PortfolioTagsConfig, TradingLimitsConfig
)

class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass

class StrategyConfigLoader:
    """
    Strategy Configuration Loader with validation and typed access.
    
    Provides centralized access to strategy risk parameters from the Strategy Matrix,
    with validation, conversion to bot-compatible formats, and override capabilities.
    """
    
    def __init__(self, strategy_matrix: Optional[StrategyMatrix] = None, 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the configuration loader.
        
        Args:
            strategy_matrix: StrategyMatrix instance (creates new if None)
            logger: Logger instance
        """
        self.strategy_matrix = strategy_matrix or StrategyMatrix()
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        
        # Cache for processed configurations
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        
    def get_strategy_config(self, strategy_name: str, 
                           override_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get comprehensive configuration for a strategy with optional overrides.
        
        Args:
            strategy_name: Name of the strategy class
            override_config: Optional configuration overrides
            
        Returns:
            Complete strategy configuration dictionary
            
        Raises:
            ConfigValidationError: If strategy not found or configuration invalid
        """
        try:
            # Check cache first
            cache_key = f"{strategy_name}_{hash(str(override_config))}"
            if cache_key in self._config_cache:
                return self._config_cache[cache_key].copy()
            
            # Get base risk profile from strategy matrix
            risk_profile = self.strategy_matrix.get_strategy_risk_profile(strategy_name)
            if not risk_profile:
                raise ConfigValidationError(f"Strategy '{strategy_name}' not found in Strategy Matrix")
            
            # Convert risk profile to bot-compatible configuration
            config = self._convert_risk_profile_to_config(risk_profile)
            
            # Apply overrides if provided
            if override_config:
                config = self._apply_configuration_overrides(config, override_config)
                self._validate_configuration(config, strategy_name)
            
            # Cache the result
            self._config_cache[cache_key] = config.copy()
            
            self.logger.debug(f"Loaded configuration for {strategy_name}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration for {strategy_name}: {e}")
            raise ConfigValidationError(f"Failed to load configuration for {strategy_name}: {str(e)}")
    
    def _convert_risk_profile_to_config(self, risk_profile: StrategyRiskProfile) -> Dict[str, Any]:
        """
        Convert StrategyRiskProfile to bot-compatible configuration format.
        
        Args:
            risk_profile: Strategy risk profile from matrix
            
        Returns:
            Bot-compatible configuration dictionary
        """
        config = {
            'strategy_configs': {
                risk_profile.strategy_name: {
                    'description': risk_profile.description,
                    'market_type_tags': risk_profile.market_type_tags,
                    'execution_timeframe': risk_profile.execution_timeframe,
                    
                    # Risk management configuration
                    'risk_management': {
                        # Stop loss configuration
                        'stop_loss_mode': risk_profile.stop_loss.mode,
                        'stop_loss_fixed_pct': risk_profile.stop_loss.fixed_pct,
                        'stop_loss_atr_multiplier': risk_profile.stop_loss.atr_multiplier,
                        'max_loss_pct': risk_profile.stop_loss.max_loss_pct,
                        
                        # Take profit configuration
                        'take_profit_mode': risk_profile.take_profit.mode,
                        'take_profit_fixed_pct': risk_profile.take_profit.fixed_pct,
                        'take_profit_progressive_levels': risk_profile.take_profit.progressive_levels,
                        'partial_exit_sizes': risk_profile.take_profit.partial_exit_sizes,
                        
                        # Trailing stop configuration
                        'trailing_stop_enabled': risk_profile.trailing_stop.enabled,
                        'trailing_stop_mode': risk_profile.trailing_stop.mode,
                        'trailing_stop_offset_pct': risk_profile.trailing_stop.offset_pct,
                        'trailing_stop_atr_multiplier': risk_profile.trailing_stop.atr_multiplier,
                        'trailing_stop_activation_pct': risk_profile.trailing_stop.activation_pct,
                        
                        # Position sizing configuration
                        'position_sizing_mode': risk_profile.position_sizing.mode,
                        'position_fixed_notional': risk_profile.position_sizing.fixed_notional,
                        'position_risk_per_trade': risk_profile.position_sizing.risk_per_trade,
                        'position_kelly_cap': risk_profile.position_sizing.kelly_cap,
                        'tick_value': risk_profile.position_sizing.tick_value,
                        'max_position_pct': risk_profile.position_sizing.max_position_pct,
                        
                        # Leverage by regime
                        'leverage_by_regime': {
                            'low': risk_profile.leverage_by_regime.low,
                            'normal': risk_profile.leverage_by_regime.normal,
                            'high': risk_profile.leverage_by_regime.high
                        },
                        
                        # Liquidity filters (using sensible defaults)
                        'min_liquidity_filter': {
                            'enabled': True,
                            'min_bid_ask_volume': 10000,
                            'max_spread_bps': 10
                        },
                        
                        'spread_slippage_guard': {
                            'enabled': True,
                            'max_spread_pct': 0.001,
                            'max_slippage_pct': 0.002
                        },
                        
                        # ATR period for calculations
                        'atr_period': 14
                    },
                    
                    # Portfolio tags and limits
                    'portfolio': {
                        'sector': risk_profile.portfolio_tags.sector,
                        'factor': risk_profile.portfolio_tags.factor,
                        'correlation_group': risk_profile.portfolio_tags.correlation_group,
                        'market_cap_tier': risk_profile.portfolio_tags.market_cap_tier
                    },
                    
                    # Trading limits
                    'trading_limits': {
                        'max_concurrent_trades': risk_profile.trading_limits.max_concurrent_trades,
                        'max_per_symbol': risk_profile.trading_limits.max_per_symbol,
                        'min_time_between_trades': risk_profile.trading_limits.min_time_between_trades,
                        'daily_trade_limit': risk_profile.trading_limits.daily_trade_limit,
                        'max_daily_drawdown_pct': risk_profile.trading_limits.max_daily_drawdown_pct
                    },
                    
                    # Volatility constraints
                    'volatility_constraints': {
                        'min_volatility_pct': risk_profile.min_volatility_pct,
                        'max_volatility_pct': risk_profile.max_volatility_pct,
                        'preferred_sessions': risk_profile.preferred_sessions
                    }
                }
            }
        }
        
        return config
    
    def _apply_configuration_overrides(self, base_config: Dict[str, Any], 
                                     overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply configuration overrides to base configuration.
        
        Args:
            base_config: Base configuration from risk profile
            overrides: Override values
            
        Returns:
            Updated configuration with overrides applied
        """
        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge dictionaries"""
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(base_config, overrides)
    
    def _validate_configuration(self, config: Dict[str, Any], strategy_name: str) -> None:
        """
        Validate strategy configuration for consistency and safety.
        
        Args:
            config: Configuration to validate
            strategy_name: Name of the strategy
            
        Raises:
            ConfigValidationError: If configuration is invalid
        """
        try:
            strategy_config = config['strategy_configs'][strategy_name]
            risk_config = strategy_config['risk_management']
            
            # Validate stop loss configuration
            if risk_config['stop_loss_mode'] not in ['fixed_pct', 'atr_mult']:
                raise ConfigValidationError(f"Invalid stop_loss_mode: {risk_config['stop_loss_mode']}")
            
            if risk_config['stop_loss_fixed_pct'] <= 0 or risk_config['stop_loss_fixed_pct'] > 0.1:
                raise ConfigValidationError(f"stop_loss_fixed_pct must be between 0 and 0.1 (10%)")
            
            if risk_config['stop_loss_atr_multiplier'] <= 0 or risk_config['stop_loss_atr_multiplier'] > 5.0:
                raise ConfigValidationError(f"stop_loss_atr_multiplier must be between 0 and 5.0")
            
            # Validate take profit configuration
            if risk_config['take_profit_mode'] not in ['fixed_pct', 'progressive_levels']:
                raise ConfigValidationError(f"Invalid take_profit_mode: {risk_config['take_profit_mode']}")
            
            if risk_config['take_profit_mode'] == 'progressive_levels':
                levels = risk_config['take_profit_progressive_levels']
                if not isinstance(levels, list) or len(levels) == 0:
                    raise ConfigValidationError("progressive_levels must be a non-empty list")
                
                if not all(isinstance(level, (int, float)) and level > 0 for level in levels):
                    raise ConfigValidationError("All progressive levels must be positive numbers")
            
            # Validate position sizing
            if risk_config['position_sizing_mode'] not in ['fixed_notional', 'vol_normalized', 'kelly_capped']:
                raise ConfigValidationError(f"Invalid position_sizing_mode: {risk_config['position_sizing_mode']}")
            
            if risk_config['position_risk_per_trade'] <= 0 or risk_config['position_risk_per_trade'] > 0.1:
                raise ConfigValidationError("position_risk_per_trade must be between 0 and 0.1 (10%)")
            
            # Validate leverage multipliers
            leverage_config = risk_config['leverage_by_regime']
            for regime, multiplier in leverage_config.items():
                if not isinstance(multiplier, (int, float)) or multiplier <= 0 or multiplier > 3.0:
                    raise ConfigValidationError(f"leverage multiplier for {regime} must be between 0 and 3.0")
            
            # Validate trading limits
            limits_config = strategy_config['trading_limits']
            if limits_config['max_concurrent_trades'] <= 0 or limits_config['max_concurrent_trades'] > 20:
                raise ConfigValidationError("max_concurrent_trades must be between 1 and 20")
            
            self.logger.debug(f"Configuration validation passed for {strategy_name}")
            
        except KeyError as e:
            raise ConfigValidationError(f"Missing required configuration key: {e}")
    
    def get_all_strategy_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get configurations for all strategies in the matrix.
        
        Returns:
            Dictionary mapping strategy names to their configurations
        """
        all_configs = {}
        
        for strategy_name in self.strategy_matrix.get_all_strategy_profiles().keys():
            try:
                config = self.get_strategy_config(strategy_name)
                all_configs[strategy_name] = config['strategy_configs'][strategy_name]
            except ConfigValidationError as e:
                self.logger.error(f"Failed to load config for {strategy_name}: {e}")
                continue
        
        return all_configs
    
    def export_strategy_config_to_json(self, strategy_name: str, 
                                      output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Export strategy configuration to JSON format.
        
        Args:
            strategy_name: Name of the strategy
            output_path: Optional output file path
            
        Returns:
            JSON string of the configuration
        """
        config = self.get_strategy_config(strategy_name)
        json_str = json.dumps(config, indent=2, default=str)
        
        if output_path:
            path = Path(output_path)
            path.write_text(json_str)
            self.logger.info(f"Exported {strategy_name} configuration to {path}")
        
        return json_str
    
    def validate_strategy_for_conditions(self, strategy_name: str, 
                                       market_conditions: Dict[str, Any]) -> bool:
        """
        Validate if a strategy is suitable for current market conditions.
        
        Args:
            strategy_name: Name of the strategy
            market_conditions: Current market conditions dictionary
            
        Returns:
            True if strategy is suitable for current conditions
        """
        try:
            risk_profile = self.strategy_matrix.get_strategy_risk_profile(strategy_name)
            if not risk_profile:
                return False
            
            # Check volatility constraints
            current_volatility = market_conditions.get('volatility_pct', 1.0)
            if not (risk_profile.min_volatility_pct <= current_volatility <= risk_profile.max_volatility_pct):
                self.logger.debug(f"{strategy_name} volatility check failed: {current_volatility}% not in [{risk_profile.min_volatility_pct}%, {risk_profile.max_volatility_pct}%]")
                return False
            
            # Check market type alignment
            current_market_type = market_conditions.get('market_type', 'UNKNOWN')
            if current_market_type not in risk_profile.market_type_tags:
                self.logger.debug(f"{strategy_name} market type check failed: {current_market_type} not in {risk_profile.market_type_tags}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating {strategy_name} for conditions: {e}")
            return False
    
    def get_correlation_aware_strategies(self, active_strategies: List[str], 
                                       max_correlation_exposure: float = 0.6) -> List[str]:
        """
        Get strategies that won't exceed correlation exposure limits.
        
        Args:
            active_strategies: Currently active strategy names
            max_correlation_exposure: Maximum exposure per correlation group (60%)
            
        Returns:
            List of strategies that can be added without exceeding correlation limits
        """
        # Get correlation groups for active strategies
        active_groups = {}
        for strategy in active_strategies:
            profile = self.strategy_matrix.get_strategy_risk_profile(strategy)
            if profile:
                group = profile.portfolio_tags.correlation_group
                active_groups[group] = active_groups.get(group, 0) + 1
        
        # Find strategies that won't exceed correlation limits
        available_strategies = []
        correlation_matrix = self.strategy_matrix.get_portfolio_correlation_matrix()
        
        for group, strategies in correlation_matrix.items():
            current_exposure = active_groups.get(group, 0)
            # For single-strategy groups, allow up to 1 strategy (100% of group)
            # For multi-strategy groups, use the exposure percentage
            max_allowed = max(1, int(len(strategies) * max_correlation_exposure))
            
            if current_exposure < max_allowed:
                for strategy in strategies:
                    if strategy not in active_strategies:
                        available_strategies.append(strategy)
        
        return available_strategies
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._config_cache.clear()
        self.logger.debug("Configuration cache cleared")