import os
import importlib
import json
import logging # Import logging for the helper function
from modules.logger import get_logger, configure_logging_session, close_all_loggers
from modules.exchange import ExchangeConnector
from modules.data_fetcher import LiveDataFetcher
from modules.order_manager import OrderManager, OrderExecutionError
from modules.performance_tracker import PerformanceTracker, TradeRecord, MarketContext, OrderDetails, RiskMetrics, TradeStatus
from modules.market_analyzer import MarketAnalyzer, MarketAnalysisError
from modules.strategy_matrix import StrategyMatrix
from modules.session_manager import SessionManager
from modules.real_time_monitor import RealTimeMonitor
from modules.advanced_risk_manager import AdvancedRiskManager, EnforceResult, EnforceAction
from modules.config_loader import StrategyConfigLoader, ConfigValidationError
from modules.trailing_tp_handler import TrailingTPHandler, PositionState, OrderType
from datetime import datetime, timezone
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
import atexit
from typing import Dict, Any, Optional, List, Tuple

# Import StrategyTemplate for type checking in dynamic_import_strategy
from strategies.strategy_template import StrategyTemplate

# Import risk utilities with fallback
try:
    from modules import risk_utilities
    RISK_UTILITIES_AVAILABLE = True
except ImportError:
    RISK_UTILITIES_AVAILABLE = False

warnings.simplefilter(action='ignore', category=FutureWarning)

CONFIG_PATH = 'config.json'
STRATEGY_DIR = 'strategies'

# Global variables for cleanup
session_manager = None
real_time_monitor = None
trailing_tp_handler = None


def cleanup_on_exit():
    """Cleanup function to ensure proper shutdown of all components"""
    global session_manager, real_time_monitor, trailing_tp_handler
    
    try:
        if trailing_tp_handler:
            trailing_tp_handler.stop_monitoring()
            
        if real_time_monitor:
            real_time_monitor.stop_monitoring()
            
        # Close all logging handlers
        close_all_loggers()
            
        if session_manager:
            # Export session data before ending sessions
            try:
                # Get active sessions for export
                active_sessions = session_manager.get_active_sessions()
                if active_sessions:
                    print("🔄 Exporting session data before shutdown...")
                    
                    # Export in both JSON and CSV formats for maximum data preservation
                    active_session_ids = list(active_sessions.keys())
                    
                    # Export JSON (comprehensive data)
                    json_file = session_manager.export_session_data(active_session_ids, format='json')
                    print(f"✅ Session data exported to: {json_file}")
                    
                    # Export CSV (tabular data for analysis)
                    csv_file = session_manager.export_session_data(active_session_ids, format='csv')
                    print(f"✅ Session data exported to: {csv_file}")
                    
                    # Also export all historical sessions for complete backup
                    all_session_ids = list(session_manager.get_session_history().keys()) + active_session_ids
                    if all_session_ids:
                        json_full = session_manager.export_session_data(all_session_ids, format='json')
                        print(f"📊 Complete session history exported to: {json_full}")
                        
                else:
                    print("ℹ️  No active sessions to export")
                    
            except Exception as export_error:
                print(f"⚠️  Warning: Failed to export session data during shutdown: {export_error}")
            
            # End active sessions after export
            session_manager.end_active_sessions("Bot shutdown")
            
    except Exception as e:
        print(f"Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup_on_exit)


def enhanced_order_validation(
    strategy_name: str,
    symbol: str,
    side: str,
    entry_price: float,
    base_size: float,
    strategy_matrix: StrategyMatrix,
    config_loader: StrategyConfigLoader,
    market_analyzer: MarketAnalyzer,
    risk_manager: AdvancedRiskManager,
    exchange: ExchangeConnector,
    account_equity: float,
    open_positions: List[Dict[str, Any]],
    logger: logging.Logger,
    directional_bias: str = 'NEUTRAL',
    bias_strength: str = 'WEAK'
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Enhanced order validation with comprehensive risk management integration.
    
    This function integrates all risk management components to validate and size orders:
    1. Pull strategy config from strategy_matrix
    2. Query MarketAnalyzer for volatility/market regime  
    3. Compute SL/TP using RiskUtilities based on configured mode
    4. Compute position size via vol_normalized or kelly_capped
    5. Call AdvancedRiskManager.enforce_portfolio_limits to approve/scale/deny
    6. Add trailing stop logic if enabled
    7. Add defensive slippage guard
    
    Args:
        strategy_name: Name of the strategy (e.g., 'StrategyATRMomentumBreakout')
        symbol: Trading symbol (e.g., 'BTCUSDT')
        side: Order side ('buy' or 'sell')
        entry_price: Intended entry price
        base_size: Base position size before risk adjustments
        strategy_matrix: StrategyMatrix instance
        config_loader: StrategyConfigLoader instance
        market_analyzer: MarketAnalyzer instance  
        risk_manager: AdvancedRiskManager instance
        exchange: ExchangeConnector instance
        account_equity: Current account equity
        open_positions: List of currently open positions
        logger: Logger instance
        directional_bias: Higher timeframe directional bias
        bias_strength: Strength of the bias (WEAK/MODERATE/STRONG)
        
    Returns:
        Tuple[bool, Dict[str, Any], str]: (approved, order_details, reason)
        - approved: True if order is approved, False if denied
        - order_details: Complete order details with computed values
        - reason: Detailed reason for approval/denial
    """
    
    logger.info(f"🔍 Enhanced Order Validation Starting for {strategy_name} on {symbol}")
    logger.info(f"   Initial Parameters: side={side}, entry_price={entry_price}, base_size={base_size}")
    
    try:
        # ===== STEP 1: Pull Strategy Configuration =====
        logger.info("📋 Step 1: Loading strategy configuration...")
        
        try:
            strategy_config = config_loader.get_strategy_config(strategy_name)
            risk_config = strategy_config['strategy_configs'][strategy_name]['risk_management']
            portfolio_config = strategy_config['strategy_configs'][strategy_name]['portfolio']
            trading_limits = strategy_config['strategy_configs'][strategy_name]['trading_limits']
            
            logger.info(f"✅ Strategy config loaded:")
            logger.info(f"   Stop Loss Mode: {risk_config['stop_loss_mode']}")
            logger.info(f"   Take Profit Mode: {risk_config['take_profit_mode']}")
            logger.info(f"   Position Sizing Mode: {risk_config['position_sizing_mode']}")
            logger.info(f"   Factor: {portfolio_config['factor']}")
            logger.info(f"   Max Concurrent: {trading_limits['max_concurrent_trades']}")
            
        except ConfigValidationError as e:
            return False, {}, f"Strategy configuration loading failed: {str(e)}"
        
        # ===== STEP 2: Query Market Conditions =====
        logger.info("📊 Step 2: Analyzing market conditions...")
        
        try:
            # Get volatility regime and market conditions
            current_volatility_pct = market_analyzer.get_atr_pct(symbol)
            volatility_regime = market_analyzer.get_vol_regime(symbol)
            market_regime = market_analyzer.get_market_regime(symbol)
            
            logger.info(f"✅ Market conditions analyzed:")
            logger.info(f"   Current Volatility: {current_volatility_pct:.2f}%")
            logger.info(f"   Volatility Regime: {volatility_regime}")
            logger.info(f"   Market Regime: {market_regime}")
            
            # Validate strategy is suitable for current conditions
            market_conditions = {
                'volatility_pct': current_volatility_pct,
                'volatility_regime': volatility_regime,
                'market_regime': market_regime
            }
            
            # Check if strategy is suitable for current volatility
            risk_profile = strategy_matrix.get_strategy_risk_profile(strategy_name)
            if risk_profile:
                min_vol = risk_profile.min_volatility_pct
                max_vol = risk_profile.max_volatility_pct
                if not (min_vol <= current_volatility_pct <= max_vol):
                    return False, {}, f"Current volatility {current_volatility_pct:.2f}% outside strategy range [{min_vol:.1f}%, {max_vol:.1f}%]"
            
        except Exception as e:
            logger.warning(f"⚠️  Market analysis failed, using defaults: {e}")
            current_volatility_pct = 1.0
            volatility_regime = 'normal'
            market_regime = 'unknown'
        
        # ===== DIRECTIONAL BIAS FILTER =====
        logger.info("🧭 Step 2.5: Applying directional bias filter...")
        
        # Block long trades in strong bearish bias
        if side.lower() == 'buy' and directional_bias in ['BEARISH', 'BEARISH_BIASED']:
            if bias_strength == 'STRONG':
                return False, {}, f"BLOCKED: Long trade rejected due to STRONG bearish bias ({directional_bias})"
            elif bias_strength == 'MODERATE' and directional_bias == 'BEARISH':
                return False, {}, f"BLOCKED: Long trade rejected due to MODERATE bearish bias"
        
        # Apply stricter criteria for long trades in any bearish conditions
        if side.lower() == 'buy' and directional_bias in ['BEARISH', 'BEARISH_BIASED', 'NEUTRAL']:
            # Get current market data for ADX check
            try:
                current_data = market_analyzer._get_cached_data(symbol, '1m', min_periods=50)
                if current_data is not None and 'adx' in current_data.columns:
                    current_adx = current_data['adx'].iloc[-1]
                    
                    # Stricter ADX threshold for long entries (30 vs normal 25)
                    min_adx_long = 30 if directional_bias in ['BEARISH', 'BEARISH_BIASED'] else 28
                    
                    if current_adx < min_adx_long:
                        return False, {}, f"BLOCKED: Long trade ADX ({current_adx:.1f}) below strict threshold ({min_adx_long}) in {directional_bias} bias"
                    
                    logger.info(f"   Long trade ADX check passed: {current_adx:.1f} >= {min_adx_long}")
                else:
                    logger.warning("   Cannot apply ADX filter for long trade - no data available")
            except Exception as adx_error:
                logger.warning(f"   ADX filter failed: {adx_error}")
        
        # Log directional bias decision
        if directional_bias != 'NEUTRAL':
            logger.info(f"   Directional bias: {directional_bias} ({bias_strength}) - {'LONG RESTRICTED' if side.lower() == 'buy' else 'SHORT FAVORED'}")
        else:
            logger.info(f"   Directional bias: NEUTRAL - No bias restrictions")
        
        # ===== MINIMUM EXPECTED MOVE CHECK =====
        logger.info("💰 Step 2.6: Checking minimum expected move vs fees...")
        
        try:
            # Get current market data for ATR calculation
            current_data = market_analyzer._get_cached_data(symbol, '1m', min_periods=20)
            if current_data is not None and 'atr_14' in current_data.columns:
                current_atr = current_data['atr_14'].iloc[-1]
                expected_move_pct = (current_atr / entry_price) * 100
                
                # Estimate trading fees (assume 0.1% total roundtrip fees)
                estimated_fees_pct = 0.1
                min_expected_move_pct = estimated_fees_pct * 2.5  # Need 2.5x fees to be profitable
                
                if expected_move_pct < min_expected_move_pct:
                    return False, {}, f"BLOCKED: Expected move ({expected_move_pct:.2f}%) too small vs fees ({min_expected_move_pct:.2f}% required)"
                
                # Additional check: avoid trading in extremely low volatility
                min_volatility_threshold = 0.3  # 0.3% minimum volatility
                if expected_move_pct < min_volatility_threshold:
                    return False, {}, f"BLOCKED: Market too quiet ({expected_move_pct:.2f}% < {min_volatility_threshold}%) - avoid noise trades"
                
                logger.info(f"   Expected move check passed: {expected_move_pct:.2f}% >= {min_expected_move_pct:.2f}%")
            else:
                logger.warning("   Cannot calculate expected move - no ATR data available")
        except Exception as move_error:
            logger.warning(f"   Expected move check failed: {move_error}")
        
        # ===== STEP 3: Compute Stop Loss and Take Profit =====
        logger.info("🛡️  Step 3: Computing stop loss and take profit levels...")
        
        sl_price = None
        tp_prices = []
        
        if RISK_UTILITIES_AVAILABLE:
            try:
                # Get current market data for ATR calculation
                current_data = market_analyzer._get_cached_data(symbol, '1m', min_periods=50)
                
                if current_data is not None and len(current_data) > 20:
                    # Calculate ATR for dynamic stop loss
                    atr_period = risk_config.get('atr_period', 14)
                    atr_series = risk_utilities.compute_atr(current_data, period=atr_period)
                    current_atr = atr_series.iloc[-1] if not atr_series.empty else entry_price * 0.02
                    
                    # Compute stop loss based on mode
                    if risk_config['stop_loss_mode'] == 'atr_mult':
                        atr_mult = risk_config['stop_loss_atr_multiplier']
                        stop_levels = risk_utilities.atr_stop_levels(
                            entry_price=entry_price,
                            side=side,
                            atr=current_atr,
                            atr_mult_sl=atr_mult,
                            atr_mult_tp=risk_config.get('take_profit_atr_multiplier', 2.0)
                        )
                        sl_price = stop_levels['stop_loss']
                        
                        logger.info(f"   ATR-based SL: {sl_price:.6f} (ATR: {current_atr:.6f}, Mult: {atr_mult}x)")
                        
                    else:  # fixed_pct mode
                        sl_pct = risk_config['stop_loss_fixed_pct']
                        if side.lower() == 'buy':
                            sl_price = entry_price * (1 - sl_pct)
                        else:
                            sl_price = entry_price * (1 + sl_pct)
                            
                        logger.info(f"   Fixed % SL: {sl_price:.6f} ({sl_pct:.1%})")
                    
                    # Compute take profit based on mode
                    if risk_config['take_profit_mode'] == 'progressive_levels':
                        tp_levels = risk_config['take_profit_progressive_levels']
                        tp_prices = risk_utilities.progressive_take_profit_levels(
                            entry_price=entry_price,
                            side=side,
                            levels=tp_levels
                        )
                        
                        logger.info(f"   Progressive TP levels: {[f'{p:.6f}' for p in tp_prices]}")
                        
                    else:  # fixed_pct mode
                        tp_pct = risk_config['take_profit_fixed_pct']
                        if side.lower() == 'buy':
                            tp_price = entry_price * (1 + tp_pct)
                        else:
                            tp_price = entry_price * (1 - tp_pct)
                        tp_prices = [tp_price]
                        
                        logger.info(f"   Fixed % TP: {tp_price:.6f} ({tp_pct:.1%})")
                
                else:
                    raise ValueError("Insufficient market data for ATR calculation")
                    
            except Exception as e:
                logger.warning(f"⚠️  RiskUtilities calculation failed, using fallback: {e}")
                # Fallback to simple percentage-based calculation
                sl_pct = risk_config.get('stop_loss_fixed_pct', 0.03)
                tp_pct = risk_config.get('take_profit_fixed_pct', 0.06)
                
                if side.lower() == 'buy':
                    sl_price = entry_price * (1 - sl_pct)
                    tp_prices = [entry_price * (1 + tp_pct)]
                else:
                    sl_price = entry_price * (1 + sl_pct)
                    tp_prices = [entry_price * (1 - tp_pct)]
                    
                logger.info(f"   Fallback SL: {sl_price:.6f}, TP: {tp_prices[0]:.6f}")
        
        else:
            # No risk utilities available, use simple percentage-based approach
            logger.warning("⚠️  RiskUtilities not available, using simple percentage calculation")
            sl_pct = risk_config.get('stop_loss_fixed_pct', 0.03)
            tp_pct = risk_config.get('take_profit_fixed_pct', 0.06)
            
            if side.lower() == 'buy':
                sl_price = entry_price * (1 - sl_pct)
                tp_prices = [entry_price * (1 + tp_pct)]
            else:
                sl_price = entry_price * (1 + sl_pct)
                tp_prices = [entry_price * (1 - tp_pct)]
        
        # ===== STEP 4: Compute Dynamic Position Size =====
        logger.info("📏 Step 4: Computing dynamic position size...")
        
        adjusted_size = base_size
        sizing_mode = risk_config['position_sizing_mode']
        
        if RISK_UTILITIES_AVAILABLE and sl_price:
            try:
                if sizing_mode == 'vol_normalized':
                    # Volatility-normalized sizing
                    risk_per_trade = risk_config['position_risk_per_trade']
                    
                    # Calculate ATR if not already done
                    if 'current_atr' not in locals():
                        current_data = market_analyzer._get_cached_data(symbol, '1m', min_periods=50)
                        if current_data is not None:
                            atr_series = risk_utilities.compute_atr(current_data, period=14)
                            current_atr = atr_series.iloc[-1] if not atr_series.empty else entry_price * 0.02
                        else:
                            current_atr = entry_price * 0.02
                    
                    tick_value = risk_config.get('tick_value', 0.01)
                    
                    adjusted_size = risk_utilities.position_size_vol_normalized(
                        account_equity=account_equity,
                        risk_per_trade=risk_per_trade,
                        atr=current_atr,
                        tick_value=tick_value
                    )
                    
                    logger.info(f"   Vol-normalized size: {adjusted_size:.6f} (Risk: {risk_per_trade:.1%}, ATR: {current_atr:.6f})")
                    
                elif sizing_mode == 'kelly_capped':
                    # Kelly criterion with cap
                    kelly_cap = risk_config['kelly_cap']
                    
                    # Estimate edge and win probability (simplified)
                    # In production, this should use historical strategy performance
                    estimated_edge = 0.05  # 5% edge assumption
                    estimated_win_prob = 0.55  # 55% win rate assumption
                    
                    kelly_fraction = risk_utilities.kelly_fraction_capped(
                        edge=estimated_edge,
                        win_prob=estimated_win_prob,
                        cap=kelly_cap
                    )
                    
                    # Calculate position size based on Kelly fraction
                    risk_amount = account_equity * kelly_fraction
                    stop_distance = abs(entry_price - sl_price) / entry_price
                    adjusted_size = risk_amount / (stop_distance * entry_price) if stop_distance > 0 else base_size
                    
                    logger.info(f"   Kelly-capped size: {adjusted_size:.6f} (Kelly: {kelly_fraction:.1%}, Edge: {estimated_edge:.1%})")
                    
                else:  # fixed_notional
                    fixed_notional = risk_config['position_fixed_notional']
                    adjusted_size = fixed_notional / entry_price
                    
                    logger.info(f"   Fixed notional size: {adjusted_size:.6f} (${fixed_notional})")
                    
            except Exception as e:
                logger.warning(f"⚠️  Dynamic sizing failed, using base size: {e}")
                adjusted_size = base_size
        
        # Apply leverage adjustment based on volatility regime
        leverage_config = risk_config['leverage_by_regime']
        regime_multiplier = leverage_config.get(volatility_regime, 1.0)
        adjusted_size *= regime_multiplier
        
        logger.info(f"   Size after regime adjustment ({volatility_regime}): {adjusted_size:.6f} (×{regime_multiplier})")
        
        # Apply max position size cap
        max_position_pct = risk_config['max_position_pct']
        max_size = (account_equity * max_position_pct / 100) / entry_price
        if adjusted_size > max_size:
            adjusted_size = max_size
            logger.info(f"   Size capped at max position limit: {adjusted_size:.6f} ({max_position_pct}% of equity)")
        
        # ===== STEP 5: Portfolio Risk Manager Validation =====
        logger.info("🏛️  Step 5: Portfolio risk manager validation...")
        
        try:
            # Prepare candidate order for risk manager
            # Convert buy/sell to long/short for risk manager
            position_side = 'long' if side.lower() == 'buy' else 'short'
            
            candidate_order = {
                'symbol': symbol,
                'side': position_side,
                'size': adjusted_size,
                'entry_price': entry_price,  # Use entry_price key name expected by risk manager
                'strategy': strategy_name,
                'factor': portfolio_config['factor'],
                'correlation_group': portfolio_config['correlation_group']
            }
            
            # Get current account state
            account_state = {
                'equity': account_equity,
                'available_balance': account_equity * 0.8,  # Simplified assumption
                'timestamp': datetime.now(timezone.utc)
            }
            
            # Call portfolio risk manager
            enforce_result = risk_manager.enforce_portfolio_limits(
                account_state=account_state,
                open_positions=open_positions,
                candidate_order=candidate_order
            )
            
            logger.info(f"   Portfolio risk result: {enforce_result.action}")
            logger.info(f"   Reason: {enforce_result.reason}")
            
            if enforce_result.action == EnforceAction.DENY:
                return False, {}, f"Portfolio risk manager denied: {enforce_result.reason}"
            
            elif enforce_result.action == EnforceAction.SCALE_DOWN:
                if enforce_result.scaled_size:
                    adjusted_size = enforce_result.scaled_size
                    logger.info(f"   Size scaled down to: {adjusted_size:.6f}")
            
            elif enforce_result.action == EnforceAction.DEFER:
                return False, {}, f"Portfolio risk manager deferred: {enforce_result.reason}"
                
        except Exception as e:
            logger.warning(f"⚠️  Portfolio risk validation failed, proceeding with caution: {e}")
        
        # ===== STEP 6: Slippage and Spread Guards =====
        logger.info("🛡️  Step 6: Checking slippage and spread guards...")
        
        try:
            # Get current market data for spread analysis
            ticker = exchange.get_ticker(symbol)
            if ticker and 'bid' in ticker and 'ask' in ticker:
                bid = float(ticker['bid'])
                ask = float(ticker['ask'])
                spread = (ask - bid) / ((ask + bid) / 2) * 100  # Spread in percentage
                
                # Check spread guard
                spread_config = risk_config.get('spread_slippage_guard', {})
                if spread_config.get('enabled', True):
                    max_spread_pct = spread_config.get('max_spread_pct', 0.1)  # 0.1% default
                    
                    if spread > max_spread_pct:
                        return False, {}, f"Spread too wide: {spread:.4f}% > {max_spread_pct:.4f}% limit"
                    
                    # Estimate slippage based on order size and current spread
                    estimated_slippage = min(spread * 0.5, 0.05)  # Simplified slippage estimation
                    max_slippage_pct = spread_config.get('max_slippage_pct', 0.2)  # 0.2% default
                    
                    if estimated_slippage > max_slippage_pct:
                        return False, {}, f"Estimated slippage too high: {estimated_slippage:.4f}% > {max_slippage_pct:.4f}% limit"
                
                logger.info(f"✅ Spread/slippage check passed: spread={spread:.4f}%")
            
        except Exception as e:
            logger.warning(f"⚠️  Spread analysis failed, proceeding with caution: {e}")
        
        # ===== STEP 7: Prepare Final Order Details =====
        logger.info("📋 Step 7: Preparing final order details...")
        
        # Calculate percentages for OrderManager compatibility
        sl_pct = abs(entry_price - sl_price) / entry_price if sl_price else 0.03
        tp_pct = abs(tp_prices[0] - entry_price) / entry_price if tp_prices else 0.06
        
        order_details = {
            'symbol': symbol,
            'side': side,
            'size': adjusted_size,
            'price': entry_price,
            'order_type': 'market',  # Can be overridden by caller
            'sl_pct': sl_pct,
            'tp_pct': tp_pct,
            'sl_price': sl_price,
            'tp_prices': tp_prices,
            'strategy_name': strategy_name,
            'volatility_regime': volatility_regime,
            'market_regime': market_regime,
            'sizing_mode': sizing_mode,
            'original_size': base_size,
            'size_adjustment_reason': f"Risk-adjusted from {base_size:.6f} to {adjusted_size:.6f}",
            
            # Trailing stop configuration
            'trailing_stop_enabled': risk_config.get('trailing_stop_enabled', False),
            'trailing_stop_mode': risk_config.get('trailing_stop_mode', 'price_pct'),
            'trailing_stop_offset_pct': risk_config.get('trailing_stop_offset_pct', 0.015),
            'trailing_stop_atr_multiplier': risk_config.get('trailing_stop_atr_multiplier', 1.5),
            
            # Risk parameters for logging/monitoring
            'risk_metadata': {
                'strategy_factor': portfolio_config['factor'],
                'correlation_group': portfolio_config['correlation_group'],
                'volatility_pct': current_volatility_pct,
                'atr_used': locals().get('current_atr', 0),
                'regime_multiplier': regime_multiplier,
                'portfolio_approved': True
            }
        }
        
        success_reason = (
            f"Order validated successfully: "
            f"Size {base_size:.6f}→{adjusted_size:.6f} ({sizing_mode}), "
            f"SL {sl_pct:.1%}, TP {tp_pct:.1%}, "
            f"Regime: {volatility_regime}, "
            f"Factor: {portfolio_config['factor']}"
        )
        
        logger.info("✅ Enhanced order validation completed successfully")
        logger.info(f"   Final size: {adjusted_size:.6f}")
        logger.info(f"   Stop loss: {sl_price:.6f} ({sl_pct:.1%})")
        logger.info(f"   Take profit: {tp_prices[0]:.6f} ({tp_pct:.1%})" if tp_prices else "   Take profit: Not set")
        logger.info(f"   Risk metadata: {order_details['risk_metadata']}")
        
        return True, order_details, success_reason
        
    except Exception as e:
        error_msg = f"Enhanced order validation failed with error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, {}, error_msg


def enhanced_order_placement_with_validation(
    strategy_name: str,
    entry_signal: Dict[str, Any],
    symbol: str,
    strategy_matrix: StrategyMatrix,
    config_loader: StrategyConfigLoader,
    market_analyzer: MarketAnalyzer,
    risk_manager: AdvancedRiskManager,
    order_manager: OrderManager,
    exchange: ExchangeConnector,
    account_equity: float,
    open_positions: List[Dict[str, Any]],
    logger: logging.Logger,
    trailing_handler: Optional[TrailingTPHandler] = None,
    directional_bias: str = 'NEUTRAL',
    bias_strength: str = 'WEAK'
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Enhanced order placement that uses comprehensive risk validation before submitting orders.
    
    This function wraps the enhanced_order_validation and integrates with OrderManager
    to provide a complete order placement pipeline with all risk checks.
    
    Args:
        strategy_name: Name of the strategy
        entry_signal: Entry signal from strategy (contains side, price, size, etc.)
        symbol: Trading symbol
        strategy_matrix: StrategyMatrix instance
        config_loader: StrategyConfigLoader instance
        market_analyzer: MarketAnalyzer instance
        risk_manager: AdvancedRiskManager instance
        order_manager: OrderManager instance
        exchange: ExchangeConnector instance
        account_equity: Current account equity
        open_positions: List of open positions
        logger: Logger instance
        
    Returns:
        Tuple[bool, Optional[Dict[str, Any]], str]: (success, order_responses, reason)
    """
    
    logger.info(f"🚀 Enhanced Order Placement Starting for {strategy_name}")
    logger.info(f"   Entry Signal: {entry_signal}")
    
    try:
        # Extract basic parameters from entry signal
        side = entry_signal.get('side')
        entry_price = entry_signal.get('price')
        base_size = entry_signal.get('size')
        order_type = entry_signal.get('order_type', 'market')
        
        if not all([side, base_size]):
            return False, None, "Missing required parameters in entry signal (side, size)"
        
        # Use current market price if no price specified (for market orders)
        if not entry_price:
            try:
                ticker = exchange.get_ticker(symbol)
                if ticker and 'last' in ticker:
                    entry_price = float(ticker['last'])
                    logger.info(f"Using current market price: {entry_price}")
                else:
                    return False, None, "Could not determine entry price and none provided"
            except Exception as e:
                return False, None, f"Failed to get market price: {str(e)}"
        
        # ===== STEP 1: Enhanced Order Validation =====
        logger.info("🔍 Running enhanced order validation...")
        
        approved, validated_order_details, validation_reason = enhanced_order_validation(
            strategy_name=strategy_name,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            base_size=base_size,
            strategy_matrix=strategy_matrix,
            config_loader=config_loader,
            market_analyzer=market_analyzer,
            risk_manager=risk_manager,
            exchange=exchange,
            account_equity=account_equity,
            open_positions=open_positions,
            logger=logger,
            directional_bias=directional_bias,
            bias_strength=bias_strength
        )
        
        if not approved:
            logger.warning(f"❌ Order validation failed: {validation_reason}")
            return False, None, f"Order validation failed: {validation_reason}"
        
        logger.info(f"✅ Order validation passed: {validation_reason}")
        
        # ===== STEP 2: Create Final Order Parameters =====
        logger.info("📋 Preparing final order parameters...")
        
        # Merge original entry signal with validated parameters
        final_order_params = entry_signal.copy()
        
        # Override with validated values
        final_order_params.update({
            'size': validated_order_details['size'],
            'sl_pct': validated_order_details['sl_pct'],
            'tp_pct': validated_order_details['tp_pct'],
            'order_type': order_type,  # Preserve original order type
            'price': entry_price if order_type != 'market' else None
        })
        
        # Add enhanced risk metadata for monitoring
        final_order_params['risk_metadata'] = validated_order_details['risk_metadata']
        final_order_params['validation_details'] = {
            'strategy_name': strategy_name,
            'original_size': base_size,
            'final_size': validated_order_details['size'],
            'sizing_mode': validated_order_details['sizing_mode'],
            'volatility_regime': validated_order_details['volatility_regime'],
            'market_regime': validated_order_details['market_regime']
        }
        
        logger.info(f"Final order parameters prepared:")
        logger.info(f"   Size: {base_size:.6f} → {validated_order_details['size']:.6f}")
        logger.info(f"   Stop Loss: {validated_order_details['sl_pct']:.1%}")
        logger.info(f"   Take Profit: {validated_order_details['tp_pct']:.1%}")
        logger.info(f"   Sizing Mode: {validated_order_details['sizing_mode']}")
        logger.info(f"   Market Regime: {validated_order_details['market_regime']}")
        
        # ===== STEP 3: Execute Order via OrderManager =====
        logger.info("📤 Executing order via OrderManager...")
        
        try:
            order_responses = order_manager.place_order_with_risk(
                symbol=symbol,
                side=final_order_params['side'],
                order_type=final_order_params.get('order_type', 'market'),
                size=final_order_params['size'],
                signal_price=final_order_params.get('price'),
                sl_pct=final_order_params['sl_pct'],
                tp_pct=final_order_params['tp_pct'],
                params=final_order_params.get('params'),
                reduce_only=final_order_params.get('reduce_only', False),
                time_in_force=final_order_params.get('time_in_force', 'GoodTillCancel')
            )
            
            logger.info("✅ Order executed successfully via OrderManager")
            logger.info(f"   Order responses: {order_responses}")
            
            # ===== STEP 4: Initialize Trailing TP Tracking =====
            if trailing_handler and order_responses and 'main_order' in order_responses:
                try:
                    main_order = order_responses['main_order']['result']
                    entry_order_id = main_order.get('orderId')
                    
                    if entry_order_id:
                        # Prepare TP levels for trailing handler
                        tp_levels_config = []
                        if validated_order_details['tp_prices']:
                            for i, tp_price in enumerate(validated_order_details['tp_prices']):
                                tp_levels_config.append({
                                    'level': i + 1,
                                    'price': tp_price,
                                    'size_pct': 0.4 if i == 0 else (0.4 if i == 1 else 0.2)  # 40/40/20 split
                                })
                        
                        # Prepare trailing config
                        trailing_config = {
                            'enabled': validated_order_details.get('trailing_stop_enabled', False),
                            'initial_offset_pct': validated_order_details.get('trailing_stop_offset_pct', 0.02),
                            'tightened_offset_pct': validated_order_details.get('trailing_stop_offset_pct', 0.015)
                        }
                        
                        # Create position tracking
                        position_id = trailing_handler.create_position(
                            symbol=symbol,
                            side=final_order_params['side'],
                            entry_price=entry_price,
                            position_size=validated_order_details['size'],
                            strategy_name=strategy_name,
                            tp_levels=tp_levels_config,
                            trailing_config=trailing_config,
                            stop_loss_pct=validated_order_details['sl_pct'],
                            session_id=validated_order_details.get('session_id', '')
                        )
                        
                        # Register entry order for fill monitoring
                        trailing_handler.order_to_symbol[entry_order_id] = symbol
                        trailing_handler.order_to_type[entry_order_id] = OrderType.ENTRY
                        
                        logger.info(f"✅ Position tracking created for {symbol} with trailing TP handler")
                        logger.info(f"   TP Levels: {len(tp_levels_config)}, Trailing: {'enabled' if trailing_config['enabled'] else 'disabled'}")
                        
                except Exception as tracking_error:
                    logger.warning(f"Failed to initialize trailing TP tracking: {tracking_error}")
                    # Continue without tracking - order was still placed successfully
            
            # Log the comprehensive trade details for monitoring
            logger.info("📊 Trade Execution Summary:")
            logger.info(f"   Strategy: {strategy_name}")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Side: {side}")
            logger.info(f"   Size: {validated_order_details['size']:.6f}")
            logger.info(f"   Entry Price: {entry_price}")
            logger.info(f"   Stop Loss: {validated_order_details['sl_price']:.6f} ({validated_order_details['sl_pct']:.1%})")
            logger.info(f"   Take Profit: {validated_order_details['tp_prices'][0]:.6f} ({validated_order_details['tp_pct']:.1%})" if validated_order_details['tp_prices'] else "   Take Profit: Not set")
            logger.info(f"   Volatility Regime: {validated_order_details['volatility_regime']}")
            logger.info(f"   Portfolio Factor: {validated_order_details['risk_metadata']['strategy_factor']}")
            logger.info(f"   Correlation Group: {validated_order_details['risk_metadata']['correlation_group']}")
            
            success_message = (
                f"Enhanced order placement successful: "
                f"{strategy_name} on {symbol}, "
                f"Size: {validated_order_details['size']:.6f}, "
                f"SL: {validated_order_details['sl_pct']:.1%}, "
                f"TP: {validated_order_details['tp_pct']:.1%}, "
                f"Regime: {validated_order_details['volatility_regime']}"
            )
            
            return True, order_responses, success_message
            
        except OrderExecutionError as oe:
            error_msg = f"OrderManager execution failed: {str(oe)}"
            logger.error(error_msg)
            return False, None, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error during order execution: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, None, error_msg
    
    except Exception as e:
        error_msg = f"Enhanced order placement failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, None, error_msg


def dry_run_enhanced_order_validation_demo(
    logger: logging.Logger,
    strategy_matrix: StrategyMatrix,
    config_loader: StrategyConfigLoader,
    market_analyzer: MarketAnalyzer,
    risk_manager: AdvancedRiskManager,
    exchange: ExchangeConnector
) -> None:
    """
    Dry-run demonstration of enhanced order validation showing all computed values.
    
    This function demonstrates the complete enhanced risk management pipeline
    without placing actual orders, logging all intermediate values and decisions.
    """
    
    logger.info("🧪 ENHANCED ORDER VALIDATION DRY-RUN DEMONSTRATION")
    logger.info("=" * 80)
    
    try:
        # Demo parameters
        demo_strategy = 'StrategyATRMomentumBreakout'
        demo_symbol = 'BTCUSDT'
        demo_side = 'buy'
        demo_entry_price = 45000.0
        demo_base_size = 0.01
        demo_account_equity = 10000.0
        demo_open_positions = []
        
        logger.info("📋 DEMO PARAMETERS:")
        logger.info(f"   Strategy: {demo_strategy}")
        logger.info(f"   Symbol: {demo_symbol}")
        logger.info(f"   Side: {demo_side}")
        logger.info(f"   Entry Price: ${demo_entry_price:,.2f}")
        logger.info(f"   Base Size: {demo_base_size}")
        logger.info(f"   Account Equity: ${demo_account_equity:,.2f}")
        logger.info("")
        
        # Run enhanced order validation
        approved, order_details, reason = enhanced_order_validation(
            strategy_name=demo_strategy,
            symbol=demo_symbol,
            side=demo_side,
            entry_price=demo_entry_price,
            base_size=demo_base_size,
            strategy_matrix=strategy_matrix,
            config_loader=config_loader,
            market_analyzer=market_analyzer,
            risk_manager=risk_manager,
            exchange=exchange,
            account_equity=demo_account_equity,
            open_positions=demo_open_positions,
            logger=logger,
            directional_bias='BEARISH_BIASED',  # Demo with bearish bias
            bias_strength='MODERATE'
        )
        
        logger.info("🏁 DRY-RUN RESULTS:")
        logger.info("=" * 80)
        
        if approved:
            logger.info("✅ ORDER APPROVED")
            logger.info(f"   Reason: {reason}")
            logger.info("")
            logger.info("📊 FINAL ORDER DETAILS:")
            logger.info(f"   Symbol: {order_details['symbol']}")
            logger.info(f"   Side: {order_details['side']}")
            logger.info(f"   Original Size: {order_details['original_size']:.6f}")
            logger.info(f"   Final Size: {order_details['size']:.6f}")
            logger.info(f"   Size Adjustment: {order_details['size_adjustment_reason']}")
            logger.info(f"   Entry Price: ${demo_entry_price:,.2f}")
            logger.info(f"   Stop Loss: ${order_details['sl_price']:.2f} ({order_details['sl_pct']:.1%})")
            logger.info(f"   Take Profit: ${order_details['tp_prices'][0]:.2f} ({order_details['tp_pct']:.1%})" if order_details['tp_prices'] else "   Take Profit: Not set")
            logger.info("")
            logger.info("🎯 RISK METADATA:")
            metadata = order_details['risk_metadata']
            logger.info(f"   Strategy Factor: {metadata['strategy_factor']}")
            logger.info(f"   Correlation Group: {metadata['correlation_group']}")
            logger.info(f"   Volatility: {metadata['volatility_pct']:.2f}%")
            logger.info(f"   ATR Used: {metadata['atr_used']:.6f}")
            logger.info(f"   Regime Multiplier: {metadata['regime_multiplier']:.2f}x")
            logger.info(f"   Portfolio Approved: {metadata['portfolio_approved']}")
            logger.info("")
            logger.info("⚙️  VALIDATION DETAILS:")
            validation = order_details.get('validation_details', {})
            logger.info(f"   Sizing Mode: {order_details['sizing_mode']}")
            logger.info(f"   Volatility Regime: {order_details['volatility_regime']}")
            logger.info(f"   Market Regime: {order_details['market_regime']}")
            logger.info(f"   Trailing Stop Enabled: {order_details['trailing_stop_enabled']}")
            
        else:
            logger.warning("❌ ORDER DENIED")
            logger.warning(f"   Reason: {reason}")
        
        logger.info("")
        logger.info("🧪 DRY-RUN DEMONSTRATION COMPLETED")
        logger.info("   No actual orders were placed - this was a validation test only")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"❌ DRY-RUN DEMONSTRATION FAILED: {str(e)}", exc_info=True)
        logger.info("=" * 80)


def convert_strategy_class_to_module_name(strategy_class_name: str) -> str:
    """
    Convert a strategy class name to its corresponding module name.
    
    Args:
        strategy_class_name: Class name like 'StrategyAdaptiveTransitionalMomentum'
        
    Returns:
        Module name like 'adaptive_transitional_momentum_strategy'
    """
    # Special case mappings for strategies that don't follow the standard naming convention
    special_mappings = {
        'StrategyDoubleEMAStochOsc': 'double_EMA_StochOsc',  # Non-standard file name
        'StrategyEMAAdx': 'ema_adx_strategy',  # Handle EMA + ADX case properly
        'StrategyEMATrendRider': 'ema_adx_strategy',  # Class name doesn't match file name
        'StrategyRSIRangeScalping': 'rsi_range_scalping_strategy',  # rsirange_scalping_strategy -> rsi_range_scalping_strategy
        'StrategyATRMomentumBreakout': 'atr_momentum_breakout_strategy',  # atrmomentum_breakout_strategy -> atr_momentum_breakout_strategy
    }
    
    # Check for special mappings first
    if strategy_class_name in special_mappings:
        return special_mappings[strategy_class_name]
    
    # Standard conversion logic
    # Remove 'Strategy' prefix
    name_without_prefix = strategy_class_name.replace('Strategy', '')
    
    # Add underscores before capitals (while capitals still exist)
    snake_case_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name_without_prefix)
    
    # Convert to lowercase
    snake_case_name = snake_case_name.lower()
    
    # Add '_strategy' suffix  
    module_name = snake_case_name + '_strategy'
    
    return module_name


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def list_strategies():
    """
    List available strategies, filtering out template files and strategies marked as hidden.
    Returns a list of tuples: (strategy_module_name, strategy_class, market_type_tags)
    """
    files = [f for f in os.listdir(STRATEGY_DIR) if f.endswith('.py') and not f.startswith('__') and 'template' not in f]
    strategy_info = []
    
    for filename in files:
        module_name = os.path.splitext(filename)[0]
        try:
            # Dynamically import to check visibility and get market type tags
            strategy_class = dynamic_import_strategy(module_name, StrategyTemplate, get_logger('strategy_discovery'))
            
            # Check if strategy should be shown in selection
            if getattr(strategy_class, 'SHOW_IN_SELECTION', True):
                market_tags = getattr(strategy_class, 'MARKET_TYPE_TAGS', [])
                strategy_info.append((module_name, strategy_class, market_tags))
                
        except Exception as e:
            # If strategy fails to import, log but don't include it
            get_logger('strategy_discovery').warning(f"Failed to import strategy {module_name}: {e}")
            continue
    
    return strategy_info

def get_strategy_parameters(strategy_class_name: str) -> dict:
    """
    Map strategy class names to their trading parameters (category only).
    Symbol, timeframe, and leverage are now selected dynamically by the user.
    """
    strategy_params = {
        'StrategyDoubleEMAStochOsc': {
            'category': 'linear'
        },
        'StrategyBreakoutAndRetest': {
            'category': 'linear'
        },
        'StrategyEMAAdx': {
            'category': 'linear'
        }
    }
    
    return strategy_params.get(strategy_class_name, {
        'category': 'linear'
    })

def select_strategies(available: list[tuple], logger_instance: logging.Logger): # Updated to handle list of tuples
    """
    Display available strategies with market type tags and prompt user to select.
    Args:
        available: List of tuples (strategy_module_name, strategy_class, market_type_tags)
        logger_instance: Logger instance
    Returns:
        List of selected strategy module names
    """
    logger_instance.info('Available strategies for selection:')
    print("\nAvailable strategies:")
    print("=" * 60)
    
    for i, (module_name, strategy_class, market_tags) in enumerate(available):
        strategy_name = strategy_class.__name__
        tags_display = f"[{', '.join(market_tags)}]" if market_tags else "[NO TAGS]"
        display_line = f"  {i+1}. {strategy_name} {tags_display}"
        print(display_line)
        logger_instance.info(display_line)
    
    print("=" * 60)
    selected_input = input('Select strategies (comma-separated indices, e.g. 1,2): ')
    logger_instance.info(f"User input for strategy selection: '{selected_input}'")
    
    indices = []
    if selected_input.strip(): # Check if input is not empty
        try:
            indices = [int(i.strip())-1 for i in selected_input.split(',') if i.strip().isdigit() and 0 <= int(i.strip())-1 < len(available)]
        except ValueError:
            logger_instance.error("Invalid input for strategy selection (non-integer value). No strategies selected.")
            return [] # Return empty if there's a non-integer value that's not filtered by isdigit
            
    selected_names = [available[i][0] for i in indices]  # Extract module names from tuples
    logger_instance.info(f"Parsed selected indices: {indices}, Corresponding names: {selected_names}")
    return selected_names

def select_symbol(analysis_results: dict, logger_instance: logging.Logger) -> str:
    """
    Let user select a symbol from the analyzed symbols.
    
    Args:
        analysis_results: Market analysis results dictionary
        logger_instance: Logger instance
        
    Returns:
        Selected symbol string
    """
    symbols = list(analysis_results.keys())
    
    logger_instance.info('Available symbols from market analysis:')
    print("\nAvailable symbols:")
    print("=" * 70)
    
    for i, symbol in enumerate(symbols):
        # Get market types for this symbol
        symbol_data = analysis_results[symbol]
        market_types = []
        for timeframe, data in symbol_data.items():
            market_type = data.get('market_type', 'UNKNOWN')
            market_types.append(f"{timeframe}:{market_type}")
        
        market_summary = " | ".join(market_types)
        display_line = f"  {i+1}. {symbol:<15} [{market_summary}]"
        print(display_line)
        logger_instance.info(display_line)
    
    print("=" * 70)
    
    while True:
        selected_input = input('Select symbol (enter number): ').strip()
        logger_instance.info(f"User input for symbol selection: '{selected_input}'")
        
        try:
            index = int(selected_input) - 1
            if 0 <= index < len(symbols):
                selected_symbol = symbols[index]
                logger_instance.info(f"Selected symbol: {selected_symbol}")
                return selected_symbol
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(symbols)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

def select_timeframe(analysis_results: dict, selected_symbol: str, logger_instance: logging.Logger) -> str:
    """
    Let user select a timeframe for the chosen symbol.
    
    Args:
        analysis_results: Market analysis results dictionary
        selected_symbol: Previously selected symbol
        logger_instance: Logger instance
        
    Returns:
        Selected timeframe string
    """
    symbol_data = analysis_results[selected_symbol]
    timeframes = list(symbol_data.keys())
    
    logger_instance.info(f'Available timeframes for {selected_symbol}:')
    print(f"\nAvailable timeframes for {selected_symbol}:")
    print("=" * 50)
    
    for i, timeframe in enumerate(timeframes):
        data = symbol_data[timeframe]
        market_type = data.get('market_type', 'UNKNOWN')
        current_price = data.get('price_range', {}).get('current', 'N/A')
        display_line = f"  {i+1}. {timeframe:<5} [Market: {market_type}, Price: ${current_price}]"
        print(display_line)
        logger_instance.info(display_line)
    
    print("=" * 50)
    
    while True:
        selected_input = input('Select timeframe (enter number): ').strip()
        logger_instance.info(f"User input for timeframe selection: '{selected_input}'")
        
        try:
            index = int(selected_input) - 1
            if 0 <= index < len(timeframes):
                selected_timeframe = timeframes[index]
                logger_instance.info(f"Selected timeframe: {selected_timeframe}")
                return selected_timeframe
            else:
                print(f"Invalid selection. Please enter a number between 1 and {len(timeframes)}")
        except ValueError:
            print("Invalid input. Please enter a number.")

def select_leverage(logger_instance: logging.Logger) -> int:
    """
    Let user select leverage for trading.
    
    Args:
        logger_instance: Logger instance
        
    Returns:
        Selected leverage as integer (1-50)
    """
    logger_instance.info('Leverage selection:')
    print("\n" + "="*60)
    print("LEVERAGE SELECTION")
    print("="*60)
    print("Choose your leverage multiplier (1-50):")
    print("  • Lower leverage (1-5): Conservative trading, lower risk")
    print("  • Medium leverage (10-25): Moderate risk/reward")
    print("  • Higher leverage (30-50): Aggressive trading, higher risk")
    print("  • Note: Higher leverage increases both potential gains and losses")
    print("="*60)
    
    while True:
        selected_input = input('Enter leverage (1-50): ').strip()
        logger_instance.info(f"User input for leverage selection: '{selected_input}'")
        
        try:
            leverage = int(selected_input)
            if 1 <= leverage <= 50:
                logger_instance.info(f"Selected leverage: {leverage}x")
                print(f"✅ Leverage set to {leverage}x")
                return leverage
            else:
                print("Invalid selection. Please enter a number between 1 and 50")
        except ValueError:
            print("Invalid input. Please enter a whole number.")

def dynamic_import_strategy(name: str, base_class_to_check: type, logger_instance: logging.Logger) -> type:
    module_name = f"strategies.{name}"
    logger_instance.debug(f"Attempting to import module: {module_name}")
    try:
        module = importlib.import_module(module_name)
        logger_instance.debug(f"Successfully imported module: {module_name}. Inspected attributes: {dir(module)}")
    except ImportError as e:
        logger_instance.error(f"Failed to import module {module_name}: {e}", exc_info=True)
        raise  # Re-raise to be caught by the main loading loop

    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        # Ensure it's a class, a subclass of base_class_to_check, and not base_class_to_check itself
        if isinstance(attribute, type) and attribute is not base_class_to_check and issubclass(attribute, base_class_to_check):
            logger_instance.debug(f"Found valid strategy class '{attribute.__name__}' in {module_name}.")
            return attribute
            
    logger_instance.error(f"No valid strategy class (subclass of {base_class_to_check.__name__}) found in {module_name}.")
    raise ImportError(f"No valid strategy class (subclass of {base_class_to_check.__name__}) found in {module_name}")

def run_market_analysis(exchange, config, logger):
    """
    Run market analysis for all configured symbols and timeframes.
    
    Args:
        exchange: ExchangeConnector instance
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Analysis results dictionary or None if analysis fails
    """
    try:
        logger.info("Starting market analysis...")
        
        # Initialize market analyzer
        market_analyzer = MarketAnalyzer(exchange, config, logger)
        
        # Run analysis for all symbols and timeframes
        analysis_results = market_analyzer.analyze_all_markets()
        
        # Get summary statistics
        summary = market_analyzer.get_market_summary(analysis_results)
        logger.info(f"Market analysis completed. Summary: {summary}")
        
        return analysis_results
        
    except MarketAnalysisError as e:
        logger.error(f"Market analysis failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during market analysis: {e}", exc_info=True)
        return None

def run_silent_market_analysis(exchange, config, symbol, timeframe, logger):
    """
    Run market analysis silently for a specific symbol and timeframe.
    Used for periodic checks without verbose output.
    
    Args:
        exchange: ExchangeConnector instance
        config: Configuration dictionary
        symbol: Symbol to analyze
        timeframe: Timeframe to analyze
        logger: Logger instance
        
    Returns:
        Market analysis results dictionary with structure: {symbol: {timeframe: {analysis_data}}}
        or None if analysis fails
    """
    try:
        logger.debug(f"Running silent market analysis for {symbol} {timeframe}")
        
        # Create a MarketAnalyzer with minimal initialization to avoid expensive symbol fetching
        # but still set up essential attributes needed for analysis
        market_analyzer = MarketAnalyzer.__new__(MarketAnalyzer)
        market_analyzer.exchange = exchange
        market_analyzer.config = config
        market_analyzer.logger = logger
        
        # Set up essential attributes without expensive operations
        market_config = config.get('market_analysis', {})
        market_analyzer.timeframes = market_config.get('timeframes', ['1m', '5m'])
        market_analyzer.use_dynamic_symbols = market_config.get('use_dynamic_symbols', False)
        market_analyzer.top_volume_count = market_config.get('top_volume_count', 10)
        market_analyzer.min_volume_usdt = market_config.get('min_volume_usdt', 1000000)
        
        # Skip the expensive symbol fetching/validation - we only need to analyze one symbol
        market_analyzer.symbols = [symbol]  # Set just the symbol we need
        
        # Analyze just the specific symbol/timeframe directly
        try:
            result = market_analyzer._analyze_symbol_timeframe(symbol, timeframe)
            market_type = result.get('market_type', 'UNKNOWN')
            
            # Log more details if analysis fails or returns UNKNOWN
            if market_type in ['UNKNOWN', 'INSUFFICIENT_DATA', 'ANALYSIS_FAILED']:
                analysis_details = result.get('analysis_details', {})
                data_points = result.get('data_points', 0)
                logger.warning(f"Silent analysis for {symbol} {timeframe} returned {market_type}. "
                             f"Data points: {data_points}, Details: {analysis_details}")
            else:
                logger.debug(f"Silent analysis result for {symbol} {timeframe}: {market_type}")
            
            # Return in the expected dictionary structure
            return {
                symbol: {
                    timeframe: result
                }
            }
        except Exception as e:
            logger.warning(f"Silent market analysis failed for {symbol} {timeframe}: {e}", exc_info=True)
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error during silent market analysis: {e}")
        return None

def check_strategy_market_compatibility(strategy_tags, current_market_type, symbol, timeframe, logger):
    """
    Check if current market type is compatible with strategy tags.
    
    Args:
        strategy_tags: List of market type tags from strategy
        current_market_type: Current market type from analysis
        symbol: Trading symbol
        timeframe: Trading timeframe
        logger: Logger instance
        
    Returns:
        bool: True if compatible, False if mismatch
    """
    if not strategy_tags or not current_market_type:
        # If no tags defined or analysis failed, assume compatible
        return True
    
    # Special handling for TEST tag - always compatible
    if 'TEST' in strategy_tags:
        return True
    
    # Check if current market type matches any strategy tag
    is_compatible = current_market_type in strategy_tags
    
    if is_compatible:
        logger.debug(f"Market compatibility check: {symbol} {timeframe} {current_market_type} matches strategy tags {strategy_tags}")
    else:
        logger.warning(f"Market compatibility mismatch: {symbol} {timeframe} is {current_market_type} but strategy expects {strategy_tags}")
    
    return is_compatible

def prompt_strategy_reselection(analysis_results, current_symbol, current_timeframe, current_market_type, available_strategies, logger):
    """
    Prompt user about market change and ask for strategy reselection.
    
    Args:
        analysis_results: Full market analysis results
        current_symbol: Currently trading symbol
        current_timeframe: Currently trading timeframe
        current_market_type: New market type detected
        available_strategies: Available strategy list
        logger: Logger instance
        
    Returns:
        tuple: (new_strategy_name, should_restart) or (None, False) to continue
    """
    print("\n" + "="*80)
    print("🚨 MARKET CONDITION CHANGE DETECTED 🚨")
    print("="*80)
    print(f"Trading pair: {current_symbol} {current_timeframe}")
    print(f"New market type: {current_market_type}")
    print(f"The market conditions have changed and may no longer match your current strategy.")
    print("\nFull market analysis:")
    
    # Print the analysis summary (reuse existing code from market_analyzer)
    try:
        if analysis_results:
            # Create temporary analyzer just to use the print function
            from modules.market_analyzer import MarketAnalyzer
            temp_analyzer = MarketAnalyzer.__new__(MarketAnalyzer)  # Create without calling __init__
            temp_analyzer.logger = logger
            temp_analyzer._print_analysis_summary(analysis_results)
    except Exception as e:
        logger.error(f"Error printing analysis summary: {e}")
        print("(Could not display full analysis)")
    
    print("\n" + "="*80)
    print("STRATEGY RESELECTION OPTIONS")
    print("="*80)
    print("1. Continue with current strategy (ignore market change)")
    print("2. Select a new strategy based on current market conditions")
    print("="*80)
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        logger.info(f"User choice for market change response: '{choice}'")
        
        if choice == "1":
            logger.info("User chose to continue with current strategy despite market change")
            return None, False
        elif choice == "2":
            logger.info("User chose to reselect strategy due to market change")
            # Let user select new strategy
            selected_strategy_names = select_strategies(available_strategies, logger)
            if selected_strategy_names:
                return selected_strategy_names[0], True  # Return first selected strategy
            else:
                print("No strategy selected. Continuing with current strategy.")
                return None, False
        else:
            print("Invalid choice. Please enter 1 or 2.")

def restart_configuration_with_new_strategy(new_strategy_name, available_strategies, analysis_results, exchange, config, bot_logger):
    """
    Restart the bot configuration process with a new strategy.
    This function handles the complete reconfiguration flow after strategy reselection.
    """
    bot_logger.info("="*60)
    bot_logger.info("RESTARTING CONFIGURATION WITH NEW STRATEGY")
    bot_logger.info("="*60)
    
    try:
        # Load the new strategy class
        bot_logger.info(f"Loading new strategy: {new_strategy_name}")
        try:
            StratClass = dynamic_import_strategy(new_strategy_name, StrategyTemplate, bot_logger)
            bot_logger.info(f"Successfully loaded strategy class: {StratClass.__name__}")
        except Exception as e:
            bot_logger.error(f"Failed to load new strategy {new_strategy_name}: {e}", exc_info=True)
            bot_logger.error("Aborting strategy change - bot will shut down")
            return
        
        # Let user select symbol and timeframe from analyzed markets
        bot_logger.info("="*60)
        bot_logger.info("SYMBOL AND TIMEFRAME SELECTION")
        bot_logger.info("="*60)
        
        selected_symbol = select_symbol(analysis_results, bot_logger)
        selected_timeframe = select_timeframe(analysis_results, selected_symbol, bot_logger)
        selected_leverage = select_leverage(bot_logger)
        
        # RECONFIGURE LOGGING FOR NEW SYMBOL
        bot_logger.info("🔄 Reconfiguring logging for strategy change...")
        configure_logging_session(selected_symbol)
        bot_logger.info(f"✅ Logging reconfigured for new symbol: {selected_symbol}")
        
        # Get strategy parameters
        strategy_params = get_strategy_parameters(StratClass.__name__)
        
        # Set up trading parameters
        symbol = selected_symbol
        timeframe = selected_timeframe
        leverage = selected_leverage
        category = strategy_params['category']
        coin_pair = symbol.replace('USDT', '/USDT')
        
        bot_logger.info(f"New configuration: {coin_pair} ({symbol}), {timeframe}, {leverage}x leverage")
        
        # Set leverage on exchange
        try:
            bot_logger.info(f"Setting leverage to {leverage}x for {symbol}")
            exchange.set_leverage(symbol, leverage, category)
            bot_logger.info(f"✅ Successfully set leverage to {leverage}x for {symbol}")
        except Exception as e:
            bot_logger.error(f"Failed to set leverage to {leverage}x for {symbol}: {e}")
            bot_logger.error("Bot will continue, but orders may fail if current leverage is insufficient")
        
        # Initialize new data fetcher
        data_fetcher = LiveDataFetcher(exchange, symbol, timeframe, logger=bot_logger)
        data = data_fetcher.fetch_initial_data()
        data_fetcher.start_websocket()
        bot_logger.info(f"Fetched initial OHLCV data: {len(data)} rows for {symbol} {timeframe}")
        
        # Initialize new order manager and performance tracker
        order_manager = OrderManager(exchange, logger=bot_logger)
        perf_tracker = PerformanceTracker(logger=bot_logger)
        
        # Initialize the new strategy instance
        try:
            strategy_specific_logger = get_logger(new_strategy_name)
            strategy_instance = StratClass(data.copy(), config, logger=strategy_specific_logger)
            bot_logger.info(f"Successfully initialized new strategy: {type(strategy_instance).__name__}")
        except Exception as e:
            bot_logger.error(f"Failed to initialize new strategy {StratClass.__name__}: {e}", exc_info=True)
            if 'data_fetcher' in locals() and data_fetcher is not None:
                data_fetcher.stop_websocket()
            return
        
        # Log initial state
        strategy_instance.log_state_change(symbol, "awaiting_entry", f"Strategy {type(strategy_instance).__name__} for {symbol}: Initialized after strategy change. Looking for new entry conditions...")
        
        bot_logger.info("="*60)
        bot_logger.info("RESUMING TRADING WITH NEW CONFIGURATION")
        bot_logger.info("="*60)
        
        # Start new trading loop with the reconfigured parameters
        run_trading_loop(
            strategy_instance, symbol, timeframe, leverage, category,
            data_fetcher, order_manager, perf_tracker, exchange, config, bot_logger
        )
        
    except Exception as e:
        bot_logger.error(f"Error during strategy reconfiguration: {e}", exc_info=True)
        bot_logger.error("Strategy change failed - bot will shut down")

def run_trading_loop(strategy_instance, symbol, timeframe, leverage, category, data_fetcher, order_manager, perf_tracker, exchange, config, bot_logger, strategy_matrix=None, config_loader=None, market_analyzer=None, risk_manager=None, directional_bias: str = 'NEUTRAL', bias_strength: str = 'WEAK'):
    """
    Main trading loop that can be reused for strategy changes.
    Enhanced with comprehensive risk management integration.
    """
    bot_logger.info("Entering enhanced main trading loop.")
    
    # Check if enhanced risk management components are available
    enhanced_risk_available = all([strategy_matrix, config_loader, market_analyzer, risk_manager])
    if enhanced_risk_available:
        bot_logger.info("✅ Enhanced risk management components available")
    else:
        bot_logger.warning("⚠️  Some enhanced risk management components missing - enhanced order validation will be limited")
        bot_logger.warning(f"   Available: strategy_matrix={strategy_matrix is not None}, config_loader={config_loader is not None}, market_analyzer={market_analyzer is not None}, risk_manager={risk_manager is not None}")
    
    # Get available strategies for potential reselection
    available_strategies = list_strategies()
    
    # Get strategy tags for compatibility checking
    primary_strategy_tags = getattr(type(strategy_instance), 'MARKET_TYPE_TAGS', [])
    
    # Initialize timing for periodic market analysis
    last_market_check = datetime.now()
    market_check_interval = timedelta(minutes=60)  # Check every 60 minutes
    
    bot_logger.info(f"Periodic market analysis will run every {market_check_interval.total_seconds()/60:.0f} minutes")
    bot_logger.info(f"Current strategy tags: {primary_strategy_tags}")
    
    # Wrap strategy in list for compatibility with existing loop logic
    strategies = [strategy_instance]
    
    # Store bias parameters for order validation (initialize before main loop)
    current_directional_bias = directional_bias
    current_bias_strength = bias_strength
    
    while True:
        bot_logger.debug("Main loop iteration started.")
        
        bot_logger.debug("Calling data_fetcher.update_data()")
        data = data_fetcher.update_data()
        bot_logger.debug("data_fetcher.update_data() returned.")
        
        # Sync active orders with exchange and process adopted orders
        bot_logger.debug(f"Calling order_manager.sync_active_orders_with_exchange for {symbol}")
        adopted_orders = order_manager.sync_active_orders_with_exchange(symbol, category=category)
        bot_logger.debug("order_manager.sync_active_orders_with_exchange() returned.")

        # Check for and cancel orphaned conditional orders more frequently
        bot_logger.debug(f"Calling order_manager.check_and_cancel_orphaned_conditional_orders for {symbol} ({category})")
        try:
            order_manager.check_and_cancel_orphaned_conditional_orders(symbol, category=category)
        except Exception as e_orphan_check:
            bot_logger.error(f"Error during check_and_cancel_orphaned_conditional_orders: {e_orphan_check}", exc_info=True)
        bot_logger.debug("order_manager.check_and_cancel_orphaned_conditional_orders() returned.")

        if adopted_orders:
            for strat in strategies: # Notify all strategies (can be refined if strategies manage specific symbols)
                for adopted_order in adopted_orders:
                    if adopted_order.get('symbol') == symbol: # Basic check
                        try:
                            bot_logger.info(f"Notifying strategy {type(strat).__name__} of adopted order {adopted_order.get('orderId')}")
                            strat.on_externally_synced_order(adopted_order, symbol)
                        except Exception as e_strat_notify:
                            bot_logger.error(f"Error notifying strategy {type(strat).__name__} of adopted order: {e_strat_notify}", exc_info=True)

        for strat in strategies:
            strat_instance = strat  # For consistency with existing variable names
            bot_logger.debug(f"Processing strategy {type(strat).__name__}")
            
            # Update strategy data efficiently - preserve indicators while adding new OHLCV rows
            if strat.data is not None and not strat.data.empty:
                # Strategy already has data - check if there are new rows to add
                if len(data) > len(strat.data):
                    # Get new rows that need to be added
                    new_rows = data.iloc[len(strat.data):]
                    
                    # Append new OHLCV rows to existing strategy data (preserving indicators)
                    if not new_rows.empty:
                        # Create empty indicator columns for new rows to match existing structure
                        new_rows_with_indicators = new_rows.copy()
                        indicator_cols = [col for col in strat.data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
                        for col in indicator_cols:
                            new_rows_with_indicators[col] = np.nan
                        
                        # Append the new rows
                        strat.data = pd.concat([strat.data, new_rows_with_indicators], ignore_index=False)
                        bot_logger.debug(f"Added {len(new_rows)} new rows to {type(strat).__name__} data, now has {len(strat.data)} rows")
                else:
                    # No new data - keep existing strategy data with indicators intact
                    bot_logger.debug(f"{type(strat).__name__} data unchanged, {len(strat.data)} rows with indicators preserved")
            else:
                # First time initialization - use fresh copy
                strat.data = data.copy()
                bot_logger.debug(f"Initialized {type(strat).__name__} data with {len(strat.data)} rows")
            
            # Use efficient indicator update if available, otherwise fall back to full init
            if hasattr(strat, 'update_indicators_for_new_row') and len(strat.data) > 1:
                strat.update_indicators_for_new_row()
            else:
                strat.init_indicators()

            # Debug: log the latest row's indicator values
            # latest_row = strat.data.iloc[-1].to_dict()
            entry_signal = strat.check_entry(symbol=symbol)
            if entry_signal:
                # Validate required fields are present in entry_signal
                required_fields = ['side', 'size', 'sl_pct', 'tp_pct']
                missing_fields = [field for field in required_fields if field not in entry_signal]
                
                if missing_fields:
                    bot_logger.error(f"Strategy {type(strat).__name__} produced invalid entry_signal missing required fields {missing_fields}: {entry_signal}")
                    continue  # Skip this signal and move to next strategy

                # Validate price field based on order type
                order_type = entry_signal.get('order_type', 'market')  # Default to market if not specified
                if order_type != 'market' and 'price' not in entry_signal:
                    bot_logger.error(f"Strategy {type(strat).__name__} produced non-market order signal without 'price': {entry_signal}")
                    continue  # Skip this signal and move to next strategy

                order_details = entry_signal.copy()

                # The strategy should now always provide sl_pct and tp_pct.
                # The OrderManager will calculate absolute SL/TP prices based on actual fill price.
                # The old block for recalculating SL/TP if not in order_details is removed.

                bot_logger.info(f"Order signal: {order_details}") # Log details before sending to Enhanced Order Placement

                # ===== ENHANCED ORDER PLACEMENT WITH COMPREHENSIVE RISK MANAGEMENT =====
                try:
                    # Get current account equity for risk calculations
                    try:
                        balance_info = exchange.get_balance()
                        account_equity = float(balance_info.get('totalEquity', 10000))  # Default fallback
                    except Exception as balance_error:
                        bot_logger.warning(f"Could not get account equity: {balance_error}, using default 10000")
                        account_equity = 10000.0
                    
                    # Get current open positions
                    try:
                        positions = exchange.get_positions()
                        open_positions = [pos for pos in positions if float(pos.get('size', 0)) != 0]
                    except Exception as pos_error:
                        bot_logger.warning(f"Could not get positions: {pos_error}, using empty list")
                        open_positions = []
                    
                    # Use enhanced order placement with validation
                    success, order_responses, placement_reason = enhanced_order_placement_with_validation(
                        strategy_name=type(strat).__name__,
                        entry_signal=order_details,
                        symbol=symbol,
                        strategy_matrix=strategy_matrix,
                        config_loader=config_loader,
                        market_analyzer=market_analyzer,
                        risk_manager=risk_manager,
                        order_manager=order_manager,
                        exchange=exchange,
                        account_equity=account_equity,
                        open_positions=open_positions,
                        logger=bot_logger,
                        trailing_handler=trailing_tp_handler,
                        directional_bias=current_directional_bias,
                        bias_strength=current_bias_strength
                    )
                    
                    # Track order placement time for minimum hold logic
                    if success and order_responses:
                        order_manager.track_order_placement(symbol)
                    
                    if not success:
                        # Enhanced validation failed - create error response for strategy
                        bot_logger.error(f"Enhanced order placement failed for {type(strat).__name__}: {placement_reason}")
                        error_response = {
                            'main_order': {
                                'result': {
                                    'orderId': None,
                                    'orderStatus': 'rejected',
                                    'error': placement_reason,
                                    'validation_type': 'enhanced_risk_management'
                                }
                            }
                        }
                        try:
                            strat.on_order_update(error_response, symbol=symbol)
                        except Exception as callback_error:
                            bot_logger.error(f"Failed to notify strategy of enhanced validation error: {callback_error}", exc_info=True)
                        continue  # Skip to next strategy
                    
                    else:
                        # Enhanced placement succeeded
                        bot_logger.info(f"✅ Enhanced order placement successful for {type(strat).__name__}: {placement_reason}")
                        
                except Exception as e:
                    # Fallback to original order placement if enhanced placement fails completely
                    bot_logger.error(f"Enhanced order placement system failed for {type(strat).__name__}: {e}", exc_info=True)
                    bot_logger.warning("Falling back to original order placement system...")
                    
                    try:
                        order_responses = order_manager.place_order_with_risk(
                         symbol=symbol,
                         side=order_details['side'],
                         order_type=order_details.get('order_type', 'market'),
                         size=order_details['size'],
                         signal_price=order_details.get('price'), # Price at the time of signal generation
                         sl_pct=order_details['sl_pct'],
                         tp_pct=order_details['tp_pct'],
                         params=order_details.get('params'), # Pass any extra params from strategy
                         reduce_only=order_details.get('reduce_only', False),
                         time_in_force=order_details.get('time_in_force', 'GoodTillCancel')
                        )
                        bot_logger.info("✅ Fallback order placement successful")
                        
                    except OrderExecutionError as oe:
                        bot_logger.error(f"Fallback order placement failed for {type(strat).__name__}: {oe}")
                        error_response = {
                            'main_order': {
                                'result': {
                                    'orderId': None,
                                    'orderStatus': 'rejected',
                                    'error': str(oe)
                                }
                            }
                        }
                        try:
                            strat.on_order_update(error_response, symbol=symbol)
                        except Exception as callback_error:
                            bot_logger.error(f"Failed to notify strategy of fallback error: {callback_error}", exc_info=True)
                        continue
                    
                    except Exception as fallback_error:
                        bot_logger.error(f"Unexpected error during fallback order placement for {type(strat).__name__}: {fallback_error}", exc_info=True)
                        error_response = {
                            'main_order': {
                                'result': {
                                    'orderId': None,
                                    'orderStatus': 'rejected',
                                    'error': f"Fallback placement failed: {str(fallback_error)}",
                                    'category': 'linear',
                                    'symbol': symbol,
                                    'side': order_details.get('side', 'N/A'),
                                }
                            }
                        }
                        try:
                            strat.on_order_update(error_response, symbol=symbol)
                        except Exception as callback_error:
                            bot_logger.error(f"Failed to notify strategy of fallback error: {callback_error}", exc_info=True)
                        continue
                
                # Now call on_order_update with the actual responses from OrderManager.
                try:
                    strat.on_order_update(order_responses, symbol=symbol)
                except Exception as callback_error:
                    bot_logger.error(f"Strategy callback error in {type(strat).__name__}.on_order_update: {callback_error}", exc_info=True)
                    continue  # move on to next strategy without crashing the bot

            # Check for open position and exit
            # Ensure strat_instance.position is a dict, as expected
            if not isinstance(strat_instance.position, dict):
                strat_instance.position = {} # Initialize if not a dict to prevent errors

            current_position_details = strat_instance.position.get(symbol)
            if current_position_details and safe_float_convert(current_position_details.get('size', 0)) != 0:
                # POSITION SYNC FIX: Verify position actually exists on exchange before attempting exit
                try:
                    positions_response = exchange.fetch_positions(symbol, category)
                    positions_list = positions_response.get('result', {}).get('list', [])
                    
                    # Find position for this symbol
                    actual_position = None
                    for pos in positions_list:
                        if pos.get('symbol') == symbol.replace('/', '').upper():
                            actual_position = pos
                            break
                    
                    actual_size = safe_float_convert(actual_position.get('size', 0)) if actual_position else 0
                    
                    if actual_size == 0:
                        # Position was already closed (likely by SL/TP) but strategy wasn't notified
                        bot_logger.warning(f"Position sync mismatch detected for {symbol}: Strategy thinks it has position {current_position_details.get('size', 0)}, but exchange shows 0. Clearing strategy position.")
                        strat_instance.clear_position(symbol)
                        continue  # Skip to next strategy since position is already closed
                    else:
                        bot_logger.debug(f"Position sync confirmed for {symbol}: Strategy size {current_position_details.get('size', 0)}, Exchange size {actual_size}")
                except Exception as e:
                    bot_logger.warning(f"Failed to verify position for {symbol}: {e}. Proceeding with exit attempt.")
                
                # Check minimum hold time before allowing exits
                if not order_manager.can_close_position(symbol):
                    bot_logger.debug(f"Position {symbol} in minimum hold period - skipping exit check")
                    continue
                
                exit_signal = strat.check_exit(symbol=symbol)
                if exit_signal:
                    bot_logger.info(f"Exit signal received from {type(strat).__name__} for {symbol}: {exit_signal}")
                    try:
                        # Pass category to execute_strategy_exit
                        exit_order_response = order_manager.execute_strategy_exit(symbol, current_position_details, category=category)
                        bot_logger.info(f"Exit order response for {type(strat).__name__}: {exit_order_response}")
                        
                        # Check if position was already closed on exchange
                        if exit_order_response.get('position_already_closed'):
                            bot_logger.info(f"✅ Position for {symbol} was already closed on exchange. Clearing strategy position.")
                            strat.clear_position(symbol)
                        else:
                            # Notify strategy of exit order update for normal exits
                            try:
                                strat.on_order_update(exit_order_response, symbol=symbol)
                            except Exception as callback_error:
                                bot_logger.error(f"Strategy callback error in {type(strat).__name__}.on_order_update (exit): {callback_error}", exc_info=True)
                        
                        # Update performance tracker after successful exit
                        if exit_order_response and exit_order_response.get('exit_order', {}).get('result', {}).get('orderStatus', '').lower() == 'filled':
                            trade_summary = {
                                'strategy': type(strat).__name__,
                                'symbol': symbol,
                                'entry_price': safe_float_convert(current_position_details.get('entry_price', 0)),
                                'exit_price': safe_float_convert(exit_order_response['exit_order']['result'].get('avgPrice', 0)),
                                'size': safe_float_convert(current_position_details.get('size', 0)),
                                'side': current_position_details.get('side'),
                                'pnl': safe_float_convert(exit_order_response.get('pnl', 0)), # Assuming OrderManager calculates this
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            }
                            perf_tracker.record_trade(trade_summary)
                            bot_logger.info(f"Trade recorded for {type(strat).__name__}: {trade_summary}")
                        
                        # Clear position from strategy after successful exit and recording
                        strat_instance.clear_position(symbol)
                        bot_logger.info(f"Position for {symbol} cleared from strategy {type(strat).__name__}.")

                    except OrderExecutionError as oe:
                        bot_logger.error(f"Exit order placement failed for {type(strat).__name__}: {oe}")
                        # Notify strategy of order failure with error response
                        error_response_exit = {
                            'exit_order': {
                                'result': {
                                    'orderId': None,
                                    'orderStatus': 'rejected',
                                    'error': str(oe)
                                }
                            }
                        }
                        try:
                            strat.on_order_update(error_response_exit, symbol=symbol)
                        except Exception as callback_error:
                            bot_logger.error(f"Failed to notify strategy of exit order error: {callback_error}", exc_info=True)
                    except Exception as e_exit:
                        bot_logger.error(f"Unexpected error during strategy exit for {type(strat).__name__}: {e_exit}", exc_info=True)
                        error_response_exit = {
                            'exit_order': {
                                'result': {
                                    'orderId': None,
                                    'orderStatus': 'rejected',
                                    'error': f"Unexpected error: {str(e_exit)}"
                                }
                            }
                        }
                        try:
                            strat.on_order_update(error_response_exit, symbol=symbol)
                        except Exception as callback_error:
                            bot_logger.error(f"Failed to notify strategy of unexpected exit order error: {callback_error}", exc_info=True)
        
        # Periodic market analysis check (every 60 minutes)
        current_time = datetime.now()
        if current_time - last_market_check >= market_check_interval:
            bot_logger.info("="*60)
            bot_logger.info("RUNNING PERIODIC MARKET ANALYSIS CHECK")
            bot_logger.info("="*60)
            
            # Run silent market analysis for current symbol/timeframe
            current_market_type = run_silent_market_analysis(exchange, config, symbol, timeframe, bot_logger)
            
            if current_market_type:
                bot_logger.info(f"Current market type for {symbol} {timeframe}: {current_market_type}")
                
                # Check if market type still matches strategy
                is_compatible = check_strategy_market_compatibility(
                    primary_strategy_tags, current_market_type, symbol, timeframe, bot_logger
                )
                
                if not is_compatible:
                    bot_logger.warning(f"Market type mismatch detected! Strategy expects {primary_strategy_tags}, but market is {current_market_type}")
                    
                    # Run full market analysis for user display
                    full_analysis = run_market_analysis(exchange, config, bot_logger)
                    
                    # Prompt user for strategy reselection
                    new_strategy_name, should_restart = prompt_strategy_reselection(
                        full_analysis, symbol, timeframe, current_market_type, available_strategies, bot_logger
                    )
                    
                    if should_restart and new_strategy_name:
                        bot_logger.info(f"User selected new strategy: {new_strategy_name}. Restarting configuration...")
                        # Clean up current resources
                        if 'data_fetcher' in locals() and data_fetcher is not None:
                            data_fetcher.stop_websocket()
                        if 'perf_tracker' in locals() and perf_tracker is not None:
                            perf_tracker.close_session()
                        
                        # Instead of shutting down, restart the configuration process
                        # This will trigger a new configuration flow with the selected strategy
                        restart_configuration_with_new_strategy(
                            new_strategy_name, available_strategies, full_analysis, 
                            exchange, config, bot_logger
                        )
                        return  # Exit current main loop, new one will start
                    else:
                        bot_logger.info("Continuing with current strategy despite market change")
                else:
                    bot_logger.info(f"Market type compatibility confirmed: {current_market_type} matches strategy tags {primary_strategy_tags}")
            else:
                bot_logger.warning("Silent market analysis failed, skipping compatibility check")
            
            # Update last check time
            last_market_check = current_time
            bot_logger.info("="*60)
            bot_logger.info("PERIODIC MARKET ANALYSIS CHECK COMPLETED")
            bot_logger.info("="*60)
        
        # Brief pause to prevent excessive CPU usage and API rate limit issues
        bot_logger.debug("Main loop iteration ended. Pausing...")
        time.sleep(0.1)

def automatic_strategy_and_timeframe_selection(analysis_results: dict, selected_symbol: str, logger_instance: logging.Logger) -> tuple:
    """
    Automatically select the optimal strategy and execution timeframe based on market conditions using the Strategy Matrix.
    
    Args:
        analysis_results: Market analysis results dictionary
        selected_symbol: The selected trading symbol
        logger_instance: Logger instance
        
    Returns:
        tuple: (strategy_class_name, execution_timeframe, strategy_description, selection_reason)
    """
    logger_instance.info("="*60)
    logger_instance.info("AUTOMATIC STRATEGY AND TIMEFRAME SELECTION")
    logger_instance.info("="*60)
    
    # Initialize Strategy Matrix
    strategy_matrix = StrategyMatrix(logger_instance)
    
    # Get market conditions for the selected symbol
    symbol_analysis = analysis_results.get(selected_symbol, {})
    
    market_5min = symbol_analysis.get('5m', {}).get('market_type', 'UNKNOWN')
    market_1min = symbol_analysis.get('1m', {}).get('market_type', 'UNKNOWN')
    analysis_1h = symbol_analysis.get('1h', {})
    
    # Determine higher-timeframe directional bias from 1-hour analysis with multiple confirmations
    directional_bias = 'NEUTRAL'
    bias_strength = 'WEAK'
    
    if analysis_1h and 'analysis_details' in analysis_1h:
        analysis_details = analysis_1h['analysis_details']
        market_type_1h = analysis_1h.get('market_type', 'UNKNOWN')
        
        # Primary bias from trend direction
        if market_type_1h == 'TRENDING':
            trend_direction = analysis_details.get('trend_direction', 'NEUTRAL')
            directional_bias = trend_direction
            
            # Check trend strength for bias confidence
            adx_1h = analysis_details.get('adx', 0)
            if adx_1h > 35:
                bias_strength = 'STRONG'
            elif adx_1h > 25:
                bias_strength = 'MODERATE'
        
        # Additional confirmation from combined regime
        combined_regime = analysis_details.get('combined_regime', '')
        if 'trending_high_vol' in combined_regime or 'trending_low_vol' in combined_regime:
            if bias_strength == 'WEAK':
                bias_strength = 'MODERATE'
    
    # Apply short bias preference in uncertain conditions (leverage what's working)
    if directional_bias == 'NEUTRAL' and bias_strength == 'WEAK':
        # Check 5m trend for short bias hints
        analysis_5min = symbol_analysis.get('5m', {})
        if market_5min == 'TRENDING' and analysis_5min:
            trend_dir_5m = analysis_5min.get('analysis_details', {}).get('trend_direction', 'NEUTRAL')
            if trend_dir_5m == 'BEARISH':
                directional_bias = 'BEARISH_BIASED'  # Prefer shorts in uncertain conditions
                bias_strength = 'MODERATE'
    
    logger_instance.info(f"Directional Bias (1h): {directional_bias} ({bias_strength})")
    
    logger_instance.info(f"Market conditions for {selected_symbol}:")
    logger_instance.info(f"  5-minute timeframe: {market_5min}")
    logger_instance.info(f"  1-minute timeframe: {market_1min}")
    
    # Validate market conditions
    if not strategy_matrix.validate_market_conditions(market_5min, market_1min):
        logger_instance.error("Invalid market conditions detected. Cannot select strategy automatically.")
        return None, None, None, "Invalid market conditions", "NEUTRAL"
    
    # Select strategy and timeframe using the matrix with detailed analysis
    analysis_5min = symbol_analysis.get('5m', {})
    analysis_1min = symbol_analysis.get('1m', {})
    selected_strategy_class, execution_timeframe, selection_reason = strategy_matrix.select_strategy_and_timeframe(
        market_5min, market_1min, analysis_5min, analysis_1min, directional_bias
    )
    strategy_description = strategy_matrix.get_strategy_description(selected_strategy_class)
    
    logger_instance.info(f"Strategy Matrix Selection:")
    logger_instance.info(f"  Selected Strategy: {selected_strategy_class}")
    logger_instance.info(f"  Execution Timeframe: {execution_timeframe}")
    logger_instance.info(f"  Description: {strategy_description}")
    logger_instance.info(f"  Reason: {selection_reason}")
    
    # Display matrix summary for reference
    matrix_summary = strategy_matrix.get_matrix_summary()
    logger_instance.debug(f"Strategy Matrix:\n{matrix_summary}")
    
    logger_instance.info("="*60)
    
    return selected_strategy_class, execution_timeframe, strategy_description, selection_reason, directional_bias, bias_strength

def check_strategy_needs_change(analysis_results: dict, selected_symbol: str, current_strategy_class: str, current_timeframe: str, logger_instance: logging.Logger, session_start_time=None, last_market_conditions=None) -> tuple:
    """
    Check if the current strategy and timeframe are still optimal for current market conditions.
    Enhanced with persistence mechanisms to prevent excessive switching.
    
    Args:
        analysis_results: Current market analysis results
        selected_symbol: The trading symbol
        current_strategy_class: Current strategy class name
        current_timeframe: Current execution timeframe
        logger_instance: Logger instance
        session_start_time: When current strategy session started
        last_market_conditions: Previous market conditions for comparison
        
    Returns:
        tuple: (needs_change: bool, new_strategy_class: str, new_timeframe: str, reason: str)
    """
    logger_instance.info("Checking if strategy/timeframe change is needed based on current market conditions...")
    
    # Initialize Strategy Matrix
    strategy_matrix = StrategyMatrix(logger_instance)
    
    # Get current market conditions
    symbol_analysis = analysis_results.get(selected_symbol, {})
    market_5min = symbol_analysis.get('5m', {}).get('market_type', 'UNKNOWN')
    market_1min = symbol_analysis.get('1m', {}).get('market_type', 'UNKNOWN')
    
    # PERSISTENCE CHECK 1: Minimum session duration (90 minutes)
    if session_start_time:
        session_duration = (datetime.now() - session_start_time).total_seconds() / 60  # minutes
        min_session_duration = 90  # 90 minutes minimum
        
        if session_duration < min_session_duration:
            logger_instance.info(f"Strategy persistence: Session duration {session_duration:.1f}min < {min_session_duration}min minimum. Checking for opposite conditions only...")
            
            # Only allow immediate switch for truly opposite conditions
            current_5min = last_market_conditions.get('5m') if last_market_conditions else 'UNKNOWN'
            current_1min = last_market_conditions.get('1m') if last_market_conditions else 'UNKNOWN'
            
            # Define opposite condition pairs
            opposite_conditions = {
                'TRENDING': 'RANGING',
                'RANGING': 'TRENDING',
                'HIGH_VOLATILITY': 'LOW_VOLATILITY',
                'LOW_VOLATILITY': 'HIGH_VOLATILITY'
            }
            
            # Check if 5m condition is truly opposite (more weight on 5m)
            is_opposite_5m = (current_5min in opposite_conditions and 
                            opposite_conditions[current_5min] == market_5min)
            
            if not is_opposite_5m:
                reason = f"Strategy persistence: Maintaining {current_strategy_class} (session {session_duration:.1f}min < {min_session_duration}min). Market: {market_5min}(5m) + {market_1min}(1m) not opposite to previous {current_5min}(5m) + {current_1min}(1m)"
                logger_instance.info(reason)
                return False, current_strategy_class, current_timeframe, reason
            else:
                logger_instance.warning(f"Opposite condition detected: {current_5min}(5m) -> {market_5min}(5m). Allowing immediate switch despite short session.")
    
    # PERSISTENCE CHECK 2: Confirmation filter - weight 5m conditions more heavily
    if last_market_conditions:
        prev_5min = last_market_conditions.get('5m', 'UNKNOWN')
        prev_1min = last_market_conditions.get('1m', 'UNKNOWN')
        
        # If 5m condition hasn't changed, be more conservative about switching
        if prev_5min == market_5min and prev_5min != 'UNKNOWN':
            logger_instance.info(f"5m condition stable ({market_5min}). Requiring stronger 1m signal for strategy change...")
            
            # Only switch if 1m condition is significantly different or volatility extreme
            if market_1min not in ['HIGH_VOLATILITY', 'LOW_VOLATILITY'] and prev_1min != 'UNKNOWN':
                reason = f"Strategy persistence: 5m condition stable ({market_5min}), 1m change ({prev_1min}->{market_1min}) not extreme enough. Maintaining {current_strategy_class}"
                logger_instance.info(reason)
                return False, current_strategy_class, current_timeframe, reason
    
    # Get optimal strategy and timeframe for current conditions with detailed analysis
    analysis_5min = symbol_analysis.get('5m', {})
    analysis_1min = symbol_analysis.get('1m', {})
    optimal_strategy_class, optimal_timeframe, selection_reason = strategy_matrix.select_strategy_and_timeframe(
        market_5min, market_1min, analysis_5min, analysis_1min
    )
    
    if optimal_strategy_class != current_strategy_class or optimal_timeframe != current_timeframe:
        reason = f"Market conditions changed significantly. Optimal setup is now {optimal_strategy_class} on {optimal_timeframe} instead of {current_strategy_class} on {current_timeframe}. {selection_reason}"
        logger_instance.warning(reason)
        return True, optimal_strategy_class, optimal_timeframe, reason
    else:
        reason = f"Current setup {current_strategy_class} on {current_timeframe} is still optimal for market conditions {market_5min}(5m) + {market_1min}(1m)"
        logger_instance.info(reason)
        return False, current_strategy_class, current_timeframe, reason

def safe_float_convert(value, default=0.0):
    """
    Safely convert a value to float, handling empty strings and None values.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        float: Converted value or default
    """
    if value is None or value == '' or value == 'null':
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def calculate_pnl_from_prices(entry_price: float, exit_price: float, size: float, side: str) -> float:
    """
    Calculate PnL from entry/exit prices and position details.
    
    Args:
        entry_price: Entry price per unit
        exit_price: Exit price per unit  
        size: Position size
        side: Position side ('buy'/'long' or 'sell'/'short')
        
    Returns:
        PnL amount (positive for profit, negative for loss)
    """
    try:
        if side.lower() in ['buy', 'long']:
            # Long position: profit when exit > entry
            pnl = (exit_price - entry_price) * size
        else:  # short position
            # Short position: profit when entry > exit
            pnl = (entry_price - exit_price) * size
        
        return pnl
    except Exception:
        return 0.0

def sync_strategy_position_with_exchange(strategy, symbol, exchange, category, logger):
    """
    Enhanced position synchronization between strategy and exchange.
    Handles mismatches more intelligently by attempting reconciliation.
    
    Args:
        strategy: Strategy instance with position tracking
        symbol: Trading symbol
        exchange: Exchange connector
        category: Trading category
        logger: Logger instance
    
    Returns:
        Dict with sync results and actions taken
    """
    sync_result = {
        'status': 'success',
        'action': 'none',
        'exchange_size': 0,
        'strategy_size': 0,
        'mismatch_detected': False,
        'reconciled': False,
        'error': None
    }
    
    try:
        # Get actual position from exchange
        positions_response = exchange.fetch_positions(symbol, category)
        positions_list = positions_response.get('result', {}).get('list', [])
        
        # Find position for this symbol
        actual_position = None
        for pos in positions_list:
            if pos.get('symbol') == symbol.replace('/', '').upper():
                actual_position = pos
                break
        
        actual_size = safe_float_convert(actual_position.get('size', 0)) if actual_position else 0
        actual_side = actual_position.get('side', '').lower() if actual_position else None
        actual_entry_price = safe_float_convert(actual_position.get('avgPrice', 0)) if actual_position else 0
        unrealized_pnl = safe_float_convert(actual_position.get('unrealisedPnl', 0)) if actual_position else 0
        
        # Get strategy position
        strategy_position = strategy.position.get(symbol)
        strategy_size = safe_float_convert(strategy_position.get('size', 0)) if strategy_position else 0
        strategy_side = strategy_position.get('side', '').lower() if strategy_position else None
        
        sync_result['exchange_size'] = actual_size
        sync_result['strategy_size'] = strategy_size
        
        # Check for mismatches
        size_tolerance = 0.0001  # Allow small differences due to rounding (reduced for better detection)
        
        if abs(actual_size - strategy_size) > size_tolerance:
            sync_result['mismatch_detected'] = True
            
            if actual_size == 0 and strategy_size != 0:
                # Case 1: Exchange position closed, strategy still thinks it's open
                sync_result['action'] = 'clear_strategy_position'
                logger.warning(f"Position sync: {symbol} closed on exchange (likely SL/TP), clearing strategy position (size: {strategy_size})")
                
                # Record the trade closure in performance tracker if we have the position details
                if strategy_position and hasattr(strategy, 'logger'):
                    try:
                        # Estimate PnL based on last known position data
                        entry_price = strategy_position.get('entry_price', 0)
                        if entry_price > 0:
                            # This is an approximation since we don't know exact exit price
                            estimated_exit_price = entry_price  # Conservative estimate
                            logger.info(f"Recording estimated trade closure for position sync: {symbol}")
                    except Exception as e:
                        logger.debug(f"Could not estimate trade closure details: {e}")
                
                strategy.clear_position(symbol)
                sync_result['reconciled'] = True
                
                # *** CRITICAL FIX: Clean up remaining conditional orders when position auto-closed ***
                # When position closes via SL/TP, the other conditional order (TP/SL) remains orphaned
                # Signal the main trading loop to handle immediate cleanup
                logger.info(f"🧹 Position auto-closed detected for {symbol} - signaling conditional order cleanup...")
                sync_result['needs_conditional_cleanup'] = True
                logger.info(f"✅ Conditional order cleanup will be handled immediately by main loop for {symbol}")
                
            elif actual_size != 0 and strategy_size == 0:
                # Case 2: Exchange has position, strategy doesn't know about it
                sync_result['action'] = 'adopt_exchange_position'
                logger.warning(f"Position sync: Found untracked position on exchange for {symbol} (size: {actual_size}, side: {actual_side})")
                
                # Adopt the exchange position
                if not hasattr(strategy, 'position'):
                    strategy.position = {}
                if not hasattr(strategy, 'order_pending'):
                    strategy.order_pending = {}
                if not hasattr(strategy, 'active_order_id'):
                    strategy.active_order_id = {}
                    
                strategy.position[symbol] = {
                    'main_order_id': f'adopted_{int(time.time())}',
                    'symbol': symbol,
                    'side': actual_side,
                    'size': actual_size,
                    'entry_price': actual_entry_price,
                    'status': 'open',
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'adopted': True,  # Mark as adopted
                    'unrealized_pnl': unrealized_pnl
                }
                strategy.order_pending[symbol] = False
                strategy.active_order_id[symbol] = strategy.position[symbol]['main_order_id']
                
                logger.info(f"Adopted exchange position: {symbol} {actual_side} {actual_size} @ {actual_entry_price}")
                sync_result['reconciled'] = True
                
            elif actual_size != 0 and strategy_size != 0:
                # Case 3: Both have positions but sizes differ (partial fills, etc.)
                size_diff = abs(actual_size - strategy_size)
                size_diff_pct = (size_diff / max(actual_size, strategy_size)) * 100
                
                if size_diff_pct > 5:  # More than 5% difference is significant
                    sync_result['action'] = 'update_strategy_size'
                    logger.warning(f"Position sync: Size mismatch for {symbol} - Exchange: {actual_size}, Strategy: {strategy_size} ({size_diff_pct:.1f}% diff)")
                    
                    # Update strategy with exchange size
                    if strategy_position:
                        strategy.position[symbol]['size'] = actual_size
                        strategy.position[symbol]['entry_price'] = actual_entry_price  # Update if exchange has better data
                        logger.info(f"Updated strategy position size: {symbol} -> {actual_size}")
                        sync_result['reconciled'] = True
                else:
                    # Small difference, just log it
                    logger.debug(f"Position sync: Minor size difference for {symbol} - Exchange: {actual_size}, Strategy: {strategy_size}")
                    sync_result['action'] = 'minor_difference_ignored'
        else:
            # Positions are in sync
            if actual_size > 0:
                logger.debug(f"Position sync: {symbol} positions are synchronized (size: {actual_size})")
            else:
                logger.debug(f"Position sync: {symbol} no positions on either side")
            sync_result['action'] = 'already_synced'
        
        return sync_result
        
    except Exception as e:
        sync_result['status'] = 'error'
        sync_result['error'] = str(e)
        logger.error(f"Error during position sync for {symbol}: {e}")
        return sync_result

def run_trading_loop_with_auto_strategy(strategy_instance, current_strategy_class, symbol, timeframe, leverage, category, data_fetcher, order_manager, perf_tracker, exchange, config, analysis_results, bot_logger, session_manager, risk_manager, real_time_monitor, directional_bias: str, bias_strength: str = 'WEAK', strategy_matrix=None, config_loader=None, market_analyzer_enhanced=None):
    """
    Main trading loop with automatic strategy switching based on market conditions.
    Enhanced with comprehensive risk management integration.
    
    Args:
        strategy_instance: Current strategy instance
        current_strategy_class: Current strategy class name
        symbol: Trading symbol
        timeframe: Trading timeframe
        leverage: Trading leverage
        category: Trading category
        data_fetcher: Data fetcher instance
        order_manager: Order manager instance
        perf_tracker: Performance tracker instance
        exchange: Exchange connector instance
        config: Configuration dictionary
        analysis_results: Initial market analysis results
        bot_logger: Logger instance
    """
    bot_logger.info("="*60)
    bot_logger.info("STARTING TRADING LOOP WITH AUTOMATIC STRATEGY MANAGEMENT")
    bot_logger.info("="*60)
    
    # Initialize timing for strategy checks
    last_strategy_check = time.time()
    strategy_check_interval = 15 * 60  # 15 minutes in seconds
    
    current_strategy = strategy_instance
    current_strategy_name = current_strategy_class
    
    # Strategy persistence tracking
    session_start_time = datetime.now()
    last_market_conditions = None
    
    # Bias parameters already initialized before main loop
    
    bot_logger.info(f"Initial strategy: {current_strategy_name}")
    bot_logger.info(f"Initial directional bias: {directional_bias} ({bias_strength})")
    bot_logger.info(f"Strategy check interval: {strategy_check_interval / 60:.0f} minutes")
    
    while True:
        try:
            current_time = time.time()
            
            # Update data using the LiveDataFetcher interface
            try:
                latest_data = data_fetcher.update_data()
                
                # Update real-time monitor with current price for P&L calculation
                if not latest_data.empty and current_strategy.position.get(symbol):
                    current_price = float(latest_data.iloc[-1]['close'])
                    real_time_monitor.update_position_pnl(symbol, current_price)
                
                # Check if strategy data has indicators (avoid overwriting them)
                if current_strategy.data is not None and not current_strategy.data.empty:
                    # Get indicator columns that exist in strategy data but not in raw OHLCV
                    base_ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    indicator_cols = [col for col in current_strategy.data.columns if col not in base_ohlcv_cols]
                    
                    if indicator_cols:
                        # Strategy data has indicators - merge carefully to preserve them
                        bot_logger.debug(f"Preserving {len(indicator_cols)} indicator columns during data update")
                        
                        # Update OHLCV columns with latest data, keeping indicators
                        for col in base_ohlcv_cols:
                            if col in latest_data.columns:
                                current_strategy.data[col] = latest_data[col].copy()
                        
                        # Ensure data length matches (trim if necessary)
                        if len(current_strategy.data) != len(latest_data):
                            current_strategy.data = current_strategy.data.iloc[-len(latest_data):].reset_index(drop=True)
                        
                        bot_logger.debug(f"Updated strategy OHLCV data while preserving indicators. Rows: {len(current_strategy.data)}")
                    else:
                        # No indicators yet - safe to replace entirely
                        current_strategy.data = latest_data.copy()
                        bot_logger.debug(f"Updated strategy data (no indicators to preserve). Rows: {len(latest_data)}")
                else:
                    # No existing data - safe to replace entirely
                    current_strategy.data = latest_data.copy()
                    bot_logger.debug(f"Initialized strategy data. Rows: {len(latest_data)}")
                
                # Now update indicators for the new row
                current_strategy.update_indicators_for_new_row()
                
            except Exception as e:
                bot_logger.debug(f"Data update failed: {e}")
                # Continue with existing data
            
            # Check for strategy change every 15 minutes (only when no active orders)
            if current_time - last_strategy_check >= strategy_check_interval:
                bot_logger.info("="*60)
                bot_logger.info("PERIODIC STRATEGY EVALUATION CHECK")
                bot_logger.info("="*60)
                
                # Check if there are any active orders
                has_active_orders = False
                if hasattr(current_strategy, 'position') and current_strategy.position.get(symbol):
                    has_active_orders = True
                    bot_logger.info(f"Active position exists for {symbol}. Skipping strategy evaluation until position closes.")
                elif hasattr(current_strategy, 'order_pending') and current_strategy.order_pending.get(symbol, False):
                    has_active_orders = True
                    bot_logger.info(f"Order pending for {symbol}. Skipping strategy evaluation until order completes.")
                
                if not has_active_orders:
                    bot_logger.info("No active orders detected. Proceeding with strategy evaluation...")
                    
                    # Run silent market analysis to get current conditions for both timeframes
                    try:
                        # Strategy matrix needs both 1m and 5m analysis, so fetch both
                        analysis_1m = run_silent_market_analysis(exchange, config, symbol, '1m', bot_logger)
                        analysis_5m = run_silent_market_analysis(exchange, config, symbol, '5m', bot_logger)
                        
                        # Combine the results
                        current_analysis = {}
                        if analysis_1m and symbol in analysis_1m:
                            current_analysis[symbol] = analysis_1m[symbol]
                        if analysis_5m and symbol in analysis_5m:
                            if symbol not in current_analysis:
                                current_analysis[symbol] = {}
                            current_analysis[symbol].update(analysis_5m[symbol])
                        
                        if current_analysis and symbol in current_analysis:
                            # Store current market conditions for persistence checking
                            current_market_conditions = {
                                '5m': current_analysis[symbol].get('5m', {}).get('market_type', 'UNKNOWN'),
                                '1m': current_analysis[symbol].get('1m', {}).get('market_type', 'UNKNOWN')
                            }
                            
                            # Check if strategy/timeframe needs to change with persistence mechanisms
                            needs_change, new_strategy_class, new_timeframe, reason = check_strategy_needs_change(
                                current_analysis, symbol, current_strategy_name, timeframe, bot_logger, 
                                session_start_time, last_market_conditions
                            )
                            
                            if needs_change:
                                bot_logger.warning(f"Strategy/timeframe change needed: {reason}")
                                
                                try:
                                    # Check if timeframe changed - if so, need to restart data fetcher
                                    timeframe_changed = new_timeframe != timeframe
                                    
                                    if timeframe_changed:
                                        bot_logger.info(f"Timeframe changing from {timeframe} to {new_timeframe}. Restarting data fetcher...")
                                        # Stop current data fetcher
                                        data_fetcher.stop_websocket()
                                        # Create new data fetcher with new timeframe
                                        data_fetcher = LiveDataFetcher(exchange, symbol, new_timeframe, logger=bot_logger)
                                        latest_data = data_fetcher.fetch_initial_data()
                                        data_fetcher.start_websocket()
                                        bot_logger.info(f"Data fetcher restarted with {new_timeframe} timeframe")
                                        # Update timeframe variable
                                        timeframe = new_timeframe
                                    else:
                                        latest_data = data_fetcher.get_data()
                                    
                                    # Load new strategy
                                    new_strategy_module = convert_strategy_class_to_module_name(new_strategy_class)
                                    
                                    bot_logger.info(f"Loading new strategy module: {new_strategy_module}")
                                    NewStratClass = dynamic_import_strategy(new_strategy_module, StrategyTemplate, bot_logger)
                                    
                                    # Create new strategy instance
                                    new_strategy_logger = get_logger(new_strategy_class.lower())
                                    new_strategy_instance = NewStratClass(latest_data.copy(), config, logger=new_strategy_logger)
                                    
                                    bot_logger.info(f"Successfully switched from {current_strategy_name} to {new_strategy_class} on {new_timeframe}")
                                    
                                    # Update session for strategy change
                                    try:
                                        # End current session
                                        session_manager.end_active_sessions(f"Strategy change: {current_strategy_name} -> {new_strategy_class}")
                                        
                                        # Create new session with updated context
                                        market_context = {}
                                        if current_analysis and symbol in current_analysis:
                                            for tf, data in current_analysis[symbol].items():
                                                market_context[f"{tf}_market_type"] = data.get('market_type', 'UNKNOWN')
                                                market_context[f"{tf}_volatility"] = data.get('volatility', 'UNKNOWN')
                                        
                                        new_session_id = session_manager.create_session(
                                            strategy_name=new_strategy_class,
                                            symbol=symbol,
                                            timeframe=new_timeframe,
                                            leverage=leverage,
                                            market_conditions=market_context,
                                            configuration={
                                                'category': category,
                                                'strategy_params': get_strategy_parameters(new_strategy_class),
                                                'selection_reason': reason
                                            }
                                        )
                                        bot_logger.info(f"New session created: {new_session_id}")
                                        
                                    except Exception as session_error:
                                        bot_logger.error(f"Failed to update session for strategy change: {session_error}")
                                    
                                    # Update current strategy references
                                    current_strategy = new_strategy_instance
                                    current_strategy_name = new_strategy_class
                                    
                                    # Reset session start time for new strategy
                                    session_start_time = datetime.now()
                                    
                                    # Update real-time monitor with new strategy info
                                    if 'real_time_monitor' in locals():
                                        real_time_monitor.set_current_strategy(new_strategy_class)
                                        # Symbol remains the same, but update market context if needed
                                        if current_analysis and symbol in current_analysis:
                                            updated_market_summary = " | ".join([f"{tf}:{data.get('market_type', 'UNKNOWN')}" 
                                                                                for tf, data in current_analysis[symbol].items() 
                                                                                if data.get('market_type') != 'UNKNOWN'])
                                            real_time_monitor.set_current_market_conditions(updated_market_summary)
                                    
                                    # Log strategy change
                                    current_strategy.log_state_change(symbol, "awaiting_entry", 
                                        f"Strategy {new_strategy_class} on {new_timeframe} for {symbol}: Switched due to market condition change. Looking for entry conditions...")
                                    
                                except Exception as e:
                                    bot_logger.error(f"Failed to switch to new strategy {new_strategy_class} on {new_timeframe}: {e}")
                                    bot_logger.info("Continuing with current strategy and timeframe")
                            else:
                                bot_logger.info("Current strategy and timeframe remain optimal for market conditions")
                        else:
                            bot_logger.warning("Failed to get current market analysis for strategy evaluation")
                    except Exception as e:
                        bot_logger.error(f"Error during strategy evaluation: {e}")
                
                # Update last check time regardless of whether we could evaluate
                last_strategy_check = current_time
                
                # Update last market conditions if we successfully analyzed them
                if 'current_market_conditions' in locals():
                    last_market_conditions = current_market_conditions.copy()
                
                bot_logger.info("="*60)
                bot_logger.info("STRATEGY EVALUATION CHECK COMPLETED")
                bot_logger.info("="*60)
            
            # Regular trading logic - check for entry
            if not hasattr(current_strategy, 'position') or not current_strategy.position.get(symbol):
                entry_signal = current_strategy.check_entry(symbol)
                if entry_signal:
                    bot_logger.info(f"Entry signal detected by {current_strategy_name}: {entry_signal}")
                    
                    # Validate entry signal has required fields
                    required_fields = ['side', 'sl_pct', 'tp_pct']
                    missing_fields = [field for field in required_fields if entry_signal.get(field) is None]
                    
                    if missing_fields:
                        bot_logger.error(f"Entry signal missing required fields {missing_fields}: {entry_signal}")
                        continue  # Skip this signal
                    
                    # Handle missing or None size - calculate default size if needed
                    signal_size = entry_signal.get('size')
                    if signal_size is None:
                        # Calculate a default position size using risk manager
                        try:
                            position_sizing = risk_manager.calculate_position_size(
                                symbol=symbol,
                                strategy_name=current_strategy_name,
                                market_context={},
                                risk_pct=1.0  # Default 1% risk
                            )
                            signal_size = position_sizing.get('recommended_size', 0.01)  # Fallback to 0.01
                            bot_logger.info(f"Strategy provided size=None, calculated default size: {signal_size}")
                            entry_signal['size'] = signal_size
                        except Exception as size_calc_error:
                            bot_logger.error(f"Failed to calculate default position size: {size_calc_error}")
                            bot_logger.info("Using fallback size of 0.01")
                            signal_size = 0.01
                            entry_signal['size'] = signal_size
                    
                    # Advanced risk management pre-trade checks
                    try:
                        # Calculate stop loss and take profit prices for risk validation
                        entry_price = entry_signal.get('price') or current_strategy.data.iloc[-1]['close']
                        sl_pct = entry_signal.get('sl_pct', 3.0)
                        tp_pct = entry_signal.get('tp_pct', 6.0)
                        side = entry_signal.get('side')
                        
                        # Calculate stop loss and take profit prices
                        if side == 'buy':
                            stop_loss_price = entry_price * (1 - sl_pct / 100)
                            take_profit_price = entry_price * (1 + tp_pct / 100)
                        else:  # sell
                            stop_loss_price = entry_price * (1 + sl_pct / 100)
                            take_profit_price = entry_price * (1 - tp_pct / 100)
                        
                        # Pre-trade risk validation
                        risk_assessment = risk_manager.validate_trade_risk(
                            symbol=symbol,
                            side=side,
                            size=signal_size,  # Use validated size
                            entry_price=entry_price,
                            stop_loss_price=stop_loss_price,
                            take_profit_price=take_profit_price,
                            leverage=leverage
                        )
                        
                        if risk_assessment.get('approved', True):
                            # Risk approved - adjust size if needed
                            adjusted_size = risk_assessment.get('adjusted_size', entry_signal.get('size'))
                            if adjusted_size != entry_signal.get('size'):
                                bot_logger.info(f"Risk manager adjusted position size from {entry_signal.get('size')} to {adjusted_size}")
                                entry_signal['size'] = adjusted_size
                        else:
                            # Risk rejected
                            risk_reason = risk_assessment.get('reason', 'Unknown risk violation')
                            bot_logger.warning(f"Trade rejected by risk manager: {risk_reason}")
                            continue  # Skip this trade
                        
                    except Exception as e:
                        bot_logger.error(f"Risk assessment failed: {e}")
                        continue  # Skip trade on risk assessment failure
                    
                    try:
                        # Extract parameters from entry_signal for place_order_with_risk method
                        side = entry_signal.get('side')
                        order_type = entry_signal.get('type', 'market')  # Default to market if not specified
                        size = entry_signal.get('size')
                        signal_price = entry_signal.get('price')
                        sl_pct = entry_signal.get('sl_pct')
                        tp_pct = entry_signal.get('tp_pct')
                        
                        # Create params dict for additional parameters
                        params = {}
                        for key, value in entry_signal.items():
                            if key not in ['side', 'type', 'size', 'price', 'sl_pct', 'tp_pct']:
                                params[key] = value
                        
                        order_responses = order_manager.place_order_with_risk(
                            symbol=symbol,
                            side=side,
                            order_type=order_type,
                            size=size,
                            signal_price=signal_price,
                            sl_pct=sl_pct,
                            tp_pct=tp_pct,
                            params=params if params else None
                        )
                        current_strategy.on_order_update(order_responses, symbol)
                        bot_logger.info(f"Orders placed successfully for {symbol}")
                    except OrderExecutionError as e:
                        bot_logger.error(f"Order execution failed: {e}")
                        current_strategy.order_pending[symbol] = False  # Reset pending state
                        current_strategy.active_order_id[symbol] = None
            
            # Check for and cancel orphaned conditional orders
            try:
                order_manager.check_and_cancel_orphaned_conditional_orders(symbol, category=category)
            except Exception as e_orphan_check:
                bot_logger.error(f"Error during orphaned order cleanup: {e_orphan_check}", exc_info=True)
            
            # Check for exit if position exists
            if hasattr(current_strategy, 'position') and current_strategy.position.get(symbol):
                # ENHANCED POSITION SYNC: Use the new intelligent sync function
                try:
                    # Capture a snapshot of the current strategy position *before* sync – used if the
                    # sync detects that the position disappeared on-exchange (SL/TP fill not routed
                    # through OrderManager).
                    import copy
                    prev_position_snapshot = copy.deepcopy(current_strategy.position.get(symbol, {}))

                    sync_result = sync_strategy_position_with_exchange(
                        current_strategy, symbol, exchange, category, bot_logger
                    )

                    # If exchange shows the position is gone but the strategy had it, we assume a
                    # protective SL/TP closed the trade. Record it now so PerformanceTracker &
                    # dashboard are up-to-date.
                    if (sync_result.get('action') == 'clear_strategy_position' and
                        prev_position_snapshot and prev_position_snapshot.get('size')):

                        entry_price   = safe_float_convert(prev_position_snapshot.get('entry_price', 0))
                        exit_price    = safe_float_convert(prev_position_snapshot.get('exit_price', entry_price))  # Fallback
                        size          = safe_float_convert(prev_position_snapshot.get('size', 0))
                        side          = prev_position_snapshot.get('side', '')

                        # PnL estimation – if we don't have exit_price, use 0 so metrics at least
                        # reflect a closed trade; more precise calculation can be added later.
                        if exit_price == entry_price:
                            calculated_pnl = 0.0
                        else:
                            if side.lower() == 'buy':
                                calculated_pnl = (exit_price - entry_price) * size
                            else:
                                calculated_pnl = (entry_price - exit_price) * size

                        trade_summary = {
                            'strategy': current_strategy_name,
                            'symbol': symbol,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'size': size,
                            'side': side,
                            'pnl': calculated_pnl,
                            'entry_timestamp': prev_position_snapshot.get('timestamp', datetime.now(timezone.utc).isoformat()),
                            'exit_timestamp': datetime.now(timezone.utc).isoformat(),
                            'exit_reason': 'sl_tp_auto_close'
                        }
                        perf_tracker.record_trade(trade_summary)
                        bot_logger.info(f"Trade recorded via position-sync for {current_strategy_name}: PnL=${calculated_pnl:.2f}")

                        # Update dashboard immediately
                        if real_time_monitor:
                            real_time_monitor.update_metrics(force_update=True)
                            bot_logger.debug("Dashboard updated after auto-closure trade recording")
                 
                    # Log sync results for analysis
                    if sync_result['mismatch_detected']:
                        bot_logger.info(f"Position sync completed for {symbol}: {sync_result['action']}")
                        bot_logger.info(f"Position sync details: {sync_result}")

                    if sync_result['action'] == 'clear_strategy_position':
                        # *** CRITICAL FIX: Handle conditional order cleanup when position auto-closed ***
                        if sync_result.get('needs_conditional_cleanup'):
                            bot_logger.info(f"🧹 Executing immediate conditional order cleanup for auto-closed position: {symbol}")
                            try:
                                order_manager._cancel_existing_conditional_orders(symbol, category=category)
                                bot_logger.info(f"✅ Successfully cleaned up remaining conditional orders for {symbol}")
                            except Exception as cleanup_error:
                                bot_logger.error(f"❌ Failed to clean up conditional orders for {symbol}: {cleanup_error}")
                                # Still continue - don't let cleanup failure stop the bot
                        continue
                    elif sync_result['action'] == 'adopt_exchange_position':
                        bot_logger.info("Position was adopted from exchange")
                    elif sync_result['action'] == 'update_strategy_size':
                        bot_logger.info(
                            f"Position size updated on exchange: {sync_result['exchange_size']} -> {sync_result['strategy_size']}"
                        )
                    elif sync_result['action'] == 'minor_difference_ignored':
                        bot_logger.debug("Position size difference within tolerance, no action taken")
                    elif sync_result['action'] == 'already_synced':
                        bot_logger.debug("Positions are properly synchronized between strategy and exchange")
                    else:
                        bot_logger.warning(f"Unexpected position sync action detected: '{sync_result['action']}'. "
                                         f"Full sync result: {sync_result}")

                    # If position was cleared or significantly changed, skip exit check
                    if sync_result['action'] in ['clear_strategy_position']:
                        continue  # Skip exit check since position was cleared
                    elif sync_result['action'] == 'adopt_exchange_position':
                        bot_logger.info(f"Adopted exchange position for {symbol}, will monitor for exits")

                # Continue with exit check only if we still have a position
                    if not current_strategy.position.get(symbol):
                         continue  # Position was cleared during sync
                     
                except Exception as e:
                    bot_logger.warning(f"Position sync failed for {symbol}: {e}. Proceeding with exit check.")
                
                # Check minimum hold time before allowing exits
                if not order_manager.can_close_position(symbol):
                    bot_logger.debug(f"Position {symbol} in minimum hold period - skipping exit check")
                    continue
                
                exit_signal = current_strategy.check_exit(symbol)
                if exit_signal:
                    bot_logger.info(f"Exit signal detected by {current_strategy_name}: {exit_signal}")
                    try:
                        # Get the position details from the strategy
                        position_to_close = current_strategy.position.get(symbol)
                        if position_to_close:
                            exit_order_responses = order_manager.execute_strategy_exit(symbol, position_to_close, category=category)
                            
                            # Check if position was already closed on exchange
                            if exit_order_responses.get('position_already_closed'):
                                bot_logger.info(f"✅ Position for {symbol} was already closed on exchange (likely SL/TP). Clearing strategy position.")
                                current_strategy.clear_position(symbol)
                                
                                # Record the trade closure for already-closed positions
                                try:
                                    entry_price = position_to_close.get('entry_price', 0)
                                    size = position_to_close.get('size', 0)
                                    side = position_to_close.get('side', 'unknown')
                                    
                                    # Since position was closed by SL/TP, we don't know exact exit price
                                    # Use entry price as conservative estimate or try to get current price
                                    try:
                                        current_price = exchange.get_current_price(symbol.replace('/', ''))
                                        exit_price = current_price if current_price else entry_price
                                    except:
                                        exit_price = entry_price
                                    
                                    # Calculate PnL based on estimated exit price
                                    calculated_pnl = calculate_pnl_from_prices(entry_price, exit_price, size, side)
                                    
                                    bot_logger.info(f"Recording trade for already-closed position: {symbol} {side} {size} @ entry=${entry_price:.2f}, est_exit=${exit_price:.2f}, PnL=${calculated_pnl:.2f}")
                                    
                                    # Create trade record
                                    trade_record = create_enhanced_trade_record(
                                        strategy_name=current_strategy_name,
                                        symbol=symbol,
                                        side=side,
                                        entry_price=entry_price,
                                        exit_price=exit_price,
                                        size=size,
                                        pnl=calculated_pnl,
                                        entry_timestamp=position_to_close.get('timestamp', datetime.now(timezone.utc).isoformat()),
                                        exit_timestamp=datetime.now(timezone.utc).isoformat(),
                                        exit_reason="Position already closed by SL/TP",
                                        leverage=leverage,
                                        session_id=session_manager.get_current_session_id() if session_manager else None
                                    )
                                    
                                    perf_tracker.record_trade(trade_record)
                                    bot_logger.info(f"✅ Trade recorded for already-closed position: PnL=${calculated_pnl:.2f}")
                                    
                                except Exception as record_error:
                                    bot_logger.error(f"Failed to record trade for already-closed position: {record_error}")
                                
                                continue  # Skip normal exit processing
                            
                            # Record trade to performance tracker after successful exit
                            # Note: execute_strategy_exit returns {'exit_market_order': response}
                            exit_market_order = exit_order_responses.get('exit_market_order', {})
                            exit_result = exit_market_order.get('result', {})
                            
                            if exit_order_responses and exit_result.get('orderStatus', '').lower() == 'filled':
                                # Calculate PnL based on entry vs exit price and position size
                                entry_price = safe_float_convert(position_to_close.get('entry_price', 0))
                                exit_price = safe_float_convert(exit_result.get('avgPrice', 0))
                                size = safe_float_convert(position_to_close.get('size', 0))
                                side = position_to_close.get('side', '')
                                
                                # Calculate PnL: (exit_price - entry_price) * size for buy, (entry_price - exit_price) * size for sell
                                if side.lower() == 'buy':
                                    calculated_pnl = (exit_price - entry_price) * size
                                else:  # sell
                                    calculated_pnl = (entry_price - exit_price) * size
                                
                                # CREATE ENHANCED TRADE RECORD with full context
                                enhanced_trade_record = create_enhanced_trade_record(
                                    strategy_name=current_strategy_name,
                                    symbol=symbol,
                                    side=side,
                                    entry_price=entry_price,
                                    exit_price=exit_price,
                                    size=size,
                                    pnl=calculated_pnl,
                                    entry_timestamp=position_to_close.get('timestamp', datetime.now(timezone.utc).isoformat()),
                                    exit_timestamp=datetime.now(timezone.utc).isoformat(),
                                    market_analysis_data=analysis_results if 'analysis_results' in locals() else None,
                                    order_response_data=exit_order_responses,
                                    position_data=position_to_close,
                                    leverage=leverage if 'leverage' in locals() else None,
                                    session_id=session_manager.get_active_session_id() if session_manager else None,
                                    exit_reason='strategy_exit_signal',
                                    sl_pct=position_to_close.get('planned_sl_pct'),
                                    tp_pct=position_to_close.get('planned_tp_pct')
                                )
                                
                                perf_tracker.record_trade(enhanced_trade_record)
                                bot_logger.info(f"✅ Enhanced trade recorded for {current_strategy_name}: PnL=${calculated_pnl:.2f}, Entry=${entry_price}, Exit=${exit_price}")
                                
                                # Immediately update real-time monitor to reflect the new trade
                                if 'real_time_monitor' in locals() and real_time_monitor:
                                    real_time_monitor.update_metrics(force_update=True)
                                    bot_logger.debug("Triggered immediate dashboard update after trade recording")
                            else:
                                bot_logger.warning(f"Exit order not filled or missing data - trade not recorded. Status: {exit_result.get('orderStatus', 'UNKNOWN')}")
                                bot_logger.debug(f"Exit order response structure: {exit_order_responses}")
                            
                            # Update strategy on successful exit
                            if exit_order_responses:
                                current_strategy.clear_position(symbol)
                                bot_logger.info(f"Position closed for {symbol}")
                        else:
                            bot_logger.error(f"Position data not found for {symbol} during exit")
                    except OrderExecutionError as e:
                        bot_logger.error(f"Exit order execution failed: {e}")
            
            # Brief pause to prevent excessive CPU usage
            time.sleep(0.1)
            
        except KeyboardInterrupt:
            bot_logger.info("Trading loop interrupted by user")
            break
        except Exception as e:
            bot_logger.error(f"Error in trading loop: {e}", exc_info=True)
            time.sleep(1)  # Pause before retrying

# Enhanced Performance Tracker Integration Functions
def create_enhanced_trade_record(
    strategy_name: str,
    symbol: str,
    side: str,
    entry_price: float,
    exit_price: float,
    size: float,
    pnl: float,
    entry_timestamp: str,
    exit_timestamp: str,
    market_analysis_data: Optional[Dict] = None,
    order_response_data: Optional[Dict] = None,
    position_data: Optional[Dict] = None,
    leverage: Optional[float] = None,
    session_id: Optional[str] = None,
    exit_reason: Optional[str] = None,
    sl_pct: Optional[float] = None,
    tp_pct: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create an enhanced trade record with comprehensive market context, order details, and risk metrics.
    
    Args:
        strategy_name: Name of the trading strategy
        symbol: Trading symbol
        side: Trade side (buy/sell)
        entry_price: Entry execution price
        exit_price: Exit execution price
        size: Position size
        pnl: Realized profit/loss
        entry_timestamp: Trade entry timestamp
        exit_timestamp: Trade exit timestamp
        market_analysis_data: Market analysis results for context
        order_response_data: Order execution response data
        position_data: Position tracking data
        leverage: Leverage used
        session_id: Current session ID
        exit_reason: Reason for trade exit
        sl_pct: Stop loss percentage used
        tp_pct: Take profit percentage used
        
    Returns:
        Enhanced trade record dictionary compatible with PerformanceTracker
    """
    try:
        # Generate unique trade ID
        trade_id = f"trade_{int(datetime.now().timestamp())}_{symbol}_{side}"
        
        # Create Market Context
        market_context = None
        if market_analysis_data and symbol in market_analysis_data:
            symbol_data = market_analysis_data[symbol]
            market_5m = symbol_data.get('5m', {}).get('market_type', 'UNKNOWN')
            market_1m = symbol_data.get('1m', {}).get('market_type', 'UNKNOWN')
            
            # Determine strategy selection reason
            from modules.strategy_matrix import StrategyMatrix
            strategy_matrix = StrategyMatrix()
            _, _, selection_reason = strategy_matrix.select_strategy_and_timeframe(market_5m, market_1m)
            
            # Determine execution timeframe from strategy matrix
            execution_timeframe = '5m' if (market_5m == 'TRENDING' and market_1m == 'TRENDING') else '1m'
            
            # Determine market session based on current time
            current_hour = datetime.now().hour
            if 0 <= current_hour < 8:
                market_session = "Asia"
            elif 8 <= current_hour < 16:
                market_session = "Europe"
            else:
                market_session = "US"
            
            market_context = MarketContext(
                market_5m=market_5m,
                market_1m=market_1m,
                strategy_selection_reason=selection_reason,
                execution_timeframe=execution_timeframe,
                volatility_regime=symbol_data.get('5m', {}).get('analysis_details', {}).get('volatility_regime'),
                trend_strength=symbol_data.get('5m', {}).get('analysis_details', {}).get('trend_strength'),
                market_session=market_session
            )
        
        # Create Order Details
        order_details = None
        if order_response_data:
            main_order = order_response_data.get('main_order', {}).get('result', {})
            sl_order = order_response_data.get('sl_order', {}).get('result', {})
            tp_order = order_response_data.get('tp_order', {}).get('result', {})
            
            # Calculate slippage if we have expected vs actual price
            slippage_pct = None
            if order_response_data.get('signal_price') and entry_price:
                expected_price = float(order_response_data['signal_price'])
                if expected_price > 0:
                    slippage_pct = abs(entry_price - expected_price) / expected_price * 100
            
            order_details = OrderDetails(
                main_order_id=main_order.get('orderId'),
                sl_order_id=sl_order.get('orderId'),
                tp_order_id=tp_order.get('orderId'),
                retry_attempts=order_response_data.get('retry_attempts', 0),
                slippage_pct=slippage_pct,
                order_type=order_response_data.get('order_type', 'market'),
                time_in_force=order_response_data.get('time_in_force', 'GoodTillCancel'),
                reduce_only=order_response_data.get('reduce_only', False)
            )
        
        # Create Risk Metrics
        risk_metrics = None
        if entry_price > 0:
            # Calculate actual SL/TP percentages
            actual_sl_pct = None
            actual_tp_pct = None
            
            if exit_price > 0:
                if side.lower() == 'buy':
                    if exit_price < entry_price:  # Loss (SL triggered)
                        actual_sl_pct = (entry_price - exit_price) / entry_price * 100
                    else:  # Profit (TP triggered or manual exit)
                        actual_tp_pct = (exit_price - entry_price) / entry_price * 100
                else:  # sell
                    if exit_price > entry_price:  # Loss (SL triggered)
                        actual_sl_pct = (exit_price - entry_price) / entry_price * 100
                    else:  # Profit (TP triggered or manual exit)
                        actual_tp_pct = (entry_price - exit_price) / entry_price * 100
            
            # Calculate risk-reward ratio
            risk_reward_ratio = None
            if sl_pct and tp_pct:
                risk_reward_ratio = tp_pct / sl_pct
            
            # Calculate position size percentage (if we have account info)
            position_size_pct = None
            if position_data and position_data.get('account_balance'):
                position_value = size * entry_price
                position_size_pct = (position_value / position_data['account_balance']) * 100
            
            risk_metrics = RiskMetrics(
                planned_sl_pct=sl_pct,
                actual_sl_pct=actual_sl_pct,
                planned_tp_pct=tp_pct,
                actual_tp_pct=actual_tp_pct,
                risk_reward_ratio=risk_reward_ratio,
                position_size_pct=position_size_pct,
                leverage_used=leverage
            )
        
        # Calculate trade duration
        trade_duration_seconds = None
        if entry_timestamp and exit_timestamp:
            try:
                entry_dt = pd.to_datetime(entry_timestamp)
                exit_dt = pd.to_datetime(exit_timestamp)
                trade_duration_seconds = (exit_dt - entry_dt).total_seconds()
            except Exception as e:
                # Log the error but don't fail the trade recording process
                logger = logging.getLogger('bot.trade_tracking')
                logger.warning(f"Failed to calculate trade duration for trade {trade_id}: {e}. "
                             f"Entry: {entry_timestamp}, Exit: {exit_timestamp}")
                trade_duration_seconds = None
        
        # Calculate return percentage
        return_pct = None
        if entry_price > 0:
            return_pct = (pnl / (size * entry_price)) * 100
        
        # Create enhanced trade record
        enhanced_trade = {
            'trade_id': trade_id,
            'strategy': strategy_name,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'entry_timestamp': entry_timestamp,
            'exit_timestamp': exit_timestamp,
            'market_context': market_context,
            'order_details': order_details,
            'risk_metrics': risk_metrics,
            'status': TradeStatus.FILLED,
            'exit_reason': exit_reason,
            'trade_duration_seconds': trade_duration_seconds,
            'return_pct': return_pct,
            'session_id': session_id
        }
        
        return enhanced_trade
        
    except Exception as e:
        # Fallback to basic format if enhancement fails
        return {
            'strategy': strategy_name,
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'entry_timestamp': entry_timestamp,
            'exit_timestamp': exit_timestamp,
            'exit_reason': exit_reason
        }

def start_trade_tracking(
    trade_id: str,
    strategy_name: str,
    symbol: str,
    side: str,
    size: float,
    entry_price: float,
    perf_tracker: 'PerformanceTracker',
    session_id: Optional[str] = None,
    market_analysis_data: Optional[Dict] = None,
    order_response_data: Optional[Dict] = None
) -> str:
    """
    Start tracking a trade in the performance tracker with PENDING status.
    
    Args:
        trade_id: Unique trade identifier
        strategy_name: Name of the trading strategy
        symbol: Trading symbol
        side: Trade side (buy/sell)
        size: Position size
        entry_price: Entry execution price
        perf_tracker: PerformanceTracker instance
        session_id: Current session ID
        market_analysis_data: Market analysis results
        order_response_data: Order execution response data
        
    Returns:
        Trade ID for tracking
    """
    try:
        # Create initial trade record with PENDING status
        initial_trade = create_enhanced_trade_record(
            strategy_name=strategy_name,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            exit_price=0.0,  # Will be updated on exit
            size=size,
            pnl=0.0,  # Will be calculated on exit
            entry_timestamp=datetime.now(timezone.utc).isoformat(),
            exit_timestamp=None,
            market_analysis_data=market_analysis_data,
            order_response_data=order_response_data,
            session_id=session_id
        )
        
        # Override status to PENDING
        initial_trade['status'] = TradeStatus.PENDING
        
        # Record the trade
        actual_trade_id = perf_tracker.record_trade(initial_trade)
        return actual_trade_id
        
    except Exception as e:
        # Return the provided trade_id if tracking setup fails
        return trade_id

def update_trade_status(
    trade_id: str,
    status: TradeStatus,
    perf_tracker: 'PerformanceTracker',
    notes: Optional[str] = None
):
    """Update trade status in the performance tracker."""
    try:
        perf_tracker.update_trade_status(trade_id, status, notes)
    except Exception as e:
        # Log error but don't fail the trading process
        logger = logging.getLogger('bot.trade_tracking')
        logger.error(f"Failed to update trade status for trade {trade_id} to {status}: {e}. "
                    f"Notes: {notes}. Trade tracking may be incomplete but trading will continue.")
        # Optionally, we could try to persist the status update to a backup file
        try:
            backup_dir = "logs/trade_status_failures"
            os.makedirs(backup_dir, exist_ok=True)
            
            failure_record = {
                'timestamp': datetime.now().isoformat(),
                'trade_id': trade_id,
                'intended_status': str(status),
                'notes': notes,
                'error': str(e)
            }
            
            backup_file = os.path.join(backup_dir, f"failed_status_updates_{datetime.now().strftime('%Y%m%d')}.json")
            
            # Append to daily backup file
            existing_data = []
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, 'r') as f:
                        existing_data = json.load(f)
                except:
                    existing_data = []
            
            existing_data.append(failure_record)
            
            with open(backup_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
                
            logger.info(f"Trade status update failure backed up to {backup_file}")
        except Exception as backup_error:
            logger.debug(f"Failed to backup trade status update failure: {backup_error}")

def complete_trade_tracking(
    trade_id: str,
    exit_price: float,
    pnl: float,
    perf_tracker: 'PerformanceTracker',
    exit_reason: Optional[str] = None,
    order_response_data: Optional[Dict] = None
):
    """
    Complete trade tracking by updating the existing trade record with exit details.
    
    Args:
        trade_id: Trade ID to update
        exit_price: Exit execution price
        pnl: Realized profit/loss
        perf_tracker: PerformanceTracker instance
        exit_reason: Reason for trade exit
        order_response_data: Exit order response data
    """
    try:
        # Get the existing trade record
        trade_record = perf_tracker.get_trade_by_id(trade_id)
        if not trade_record:
            return
        
        # Update exit details
        trade_record.exit_price = exit_price
        trade_record.pnl = pnl
        trade_record.exit_timestamp = datetime.now(timezone.utc).isoformat()
        trade_record.status = TradeStatus.FILLED
        trade_record.exit_reason = exit_reason
        
        # Calculate trade duration
        if trade_record.entry_timestamp:
            try:
                entry_dt = pd.to_datetime(trade_record.entry_timestamp)
                exit_dt = pd.to_datetime(trade_record.exit_timestamp)
                trade_record.trade_duration_seconds = (exit_dt - entry_dt).total_seconds()
            except Exception as e:
                # Log error but continue with trade record completion
                logger = logging.getLogger('bot.trade_tracking')
                logger.warning(f"Failed to calculate trade duration for trade {trade_id} during completion: {e}. "
                             f"Entry: {trade_record.entry_timestamp}, Exit: {trade_record.exit_timestamp}")
                trade_record.trade_duration_seconds = None
        
        # Calculate return percentage
        if trade_record.entry_price > 0:
            trade_record.return_pct = (pnl / (trade_record.size * trade_record.entry_price)) * 100
        
        # Update risk metrics with actual values
        if trade_record.risk_metrics and exit_price > 0:
            if trade_record.side.lower() == 'buy':
                if exit_price < trade_record.entry_price:  # Loss
                    trade_record.risk_metrics.actual_sl_pct = (trade_record.entry_price - exit_price) / trade_record.entry_price * 100
                else:  # Profit
                    trade_record.risk_metrics.actual_tp_pct = (exit_price - trade_record.entry_price) / trade_record.entry_price * 100
            else:  # sell
                if exit_price > trade_record.entry_price:  # Loss
                    trade_record.risk_metrics.actual_sl_pct = (exit_price - trade_record.entry_price) / trade_record.entry_price * 100
                else:  # Profit
                    trade_record.risk_metrics.actual_tp_pct = (trade_record.entry_price - exit_price) / trade_record.entry_price * 100
        
        # Update order details if provided
        if order_response_data and trade_record.order_details:
            exit_order = order_response_data.get('exit_order', {}).get('result', {})
            # Could add exit order details here if needed
        
        perf_tracker.logger.info(f"✅ TRADE COMPLETED {trade_id}: {trade_record.strategy} {trade_record.side} "
                               f"{trade_record.symbol} PnL: ${pnl:.2f} Duration: {trade_record.trade_duration_seconds:.1f}s")
        
    except Exception as e:
        perf_tracker.logger.error(f"Failed to complete trade tracking for {trade_id}: {e}")

def get_performance_analytics(perf_tracker: 'PerformanceTracker') -> Dict[str, Any]:
    """
    Get comprehensive performance analytics using the enhanced tracker capabilities.
    
    Args:
        perf_tracker: PerformanceTracker instance
        
    Returns:
        Dictionary with comprehensive performance analytics
    """
    try:
        analytics = {
            'comprehensive_stats': perf_tracker.get_comprehensive_statistics(),
            'strategy_performance': perf_tracker.get_strategy_performance(),
            'risk_metrics_summary': perf_tracker.get_risk_metrics_summary(),
            'market_context_performance': perf_tracker.get_market_context_performance()
        }
        
        # Add advanced analytics if enough trades
        if len(perf_tracker.trades) >= 10:
            analytics['rolling_sharpe'] = perf_tracker.rolling_sharpe(window=10)
            analytics['rolling_drawdown'] = perf_tracker.rolling_drawdown_curve(window=10)
        
        return analytics
        
    except Exception as e:
        perf_tracker.logger.error(f"Failed to get performance analytics: {e}")
        return {}

def check_existing_orders_and_positions(exchange, config, logger, symbol=None):
    """
    Check for existing orders and positions for a specific symbol (or all if None) and ask user what to do with them.
    
    Args:
        exchange: Exchange connector instance
        config: Bot configuration
        logger: Logger instance  
        symbol: Specific symbol to check (e.g., 'SOLUSDT'). If None, checks all symbols.
        
    Returns:
        Dict with recovered orders/positions info or None if user chooses to ignore
    """
    symbol_text = f" FOR {symbol}" if symbol else ""
    logger.info("="*60)
    logger.info(f"CHECKING FOR EXISTING ORDERS AND POSITIONS{symbol_text}")
    logger.info("="*60)
    
    try:
        # Get all open orders
        open_orders_response = exchange.fetch_all_open_orders()
        open_orders = []
        
        if isinstance(open_orders_response, dict) and 'result' in open_orders_response:
            open_orders = open_orders_response['result'].get('list', [])
        elif isinstance(open_orders_response, dict) and 'list' in open_orders_response:
            open_orders = open_orders_response.get('list', [])
        elif isinstance(open_orders_response, list):
            open_orders = open_orders_response
        
        # Get all positions
        positions_response = exchange.fetch_all_positions()
        positions = []
        
        if isinstance(positions_response, dict) and 'result' in positions_response:
            positions = positions_response['result'].get('list', [])
        elif isinstance(positions_response, dict) and 'list' in positions_response:
            positions = positions_response.get('list', [])
        elif isinstance(positions_response, list):
            positions = positions_response
        
        # Filter for specific symbol if provided
        if symbol:
            target_symbol = symbol.replace("/", "").upper()  # Normalize symbol (SOLUSDT)
            open_orders = [order for order in open_orders if order.get('symbol', '').upper() == target_symbol]
            positions = [pos for pos in positions if pos.get('symbol', '').upper() == target_symbol]
        
        # Filter for non-zero positions
        active_positions = [pos for pos in positions if float(pos.get('size', 0)) != 0]
        
        # Filter for relevant open orders (not just SL/TP orphaned ones)
        main_orders = []
        conditional_orders = []
        
        for order in open_orders:
            order_type = order.get('orderType', '').lower()
            stop_order_type = order.get('stopOrderType', '')
            
            if stop_order_type in ['Stop', 'TakeProfit', 'StopLoss']:
                conditional_orders.append(order)  
            else:
                main_orders.append(order)
        
        # Check if there's anything to recover
        if not main_orders and not active_positions:
            symbol_msg = f" for {symbol}" if symbol else ""
            logger.info(f"✅ No existing orders or positions found{symbol_msg}. Starting fresh.")
            return None
        
        # Display findings
        symbol_msg = f" for {symbol}" if symbol else ""
        logger.info(f"🔍 Found existing trading activity{symbol_msg}:")
        if main_orders:
            logger.info(f"   📋 {len(main_orders)} open orders")
        if active_positions:
            logger.info(f"   📊 {len(active_positions)} active positions")  
        if conditional_orders:
            logger.info(f"   ⚠️  {len(conditional_orders)} conditional orders (SL/TP)")
        
        symbol_header = f" FOR {symbol}" if symbol else ""
        print(f"\n" + "="*70)
        print(f"🚨 EXISTING TRADING ACTIVITY DETECTED{symbol_header}")
        print("="*70)
        
        if main_orders:
            print("\n📋 OPEN ORDERS:")
            for i, order in enumerate(main_orders):
                symbol = order.get('symbol', 'N/A')
                side = order.get('side', 'N/A').upper()
                size = order.get('qty', 'N/A')
                price = order.get('price', 'N/A')
                order_type = order.get('orderType', 'N/A')
                order_id = order.get('orderId', 'N/A')
                
                print(f"   {i+1}. {symbol} - {side} {size} @ ${price} ({order_type}) [ID: {order_id[:8]}...]")
        
        if active_positions:
            print("\n📊 ACTIVE POSITIONS:")
            for i, pos in enumerate(active_positions):
                symbol = pos.get('symbol', 'N/A')
                side = pos.get('side', 'N/A').upper()
                size = pos.get('size', 'N/A')
                entry_price = pos.get('avgPrice', 'N/A')
                unrealized_pnl = float(pos.get('unrealisedPnl', 0))
                
                pnl_display = f"${unrealized_pnl:,.2f}" if unrealized_pnl != 0 else "$0.00"
                pnl_color = "📈" if unrealized_pnl > 0 else "📉" if unrealized_pnl < 0 else "➡️"
                
                print(f"   {i+1}. {symbol} - {side} {size} @ ${entry_price} | P&L: {pnl_color} {pnl_display}")
        
        print("\n" + "="*70)
        print("RECOVERY OPTIONS:")
        print("="*70)
        print("1. 🔄 ADOPT & TRACK - Import existing orders/positions into bot for monitoring")
        print("2. 🚫 IGNORE & CONTINUE - Start fresh (existing orders/positions remain but untracked)")
        print("3. ❌ CANCEL & CLEAN - Cancel all orders, close positions, then start fresh")
        print("4. 🛑 EXIT - Stop bot startup to handle manually")
        print("="*70)
        
        while True:
            choice = input("\nSelect option (1-4): ").strip()
            logger.info(f"User recovery choice: '{choice}'")
            
            if choice == "1":
                logger.info("User chose to adopt and track existing orders/positions")
                return {
                    'action': 'adopt',
                    'main_orders': main_orders,
                    'positions': active_positions,
                    'conditional_orders': conditional_orders
                }
            elif choice == "2":
                logger.info("User chose to ignore existing orders/positions")
                print("⚠️  Note: Existing orders and positions will remain active but won't be tracked by the bot.")
                print("   You can manage them manually through the exchange interface.")
                return None
            elif choice == "3":
                logger.info("User chose to cancel and clean existing orders/positions")
                return {
                    'action': 'cancel_and_clean',
                    'main_orders': main_orders,
                    'positions': active_positions,
                    'conditional_orders': conditional_orders
                }
            elif choice == "4":
                logger.info("User chose to exit bot startup")
                print("👋 Bot startup cancelled. Handle existing orders/positions manually and restart when ready.")
                return {'action': 'exit'}
            else:
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
    
    except Exception as e:
        logger.error(f"Error checking existing orders/positions: {e}")
        logger.warning("Continuing with normal startup...")
        return None

def handle_recovery_action(recovery_info, exchange, config, logger):
    """Handle the user's choice for recovering existing orders/positions."""
    if not recovery_info:
        return True  # Continue normal startup
    
    action = recovery_info.get('action')
    
    if action == 'exit':
        logger.info("Exiting bot as requested by user")
        return False  # Stop startup
    
    elif action == 'cancel_and_clean':
        logger.info("Cancelling all orders and closing positions...")
        
        try:
            # Cancel all open orders
            for order in recovery_info.get('main_orders', []) + recovery_info.get('conditional_orders', []):
                try:
                    symbol = order.get('symbol')
                    order_id = order.get('orderId')
                    logger.info(f"Cancelling order {order_id} for {symbol}")
                    
                    cancel_response = exchange.cancel_order(symbol=symbol, order_id=order_id, params={'category': 'linear'})
                    logger.info(f"✅ Cancelled order {order_id}")
                except Exception as e:
                    logger.warning(f"Failed to cancel order {order_id}: {e}")
            
            # Close all positions  
            for position in recovery_info.get('positions', []):
                try:
                    symbol = position.get('symbol')
                    side = position.get('side')
                    size = float(position.get('size', 0))
                    
                    if size > 0:
                        # Place market order to close position
                        close_side = 'Sell' if side.lower() == 'buy' else 'Buy'
                        logger.info(f"Closing position {symbol} {side} {size}")
                        
                        close_response = exchange.place_order(
                            symbol=symbol,
                            side=close_side,
                            orderType='Market',
                            qty=str(size),
                            params={'category': 'linear', 'reduceOnly': True}
                        )
                        logger.info(f"✅ Closed position {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to close position {symbol}: {e}")
            
            logger.info("✅ Cleanup completed. Starting fresh.")
            return True  # Continue normal startup
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return True  # Continue despite errors
    
    elif action == 'adopt':
        logger.info("Adoption selected - will integrate existing orders/positions into strategy")
        # The actual adoption logic will be handled in the strategy initialization
        return True
    
    return True

def main():
    global session_manager, real_time_monitor, trailing_tp_handler
    
    config = load_config()
    # Initialize the main bot logger
    bot_logger = get_logger('bot') # Renamed to bot_logger for clarity
    bot_logger.info('Bot starting up.')
    
    # Exchange
    ex_cfg = config['bybit']
    exchange = ExchangeConnector(api_key=ex_cfg['api_key'], api_secret=ex_cfg['api_secret'], testnet=False, logger=bot_logger)
    
    # Note: Order detection moved to after symbol selection for better UX
    
    # Initialize core management modules
    bot_logger.info("="*60)
    bot_logger.info("INITIALIZING CORE MODULES")
    bot_logger.info("="*60)
    
    # Initialize PerformanceTracker first (required by other modules)
    perf_tracker = PerformanceTracker(logger=bot_logger)
    bot_logger.info("✅ PerformanceTracker initialized")
    
    # Initialize SessionManager
    session_manager = SessionManager(
        base_dir="sessions",
        logger=bot_logger
    )
    bot_logger.info("✅ SessionManager initialized")
    
    # Initialize AdvancedRiskManager
    risk_manager = AdvancedRiskManager(
        exchange=exchange,
        performance_tracker=perf_tracker,
        session_manager=session_manager,
        strategy_matrix=StrategyMatrix(bot_logger),
        logger=bot_logger
    )
    bot_logger.info("✅ AdvancedRiskManager initialized")
    
    # Initialize Strategy Matrix and Config Loader
    strategy_matrix = StrategyMatrix(logger=bot_logger)
    config_loader = StrategyConfigLoader(strategy_matrix=strategy_matrix, logger=bot_logger)
    bot_logger.info("✅ Strategy Matrix and Config Loader initialized")
    
    # Initialize MarketAnalyzer for enhanced risk management
    market_analyzer = MarketAnalyzer(exchange, config, bot_logger)
    bot_logger.info("✅ MarketAnalyzer initialized for enhanced risk management")
    
    # Run dry-run demonstration of enhanced order validation
    bot_logger.info("="*60)
    bot_logger.info("RUNNING ENHANCED RISK MANAGEMENT DRY-RUN DEMONSTRATION")
    bot_logger.info("="*60)
    
    try:
        dry_run_enhanced_order_validation_demo(
            logger=bot_logger,
            strategy_matrix=strategy_matrix,
            config_loader=config_loader,
            market_analyzer=market_analyzer,
            risk_manager=risk_manager,
            exchange=exchange
        )
    except Exception as demo_error:
        bot_logger.warning(f"Dry-run demonstration failed: {demo_error}")
        bot_logger.info("Continuing with normal bot startup...")
    
    bot_logger.info("="*60)
    
    # Initialize Trailing TP Handler
    # Ensure state directory exists
    import os
    os.makedirs("state", exist_ok=True)
    
    trailing_tp_handler = TrailingTPHandler(
        exchange_connector=exchange,
        order_manager=None,  # Will be set later when OrderManager is created
        logger=bot_logger,
        state_file="state/trailing_tp_state.json",
        monitoring_interval=5.0
    )
    bot_logger.info("✅ TrailingTPHandler initialized")
    
    # Initialize RealTimeMonitor (but don't start yet - will start after user selections)
    real_time_monitor = RealTimeMonitor(
        performance_tracker=perf_tracker,
        session_manager=session_manager,
        logger=bot_logger
    )
    bot_logger.info("✅ RealTimeMonitor initialized (will start after setup)")
    
    # RUN MARKET ANALYSIS FIRST - before strategy selection
    bot_logger.info("="*60)
    bot_logger.info("RUNNING STARTUP MARKET ANALYSIS")
    bot_logger.info("="*60)
    
    analysis_results = run_market_analysis(exchange, config, bot_logger)
    
    if analysis_results:
        bot_logger.info("Market analysis completed successfully")
    else:
        bot_logger.warning("Market analysis failed or returned no results")
    
    bot_logger.info("="*60)
    bot_logger.info("CONTINUING WITH STRATEGY SETUP")
    bot_logger.info("="*60)
    
    # User selects symbol and timeframe from analyzed markets (NO MANUAL STRATEGY SELECTION)
    if not analysis_results:
        bot_logger.error("No market analysis results available for symbol selection. Bot will exit.")
        return
    
    bot_logger.info("="*60)
    bot_logger.info("SYMBOL AND LEVERAGE SELECTION")
    bot_logger.info("="*60)
    
    # Let user select symbol from analyzed markets
    selected_symbol = select_symbol(analysis_results, bot_logger)
    
    # CONFIGURE LOGGING FOR THIS TRADING SESSION
    bot_logger.info("="*60)
    bot_logger.info("CONFIGURING SESSION LOGGING")
    bot_logger.info("="*60)
    
    # Configure logging with symbol and current date
    configure_logging_session(selected_symbol)
    
    # Log session information
    bot_logger.info(f"✅ Logging configured for trading session")
    bot_logger.info(f"   Symbol: {selected_symbol}")
    bot_logger.info(f"   Date: {datetime.now().strftime('%Y-%m-%d')}")
    bot_logger.info(f"   Session ID: {selected_symbol}_{datetime.now().strftime('%Y%m%d')}")
    
    # CHECK FOR EXISTING ORDERS/POSITIONS FOR THE SELECTED SYMBOL
    recovery_info = check_existing_orders_and_positions(exchange, config, bot_logger, selected_symbol)
    
    # Handle user's recovery choice  
    if not handle_recovery_action(recovery_info, exchange, config, bot_logger):
        bot_logger.info("Bot startup cancelled by user")
        return
    
    # Let user select leverage
    selected_leverage = select_leverage(bot_logger)
    
    # AUTOMATIC STRATEGY AND TIMEFRAME SELECTION based on market conditions
    selected_strategy_class, selected_timeframe, strategy_description, selection_reason, directional_bias, bias_strength = automatic_strategy_and_timeframe_selection(
        analysis_results, selected_symbol, bot_logger
    )
    
    if not selected_strategy_class or not selected_timeframe:
        bot_logger.error("Automatic strategy and timeframe selection failed. Bot will exit.")
        return
    
    bot_logger.info(f"Automatically selected strategy: {selected_strategy_class}")
    bot_logger.info(f"Automatically selected timeframe: {selected_timeframe}")
    bot_logger.info(f"Strategy description: {strategy_description}")
    
    # Load the selected strategy class
    try:
        # Convert class name to module name using helper function
        strategy_module_name = convert_strategy_class_to_module_name(selected_strategy_class)
        
        bot_logger.info(f"Attempting to load strategy module: {strategy_module_name}")
        StratClass = dynamic_import_strategy(strategy_module_name, StrategyTemplate, bot_logger)
        bot_logger.info(f"Successfully loaded strategy class: {StratClass.__name__}")
    except Exception as e:
        bot_logger.error(f"Failed to load automatically selected strategy {selected_strategy_class}: {e}")
        bot_logger.error("Falling back to StrategyBreakoutAndRetest with its intended timeframe as a working strategy")
        try:
            StratClass = dynamic_import_strategy('breakout_and_retest_strategy', StrategyTemplate, bot_logger)
            selected_strategy_class = 'StrategyBreakoutAndRetest'
            # Use the strategy's intended timeframe from risk profile
            fallback_matrix = StrategyMatrix(bot_logger)
            fallback_profile = fallback_matrix.get_strategy_risk_profile('StrategyBreakoutAndRetest')
            selected_timeframe = fallback_profile.execution_timeframe if fallback_profile else '1m'
            bot_logger.info(f"Successfully loaded fallback strategy: {StratClass.__name__} on {selected_timeframe}")
        except Exception as fallback_error:
            bot_logger.error(f"Even fallback strategy failed to load: {fallback_error}")
            return
    
    # Get strategy-specific parameters (category only, leverage now user-selected)
    strategy_params = get_strategy_parameters(StratClass.__name__)
    
    # Use user selections and automatically selected strategy/timeframe
    symbol = selected_symbol
    timeframe = selected_timeframe  # Automatically selected timeframe
    leverage = selected_leverage  # Use user-selected leverage
    category = strategy_params['category']
    coin_pair = symbol.replace('USDT', '/USDT')  # Convert format for display
    
    bot_logger.info(f"Final trading parameters: {coin_pair} ({symbol}), {timeframe}, {leverage}x leverage")
    bot_logger.info(f"Selected strategy: {selected_strategy_class}")
    bot_logger.info(f"Selected timeframe: {timeframe}")
    
    # Set leverage on the exchange for the selected symbol
    try:
        bot_logger.info(f"Setting leverage to {leverage}x for {symbol}")
        leverage_response = exchange.set_leverage(symbol, leverage, category)
        
        # Check if leverage was bypassed due to timestamp issues
        if leverage_response.get('result', {}).get('bypass_reason') == 'timestamp_sync_failure':
            current_leverage = leverage_response.get('result', {}).get('current_leverage', 'unknown')
            bot_logger.warning(f"⚠️  Leverage setting bypassed due to timestamp sync issues")
            bot_logger.info(f"📊 Current leverage for {symbol}: {current_leverage}x")
            bot_logger.info(f"🚀 Bot will continue trading with existing leverage settings")
        else:
            bot_logger.info(f"✅ Successfully set leverage to {leverage}x for {symbol}")
    except Exception as e:
        bot_logger.error(f"Failed to set leverage to {leverage}x for {symbol}: {e}")
        bot_logger.error("Bot will continue, but orders may fail if current leverage is insufficient")
    
    bot_logger.info("="*60)
    
    # Data fetcher with strategy-determined parameters
    data_fetcher = LiveDataFetcher(exchange, symbol, timeframe, logger=bot_logger)
    data = data_fetcher.fetch_initial_data()
    # Start WebSocket for live data
    data_fetcher.start_websocket()
    bot_logger.info(f"Fetched initial OHLCV data: {len(data)} rows for {symbol} {timeframe}")
    
    # Order manager (enhanced with risk management)
    order_manager = OrderManager(exchange, logger=bot_logger)
    # Inject risk manager into order manager
    order_manager.risk_manager = risk_manager
    
    # Wire TrailingTPHandler to OrderManager and start monitoring
    if trailing_tp_handler:
        trailing_tp_handler.order_manager = order_manager
        trailing_tp_handler.start_monitoring()
        bot_logger.info("✅ TrailingTPHandler wired to OrderManager and monitoring started")

    # Initialize the selected strategy instance
    try:
        # Create strategy-specific logger
        strategy_logger = get_logger(selected_strategy_class.lower())
        strategy_instance = StratClass(data.copy(), config, logger=strategy_logger)
        bot_logger.info(f"Successfully initialized strategy: {type(strategy_instance).__name__}")
        
        # HANDLE ADOPTION OF EXISTING ORDERS/POSITIONS
        if recovery_info and recovery_info.get('action') == 'adopt':
            bot_logger.info("="*60)
            bot_logger.info("ADOPTING EXISTING ORDERS AND POSITIONS")
            bot_logger.info("="*60)
            
            try:
                adopted_count = 0
                
                # Adopt existing positions
                for position in recovery_info.get('positions', []):
                    symbol_to_adopt = position.get('symbol')
                    
                    # Only adopt positions for the selected symbol
                    if symbol_to_adopt == symbol.replace('/', ''):
                        side = position.get('side', '').lower()
                        size = float(position.get('size', 0))
                        entry_price = float(position.get('avgPrice', 0))
                        unrealized_pnl = float(position.get('unrealisedPnl', 0))
                        
                        # Initialize strategy position tracking if not exists
                        if not hasattr(strategy_instance, 'position'):
                            strategy_instance.position = {}
                        if not hasattr(strategy_instance, 'order_pending'):
                            strategy_instance.order_pending = {}
                        if not hasattr(strategy_instance, 'active_order_id'):
                            strategy_instance.active_order_id = {}
                        
                        # Create adopted position record
                        strategy_instance.position[symbol] = {
                            'main_order_id': f'adopted_{int(time.time())}',
                            'symbol': symbol,
                            'side': side,
                            'size': size,
                            'entry_price': entry_price,
                            'status': 'open',
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'adopted': True,
                            'unrealized_pnl': unrealized_pnl,
                            'exchange_position_data': position  # Store original exchange data
                        }
                        strategy_instance.order_pending[symbol] = False
                        strategy_instance.active_order_id[symbol] = strategy_instance.position[symbol]['main_order_id']
                        
                        bot_logger.info(f"✅ Adopted position: {symbol} {side.upper()} {size} @ ${entry_price} (P&L: ${unrealized_pnl:.2f})")
                        adopted_count += 1
                
                # Adopt pending orders (for the selected symbol only)
                for order in recovery_info.get('main_orders', []):
                    order_symbol = order.get('symbol')
                    
                    if order_symbol == symbol.replace('/', ''):
                        # For now, just log the pending orders - full order adoption could be complex
                        # depending on the strategy's order management logic
                        side = order.get('side', '').upper()
                        size = order.get('qty', 'N/A')
                        price = order.get('price', 'N/A')
                        order_type = order.get('orderType', 'N/A')
                        order_id = order.get('orderId', 'N/A')
                        
                        bot_logger.info(f"ℹ️  Detected pending order: {order_symbol} {side} {size} @ ${price} ({order_type})")
                        bot_logger.info(f"   Order ID: {order_id} - Will be monitored but not directly managed by strategy")
                
                if adopted_count > 0:
                    bot_logger.info(f"✅ Successfully adopted {adopted_count} position(s) for tracking")
                    # Log the adoption in strategy state
                    strategy_instance.log_state_change(symbol, "position_adopted", 
                        f"Strategy {type(strategy_instance).__name__} for {symbol}: Adopted existing position on restart. Monitoring for exit conditions...")
                else:
                    bot_logger.info("ℹ️  No positions adopted (none matched the selected trading symbol)")
                    
            except Exception as adoption_error:
                bot_logger.error(f"Error during position adoption: {adoption_error}")
                bot_logger.info("Continuing with normal strategy initialization...")
        
    except Exception as e:
        bot_logger.error(f"Failed to initialize strategy class {StratClass.__name__}: {e}", exc_info=True)
        if 'data_fetcher' in locals() and data_fetcher is not None:
            data_fetcher.stop_websocket()
        bot_logger.info('Bot session closed due to strategy initialization failure.')
        return

    # Create trading session
    bot_logger.info("="*60)
    bot_logger.info("CREATING TRADING SESSION")
    bot_logger.info("="*60)
    
    # Prepare market context for session
    market_context = {}
    if analysis_results and symbol in analysis_results:
        for tf, data in analysis_results[symbol].items():
            market_context[f"{tf}_market_type"] = data.get('market_type', 'UNKNOWN')
            market_context[f"{tf}_volatility"] = data.get('volatility', 'UNKNOWN')
    
    # Create session
    session_id = session_manager.create_session(
        strategy_name=selected_strategy_class,
        symbol=symbol,
        timeframe=timeframe,
        leverage=leverage,
        market_conditions=market_context,
        configuration={
            'category': category,
            'strategy_params': strategy_params,
            'selection_reason': selection_reason if 'selection_reason' in locals() else 'Automatic selection'
        }
    )
    
    bot_logger.info(f"✅ Trading session created: {session_id}")
    bot_logger.info(f"Strategy: {selected_strategy_class}")
    bot_logger.info(f"Symbol: {symbol} | Timeframe: {timeframe} | Leverage: {leverage}x")
    bot_logger.info(f"Market Context: {market_context}")

    # NOW start real-time monitoring (after all setup is complete)
    bot_logger.info("="*60)
    bot_logger.info("STARTING REAL-TIME MONITORING")
    bot_logger.info("="*60)
    
    # Set strategy, symbol, and market context for dashboard
    real_time_monitor.set_current_strategy(selected_strategy_class)
    real_time_monitor.set_current_symbol(symbol)  # This should display as "Symbol: BTCUSDT" in dashboard
    market_summary = " | ".join([f"{tf}:{ctx}" for tf, ctx in market_context.items() if ctx != 'UNKNOWN'])
    if market_summary:
        real_time_monitor.set_current_market_conditions(market_summary)
    
    # Log dashboard setup for verification
    bot_logger.info(f"📊 Dashboard configured:")
    bot_logger.info(f"   Strategy: {selected_strategy_class}")
    bot_logger.info(f"   Symbol: {symbol}")
    bot_logger.info(f"   Market conditions: {market_summary if market_summary else 'Not set'}")
    
    # Register the strategy for position tracking
    real_time_monitor.add_strategy_for_tracking(strategy_instance)
    bot_logger.info("✅ Strategy registered with RealTimeMonitor for position tracking")
    
    real_time_monitor.start_monitoring()
    bot_logger.info("✅ RealTimeMonitor started - Live dashboard active")
    bot_logger.info("📊 Dashboard will show updates every 30 seconds (faster during active trading)")

    # Initial state logging
    strategy_instance.log_state_change(symbol, "awaiting_entry", f"Strategy {type(strategy_instance).__name__} for {symbol}: Initialized. Looking for new entry conditions...")

    # Main trading loop with automatic strategy switching
    try:
        # Start trading loop with automatic strategy management
        run_trading_loop_with_auto_strategy(
            strategy_instance, selected_strategy_class, symbol, timeframe, leverage, category,
            data_fetcher, order_manager, perf_tracker, exchange, config, analysis_results, bot_logger,
            session_manager, risk_manager, real_time_monitor, directional_bias, bias_strength, strategy_matrix, config_loader, market_analyzer
        )
    except KeyboardInterrupt:
        bot_logger.info('Bot shutting down (KeyboardInterrupt).')
    except Exception as exc:
        bot_logger.error(f'Bot crashed: {exc}', exc_info=True) # Added exc_info=True
        PerformanceTracker.persist_on_exception(perf_tracker)
        raise 
    finally:
        # Cleanup all modules
        try:
            if 'real_time_monitor' in locals() and real_time_monitor is not None:
                real_time_monitor.stop_monitoring()
                bot_logger.info("✅ RealTimeMonitor stopped")
                
            # Export session data before ending sessions (backup export mechanism)
            if 'session_manager' in locals() and session_manager is not None:
                try:
                    active_sessions = session_manager.get_active_sessions()
                    if active_sessions:
                        bot_logger.info("🔄 Exporting session data before final cleanup...")
                        
                        # Export active sessions in both formats
                        active_session_ids = list(active_sessions.keys())
                        
                        json_file = session_manager.export_session_data(active_session_ids, format='json')
                        bot_logger.info(f"✅ Final session export (JSON): {json_file}")
                        
                        csv_file = session_manager.export_session_data(active_session_ids, format='csv')
                        bot_logger.info(f"✅ Final session export (CSV): {csv_file}")
                        
                except Exception as export_error:
                    bot_logger.warning(f"⚠️  Session export failed during cleanup: {export_error}")
                
                # End sessions after export attempt
                session_manager.end_active_sessions("Bot shutdown")
                bot_logger.info("✅ Active sessions ended")
                
            if 'trailing_tp_handler' in locals() and trailing_tp_handler is not None:
                trailing_tp_handler.stop_monitoring()
                bot_logger.info("✅ TrailingTPHandler monitoring stopped")
                
            if 'data_fetcher' in locals() and data_fetcher is not None:
                data_fetcher.stop_websocket()
                bot_logger.info("✅ WebSocket data feed stopped")
                
            if 'perf_tracker' in locals() and perf_tracker is not None:
                perf_tracker.close_session()
                bot_logger.info("✅ Performance tracker session closed")
            
            # Close all logging handlers gracefully
            bot_logger.info("🔚 Closing all logging handlers...")
            close_all_loggers()
            print("✅ All logging handlers closed")  # Use print since loggers are closed
                
        except Exception as cleanup_error:
            bot_logger.error(f"Error during cleanup: {cleanup_error}")
            
        bot_logger.info('Bot session closed.')

if __name__ == '__main__':
    main() 