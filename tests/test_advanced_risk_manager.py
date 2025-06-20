import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.advanced_risk_manager import (
    AdvancedRiskManager, RiskLimits, PositionRisk, PortfolioRisk,
    RiskLevel, RiskViolationType
)
from modules.performance_tracker import PerformanceTracker, TradeRecord

class TestAdvancedRiskManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_exchange = Mock()
        self.mock_performance_tracker = Mock(spec=PerformanceTracker)
        self.mock_logger = Mock()
        
        # Create risk limits for testing
        self.test_risk_limits = RiskLimits(
            max_daily_loss_pct=5.0,
            max_drawdown_pct=10.0,
            max_position_size_pct=10.0,
            max_total_exposure_pct=30.0,
            max_leverage=25.0,
            max_consecutive_losses=5,
            min_risk_reward_ratio=1.5
        )
        
        self.risk_manager = AdvancedRiskManager(
            exchange=self.mock_exchange,
            performance_tracker=self.mock_performance_tracker,
            logger=self.mock_logger,
            risk_limits=self.test_risk_limits
        )
        
        # Mock account info
        self.mock_account_info = {
            'total_balance': 10000.0,
            'available_balance': 9000.0,
            'unrealized_pnl': 0.0
        }

    def test_initialization(self):
        """Test AdvancedRiskManager initialization"""
        self.assertIsNotNone(self.risk_manager)
        self.assertEqual(self.risk_manager.risk_limits.max_daily_loss_pct, 5.0)
        self.assertEqual(self.risk_manager.risk_limits.max_leverage, 25.0)
        self.assertFalse(self.risk_manager.emergency_stop_active)
        self.assertEqual(len(self.risk_manager.positions), 0)

    def test_risk_limits_dataclass(self):
        """Test RiskLimits dataclass"""
        limits = RiskLimits()
        self.assertEqual(limits.max_daily_loss_pct, 5.0)
        self.assertEqual(limits.max_drawdown_pct, 10.0)
        self.assertEqual(limits.max_position_size_pct, 10.0)
        
        # Test custom limits
        custom_limits = RiskLimits(max_daily_loss_pct=3.0, max_leverage=10.0)
        self.assertEqual(custom_limits.max_daily_loss_pct, 3.0)
        self.assertEqual(custom_limits.max_leverage, 10.0)

    @patch.object(AdvancedRiskManager, '_get_account_info')
    @patch.object(AdvancedRiskManager, '_get_current_price')
    def test_calculate_position_size_basic(self, mock_get_price, mock_get_account):
        """Test basic position size calculation"""
        mock_get_account.return_value = self.mock_account_info
        mock_get_price.return_value = 50000.0
        
        market_context = {
            'market_5m': 'TRENDING',
            'market_1m': 'TRENDING'
        }
        
        result = self.risk_manager.calculate_position_size(
            symbol='BTCUSDT',
            strategy_name='test_strategy',
            market_context=market_context,
            risk_pct=2.0
        )
        
        self.assertIn('recommended_size', result)
        self.assertIn('risk_amount', result)
        self.assertIn('risk_pct', result)
        self.assertGreater(result['risk_amount'], 0)
        self.assertEqual(result['current_price'], 50000.0)
        self.assertEqual(result['account_balance'], 10000.0)

    @patch.object(AdvancedRiskManager, '_get_account_info')
    def test_calculate_position_size_high_volatility_adjustment(self, mock_get_account):
        """Test position size adjustment for high volatility"""
        mock_get_account.return_value = self.mock_account_info
        
        with patch.object(self.risk_manager, '_get_current_price', return_value=50000.0):
            market_context = {
                'market_5m': 'HIGH_VOLATILITY',
                'market_1m': 'TRENDING'
            }
            
            result = self.risk_manager.calculate_position_size(
                symbol='BTCUSDT',
                strategy_name='test_strategy',
                market_context=market_context,
                risk_pct=2.0
            )
            
            # Should have market adjustment of 0.7 for high volatility
            self.assertEqual(result['market_adjustment'], 0.7)

    @patch.object(AdvancedRiskManager, '_get_account_info')
    def test_calculate_position_size_fallback(self, mock_get_account):
        """Test position size calculation fallback when account info fails"""
        mock_get_account.return_value = None
        
        market_context = {'market_5m': 'TRENDING', 'market_1m': 'TRENDING'}
        
        result = self.risk_manager.calculate_position_size(
            symbol='BTCUSDT',
            strategy_name='test_strategy',
            market_context=market_context,
            risk_pct=2.0
        )
        
        self.assertIn('error', result)
        self.assertEqual(result['recommended_size'], 0.01)

    @patch.object(AdvancedRiskManager, '_get_account_info')
    def test_validate_trade_risk_approved(self, mock_get_account):
        """Test trade risk validation that should be approved"""
        mock_get_account.return_value = self.mock_account_info
        
        with patch.object(self.risk_manager, '_get_daily_pnl', return_value=0.0), \
             patch.object(self.risk_manager, '_get_consecutive_losses', return_value=0), \
             patch.object(self.risk_manager, '_get_total_exposure', return_value=1000.0):
            
            result = self.risk_manager.validate_trade_risk(
                symbol='BTCUSDT',
                side='long',
                size=0.1,
                entry_price=50000.0,
                stop_loss_price=49000.0,
                take_profit_price=52000.0,
                leverage=10.0
            )
            
            self.assertTrue(result['approved'])
            self.assertEqual(len(result['violations']), 0)
            self.assertLess(result['risk_score'], 50.0)

    @patch.object(AdvancedRiskManager, '_get_account_info')
    def test_validate_trade_risk_position_size_violation(self, mock_get_account):
        """Test trade risk validation with position size violation"""
        mock_get_account.return_value = self.mock_account_info
        
        with patch.object(self.risk_manager, '_get_daily_pnl', return_value=0.0), \
             patch.object(self.risk_manager, '_get_consecutive_losses', return_value=0), \
             patch.object(self.risk_manager, '_get_total_exposure', return_value=1000.0):
            
            # Large position size that exceeds limit
            result = self.risk_manager.validate_trade_risk(
                symbol='BTCUSDT',
                side='long',
                size=5.0,  # Large size
                entry_price=50000.0,
                leverage=1.0
            )
            
            self.assertFalse(result['approved'])
            self.assertGreater(len(result['violations']), 0)
            self.assertEqual(result['violations'][0]['type'], RiskViolationType.POSITION_SIZE_LIMIT.value)

    @patch.object(AdvancedRiskManager, '_get_account_info')
    def test_validate_trade_risk_leverage_violation(self, mock_get_account):
        """Test trade risk validation with leverage violation"""
        mock_get_account.return_value = self.mock_account_info
        
        with patch.object(self.risk_manager, '_get_daily_pnl', return_value=0.0), \
             patch.object(self.risk_manager, '_get_consecutive_losses', return_value=0), \
             patch.object(self.risk_manager, '_get_total_exposure', return_value=1000.0):
            
            result = self.risk_manager.validate_trade_risk(
                symbol='BTCUSDT',
                side='long',
                size=0.1,
                entry_price=50000.0,
                leverage=50.0  # Exceeds max leverage of 25
            )
            
            self.assertFalse(result['approved'])
            violations = [v for v in result['violations'] if v['type'] == RiskViolationType.LEVERAGE_LIMIT.value]
            self.assertGreater(len(violations), 0)

    @patch.object(AdvancedRiskManager, '_get_account_info')
    def test_validate_trade_risk_daily_loss_violation(self, mock_get_account):
        """Test trade risk validation with daily loss violation"""
        mock_get_account.return_value = self.mock_account_info
        
        with patch.object(self.risk_manager, '_get_daily_pnl', return_value=-600.0), \
             patch.object(self.risk_manager, '_get_consecutive_losses', return_value=0), \
             patch.object(self.risk_manager, '_get_total_exposure', return_value=1000.0):
            
            result = self.risk_manager.validate_trade_risk(
                symbol='BTCUSDT',
                side='long',
                size=0.1,
                entry_price=50000.0,
                leverage=10.0
            )
            
            self.assertFalse(result['approved'])
            violations = [v for v in result['violations'] if v['type'] == RiskViolationType.DAILY_LOSS_LIMIT.value]
            self.assertGreater(len(violations), 0)

    @patch.object(AdvancedRiskManager, '_get_account_info')
    def test_validate_trade_risk_consecutive_losses_violation(self, mock_get_account):
        """Test trade risk validation with consecutive losses violation"""
        mock_get_account.return_value = self.mock_account_info
        
        with patch.object(self.risk_manager, '_get_daily_pnl', return_value=0.0), \
             patch.object(self.risk_manager, '_get_consecutive_losses', return_value=5), \
             patch.object(self.risk_manager, '_get_total_exposure', return_value=1000.0):
            
            result = self.risk_manager.validate_trade_risk(
                symbol='BTCUSDT',
                side='long',
                size=0.1,
                entry_price=50000.0,
                leverage=10.0
            )
            
            self.assertFalse(result['approved'])
            violations = [v for v in result['violations'] if v['type'] == RiskViolationType.CONSECUTIVE_LOSSES.value]
            self.assertGreater(len(violations), 0)

    def test_validate_trade_risk_emergency_stop(self):
        """Test trade risk validation when emergency stop is active"""
        self.risk_manager.emergency_stop_active = True
        
        result = self.risk_manager.validate_trade_risk(
            symbol='BTCUSDT',
            side='long',
            size=0.1,
            entry_price=50000.0
        )
        
        self.assertFalse(result['approved'])
        self.assertEqual(result['violations'][0]['type'], 'EMERGENCY_STOP_ACTIVE')

    def test_validate_trade_risk_low_risk_reward_warning(self):
        """Test trade risk validation with low risk/reward ratio warning"""
        with patch.object(self.risk_manager, '_get_account_info', return_value=self.mock_account_info), \
             patch.object(self.risk_manager, '_get_daily_pnl', return_value=0.0), \
             patch.object(self.risk_manager, '_get_consecutive_losses', return_value=0), \
             patch.object(self.risk_manager, '_get_total_exposure', return_value=1000.0):
            
            result = self.risk_manager.validate_trade_risk(
                symbol='BTCUSDT',
                side='long',
                size=0.1,
                entry_price=50000.0,
                stop_loss_price=49000.0,
                take_profit_price=50500.0,  # Low reward
                leverage=10.0
            )
            
            self.assertTrue(result['approved'])  # Should still be approved
            warnings = [w for w in result['warnings'] if w['type'] == 'LOW_RISK_REWARD_RATIO']
            self.assertGreater(len(warnings), 0)

    @patch.object(AdvancedRiskManager, '_get_account_info')
    def test_get_portfolio_risk_assessment(self, mock_get_account):
        """Test portfolio risk assessment calculation"""
        mock_get_account.return_value = self.mock_account_info
        
        with patch.object(self.risk_manager, '_get_total_exposure', return_value=2000.0), \
             patch.object(self.risk_manager, '_get_net_exposure', return_value=1500.0), \
             patch.object(self.risk_manager, '_get_unrealized_pnl', return_value=100.0), \
             patch.object(self.risk_manager, '_get_daily_pnl', return_value=50.0), \
             patch.object(self.risk_manager, '_calculate_current_drawdown_pct', return_value=2.0):
            
            portfolio_risk = self.risk_manager.get_portfolio_risk_assessment()
            
            self.assertIsInstance(portfolio_risk, PortfolioRisk)
            self.assertEqual(portfolio_risk.total_account_value, 10000.0)
            self.assertEqual(portfolio_risk.total_exposure, 2000.0)
            self.assertEqual(portfolio_risk.total_exposure_pct, 20.0)
            self.assertEqual(portfolio_risk.net_exposure, 1500.0)
            self.assertEqual(portfolio_risk.unrealized_pnl, 100.0)
            self.assertEqual(portfolio_risk.daily_pnl, 50.0)
            self.assertEqual(portfolio_risk.current_drawdown_pct, 2.0)

    def test_get_portfolio_risk_assessment_caching(self):
        """Test portfolio risk assessment caching"""
        with patch.object(self.risk_manager, '_get_account_info', return_value=self.mock_account_info), \
             patch.object(self.risk_manager, '_get_total_exposure', return_value=2000.0), \
             patch.object(self.risk_manager, '_get_net_exposure', return_value=1500.0), \
             patch.object(self.risk_manager, '_get_unrealized_pnl', return_value=100.0), \
             patch.object(self.risk_manager, '_get_daily_pnl', return_value=50.0), \
             patch.object(self.risk_manager, '_calculate_current_drawdown_pct', return_value=2.0):
            
            # First call
            portfolio_risk1 = self.risk_manager.get_portfolio_risk_assessment()
            
            # Second call should use cache (within cache validity period)
            portfolio_risk2 = self.risk_manager.get_portfolio_risk_assessment()
            
            self.assertEqual(portfolio_risk1.total_account_value, portfolio_risk2.total_account_value)
            self.assertIsNotNone(self.risk_manager._portfolio_risk_cache)

    @patch.object(AdvancedRiskManager, 'get_portfolio_risk_assessment')
    def test_check_emergency_conditions_drawdown(self, mock_portfolio_risk):
        """Test emergency conditions check for critical drawdown"""
        mock_portfolio_risk.return_value = PortfolioRisk(
            total_account_value=10000.0,
            total_exposure=2000.0,
            total_exposure_pct=20.0,
            net_exposure=1500.0,
            net_exposure_pct=15.0,
            unrealized_pnl=-500.0,
            unrealized_pnl_pct=-5.0,
            daily_pnl=-300.0,
            daily_pnl_pct=-3.0,
            current_drawdown_pct=12.0  # Exceeds 10% limit
        )
        
        emergency = self.risk_manager.check_emergency_conditions()
        
        self.assertTrue(emergency['emergency_stop_required'])
        self.assertGreater(len(emergency['conditions']), 0)
        self.assertIn('STOP_ALL_TRADING', emergency['recommended_actions'])

    @patch.object(AdvancedRiskManager, 'get_portfolio_risk_assessment')
    def test_check_emergency_conditions_daily_loss(self, mock_portfolio_risk):
        """Test emergency conditions check for daily loss limit"""
        mock_portfolio_risk.return_value = PortfolioRisk(
            total_account_value=10000.0,
            total_exposure=2000.0,
            total_exposure_pct=20.0,
            net_exposure=1500.0,
            net_exposure_pct=15.0,
            unrealized_pnl=-300.0,
            unrealized_pnl_pct=-3.0,
            daily_pnl=-600.0,
            daily_pnl_pct=-6.0,  # Exceeds 5% daily loss limit
            current_drawdown_pct=3.0
        )
        
        emergency = self.risk_manager.check_emergency_conditions()
        
        self.assertTrue(emergency['emergency_stop_required'])
        conditions = [c for c in emergency['conditions'] if c['type'] == 'DAILY_LOSS_LIMIT_EXCEEDED']
        self.assertGreater(len(conditions), 0)

    @patch.object(AdvancedRiskManager, 'get_portfolio_risk_assessment')
    def test_check_emergency_conditions_excessive_exposure(self, mock_portfolio_risk):
        """Test emergency conditions check for excessive exposure"""
        mock_portfolio_risk.return_value = PortfolioRisk(
            total_account_value=10000.0,
            total_exposure=4000.0,
            total_exposure_pct=40.0,  # Exceeds 30% * 1.2 = 36% critical limit
            net_exposure=3500.0,
            net_exposure_pct=35.0,
            unrealized_pnl=100.0,
            unrealized_pnl_pct=1.0,
            daily_pnl=50.0,
            daily_pnl_pct=0.5,
            current_drawdown_pct=2.0
        )
        
        emergency = self.risk_manager.check_emergency_conditions()
        
        self.assertTrue(emergency['emergency_stop_required'])
        conditions = [c for c in emergency['conditions'] if c['type'] == 'EXCESSIVE_EXPOSURE']
        self.assertGreater(len(conditions), 0)

    def test_activate_emergency_stop(self):
        """Test emergency stop activation"""
        reason = "Critical drawdown exceeded"
        
        self.risk_manager.activate_emergency_stop(reason)
        
        self.assertTrue(self.risk_manager.emergency_stop_active)
        self.assertGreater(len(self.risk_manager.risk_violations), 0)
        self.assertEqual(self.risk_manager.risk_violations[-1]['type'], 'EMERGENCY_STOP')
        self.assertEqual(self.risk_manager.risk_violations[-1]['reason'], reason)

    def test_deactivate_emergency_stop(self):
        """Test emergency stop deactivation"""
        # First activate
        self.risk_manager.activate_emergency_stop("Test activation")
        self.assertTrue(self.risk_manager.emergency_stop_active)
        
        # Then deactivate
        reason = "Risk levels normalized"
        self.risk_manager.deactivate_emergency_stop(reason)
        
        self.assertFalse(self.risk_manager.emergency_stop_active)

    def test_calculate_risk_reward_ratio_long(self):
        """Test risk/reward ratio calculation for long position"""
        ratio = self.risk_manager._calculate_risk_reward_ratio(
            entry_price=50000.0,
            stop_loss_price=49000.0,
            take_profit_price=52000.0,
            side='long'
        )
        
        # Risk = 1000, Reward = 2000, Ratio = 2.0
        self.assertEqual(ratio, 2.0)

    def test_calculate_risk_reward_ratio_short(self):
        """Test risk/reward ratio calculation for short position"""
        ratio = self.risk_manager._calculate_risk_reward_ratio(
            entry_price=50000.0,
            stop_loss_price=51000.0,
            take_profit_price=48000.0,
            side='short'
        )
        
        # Risk = 1000, Reward = 2000, Ratio = 2.0
        self.assertEqual(ratio, 2.0)

    def test_market_condition_adjustments(self):
        """Test market condition adjustments"""
        # High volatility
        high_vol_adj = self.risk_manager._get_market_condition_adjustment({
            'market_5m': 'HIGH_VOLATILITY',
            'market_1m': 'TRENDING'
        })
        self.assertEqual(high_vol_adj, 0.7)
        
        # Transitional
        trans_adj = self.risk_manager._get_market_condition_adjustment({
            'market_5m': 'TRANSITIONAL',
            'market_1m': 'RANGING'
        })
        self.assertEqual(trans_adj, 0.8)
        
        # Trending
        trend_adj = self.risk_manager._get_market_condition_adjustment({
            'market_5m': 'TRENDING',
            'market_1m': 'TRENDING'
        })
        self.assertEqual(trend_adj, 1.0)

    def test_volatility_adjustments(self):
        """Test volatility-based adjustments"""
        # High volatility
        high_vol_adj = self.risk_manager._get_volatility_adjustment({
            'volatility_score': 2.5
        })
        self.assertEqual(high_vol_adj, 0.6)
        
        # Medium volatility
        med_vol_adj = self.risk_manager._get_volatility_adjustment({
            'volatility_score': 1.7
        })
        self.assertEqual(med_vol_adj, 0.8)
        
        # Low volatility
        low_vol_adj = self.risk_manager._get_volatility_adjustment({
            'volatility_score': 0.3
        })
        self.assertEqual(low_vol_adj, 1.2)
        
        # No volatility data
        no_vol_adj = self.risk_manager._get_volatility_adjustment(None)
        self.assertEqual(no_vol_adj, 1.0)

    def test_strategy_risk_adjustments(self):
        """Test strategy-specific risk adjustments"""
        # High risk strategy
        high_risk_adj = self.risk_manager._get_strategy_risk_adjustment('volatility_reversal_strategy')
        self.assertEqual(high_risk_adj, 0.7)
        
        # Medium risk strategy
        med_risk_adj = self.risk_manager._get_strategy_risk_adjustment('momentum_strategy')
        self.assertEqual(med_risk_adj, 0.9)
        
        # Normal strategy
        normal_adj = self.risk_manager._get_strategy_risk_adjustment('ema_crossover_strategy')
        self.assertEqual(normal_adj, 1.0)

    @patch.object(AdvancedRiskManager, 'get_portfolio_risk_assessment')
    def test_portfolio_risk_adjustments(self, mock_portfolio_risk):
        """Test portfolio risk-based adjustments"""
        # High exposure
        mock_portfolio_risk.return_value = PortfolioRisk(
            total_account_value=10000.0,
            total_exposure=2200.0,
            total_exposure_pct=22.0,
            net_exposure=2000.0,
            net_exposure_pct=20.0,
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            daily_pnl=0.0,
            daily_pnl_pct=0.0,
            current_drawdown_pct=1.0
        )
        
        adj = self.risk_manager._get_portfolio_risk_adjustment()
        self.assertEqual(adj, 0.7)

    def test_get_daily_pnl_with_performance_tracker(self):
        """Test daily P&L calculation with performance tracker"""
        # Mock trades for today
        today = datetime.now(timezone.utc)
        mock_trades = [
            Mock(pnl=100.0, entry_timestamp=today.isoformat()),
            Mock(pnl=-50.0, entry_timestamp=today.isoformat()),
            Mock(pnl=25.0, entry_timestamp=(today - timedelta(days=1)).isoformat())  # Yesterday
        ]
        
        self.mock_performance_tracker.trades = mock_trades
        self.mock_performance_tracker.get_comprehensive_statistics.return_value = {'total_trades': 3}
        
        daily_pnl = self.risk_manager._get_daily_pnl()
        
        # Should only include today's trades: 100 + (-50) = 50
        self.assertEqual(daily_pnl, 50.0)

    def test_get_consecutive_losses(self):
        """Test consecutive losses retrieval"""
        self.mock_performance_tracker.consecutive_losses = 3
        
        consecutive_losses = self.risk_manager._get_consecutive_losses()
        
        self.assertEqual(consecutive_losses, 3)

    def test_assess_overall_risk_level(self):
        """Test overall risk level assessment"""
        # Conservative risk
        conservative_risk = PortfolioRisk(
            total_account_value=10000.0,
            total_exposure=1000.0,
            total_exposure_pct=10.0,
            net_exposure=800.0,
            net_exposure_pct=8.0,
            unrealized_pnl=50.0,
            unrealized_pnl_pct=0.5,
            daily_pnl=25.0,
            daily_pnl_pct=0.25,
            current_drawdown_pct=1.0
        )
        
        risk_level = self.risk_manager._assess_overall_risk_level(conservative_risk)
        self.assertEqual(risk_level, RiskLevel.CONSERVATIVE.value)
        
        # Aggressive risk
        aggressive_risk = PortfolioRisk(
            total_account_value=10000.0,
            total_exposure=3000.0,
            total_exposure_pct=30.0,
            net_exposure=2500.0,
            net_exposure_pct=25.0,
            unrealized_pnl=-500.0,
            unrealized_pnl_pct=-5.0,
            daily_pnl=-400.0,
            daily_pnl_pct=-4.0,
            current_drawdown_pct=8.0
        )
        
        risk_level = self.risk_manager._assess_overall_risk_level(aggressive_risk)
        self.assertEqual(risk_level, RiskLevel.AGGRESSIVE.value)

    @patch.object(AdvancedRiskManager, 'get_portfolio_risk_assessment')
    @patch.object(AdvancedRiskManager, 'check_emergency_conditions')
    def test_get_risk_summary(self, mock_emergency, mock_portfolio):
        """Test comprehensive risk summary generation"""
        mock_portfolio.return_value = PortfolioRisk(
            total_account_value=10000.0,
            total_exposure=2000.0,
            total_exposure_pct=20.0,
            net_exposure=1500.0,
            net_exposure_pct=15.0,
            unrealized_pnl=100.0,
            unrealized_pnl_pct=1.0,
            daily_pnl=50.0,
            daily_pnl_pct=0.5,
            current_drawdown_pct=2.0
        )
        
        mock_emergency.return_value = {
            'emergency_stop_required': False,
            'conditions': [],
            'recommended_actions': []
        }
        
        summary = self.risk_manager.get_risk_summary()
        
        self.assertIn('timestamp', summary)
        self.assertIn('emergency_stop_active', summary)
        self.assertIn('portfolio_risk', summary)
        self.assertIn('emergency_conditions', summary)
        self.assertIn('risk_limits', summary)
        self.assertIn('risk_level', summary)
        self.assertEqual(summary['emergency_stop_active'], False)
        self.assertEqual(summary['active_positions'], 0)

    def test_trade_frequency_violation_check(self):
        """Test trade frequency violation checking"""
        # Mock recent trades (within last hour)
        recent_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        mock_trades = [
            Mock(entry_timestamp=recent_time.isoformat()) for _ in range(12)  # Exceeds limit of 10
        ]
        
        self.mock_performance_tracker.trades = mock_trades
        
        violation = self.risk_manager._check_trade_frequency_violation()
        
        self.assertTrue(violation)

    def test_var_95_calculation(self):
        """Test Value at Risk calculation"""
        # Mock trades with returns
        mock_trades = [
            Mock(return_pct=2.0),
            Mock(return_pct=-1.5),
            Mock(return_pct=1.0),
            Mock(return_pct=-3.0),
            Mock(return_pct=0.5),
            Mock(return_pct=-2.0),
            Mock(return_pct=1.5),
            Mock(return_pct=-1.0),
            Mock(return_pct=0.8),
            Mock(return_pct=-0.5),
            Mock(return_pct=2.5),
            Mock(return_pct=-4.0),  # This should be the 5th percentile
            Mock(return_pct=1.2),
            Mock(return_pct=-0.8),
            Mock(return_pct=0.3),
            Mock(return_pct=-1.2),
            Mock(return_pct=1.8),
            Mock(return_pct=-2.5),
            Mock(return_pct=0.7),
            Mock(return_pct=-0.3)
        ]
        
        self.mock_performance_tracker.trades = mock_trades
        
        var_95 = self.risk_manager._calculate_var_95()
        
        self.assertIsNotNone(var_95)
        self.assertLess(var_95, 0)  # VaR should be negative

    def test_var_95_insufficient_data(self):
        """Test VaR calculation with insufficient data"""
        # Only 5 trades (less than minimum 10)
        mock_trades = [Mock(return_pct=1.0) for _ in range(5)]
        self.mock_performance_tracker.trades = mock_trades
        
        var_95 = self.risk_manager._calculate_var_95()
        
        self.assertIsNone(var_95)

    def test_concentration_risk_calculation(self):
        """Test concentration risk calculation"""
        # Add positions with different sizes
        self.risk_manager.positions = {
            'BTCUSDT': Mock(position_value=2000.0),
            'ETHUSDT': Mock(position_value=1000.0),
            'ADAUSDT': Mock(position_value=500.0)
        }
        
        concentration_risk = self.risk_manager._calculate_concentration_risk()
        
        # Largest position (2000) / Total (3500) * 100 = 57.14%
        self.assertAlmostEqual(concentration_risk, 57.14, places=1)

    def test_concentration_risk_no_positions(self):
        """Test concentration risk with no positions"""
        self.risk_manager.positions = {}
        
        concentration_risk = self.risk_manager._calculate_concentration_risk()
        
        self.assertEqual(concentration_risk, 0.0)

if __name__ == '__main__':
    unittest.main() 