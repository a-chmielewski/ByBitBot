#!/usr/bin/env python3
"""
Unit tests for TrailingTPHandler state machine transitions.

This module provides comprehensive tests for:
- State machine transitions
- Order fill processing
- Progressive take profit execution
- Trailing stop logic
- State persistence and recovery
- Error handling and edge cases
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

from modules.trailing_tp_handler import (
    TrailingTPHandler, PositionState, OrderType, PositionData, 
    ProgressiveTPLevel, TrailingStopConfig
)


class TestTrailingTPHandler(unittest.TestCase):
    """Test suite for TrailingTPHandler"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary state file
        self.temp_dir = tempfile.mkdtemp()
        self.state_file = Path(self.temp_dir) / "test_state.json"
        
        # Mock dependencies
        self.mock_exchange = Mock()
        self.mock_order_manager = Mock()
        self.mock_logger = Mock()
        
        # Configure mocks
        self.mock_exchange.get_ticker.return_value = {'last': 45000.0, 'bid': 44995.0, 'ask': 45005.0}
        self.mock_order_manager.place_order.return_value = {'orderId': 'test_order_123'}
        self.mock_order_manager.cancel_order.return_value = True
        
        # Create handler instance
        self.handler = TrailingTPHandler(
            exchange_connector=self.mock_exchange,
            order_manager=self.mock_order_manager,
            logger=self.mock_logger,
            state_file=str(self.state_file),
            monitoring_interval=1.0
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.handler.stop_monitoring()
        if self.state_file.exists():
            self.state_file.unlink()
        os.rmdir(self.temp_dir)
    
    def test_position_creation(self):
        """Test position creation and initial state"""
        # Create position
        position_id = self.handler.create_position(
            symbol='BTCUSDT',
            side='buy',
            entry_price=45000.0,
            position_size=0.01,
            strategy_name='TestStrategy',
            tp_levels=[
                {'level': 1, 'price': 45500.0, 'size_pct': 0.4},
                {'level': 2, 'price': 46000.0, 'size_pct': 0.4},
                {'level': 3, 'price': 46500.0, 'size_pct': 0.2}
            ],
            trailing_config={'enabled': True, 'initial_offset_pct': 0.02},
            stop_loss_pct=0.02,
            session_id='test_session'
        )
        
        # Verify position created
        self.assertEqual(position_id, 'BTCUSDT')
        self.assertIn('BTCUSDT', self.handler.positions)
        
        position = self.handler.positions['BTCUSDT']
        self.assertEqual(position.state, PositionState.ENTRY_PENDING)
        self.assertEqual(position.symbol, 'BTCUSDT')
        self.assertEqual(position.side, 'buy')
        self.assertEqual(position.entry_price, 45000.0)
        self.assertEqual(position.position_size, 0.01)
        self.assertEqual(position.remaining_size, 0.01)
        self.assertEqual(len(position.tp_levels), 3)
        self.assertTrue(position.trailing_config.enabled)
    
    def test_entry_order_fill(self):
        """Test entry order fill processing"""
        # Create position
        self.handler.create_position(
            symbol='BTCUSDT', side='buy', entry_price=45000.0, position_size=0.01,
            strategy_name='TestStrategy', tp_levels=[
                {'level': 1, 'price': 45500.0, 'size_pct': 0.4}
            ],
            trailing_config={'enabled': True}, stop_loss_pct=0.02
        )
        
        # Register entry order
        entry_order_id = 'entry_123'
        self.handler.order_to_symbol[entry_order_id] = 'BTCUSDT'
        self.handler.order_to_type[entry_order_id] = OrderType.ENTRY
        
        # Process entry fill
        fill_data = {'size': 0.01, 'price': 45000.0}
        self.handler.on_order_fill(entry_order_id, fill_data)
        
        # Verify state transition
        position = self.handler.positions['BTCUSDT']
        self.assertEqual(position.state, PositionState.ACTIVE)
        self.assertEqual(position.filled_size, 0.01)
        self.assertEqual(position.average_fill_price, 45000.0)
        self.assertEqual(position.remaining_size, 0.01)
        
        # Verify TP orders submitted (mocked)
        self.mock_order_manager.place_order.assert_called()
    
    def test_partial_entry_fill(self):
        """Test partial entry order fill"""
        # Create position
        self.handler.create_position(
            symbol='BTCUSDT', side='buy', entry_price=45000.0, position_size=0.01,
            strategy_name='TestStrategy', tp_levels=[],
            trailing_config={'enabled': False}, stop_loss_pct=0.02
        )
        
        # Register entry order
        entry_order_id = 'entry_123'
        self.handler.order_to_symbol[entry_order_id] = 'BTCUSDT'
        self.handler.order_to_type[entry_order_id] = OrderType.ENTRY
        
        # Process partial fill
        fill_data = {'size': 0.005, 'price': 45000.0}
        self.handler.on_order_fill(entry_order_id, fill_data)
        
        # Verify partial fill state
        position = self.handler.positions['BTCUSDT']
        self.assertEqual(position.state, PositionState.ENTRY_PENDING)  # Still pending
        self.assertEqual(position.filled_size, 0.005)
        self.assertEqual(position.remaining_size, 0.005)
        
        # Process remaining fill
        fill_data = {'size': 0.005, 'price': 45010.0}
        self.handler.on_order_fill(entry_order_id, fill_data)
        
        # Verify fully filled
        position = self.handler.positions['BTCUSDT']
        self.assertEqual(position.state, PositionState.ACTIVE)
        self.assertEqual(position.filled_size, 0.01)
        self.assertEqual(position.average_fill_price, 45005.0)  # Average of 45000 and 45010
    
    def test_tp1_fill_breakeven_move(self):
        """Test TP1 fill triggers breakeven move"""
        # Create and activate position
        self.handler.create_position(
            symbol='BTCUSDT', side='buy', entry_price=45000.0, position_size=0.01,
            strategy_name='TestStrategy', tp_levels=[
                {'level': 1, 'price': 45500.0, 'size_pct': 0.4},
                {'level': 2, 'price': 46000.0, 'size_pct': 0.6}
            ],
            trailing_config={'enabled': True}, stop_loss_pct=0.02
        )
        
        position = self.handler.positions['BTCUSDT']
        position.state = PositionState.ACTIVE
        position.filled_size = 0.01
        position.average_fill_price = 45000.0
        position.tp_levels[0].order_id = 'tp1_123'
        
        # Register TP1 order
        self.handler.order_to_symbol['tp1_123'] = 'BTCUSDT'
        self.handler.order_to_type['tp1_123'] = OrderType.TAKE_PROFIT_1
        
        # Process TP1 fill
        fill_data = {'size': 0.004, 'price': 45500.0}
        self.handler.on_order_fill('tp1_123', fill_data)
        
        # Verify state transition
        self.assertEqual(position.state, PositionState.TP1_PARTIAL)
        self.assertEqual(position.remaining_size, 0.006)  # 0.01 - 0.004
        self.assertTrue(position.tp_levels[0].is_filled)
        self.assertTrue(position.breakeven_moved)
        
        # Verify stop loss moved to breakeven (mocked)
        self.mock_order_manager.place_order.assert_called()
    
    def test_tp2_fill_trailing_activation(self):
        """Test TP2 fill triggers trailing stop activation"""
        # Create and setup position in TP1_PARTIAL state
        self.handler.create_position(
            symbol='BTCUSDT', side='buy', entry_price=45000.0, position_size=0.01,
            strategy_name='TestStrategy', tp_levels=[
                {'level': 1, 'price': 45500.0, 'size_pct': 0.4},
                {'level': 2, 'price': 46000.0, 'size_pct': 0.4}
            ],
            trailing_config={'enabled': True, 'initial_offset_pct': 0.02, 'tightened_offset_pct': 0.015},
            stop_loss_pct=0.02
        )
        
        position = self.handler.positions['BTCUSDT']
        position.state = PositionState.TP1_PARTIAL
        position.filled_size = 0.01
        position.remaining_size = 0.006
        position.tp_levels[0].is_filled = True
        position.tp_levels[1].order_id = 'tp2_123'
        position.breakeven_moved = True
        
        # Register TP2 order
        self.handler.order_to_symbol['tp2_123'] = 'BTCUSDT'
        self.handler.order_to_type['tp2_123'] = OrderType.TAKE_PROFIT_2
        
        # Process TP2 fill
        fill_data = {'size': 0.004, 'price': 46000.0}
        self.handler.on_order_fill('tp2_123', fill_data)
        
        # Verify trailing activation
        self.assertEqual(position.state, PositionState.TRAILING_ACTIVE)
        self.assertEqual(position.remaining_size, 0.002)  # 0.006 - 0.004
        self.assertTrue(position.tp_levels[1].is_filled)
        self.assertTrue(position.trailing_activated)
        self.assertEqual(position.trailing_config.initial_offset_pct, 0.015)  # Tightened
    
    def test_stop_loss_fill_position_close(self):
        """Test stop loss fill closes position"""
        # Create and setup position
        self.handler.create_position(
            symbol='BTCUSDT', side='buy', entry_price=45000.0, position_size=0.01,
            strategy_name='TestStrategy', tp_levels=[],
            trailing_config={'enabled': False}, stop_loss_pct=0.02
        )
        
        position = self.handler.positions['BTCUSDT']
        position.state = PositionState.ACTIVE
        position.remaining_size = 0.01
        position.stop_loss_order_id = 'sl_123'
        
        # Register stop loss order
        self.handler.order_to_symbol['sl_123'] = 'BTCUSDT'
        self.handler.order_to_type['sl_123'] = OrderType.STOP_LOSS
        
        # Process stop loss fill
        fill_data = {'size': 0.01, 'price': 44100.0}
        self.handler.on_order_fill('sl_123', fill_data)
        
        # Verify position closed
        self.assertEqual(position.state, PositionState.CLOSED)
        self.assertEqual(position.remaining_size, 0.0)
        
        # Verify remaining orders cancelled (mocked)
        self.mock_order_manager.cancel_order.assert_called()
    
    def test_trailing_stop_update_long_position(self):
        """Test trailing stop updates for long position"""
        # Create position with trailing enabled
        self.handler.create_position(
            symbol='BTCUSDT', side='buy', entry_price=45000.0, position_size=0.01,
            strategy_name='TestStrategy', tp_levels=[],
            trailing_config={'enabled': True, 'initial_offset_pct': 0.02},
            stop_loss_pct=0.02
        )
        
        position = self.handler.positions['BTCUSDT']
        position.state = PositionState.TRAILING_ACTIVE
        position.trailing_activated = True
        position.current_stop_price = 44100.0  # Initial stop
        
        # Mock higher price to trigger trailing update
        self.mock_exchange.get_ticker.return_value = {'last': 46000.0}
        
        # Update trailing stops
        self.handler._update_single_trailing_stop(position)
        
        # Verify trailing stop moved higher
        expected_stop = 46000.0 * (1 - 0.02)  # 45080.0
        self.assertEqual(position.current_stop_price, expected_stop)
        self.assertEqual(position.trailing_config.highest_price, 46000.0)
        
        # Verify stop loss order updated (mocked)
        self.mock_order_manager.place_order.assert_called()
    
    def test_trailing_stop_update_short_position(self):
        """Test trailing stop updates for short position"""
        # Create short position with trailing enabled
        self.handler.create_position(
            symbol='BTCUSDT', side='sell', entry_price=45000.0, position_size=0.01,
            strategy_name='TestStrategy', tp_levels=[],
            trailing_config={'enabled': True, 'initial_offset_pct': 0.02},
            stop_loss_pct=0.02
        )
        
        position = self.handler.positions['BTCUSDT']
        position.state = PositionState.TRAILING_ACTIVE
        position.trailing_activated = True
        position.current_stop_price = 45900.0  # Initial stop
        
        # Mock lower price to trigger trailing update
        self.mock_exchange.get_ticker.return_value = {'last': 44000.0}
        
        # Update trailing stops
        self.handler._update_single_trailing_stop(position)
        
        # Verify trailing stop moved lower
        expected_stop = 44000.0 * (1 + 0.02)  # 44880.0
        self.assertEqual(position.current_stop_price, expected_stop)
        self.assertEqual(position.trailing_config.lowest_price, 44000.0)
    
    def test_emergency_close_position(self):
        """Test emergency position close"""
        # Create active position
        self.handler.create_position(
            symbol='BTCUSDT', side='buy', entry_price=45000.0, position_size=0.01,
            strategy_name='TestStrategy', tp_levels=[
                {'level': 1, 'price': 45500.0, 'size_pct': 0.5}
            ],
            trailing_config={'enabled': True}, stop_loss_pct=0.02
        )
        
        position = self.handler.positions['BTCUSDT']
        position.state = PositionState.ACTIVE
        position.remaining_size = 0.01
        position.tp_levels[0].order_id = 'tp1_123'
        position.stop_loss_order_id = 'sl_123'
        
        # Emergency close
        result = self.handler.emergency_close_position('BTCUSDT', 'Circuit breaker triggered')
        
        # Verify emergency close successful
        self.assertTrue(result)
        self.assertEqual(position.state, PositionState.CLOSING)
        
        # Verify orders cancelled and market close order placed
        self.assertEqual(self.mock_order_manager.cancel_order.call_count, 2)  # TP and SL
        self.mock_order_manager.place_order.assert_called_with(
            symbol='BTCUSDT',
            side='sell',
            order_type='market',
            size=0.01,
            reduce_only=True
        )
    
    def test_state_persistence(self):
        """Test position state save and load"""
        # Create position
        position_id = self.handler.create_position(
            symbol='BTCUSDT', side='buy', entry_price=45000.0, position_size=0.01,
            strategy_name='TestStrategy', tp_levels=[
                {'level': 1, 'price': 45500.0, 'size_pct': 0.4}
            ],
            trailing_config={'enabled': True}, stop_loss_pct=0.02,
            session_id='test_session_123'
        )
        
        # Modify position state
        position = self.handler.positions['BTCUSDT']
        position.state = PositionState.ACTIVE
        position.filled_size = 0.01
        position.average_fill_price = 45010.0
        position.breakeven_moved = True
        
        # Save state
        self.handler.save_state()
        
        # Verify state file created
        self.assertTrue(self.state_file.exists())
        
        # Create new handler and load state
        new_handler = TrailingTPHandler(
            exchange_connector=self.mock_exchange,
            order_manager=self.mock_order_manager,
            logger=self.mock_logger,
            state_file=str(self.state_file)
        )
        
        # Verify state restored
        self.assertIn('BTCUSDT', new_handler.positions)
        restored_position = new_handler.positions['BTCUSDT']
        self.assertEqual(restored_position.state, PositionState.ACTIVE)
        self.assertEqual(restored_position.filled_size, 0.01)
        self.assertEqual(restored_position.average_fill_price, 45010.0)
        self.assertTrue(restored_position.breakeven_moved)
        self.assertEqual(restored_position.session_id, 'test_session_123')
    
    def test_position_status_reporting(self):
        """Test comprehensive position status reporting"""
        # Create and setup position
        self.handler.create_position(
            symbol='BTCUSDT', side='buy', entry_price=45000.0, position_size=0.01,
            strategy_name='TestStrategy', tp_levels=[
                {'level': 1, 'price': 45500.0, 'size_pct': 0.4},
                {'level': 2, 'price': 46000.0, 'size_pct': 0.6}
            ],
            trailing_config={'enabled': True, 'initial_offset_pct': 0.02},
            stop_loss_pct=0.02
        )
        
        position = self.handler.positions['BTCUSDT']
        position.state = PositionState.TP1_PARTIAL
        position.filled_size = 0.01
        position.remaining_size = 0.006
        position.tp_levels[0].is_filled = True
        position.breakeven_moved = True
        
        # Get position status
        status = self.handler.get_position_status('BTCUSDT')
        
        # Verify status completeness
        self.assertIsNotNone(status)
        self.assertEqual(status['symbol'], 'BTCUSDT')
        self.assertEqual(status['state'], 'tp1_partial')
        self.assertEqual(status['side'], 'buy')
        self.assertEqual(status['position_size'], 0.01)
        self.assertEqual(status['remaining_size'], 0.006)
        self.assertEqual(status['filled_tp_levels'], 1)
        self.assertEqual(status['pending_tp_levels'], 1)
        self.assertTrue(status['breakeven_moved'])
        
        # Verify TP details
        tp_details = status['tp_details']
        self.assertEqual(len(tp_details), 2)
        self.assertTrue(tp_details[0]['filled'])
        self.assertFalse(tp_details[1]['filled'])
    
    def test_invalid_order_fill_handling(self):
        """Test handling of invalid order fills"""
        # Process fill for non-existent order
        self.handler.on_order_fill('invalid_order_123', {'size': 0.01, 'price': 45000.0})
        
        # Should not crash and should log debug message
        self.mock_logger.debug.assert_called()
    
    def test_multiple_positions_handling(self):
        """Test handling multiple positions simultaneously"""
        # Create multiple positions
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        
        for symbol in symbols:
            self.handler.create_position(
                symbol=symbol, side='buy', entry_price=1000.0, position_size=0.1,
                strategy_name='TestStrategy', tp_levels=[],
                trailing_config={'enabled': True}, stop_loss_pct=0.02
            )
        
        # Verify all positions created
        self.assertEqual(len(self.handler.positions), 3)
        for symbol in symbols:
            self.assertIn(symbol, self.handler.positions)
            self.assertEqual(self.handler.positions[symbol].state, PositionState.ENTRY_PENDING)
        
        # Get all positions status
        all_status = self.handler.get_all_positions_status()
        self.assertEqual(len(all_status), 3)
        for symbol in symbols:
            self.assertIn(symbol, all_status)
    
    def test_cleanup_closed_positions(self):
        """Test cleanup of old closed positions"""
        # Create position and mark as closed
        self.handler.create_position(
            symbol='BTCUSDT', side='buy', entry_price=45000.0, position_size=0.01,
            strategy_name='TestStrategy', tp_levels=[],
            trailing_config={'enabled': False}, stop_loss_pct=0.02
        )
        
        position = self.handler.positions['BTCUSDT']
        position.state = PositionState.CLOSED
        position.last_updated_at = datetime.now(timezone.utc) - timedelta(hours=25)  # Old
        
        # Create recent closed position
        self.handler.create_position(
            symbol='ETHUSDT', side='buy', entry_price=3000.0, position_size=0.1,
            strategy_name='TestStrategy', tp_levels=[],
            trailing_config={'enabled': False}, stop_loss_pct=0.02
        )
        
        eth_position = self.handler.positions['ETHUSDT']
        eth_position.state = PositionState.CLOSED
        eth_position.last_updated_at = datetime.now(timezone.utc) - timedelta(hours=1)  # Recent
        
        # Cleanup old positions
        self.handler.cleanup_closed_positions(max_age_hours=24)
        
        # Verify old position removed, recent kept
        self.assertNotIn('BTCUSDT', self.handler.positions)
        self.assertIn('ETHUSDT', self.handler.positions)
    
    def test_state_machine_transition_validation(self):
        """Test state machine transition validation"""
        # Create position
        self.handler.create_position(
            symbol='BTCUSDT', side='buy', entry_price=45000.0, position_size=0.01,
            strategy_name='TestStrategy', tp_levels=[
                {'level': 1, 'price': 45500.0, 'size_pct': 0.5},
                {'level': 2, 'price': 46000.0, 'size_pct': 0.5}
            ],
            trailing_config={'enabled': True}, stop_loss_pct=0.02
        )
        
        position = self.handler.positions['BTCUSDT']
        
        # Test state transitions
        test_cases = [
            # (filled_tps, remaining_size, expected_state)
            (0, 0.01, PositionState.ACTIVE),
            (1, 0.005, PositionState.TP1_PARTIAL),
            (2, 0.0, PositionState.CLOSED),
        ]
        
        for filled_tp_count, remaining_size, expected_state in test_cases:
            # Setup position state
            for i in range(filled_tp_count):
                position.tp_levels[i].is_filled = True
            position.remaining_size = remaining_size
            
            # Update state
            self.handler._update_position_state(position)
            
            # Verify expected state
            self.assertEqual(position.state, expected_state, 
                           f"Failed for filled_tps={filled_tp_count}, remaining={remaining_size}")


class TestTrailingTPHandlerIntegration(unittest.TestCase):
    """Integration tests for TrailingTPHandler with realistic scenarios"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.state_file = Path(self.temp_dir) / "integration_state.json"
        
        # More realistic mocks
        self.mock_exchange = Mock()
        self.mock_order_manager = Mock()
        self.mock_logger = Mock()
        
        # Price simulation
        self.current_price = 45000.0
        self.mock_exchange.get_ticker.side_effect = self._get_ticker_response
        
        # Order ID counter for realistic responses
        self.order_id_counter = 1000
        self.mock_order_manager.place_order.side_effect = self._place_order_response
        
        self.handler = TrailingTPHandler(
            exchange_connector=self.mock_exchange,
            order_manager=self.mock_order_manager,
            logger=self.mock_logger,
            state_file=str(self.state_file),
            monitoring_interval=0.1  # Fast for testing
        )
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        self.handler.stop_monitoring()
        if self.state_file.exists():
            self.state_file.unlink()
        os.rmdir(self.temp_dir)
    
    def _get_ticker_response(self, symbol):
        """Mock ticker response with current price"""
        return {
            'last': self.current_price,
            'bid': self.current_price - 2.5,
            'ask': self.current_price + 2.5
        }
    
    def _place_order_response(self, **kwargs):
        """Mock order placement response"""
        self.order_id_counter += 1
        return {'orderId': f'order_{self.order_id_counter}'}
    
    def test_complete_trading_lifecycle(self):
        """Test complete trading lifecycle from entry to close"""
        # Create position
        position_id = self.handler.create_position(
            symbol='BTCUSDT',
            side='buy',
            entry_price=45000.0,
            position_size=0.01,
            strategy_name='IntegrationTest',
            tp_levels=[
                {'level': 1, 'price': 45500.0, 'size_pct': 0.4},
                {'level': 2, 'price': 46000.0, 'size_pct': 0.4},
                {'level': 3, 'price': 46500.0, 'size_pct': 0.2}
            ],
            trailing_config={'enabled': True, 'initial_offset_pct': 0.02, 'tightened_offset_pct': 0.015},
            stop_loss_pct=0.02
        )
        
        position = self.handler.positions['BTCUSDT']
        
        # Step 1: Entry fill
        entry_order_id = 'entry_order_123'
        self.handler.order_to_symbol[entry_order_id] = 'BTCUSDT'
        self.handler.order_to_type[entry_order_id] = OrderType.ENTRY
        
        self.handler.on_order_fill(entry_order_id, {'size': 0.01, 'price': 45000.0})
        self.assertEqual(position.state, PositionState.ACTIVE)
        
        # Step 2: TP1 fill (40% of position)
        position.tp_levels[0].order_id = 'tp1_order_456'
        self.handler.order_to_symbol['tp1_order_456'] = 'BTCUSDT'
        self.handler.order_to_type['tp1_order_456'] = OrderType.TAKE_PROFIT_1
        
        self.handler.on_order_fill('tp1_order_456', {'size': 0.004, 'price': 45500.0})
        self.assertEqual(position.state, PositionState.TP1_PARTIAL)
        self.assertTrue(position.breakeven_moved)
        self.assertEqual(position.remaining_size, 0.006)
        
        # Step 3: TP2 fill (40% of remaining 60%)
        position.tp_levels[1].order_id = 'tp2_order_789'
        self.handler.order_to_symbol['tp2_order_789'] = 'BTCUSDT'
        self.handler.order_to_type['tp2_order_789'] = OrderType.TAKE_PROFIT_2
        
        self.handler.on_order_fill('tp2_order_789', {'size': 0.004, 'price': 46000.0})
        self.assertEqual(position.state, PositionState.TRAILING_ACTIVE)
        self.assertTrue(position.trailing_activated)
        self.assertEqual(position.trailing_config.initial_offset_pct, 0.015)  # Tightened
        self.assertEqual(position.remaining_size, 0.002)
        
        # Step 4: Price rises, trailing stop should update
        self.current_price = 47000.0
        self.handler._update_single_trailing_stop(position)
        
        expected_stop = 47000.0 * (1 - 0.015)  # 46295.0
        self.assertEqual(position.current_stop_price, expected_stop)
        self.assertEqual(position.trailing_config.highest_price, 47000.0)
        
        # Step 5: Final exit via trailing stop
        position.stop_loss_order_id = 'sl_final_999'
        self.handler.order_to_symbol['sl_final_999'] = 'BTCUSDT'
        self.handler.order_to_type['sl_final_999'] = OrderType.STOP_LOSS
        
        self.handler.on_order_fill('sl_final_999', {'size': 0.002, 'price': 46295.0})
        self.assertEqual(position.state, PositionState.CLOSED)
        self.assertEqual(position.remaining_size, 0.0)
        
        # Verify order placement calls were made appropriately
        # Entry creates TP orders + initial SL, breakeven move, trailing updates
        self.assertGreater(self.mock_order_manager.place_order.call_count, 5)


def run_trailing_tp_tests():
    """Run all TrailingTPHandler tests"""
    print("🧪 TRAILING TP HANDLER UNIT TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestTrailingTPHandler))
    suite.addTest(unittest.makeSuite(TestTrailingTPHandlerIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 TEST RESULTS:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("✅ ALL TESTS PASSED - TrailingTPHandler is ready for production!")
    else:
        print("❌ Some tests failed - please review and fix issues")
    
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    run_trailing_tp_tests()