"""
Trailing Stop & Progressive Take Profit Handler

This module provides a broker-agnostic, state machine-based system for managing
trailing stops and progressive take profit orders. It's designed to be:
- Idempotent and safe on restarts
- Replayable from logs
- Integrated with the enhanced risk management system
- Fully testable with comprehensive state transitions
"""

import logging
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path

class PositionState(Enum):
    """Position lifecycle states"""
    INACTIVE = "inactive"           # No position
    ENTRY_PENDING = "entry_pending" # Entry order submitted, not filled
    ACTIVE = "active"               # Position open, initial state
    TP1_PARTIAL = "tp1_partial"     # First TP hit, stop moved to breakeven
    TP2_PARTIAL = "tp2_partial"     # Second TP hit, trailing tightened
    TP3_PARTIAL = "tp3_partial"     # Third TP hit (if applicable)
    TRAILING_ACTIVE = "trailing_active"  # Pure trailing mode
    CLOSING = "closing"             # Exit order submitted
    CLOSED = "closed"               # Position fully closed
    ERROR = "error"                 # Error state requiring intervention

class OrderType(Enum):
    """Order types tracked by the handler"""
    ENTRY = "entry"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT_1 = "take_profit_1"
    TAKE_PROFIT_2 = "take_profit_2"
    TAKE_PROFIT_3 = "take_profit_3"
    TRAILING_STOP = "trailing_stop"
    EXIT = "exit"

@dataclass
class ProgressiveTPLevel:
    """Progressive take profit level configuration"""
    level: int                    # 1, 2, 3, etc.
    price: float                 # Target price
    size_percentage: float       # Percentage of position to close (0.0-1.0)
    size_amount: float          # Actual size to close
    order_id: Optional[str] = None    # Exchange order ID when submitted
    filled_size: float = 0.0    # Amount filled
    filled_timestamp: Optional[datetime] = None
    is_filled: bool = False     # Whether this level is completely filled

@dataclass
class TrailingStopConfig:
    """Trailing stop configuration"""
    enabled: bool = True
    initial_offset_pct: float = 0.02    # Initial trailing offset (2%)
    tightened_offset_pct: float = 0.015  # Tightened offset after TP2 (1.5%)
    activation_price: Optional[float] = None  # Price at which trailing activates
    current_stop_price: Optional[float] = None
    highest_price: Optional[float] = None    # For long positions
    lowest_price: Optional[float] = None     # For short positions
    last_update_time: Optional[datetime] = None

@dataclass
class PositionData:
    """Complete position data and state"""
    # Core position info
    symbol: str
    side: str                    # 'buy' or 'sell'
    entry_price: float
    position_size: float
    strategy_name: str
    
    # State management  
    state: PositionState = PositionState.INACTIVE
    state_changed_at: datetime = None
    
    # Order tracking
    entry_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    current_stop_price: Optional[float] = None
    
    # Progressive TP configuration
    tp_levels: List[ProgressiveTPLevel] = None
    
    # Trailing stop configuration
    trailing_config: TrailingStopConfig = None
    
    # Fill tracking
    filled_size: float = 0.0
    remaining_size: float = 0.0
    average_fill_price: float = 0.0
    
    # Risk and metadata
    original_stop_loss_pct: float = 0.02
    original_take_profit_pct: float = 0.04
    breakeven_moved: bool = False
    trailing_activated: bool = False
    
    # Persistence
    created_at: datetime = None
    last_updated_at: datetime = None
    session_id: str = ""
    
    def __post_init__(self):
        if self.state_changed_at is None:
            self.state_changed_at = datetime.now(timezone.utc)
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.last_updated_at is None:
            self.last_updated_at = datetime.now(timezone.utc)
        if self.tp_levels is None:
            self.tp_levels = []
        if self.trailing_config is None:
            self.trailing_config = TrailingStopConfig()
        if self.remaining_size == 0.0:
            self.remaining_size = self.position_size

class TrailingTPHandler:
    """
    Broker-agnostic trailing stop and progressive take profit handler.
    
    This class manages the complete lifecycle of positions with:
    - Progressive take profit execution
    - Dynamic stop loss management  
    - Trailing stop logic
    - State persistence for restart safety
    - Comprehensive logging and replay capability
    """
    
    def __init__(self, 
                 exchange_connector,
                 order_manager,
                 logger: Optional[logging.Logger] = None,
                 state_file: str = "trailing_tp_state.json",
                 monitoring_interval: float = 5.0):
        """
        Initialize the Trailing TP Handler.
        
        Args:
            exchange_connector: Exchange connector for market data and orders
            order_manager: Order manager for placing/canceling orders
            logger: Logger instance
            state_file: File path for state persistence
            monitoring_interval: Seconds between monitoring cycles
        """
        self.exchange = exchange_connector
        self.order_manager = order_manager
        # Use consistent logger name for the session
        from modules.logger import get_logger
        self.logger = logger or get_logger('trailing_tp_handler')
        self.state_file = Path(state_file)
        self.monitoring_interval = monitoring_interval
        
        # Position tracking
        self.positions: Dict[str, PositionData] = {}  # symbol -> PositionData
        self.order_to_symbol: Dict[str, str] = {}     # order_id -> symbol
        self.order_to_type: Dict[str, OrderType] = {} # order_id -> OrderType
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Monitoring
        self._monitoring_active = False
        self._monitoring_thread = None
        
        # Load persisted state
        self.load_state()
        
        self.logger.info("TrailingTPHandler initialized")
    
    def create_position(self, 
                       symbol: str,
                       side: str,
                       entry_price: float,
                       position_size: float,
                       strategy_name: str,
                       tp_levels: List[Dict[str, Any]],
                       trailing_config: Dict[str, Any],
                       stop_loss_pct: float,
                       session_id: str = "") -> str:
        """
        Create a new position for tracking.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            entry_price: Entry price
            position_size: Position size
            strategy_name: Name of the strategy
            tp_levels: List of TP level configs [{'level': 1, 'price': 45500, 'size_pct': 0.4}, ...]
            trailing_config: Trailing stop configuration
            stop_loss_pct: Initial stop loss percentage
            session_id: Trading session ID
            
        Returns:
            Position ID (symbol for now)
        """
        with self._lock:
            # Convert TP levels to ProgressiveTPLevel objects
            progressive_levels = []
            for tp_config in tp_levels:
                level_obj = ProgressiveTPLevel(
                    level=tp_config['level'],
                    price=float(tp_config['price']),
                    size_percentage=float(tp_config['size_pct']),
                    size_amount=position_size * float(tp_config['size_pct'])
                )
                progressive_levels.append(level_obj)
            
            # Create trailing configuration
            trailing_cfg = TrailingStopConfig(
                enabled=trailing_config.get('enabled', True),
                initial_offset_pct=trailing_config.get('initial_offset_pct', 0.02),
                tightened_offset_pct=trailing_config.get('tightened_offset_pct', 0.015)
            )
            
            # Create position data
            position = PositionData(
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                position_size=position_size,
                strategy_name=strategy_name,
                state=PositionState.ENTRY_PENDING,
                tp_levels=progressive_levels,
                trailing_config=trailing_cfg,
                original_stop_loss_pct=stop_loss_pct,
                session_id=session_id
            )
            
            self.positions[symbol] = position
            
            self.logger.info(f"Created position tracking for {symbol}: {side} {position_size} @ {entry_price}")
            self.logger.info(f"  TP Levels: {len(progressive_levels)}")
            self.logger.info(f"  Trailing: {'enabled' if trailing_cfg.enabled else 'disabled'}")
            
            self.save_state()
            
            return symbol
    
    def on_order_fill(self, order_id: str, fill_data: Dict[str, Any]) -> None:
        """
        Process order fill notification.
        
        Args:
            order_id: Exchange order ID that was filled
            fill_data: Fill information from exchange
        """
        with self._lock:
            try:
                if order_id not in self.order_to_symbol:
                    self.logger.debug(f"Ignoring fill for untracked order: {order_id}")
                    return
                
                symbol = self.order_to_symbol[order_id]
                order_type = self.order_to_type.get(order_id, OrderType.ENTRY)
                
                if symbol not in self.positions:
                    self.logger.warning(f"Fill for {order_id} but no position found for {symbol}")
                    return
                
                position = self.positions[symbol]
                fill_size = float(fill_data.get('size', 0))
                fill_price = float(fill_data.get('price', 0))
                
                self.logger.info(f"Processing fill: {symbol} {order_type.value} {fill_size} @ {fill_price}")
                
                # Process fill based on order type
                if order_type == OrderType.ENTRY:
                    self._process_entry_fill(position, fill_size, fill_price, fill_data)
                elif order_type in [OrderType.TAKE_PROFIT_1, OrderType.TAKE_PROFIT_2, OrderType.TAKE_PROFIT_3]:
                    self._process_tp_fill(position, order_type, fill_size, fill_price, fill_data)
                elif order_type == OrderType.STOP_LOSS:
                    self._process_stop_loss_fill(position, fill_size, fill_price, fill_data)
                elif order_type == OrderType.EXIT:
                    self._process_exit_fill(position, fill_size, fill_price, fill_data)
                
                # Update position state
                self._update_position_state(position)
                self.save_state()
                
            except Exception as e:
                self.logger.error(f"Error processing fill for {order_id}: {e}", exc_info=True)
    
    def _process_entry_fill(self, position: PositionData, fill_size: float, fill_price: float, fill_data: Dict[str, Any]) -> None:
        """Process entry order fill"""
        position.filled_size += fill_size
        position.remaining_size = position.position_size  # Remaining size for closing, not for entry
        
        # Update average fill price
        if position.average_fill_price == 0:
            position.average_fill_price = fill_price
        else:
            total_filled_value = (position.filled_size - fill_size) * position.average_fill_price + fill_size * fill_price
            position.average_fill_price = total_filled_value / position.filled_size
        
        self.logger.info(f"Entry fill: {position.symbol} filled {fill_size}, total {position.filled_size}/{position.position_size}")
        
        # If entry is fully filled, activate position management
        if position.filled_size >= position.position_size * 0.99:  # Allow for minor rounding
            position.state = PositionState.ACTIVE
            position.state_changed_at = datetime.now(timezone.utc)
            
            # Submit progressive TP orders
            self._submit_progressive_tp_orders(position)
            
            # Set initial stop loss
            self._update_stop_loss(position, breakeven=False)
            
            self.logger.info(f"Position {position.symbol} fully entered, progressive TPs submitted")
        else:
            # Partial fill - remain in ENTRY_PENDING
            self.logger.info(f"Position {position.symbol} partially filled, remaining in ENTRY_PENDING state")
    
    def _process_tp_fill(self, position: PositionData, order_type: OrderType, fill_size: float, fill_price: float, fill_data: Dict[str, Any]) -> None:
        """Process take profit order fill"""
        # Find the corresponding TP level
        tp_level_num = int(order_type.value.split('_')[-1])  # Extract number from 'take_profit_1'
        tp_level = None
        
        for level in position.tp_levels:
            if level.level == tp_level_num:
                tp_level = level
                break
        
        if not tp_level:
            self.logger.error(f"Could not find TP level {tp_level_num} for {position.symbol}")
            return
        
        # Update TP level fill data
        tp_level.filled_size += fill_size
        tp_level.filled_timestamp = datetime.now(timezone.utc)
        
        if tp_level.filled_size >= tp_level.size_amount * 0.99:  # Allow for minor rounding
            tp_level.is_filled = True
        
        # Update position remaining size
        position.remaining_size -= fill_size
        
        self.logger.info(f"TP{tp_level_num} fill: {position.symbol} closed {fill_size} @ {fill_price}, remaining: {position.remaining_size}")
        
        # Handle TP level-specific actions
        if tp_level_num == 1 and tp_level.is_filled:
            # Move stop to breakeven on TP1 fill
            self._move_stop_to_breakeven(position)
            
        elif tp_level_num == 2 and tp_level.is_filled:
            # Tighten trailing stop on TP2 fill
            self._tighten_trailing_stop(position)
            
        # Check if position should transition to pure trailing mode
        filled_tp_count = sum(1 for level in position.tp_levels if level.is_filled)
        if filled_tp_count >= 2:  # After TP2, activate pure trailing
            if not position.trailing_activated:
                position.trailing_activated = True
                self.logger.info(f"Activating pure trailing mode for {position.symbol}")
                # Force state update to transition to TRAILING_ACTIVE
                self._update_position_state(position)
    
    def _process_stop_loss_fill(self, position: PositionData, fill_size: float, fill_price: float, fill_data: Dict[str, Any]) -> None:
        """Process stop loss order fill"""
        position.remaining_size -= fill_size
        
        self.logger.info(f"Stop loss fill: {position.symbol} closed {fill_size} @ {fill_price}")
        
        if position.remaining_size <= position.position_size * 0.01:  # Position fully closed
            position.state = PositionState.CLOSED
            position.state_changed_at = datetime.now(timezone.utc)
            self._cancel_remaining_orders(position)
    
    def _process_exit_fill(self, position: PositionData, fill_size: float, fill_price: float, fill_data: Dict[str, Any]) -> None:
        """Process manual exit order fill"""
        position.remaining_size -= fill_size
        
        self.logger.info(f"Exit fill: {position.symbol} closed {fill_size} @ {fill_price}")
        
        if position.remaining_size <= position.position_size * 0.01:  # Position fully closed
            position.state = PositionState.CLOSED
            position.state_changed_at = datetime.now(timezone.utc)
            self._cancel_remaining_orders(position)
    
    def _submit_progressive_tp_orders(self, position: PositionData) -> None:
        """Submit progressive take profit orders"""
        try:
            for tp_level in position.tp_levels:
                if tp_level.order_id:  # Already submitted
                    continue
                
                # Determine order side (opposite of position)
                order_side = 'sell' if position.side == 'buy' else 'buy'
                
                # Submit TP order
                order_response = self.order_manager.place_order(
                    symbol=position.symbol,
                    side=order_side,
                    order_type='limit',
                    size=tp_level.size_amount,
                    price=tp_level.price,
                    reduce_only=True,
                    time_in_force='GoodTillCancel'
                )
                
                if order_response and 'orderId' in order_response:
                    tp_level.order_id = order_response['orderId']
                    
                    # Track order for fill monitoring
                    order_type = OrderType.TAKE_PROFIT_1 if tp_level.level == 1 else (
                        OrderType.TAKE_PROFIT_2 if tp_level.level == 2 else OrderType.TAKE_PROFIT_3
                    )
                    self.order_to_symbol[tp_level.order_id] = position.symbol
                    self.order_to_type[tp_level.order_id] = order_type
                    
                    self.logger.info(f"Submitted TP{tp_level.level} for {position.symbol}: {tp_level.size_amount} @ {tp_level.price}")
                else:
                    self.logger.error(f"Failed to submit TP{tp_level.level} for {position.symbol}")
                    
        except Exception as e:
            self.logger.error(f"Error submitting progressive TP orders for {position.symbol}: {e}", exc_info=True)
    
    def _update_stop_loss(self, position: PositionData, breakeven: bool = False) -> None:
        """Update stop loss order"""
        try:
            # Cancel existing stop loss
            if position.stop_loss_order_id:
                try:
                    self.order_manager.cancel_order(position.symbol, position.stop_loss_order_id)
                    self.logger.debug(f"Cancelled previous stop loss for {position.symbol}")
                except Exception as e:
                    self.logger.warning(f"Could not cancel previous stop loss: {e}")
            
            # Calculate new stop price
            if breakeven:
                stop_price = position.average_fill_price
                self.logger.info(f"Moving stop to breakeven for {position.symbol}: {stop_price}")
            else:
                # Initial stop loss
                stop_offset = position.original_stop_loss_pct
                if position.side == 'buy':
                    stop_price = position.entry_price * (1 - stop_offset)
                else:
                    stop_price = position.entry_price * (1 + stop_offset)
                self.logger.info(f"Setting initial stop loss for {position.symbol}: {stop_price}")
            
            # Submit new stop loss order
            order_side = 'sell' if position.side == 'buy' else 'buy'
            
            order_response = self.order_manager.place_order(
                symbol=position.symbol,
                side=order_side,
                order_type='stop_market',
                size=position.remaining_size,
                stop_price=stop_price,
                reduce_only=True,
                time_in_force='GoodTillCancel'
            )
            
            if order_response and 'orderId' in order_response:
                position.stop_loss_order_id = order_response['orderId']
                position.current_stop_price = stop_price
                
                # Track order for fill monitoring
                self.order_to_symbol[position.stop_loss_order_id] = position.symbol
                self.order_to_type[position.stop_loss_order_id] = OrderType.STOP_LOSS
                
                if breakeven:
                    position.breakeven_moved = True
                    
            else:
                self.logger.error(f"Failed to submit stop loss for {position.symbol}")
                
        except Exception as e:
            self.logger.error(f"Error updating stop loss for {position.symbol}: {e}", exc_info=True)
    
    def _move_stop_to_breakeven(self, position: PositionData) -> None:
        """Move stop loss to breakeven after TP1 fill"""
        if not position.breakeven_moved:
            self._update_stop_loss(position, breakeven=True)
            self.logger.info(f"Stop moved to breakeven for {position.symbol} after TP1 fill")
    
    def _tighten_trailing_stop(self, position: PositionData) -> None:
        """Tighten trailing stop offset after TP2 fill"""
        if position.trailing_config.enabled:
            # Update trailing configuration
            position.trailing_config.initial_offset_pct = position.trailing_config.tightened_offset_pct
            position.trailing_activated = True
            
            # Initialize trailing prices
            current_price = self._get_current_price(position.symbol)
            if current_price:
                if position.side == 'buy':
                    position.trailing_config.highest_price = current_price
                    new_stop = current_price * (1 - position.trailing_config.tightened_offset_pct)
                else:
                    position.trailing_config.lowest_price = current_price
                    new_stop = current_price * (1 + position.trailing_config.tightened_offset_pct)
                
                # Update stop loss if the new trailing stop is better
                if (position.side == 'buy' and new_stop > position.current_stop_price) or \
                   (position.side == 'sell' and new_stop < position.current_stop_price):
                    position.current_stop_price = new_stop
                    self._update_stop_loss_order(position, new_stop)
            
            self.logger.info(f"Tightened trailing stop for {position.symbol} to {position.trailing_config.tightened_offset_pct:.1%}")
    
    def _update_trailing_stops(self) -> None:
        """Update all active trailing stops based on current prices"""
        with self._lock:
            for symbol, position in self.positions.items():
                if (position.state in [PositionState.ACTIVE, PositionState.TP1_PARTIAL, PositionState.TP2_PARTIAL, PositionState.TRAILING_ACTIVE] and
                    position.trailing_config.enabled and position.trailing_activated):
                    
                    self._update_single_trailing_stop(position)
    
    def _update_single_trailing_stop(self, position: PositionData) -> None:
        """Update trailing stop for a single position"""
        try:
            current_price = self._get_current_price(position.symbol)
            if not current_price:
                return
            
            offset_pct = position.trailing_config.initial_offset_pct
            updated = False
            
            if position.side == 'buy':
                # Long position - trail up
                if position.trailing_config.highest_price is None or current_price > position.trailing_config.highest_price:
                    position.trailing_config.highest_price = current_price
                    new_stop = current_price * (1 - offset_pct)
                    
                    if new_stop > position.current_stop_price:
                        position.current_stop_price = new_stop
                        updated = True
                        
            else:
                # Short position - trail down
                if position.trailing_config.lowest_price is None or current_price < position.trailing_config.lowest_price:
                    position.trailing_config.lowest_price = current_price
                    new_stop = current_price * (1 + offset_pct)
                    
                    if new_stop < position.current_stop_price:
                        position.current_stop_price = new_stop
                        updated = True
            
            if updated:
                position.trailing_config.last_update_time = datetime.now(timezone.utc)
                self._update_stop_loss_order(position, position.current_stop_price)
                
                self.logger.debug(f"Updated trailing stop for {position.symbol}: {position.current_stop_price}")
                
        except Exception as e:
            self.logger.error(f"Error updating trailing stop for {position.symbol}: {e}")
    
    def _update_stop_loss_order(self, position: PositionData, new_stop_price: float) -> None:
        """Update the actual stop loss order on the exchange"""
        try:
            # Cancel existing stop loss
            if position.stop_loss_order_id:
                self.order_manager.cancel_order(position.symbol, position.stop_loss_order_id)
            
            # Submit new stop loss
            order_side = 'sell' if position.side == 'buy' else 'buy'
            
            order_response = self.order_manager.place_order(
                symbol=position.symbol,
                side=order_side,
                order_type='stop_market',
                size=position.remaining_size,
                stop_price=new_stop_price,
                reduce_only=True
            )
            
            if order_response and 'orderId' in order_response:
                position.stop_loss_order_id = order_response['orderId']
                self.order_to_symbol[position.stop_loss_order_id] = position.symbol
                self.order_to_type[position.stop_loss_order_id] = OrderType.STOP_LOSS
                
        except Exception as e:
            self.logger.error(f"Error updating stop loss order for {position.symbol}: {e}")
    
    def _update_position_state(self, position: PositionData) -> None:
        """Update position state based on current conditions"""
        old_state = position.state
        
        # Count filled TP levels
        filled_tp_count = sum(1 for level in position.tp_levels if level.is_filled)
        
        # Determine new state
        # Position is closed if remaining size is very small (accounting for floating point precision)
        if position.remaining_size <= position.position_size * 0.01:  # Less than 1% remaining
            position.state = PositionState.CLOSED
        # Check if transitioning to trailing mode first (takes precedence)
        elif filled_tp_count >= 2 and position.trailing_activated:
            position.state = PositionState.TRAILING_ACTIVE
        elif filled_tp_count == 0:
            position.state = PositionState.ACTIVE
        elif filled_tp_count == 1:
            position.state = PositionState.TP1_PARTIAL
        elif filled_tp_count == 2:
            position.state = PositionState.TP2_PARTIAL
        elif filled_tp_count >= 3:
            position.state = PositionState.TP3_PARTIAL
        
        # Log state changes
        if old_state != position.state:
            position.state_changed_at = datetime.now(timezone.utc)
            position.last_updated_at = datetime.now(timezone.utc)
            self.logger.info(f"Position {position.symbol} state: {old_state.value} -> {position.state.value}")
    
    def _cancel_remaining_orders(self, position: PositionData) -> None:
        """Cancel all remaining orders for a position"""
        try:
            orders_to_cancel = []
            
            # Collect unfilled TP orders
            for tp_level in position.tp_levels:
                if tp_level.order_id and not tp_level.is_filled:
                    orders_to_cancel.append(tp_level.order_id)
            
            # Add stop loss if exists
            if position.stop_loss_order_id:
                orders_to_cancel.append(position.stop_loss_order_id)
            
            # Cancel all orders
            for order_id in orders_to_cancel:
                try:
                    self.order_manager.cancel_order(position.symbol, order_id)
                    self.logger.info(f"Cancelled order {order_id} for {position.symbol}")
                except Exception as e:
                    self.logger.warning(f"Could not cancel order {order_id}: {e}")
            
            # Clean up tracking
            for order_id in orders_to_cancel:
                self.order_to_symbol.pop(order_id, None)
                self.order_to_type.pop(order_id, None)
                
        except Exception as e:
            self.logger.error(f"Error cancelling remaining orders for {position.symbol}: {e}")
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        try:
            ticker = self.exchange.get_ticker(symbol)
            if ticker and 'last' in ticker:
                return float(ticker['last'])
        except Exception as e:
            self.logger.debug(f"Could not get current price for {symbol}: {e}")
        return None
    
    def start_monitoring(self) -> None:
        """Start the monitoring thread"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info("Trailing TP monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop the monitoring thread"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        
        self.logger.info("Trailing TP monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                self._update_trailing_stops()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(self.monitoring_interval)
    
    def emergency_close_position(self, symbol: str, reason: str = "Emergency close") -> bool:
        """
        Emergency close a position, cancelling all orders and closing at market.
        
        Args:
            symbol: Symbol to close
            reason: Reason for emergency close
            
        Returns:
            True if successful
        """
        with self._lock:
            if symbol not in self.positions:
                self.logger.warning(f"No position found for emergency close: {symbol}")
                return False
            
            position = self.positions[symbol]
            
            try:
                self.logger.warning(f"EMERGENCY CLOSE: {symbol} - {reason}")
                
                # Cancel all orders
                self._cancel_remaining_orders(position)
                
                # Close remaining position at market
                if position.remaining_size > position.position_size * 0.01:
                    order_side = 'sell' if position.side == 'buy' else 'buy'
                    
                    order_response = self.order_manager.place_order(
                        symbol=symbol,
                        side=order_side,
                        order_type='market',
                        size=position.remaining_size,
                        reduce_only=True
                    )
                    
                    if order_response:
                        self.logger.info(f"Emergency market close order submitted for {symbol}")
                
                # Update position state
                position.state = PositionState.CLOSING
                position.state_changed_at = datetime.now(timezone.utc)
                
                self.save_state()
                return True
                
            except Exception as e:
                self.logger.error(f"Error during emergency close of {symbol}: {e}", exc_info=True)
                position.state = PositionState.ERROR
                return False
    
    def get_position_status(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive position status"""
        with self._lock:
            if symbol not in self.positions:
                return None
            
            position = self.positions[symbol]
            
            # Calculate filled TP levels
            filled_tps = [level for level in position.tp_levels if level.is_filled]
            pending_tps = [level for level in position.tp_levels if not level.is_filled]
            
            return {
                'symbol': position.symbol,
                'state': position.state.value,
                'side': position.side,
                'entry_price': position.entry_price,
                'position_size': position.position_size,
                'remaining_size': position.remaining_size,
                'filled_size': position.filled_size,
                'average_fill_price': position.average_fill_price,
                'current_stop_price': position.current_stop_price,
                'breakeven_moved': position.breakeven_moved,
                'trailing_activated': position.trailing_activated,
                'filled_tp_levels': len(filled_tps),
                'pending_tp_levels': len(pending_tps),
                'tp_details': [
                    {
                        'level': level.level,
                        'price': level.price,
                        'size_pct': level.size_percentage,
                        'filled': level.is_filled,
                        'order_id': level.order_id
                    }
                    for level in position.tp_levels
                ],
                'trailing_config': {
                    'enabled': position.trailing_config.enabled,
                    'offset_pct': position.trailing_config.initial_offset_pct,
                    'highest_price': position.trailing_config.highest_price,
                    'lowest_price': position.trailing_config.lowest_price
                },
                'created_at': position.created_at.isoformat() if position.created_at else None,
                'last_updated': position.last_updated_at.isoformat() if position.last_updated_at else None
            }
    
    def save_state(self) -> None:
        """Save current state to file for restart safety"""
        try:
            state_data = {
                'positions': {},
                'order_tracking': {
                    'order_to_symbol': self.order_to_symbol,
                    'order_to_type': {k: v.value for k, v in self.order_to_type.items()}
                },
                'saved_at': datetime.now(timezone.utc).isoformat()
            }
            
            # Convert positions to serializable format
            for symbol, position in self.positions.items():
                pos_dict = asdict(position)
                
                # Convert enums and datetime objects
                pos_dict['state'] = position.state.value
                pos_dict['state_changed_at'] = position.state_changed_at.isoformat() if position.state_changed_at else None
                pos_dict['created_at'] = position.created_at.isoformat() if position.created_at else None
                pos_dict['last_updated_at'] = position.last_updated_at.isoformat() if position.last_updated_at else None
                
                # Convert TP levels
                for tp_level in pos_dict['tp_levels']:
                    tp_level['filled_timestamp'] = tp_level['filled_timestamp'].isoformat() if tp_level['filled_timestamp'] else None
                
                # Convert trailing config datetime
                if pos_dict['trailing_config']['last_update_time']:
                    pos_dict['trailing_config']['last_update_time'] = position.trailing_config.last_update_time.isoformat()
                
                state_data['positions'][symbol] = pos_dict
            
            # Write to file atomically
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            temp_file.replace(self.state_file)
            
        except Exception as e:
            self.logger.error(f"Error saving state: {e}", exc_info=True)
    
    def load_state(self) -> None:
        """Load state from file for restart safety"""
        try:
            if not self.state_file.exists():
                self.logger.info("No previous state file found, starting fresh")
                return
            
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)
            
            # Restore order tracking
            order_tracking = state_data.get('order_tracking', {})
            self.order_to_symbol = order_tracking.get('order_to_symbol', {})
            
            order_type_data = order_tracking.get('order_to_type', {})
            self.order_to_type = {k: OrderType(v) for k, v in order_type_data.items()}
            
            # Restore positions
            positions_data = state_data.get('positions', {})
            for symbol, pos_data in positions_data.items():
                try:
                    # Convert back from serialized format
                    pos_data['state'] = PositionState(pos_data['state'])
                    
                    # Convert datetime strings back to datetime objects
                    if pos_data['state_changed_at']:
                        pos_data['state_changed_at'] = datetime.fromisoformat(pos_data['state_changed_at'])
                    if pos_data['created_at']:
                        pos_data['created_at'] = datetime.fromisoformat(pos_data['created_at'])
                    if pos_data['last_updated_at']:
                        pos_data['last_updated_at'] = datetime.fromisoformat(pos_data['last_updated_at'])
                    
                    # Convert TP levels
                    tp_levels = []
                    for tp_data in pos_data['tp_levels']:
                        if tp_data['filled_timestamp']:
                            tp_data['filled_timestamp'] = datetime.fromisoformat(tp_data['filled_timestamp'])
                        tp_levels.append(ProgressiveTPLevel(**tp_data))
                    pos_data['tp_levels'] = tp_levels
                    
                    # Convert trailing config
                    trailing_data = pos_data['trailing_config']
                    if trailing_data['last_update_time']:
                        trailing_data['last_update_time'] = datetime.fromisoformat(trailing_data['last_update_time'])
                    pos_data['trailing_config'] = TrailingStopConfig(**trailing_data)
                    
                    # Create position object
                    position = PositionData(**pos_data)
                    self.positions[symbol] = position
                    
                except Exception as e:
                    self.logger.error(f"Error restoring position {symbol}: {e}")
                    continue
            
            saved_at = state_data.get('saved_at')
            self.logger.info(f"Restored state from {saved_at}: {len(self.positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}", exc_info=True)
    
    def cleanup_closed_positions(self, max_age_hours: int = 24) -> None:
        """Clean up old closed positions from memory and state"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        
        with self._lock:
            symbols_to_remove = []
            
            for symbol, position in self.positions.items():
                if (position.state == PositionState.CLOSED and 
                    position.last_updated_at < cutoff_time):
                    symbols_to_remove.append(symbol)
            
            for symbol in symbols_to_remove:
                del self.positions[symbol]
                self.logger.info(f"Cleaned up closed position: {symbol}")
            
            if symbols_to_remove:
                self.save_state()
    
    def get_all_positions_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all tracked positions"""
        with self._lock:
            return {symbol: self.get_position_status(symbol) 
                   for symbol in self.positions.keys()}