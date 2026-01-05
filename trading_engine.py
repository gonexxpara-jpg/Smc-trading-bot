"""
Trading Engine Module for SMC Trading Bot
Handles multi-user trading execution, position management, order handling, and risk management
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, asdict, field
from decimal import Decimal
import asyncio
from abc import ABC, abstractmethod
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Models
# ============================================================================

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class PositionSide(Enum):
    """Position side enumeration"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class TimeInForce(Enum):
    """Time in force enumeration"""
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    DAY = "day"  # Day order


@dataclass
class Order:
    """Order data model"""
    order_id: str
    user_id: str
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    executed_price: float = 0.0
    commission: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert order to dictionary"""
        data = asdict(self)
        data['order_type'] = self.order_type.value
        data['status'] = self.status.value
        data['side'] = self.side
        data['time_in_force'] = self.time_in_force.value
        return data


@dataclass
class Position:
    """Position data model"""
    position_id: str
    user_id: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission_paid: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = field(default_factory=dict)

    def calculate_pnl(self, current_price: float) -> Tuple[float, float]:
        """Calculate unrealized and realized PnL"""
        if self.side == PositionSide.LONG:
            unrealized_pnl = (current_price - self.entry_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            unrealized_pnl = (self.entry_price - current_price) * self.quantity
        else:
            unrealized_pnl = 0.0

        return unrealized_pnl, self.realized_pnl

    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        data = asdict(self)
        data['side'] = self.side.value
        return data


@dataclass
class User:
    """User account data model"""
    user_id: str
    username: str
    email: str
    account_balance: float
    available_balance: float
    used_margin: float = 0.0
    total_commission: float = 0.0
    risk_limit_percentage: float = 5.0  # Max risk per trade
    max_positions: int = 10
    leverage: float = 1.0
    is_active: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        """Convert user to dictionary"""
        return asdict(self)


@dataclass
class RiskMetrics:
    """Risk management metrics"""
    user_id: str
    total_exposure: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
        """Convert risk metrics to dictionary"""
        return asdict(self)


# ============================================================================
# Trading Engine Core
# ============================================================================

class TradingEngine:
    """
    Main trading engine for multi-user platform
    Manages all trading operations, positions, orders, and risk
    """

    def __init__(self, commission_rate: float = 0.001):
        """
        Initialize trading engine
        
        Args:
            commission_rate: Commission percentage per trade (default 0.1%)
        """
        self.commission_rate = commission_rate
        self.users: Dict[str, User] = {}
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, List[Position]] = {}  # user_id -> positions
        self.order_history: Dict[str, List[Order]] = {}  # user_id -> order history
        self.risk_metrics: Dict[str, RiskMetrics] = {}
        self.market_prices: Dict[str, float] = {}  # symbol -> current price
        self.logger = logger

    # ========================================================================
    # User Management
    # ========================================================================

    def register_user(self, user_id: str, username: str, email: str, 
                     initial_balance: float, leverage: float = 1.0) -> User:
        """
        Register a new user
        
        Args:
            user_id: Unique user identifier
            username: Username
            email: Email address
            initial_balance: Initial account balance
            leverage: Trading leverage (default 1.0)
            
        Returns:
            User object
        """
        if user_id in self.users:
            raise ValueError(f"User {user_id} already registered")

        user = User(
            user_id=user_id,
            username=username,
            email=email,
            account_balance=initial_balance,
            available_balance=initial_balance,
            leverage=leverage
        )

        self.users[user_id] = user
        self.positions[user_id] = []
        self.order_history[user_id] = []
        self.logger.info(f"User registered: {user_id} ({username})")
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)

    def update_user_balance(self, user_id: str, amount: float, 
                           operation: str = "deposit") -> bool:
        """
        Update user balance
        
        Args:
            user_id: User ID
            amount: Amount to add/subtract
            operation: 'deposit' or 'withdraw'
            
        Returns:
            True if successful
        """
        user = self.get_user(user_id)
        if not user:
            self.logger.error(f"User not found: {user_id}")
            return False

        if operation == "deposit":
            user.account_balance += amount
            user.available_balance += amount
        elif operation == "withdraw":
            if user.available_balance < amount:
                self.logger.error(f"Insufficient balance for withdrawal")
                return False
            user.account_balance -= amount
            user.available_balance -= amount

        user.updated_at = datetime.utcnow().isoformat()
        self.logger.info(f"User balance updated: {user_id}, {operation}: {amount}")
        return True

    # ========================================================================
    # Order Management
    # ========================================================================

    def create_order(self, user_id: str, symbol: str, order_type: OrderType,
                    side: str, quantity: float, price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    time_in_force: TimeInForce = TimeInForce.GTC,
                    metadata: Optional[Dict] = None) -> Order:
        """
        Create a new order
        
        Args:
            user_id: User ID
            symbol: Trading symbol
            order_type: Type of order
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force
            metadata: Additional metadata
            
        Returns:
            Order object
        """
        user = self.get_user(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")

        if quantity <= 0:
            raise ValueError("Quantity must be positive")

        if side not in ['buy', 'sell']:
            raise ValueError("Side must be 'buy' or 'sell'")

        order_id = f"{user_id}_{symbol}_{int(datetime.utcnow().timestamp()*1000)}"
        
        order = Order(
            order_id=order_id,
            user_id=user_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            metadata=metadata or {}
        )

        # Validation
        if not self._validate_order(user, order):
            order.status = OrderStatus.REJECTED
            self.logger.warning(f"Order rejected: {order_id}")
        else:
            order.status = OrderStatus.OPEN

        self.orders[order_id] = order
        if user_id not in self.order_history:
            self.order_history[user_id] = []
        self.order_history[user_id].append(order)

        self.logger.info(f"Order created: {order_id} - {side} {quantity} {symbol} @ {price}")
        return order

    def _validate_order(self, user: User, order: Order) -> bool:
        """
        Validate order against user risk limits
        
        Args:
            user: User object
            order: Order to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check balance
        estimated_cost = order.quantity * (order.price or 1.0) * (1 + self.commission_rate)
        if estimated_cost > user.available_balance:
            self.logger.warning(f"Insufficient balance for order: {order.order_id}")
            return False

        # Check position count
        user_positions = self.positions.get(user.user_id, [])
        if len(user_positions) >= user.max_positions:
            self.logger.warning(f"Max positions reached for user: {user.user_id}")
            return False

        # Check risk limit
        if not self._check_risk_limit(user, order):
            return False

        return True

    def _check_risk_limit(self, user: User, order: Order) -> bool:
        """Check if order respects risk management limits"""
        # Risk = (quantity * price * risk_percentage)
        risk_amount = order.quantity * (order.price or 1.0) * (user.risk_limit_percentage / 100)
        max_risk = user.account_balance * (user.risk_limit_percentage / 100)

        if risk_amount > max_risk:
            self.logger.warning(f"Order exceeds risk limit: {order.order_id}")
            return False

        return True

    def execute_order(self, order_id: str, executed_price: float) -> bool:
        """
        Execute an order at specified price
        
        Args:
            order_id: Order ID
            executed_price: Execution price
            
        Returns:
            True if successful
        """
        order = self.orders.get(order_id)
        if not order:
            self.logger.error(f"Order not found: {order_id}")
            return False

        if order.status == OrderStatus.FILLED:
            self.logger.warning(f"Order already filled: {order_id}")
            return False

        user = self.get_user(order.user_id)
        if not user:
            return False

        # Update order
        order.filled_quantity = order.quantity
        order.executed_price = executed_price
        order.status = OrderStatus.FILLED
        order.updated_at = datetime.utcnow().isoformat()

        # Calculate commission
        commission = order.quantity * executed_price * self.commission_rate
        order.commission = commission
        user.total_commission += commission

        # Create or update position
        self._update_position(order, executed_price)

        # Update user balance
        trade_value = order.quantity * executed_price + commission
        if order.side == 'buy':
            user.available_balance -= trade_value
            user.used_margin += trade_value / user.leverage
        else:
            user.available_balance += trade_value
            user.used_margin -= trade_value / user.leverage

        self.logger.info(f"Order executed: {order_id} @ {executed_price}")
        return True

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order
        
        Args:
            order_id: Order ID
            
        Returns:
            True if successful
        """
        order = self.orders.get(order_id)
        if not order:
            self.logger.error(f"Order not found: {order_id}")
            return False

        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self.logger.warning(f"Cannot cancel order with status: {order.status.value}")
            return False

        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.utcnow().isoformat()

        self.logger.info(f"Order cancelled: {order_id}")
        return True

    def modify_order(self, order_id: str, quantity: Optional[float] = None,
                    price: Optional[float] = None) -> bool:
        """
        Modify an open order
        
        Args:
            order_id: Order ID
            quantity: New quantity
            price: New price
            
        Returns:
            True if successful
        """
        order = self.orders.get(order_id)
        if not order:
            self.logger.error(f"Order not found: {order_id}")
            return False

        if order.status != OrderStatus.OPEN:
            self.logger.warning(f"Cannot modify order with status: {order.status.value}")
            return False

        if quantity and quantity > 0:
            order.quantity = quantity
        if price:
            order.price = price

        order.updated_at = datetime.utcnow().isoformat()
        self.logger.info(f"Order modified: {order_id}")
        return True

    def get_user_orders(self, user_id: str, status: Optional[OrderStatus] = None) -> List[Order]:
        """Get user's orders, optionally filtered by status"""
        user_orders = self.order_history.get(user_id, [])
        if status:
            return [o for o in user_orders if o.status == status]
        return user_orders

    # ========================================================================
    # Position Management
    # ========================================================================

    def _update_position(self, order: Order, execution_price: float) -> Position:
        """
        Update position after order execution
        
        Args:
            order: Executed order
            execution_price: Execution price
            
        Returns:
            Updated position
        """
        user_id = order.user_id
        symbol = order.symbol
        side = PositionSide.LONG if order.side == 'buy' else PositionSide.SHORT

        # Find existing position
        positions = self.positions.get(user_id, [])
        existing_position = next(
            (p for p in positions if p.symbol == symbol and p.side == side),
            None
        )

        if existing_position:
            # Update existing position
            total_quantity = existing_position.quantity + order.quantity
            total_cost = (existing_position.entry_price * existing_position.quantity +
                         execution_price * order.quantity)
            new_entry_price = total_cost / total_quantity

            existing_position.quantity = total_quantity
            existing_position.entry_price = new_entry_price
            existing_position.commission_paid += order.commission
            existing_position.updated_at = datetime.utcnow().isoformat()

            return existing_position
        else:
            # Create new position
            position_id = f"pos_{user_id}_{symbol}_{int(datetime.utcnow().timestamp()*1000)}"
            position = Position(
                position_id=position_id,
                user_id=user_id,
                symbol=symbol,
                side=side,
                quantity=order.quantity,
                entry_price=execution_price,
                current_price=execution_price,
                commission_paid=order.commission
            )

            if user_id not in self.positions:
                self.positions[user_id] = []
            self.positions[user_id].append(position)

            self.logger.info(f"Position created: {position_id}")
            return position

    def close_position(self, position_id: str, execution_price: float) -> bool:
        """
        Close a position
        
        Args:
            position_id: Position ID
            execution_price: Close price
            
        Returns:
            True if successful
        """
        # Find position
        position = None
        user_id = None
        for uid, positions in self.positions.items():
            pos = next((p for p in positions if p.position_id == position_id), None)
            if pos:
                position = pos
                user_id = uid
                break

        if not position:
            self.logger.error(f"Position not found: {position_id}")
            return False

        # Calculate realized PnL
        if position.side == PositionSide.LONG:
            realized_pnl = (execution_price - position.entry_price) * position.quantity
        else:
            realized_pnl = (position.entry_price - execution_price) * position.quantity

        position.realized_pnl = realized_pnl
        position.quantity = 0
        position.updated_at = datetime.utcnow().isoformat()

        # Update user balance
        user = self.get_user(user_id)
        if user:
            user.available_balance += realized_pnl - position.commission_paid

        self.logger.info(f"Position closed: {position_id}, PnL: {realized_pnl}")
        return True

    def get_user_positions(self, user_id: str, open_only: bool = True) -> List[Position]:
        """
        Get user's positions
        
        Args:
            user_id: User ID
            open_only: If True, return only open positions
            
        Returns:
            List of positions
        """
        positions = self.positions.get(user_id, [])
        if open_only:
            return [p for p in positions if p.quantity > 0]
        return positions

    def update_position_prices(self, symbol: str, current_price: float) -> None:
        """
        Update prices for all positions of a symbol
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
        """
        self.market_prices[symbol] = current_price

        for user_id, positions in self.positions.items():
            for position in positions:
                if position.symbol == symbol:
                    position.current_price = current_price
                    unrealized_pnl, realized_pnl = position.calculate_pnl(current_price)
                    position.unrealized_pnl = unrealized_pnl

    def set_stop_loss(self, position_id: str, stop_loss_price: float) -> bool:
        """Set stop loss for a position"""
        position = self._find_position(position_id)
        if not position:
            return False

        position.stop_loss = stop_loss_price
        position.updated_at = datetime.utcnow().isoformat()
        self.logger.info(f"Stop loss set for {position_id}: {stop_loss_price}")
        return True

    def set_take_profit(self, position_id: str, take_profit_price: float) -> bool:
        """Set take profit for a position"""
        position = self._find_position(position_id)
        if not position:
            return False

        position.take_profit = take_profit_price
        position.updated_at = datetime.utcnow().isoformat()
        self.logger.info(f"Take profit set for {position_id}: {take_profit_price}")
        return True

    def _find_position(self, position_id: str) -> Optional[Position]:
        """Find position by ID"""
        for positions in self.positions.values():
            for position in positions:
                if position.position_id == position_id:
                    return position
        return None

    # ========================================================================
    # Risk Management
    # ========================================================================

    def calculate_risk_metrics(self, user_id: str) -> RiskMetrics:
        """
        Calculate risk metrics for a user
        
        Args:
            user_id: User ID
            
        Returns:
            RiskMetrics object
        """
        user = self.get_user(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")

        positions = self.get_user_positions(user_id, open_only=True)
        orders = self.get_user_orders(user_id)

        # Calculate total exposure
        total_exposure = sum(
            p.quantity * p.current_price for p in positions
        )

        # Calculate metrics from order history
        completed_orders = [o for o in orders if o.status == OrderStatus.FILLED]
        winning_trades = 0
        losing_trades = 0
        total_profit = 0.0
        total_loss = 0.0

        for order in completed_orders:
            if order.executed_price > order.price:
                winning_trades += 1
                total_profit += (order.executed_price - order.price) * order.quantity
            elif order.executed_price < order.price:
                losing_trades += 1
                total_loss += (order.price - order.executed_price) * order.quantity

        total_trades = len(completed_orders)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_win = total_profit / winning_trades if winning_trades > 0 else 0
        avg_loss = total_loss / losing_trades if losing_trades > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else (
            total_profit if total_profit > 0 else 0
        )

        # Simplified max drawdown (could be enhanced)
        max_drawdown = 0.0  # Implement based on equity curve

        # Simplified Sharpe ratio (could be enhanced)
        sharpe_ratio = 0.0  # Implement based on returns

        metrics = RiskMetrics(
            user_id=user_id,
            total_exposure=total_exposure,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades
        )

        self.risk_metrics[user_id] = metrics
        self.logger.info(f"Risk metrics calculated for {user_id}")
        return metrics

    def check_risk_alerts(self, user_id: str) -> List[str]:
        """
        Check for risk alerts
        
        Args:
            user_id: User ID
            
        Returns:
            List of alert messages
        """
        alerts = []
        user = self.get_user(user_id)
        if not user:
            return alerts

        positions = self.get_user_positions(user_id, open_only=True)
        total_exposure = sum(p.quantity * p.current_price for p in positions)

        # Check exposure limit (e.g., 50% of account)
        if total_exposure > user.account_balance * 0.5:
            alerts.append(f"High exposure alert: {total_exposure} ({total_exposure/user.account_balance*100:.1f}% of account)")

        # Check individual position drawdowns
        for position in positions:
            if position.unrealized_pnl < 0:
                loss_percent = abs(position.unrealized_pnl) / (position.entry_price * position.quantity) * 100
                if loss_percent > 5:  # 5% loss threshold
                    alerts.append(f"Position {position.symbol} down {loss_percent:.1f}%")

        # Check if margin ratio is too high
        if user.used_margin > 0:
            margin_ratio = user.used_margin / user.account_balance
            if margin_ratio > 0.8:
                alerts.append(f"High margin usage: {margin_ratio*100:.1f}%")

        return alerts

    def enforce_stop_loss_and_take_profit(self) -> List[Tuple[str, str]]:
        """
        Check and enforce stop loss and take profit orders
        
        Returns:
            List of (position_id, action) tuples
        """
        actions = []

        for user_id, positions in self.positions.items():
            for position in positions:
                if position.quantity == 0:
                    continue

                # Check stop loss
                if position.stop_loss:
                    if position.side == PositionSide.LONG and position.current_price <= position.stop_loss:
                        self.close_position(position.position_id, position.stop_loss)
                        actions.append((position.position_id, "stop_loss_triggered"))
                        self.logger.warning(f"Stop loss triggered for {position.position_id}")

                    elif position.side == PositionSide.SHORT and position.current_price >= position.stop_loss:
                        self.close_position(position.position_id, position.stop_loss)
                        actions.append((position.position_id, "stop_loss_triggered"))
                        self.logger.warning(f"Stop loss triggered for {position.position_id}")

                # Check take profit
                if position.take_profit:
                    if position.side == PositionSide.LONG and position.current_price >= position.take_profit:
                        self.close_position(position.position_id, position.take_profit)
                        actions.append((position.position_id, "take_profit_triggered"))
                        self.logger.info(f"Take profit triggered for {position.position_id}")

                    elif position.side == PositionSide.SHORT and position.current_price <= position.take_profit:
                        self.close_position(position.position_id, position.take_profit)
                        actions.append((position.position_id, "take_profit_triggered"))
                        self.logger.info(f"Take profit triggered for {position.position_id}")

        return actions

    # ========================================================================
    # Account Summary
    # ========================================================================

    def get_account_summary(self, user_id: str) -> Dict:
        """
        Get comprehensive account summary
        
        Args:
            user_id: User ID
            
        Returns:
            Account summary dictionary
        """
        user = self.get_user(user_id)
        if not user:
            raise ValueError(f"User not found: {user_id}")

        positions = self.get_user_positions(user_id, open_only=True)
        orders = self.get_user_orders(user_id)
        risk_metrics = self.risk_metrics.get(user_id)

        total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        total_realized_pnl = sum(p.realized_pnl for p in self.get_user_positions(user_id, open_only=False))

        return {
            "user_id": user_id,
            "username": user.username,
            "account_balance": user.account_balance,
            "available_balance": user.available_balance,
            "used_margin": user.used_margin,
            "total_commission": user.total_commission,
            "open_positions": len(positions),
            "total_positions": len(self.get_user_positions(user_id, open_only=False)),
            "open_orders": len([o for o in orders if o.status == OrderStatus.OPEN]),
            "total_orders": len(orders),
            "unrealized_pnl": total_unrealized_pnl,
            "realized_pnl": total_realized_pnl,
            "total_pnl": total_unrealized_pnl + total_realized_pnl,
            "equity": user.account_balance + total_unrealized_pnl,
            "risk_metrics": risk_metrics.to_dict() if risk_metrics else None,
            "timestamp": datetime.utcnow().isoformat()
        }

    def export_data(self, user_id: str) -> Dict:
        """Export all user data"""
        return {
            "user": self.get_user(user_id).to_dict(),
            "positions": [p.to_dict() for p in self.get_user_positions(user_id, open_only=False)],
            "orders": [o.to_dict() for o in self.get_user_orders(user_id)],
            "summary": self.get_account_summary(user_id),
            "timestamp": datetime.utcnow().isoformat()
        }


# ============================================================================
# Trading Strategy Base Class
# ============================================================================

class TradingStrategy(ABC):
    """Base class for trading strategies"""

    def __init__(self, engine: TradingEngine, user_id: str):
        """Initialize strategy"""
        self.engine = engine
        self.user_id = user_id
        self.logger = logger

    @abstractmethod
    def analyze(self, market_data: Dict) -> Dict:
        """
        Analyze market data and generate signals
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Analysis results with signal
        """
        pass

    @abstractmethod
    def execute(self, signal: Dict) -> Optional[Order]:
        """
        Execute trade based on signal
        
        Args:
            signal: Trading signal
            
        Returns:
            Order object if created
        """
        pass


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Initialize engine
    engine = TradingEngine(commission_rate=0.001)

    # Register users
    user1 = engine.register_user(
        user_id="user_001",
        username="trader_john",
        email="john@example.com",
        initial_balance=10000.0
    )

    user2 = engine.register_user(
        user_id="user_002",
        username="trader_jane",
        email="jane@example.com",
        initial_balance=5000.0
    )

    # Create orders
    order1 = engine.create_order(
        user_id="user_001",
        symbol="BTC/USD",
        order_type=OrderType.LIMIT,
        side="buy",
        quantity=0.5,
        price=40000.0
    )

    # Execute order
    engine.execute_order(order1.order_id, 40000.0)

    # Update position prices
    engine.update_position_prices("BTC/USD", 41000.0)

    # Get account summary
    summary = engine.get_account_summary("user_001")
    print(json.dumps(summary, indent=2))

    # Calculate risk metrics
    metrics = engine.calculate_risk_metrics("user_001")
    print(f"\nRisk Metrics: {metrics.to_dict()}")

    # Check risk alerts
    alerts = engine.check_risk_alerts("user_001")
    print(f"\nRisk Alerts: {alerts}")
