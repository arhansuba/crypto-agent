from typing import Dict, List, Optional
from datetime import datetime
import logging
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class Position:
    token_address: str
    amount: Decimal
    entry_price: Decimal
    current_price: Decimal
    timestamp: datetime

class PortfolioManager:
    def __init__(self, config: Dict):
        """Initialize portfolio manager"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._positions = {}
        self._risk_limits = {
            RiskLevel.LOW: Decimal("0.02"),    # 2% per trade
            RiskLevel.MEDIUM: Decimal("0.05"), # 5% per trade
            RiskLevel.HIGH: Decimal("0.10")    # 10% per trade
        }

    async def add_position(self, user_id: int, position_data: Dict) -> Dict:
        """Add new position to portfolio"""
        try:
            position = Position(
                token_address=position_data["token_address"],
                amount=Decimal(str(position_data["amount"])),
                entry_price=Decimal(str(position_data["entry_price"])),
                current_price=Decimal(str(position_data["current_price"])),
                timestamp=datetime.utcnow()
            )

            # Initialize user portfolio if not exists
            if user_id not in self._positions:
                self._positions[user_id] = {}

            self._positions[user_id][position.token_address] = position

            # Calculate initial risk metrics
            risk_metrics = await self._calculate_position_risk(position)

            return {
                "status": "success",
                "position": position,
                "risk_metrics": risk_metrics
            }

        except Exception as e:
            self.logger.error(f"Failed to add position: {e}")
            return {"status": "error", "message": str(e)}

    async def update_position(self, user_id: int, token_address: str, updates: Dict) -> Dict:
        """Update existing position"""
        try:
            if not self._positions.get(user_id, {}).get(token_address):
                return {"status": "error", "message": "Position not found"}

            position = self._positions[user_id][token_address]

            # Update position attributes
            for key, value in updates.items():
                if hasattr(position, key):
                    setattr(position, key, Decimal(str(value)) if isinstance(value, (int, float)) else value)

            # Recalculate risk metrics
            risk_metrics = await self._calculate_position_risk(position)

            return {
                "status": "success",
                "position": position,
                "risk_metrics": risk_metrics
            }

        except Exception as e:
            self.logger.error(f"Failed to update position: {e}")
            return {"status": "error", "message": str(e)}

    async def get_portfolio_overview(self, user_id: int) -> Dict:
        """Get portfolio overview with risk metrics"""
        try:
            if user_id not in self._positions:
                return {
                    "status": "success",
                    "portfolio": {
                        "total_value": Decimal("0"),
                        "positions": [],
                        "risk_metrics": self._get_empty_risk_metrics()
                    }
                }

            positions = self._positions[user_id]
            total_value = sum(p.amount * p.current_price for p in positions.values())
            position_data = []

            for position in positions.values():
                pnl = self._calculate_pnl(position)
                risk_metrics = await self._calculate_position_risk(position)

                position_data.append({
                    "token_address": position.token_address,
                    "amount": str(position.amount),
                    "current_value": str(position.amount * position.current_price),
                    "pnl_percentage": str(pnl["pnl_percentage"]),
                    "pnl_value": str(pnl["pnl_value"]),
                    "risk_metrics": risk_metrics
                })

            portfolio_risk = await self._calculate_portfolio_risk(user_id)

            return {
                "status": "success",
                "portfolio": {
                    "total_value": str(total_value),
                    "positions": position_data,
                    "risk_metrics": portfolio_risk
                }
            }

        except Exception as e:
            self.logger.error(f"Failed to get portfolio overview: {e}")
            return {"status": "error", "message": str(e)}

    async def calculate_position_size(
        self,
        user_id: int,
        capital: Decimal,
        risk_level: RiskLevel,
        stop_loss_percentage: Decimal
    ) -> Dict:
        """Calculate safe position size based on risk parameters"""
        try:
            # Get risk limit based on risk level
            risk_limit = self._risk_limits[risk_level]

            # Calculate maximum loss amount
            max_loss = capital * risk_limit

            # Calculate position size based on stop loss
            position_size = max_loss / (stop_loss_percentage / Decimal("100"))

            return {
                "status": "success",
                "position_size": str(position_size),
                "max_loss": str(max_loss),
                "risk_percentage": str(risk_limit * Decimal("100"))
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate position size: {e}")
            return {"status": "error", "message": str(e)}

    def _calculate_pnl(self, position: Position) -> Dict:
        """Calculate position P&L"""
        current_value = position.amount * position.current_price
        entry_value = position.amount * position.entry_price
        pnl_value = current_value - entry_value
        pnl_percentage = (pnl_value / entry_value) * Decimal("100") if entry_value != 0 else Decimal("0")

        return {
            "pnl_value": pnl_value,
            "pnl_percentage": pnl_percentage
        }

    async def _calculate_position_risk(self, position: Position) -> Dict:
        """Calculate risk metrics for position"""
        try:
            current_value = position.amount * position.current_price
            entry_value = position.amount * position.entry_price

            return {
                "value_at_risk": str(current_value * Decimal("0.05")),  # 5% VaR
                "max_drawdown": str(max(Decimal("0"), entry_value - current_value)),
                "exposure_percentage": str((current_value / self._get_total_portfolio_value(position)) * Decimal("100")),
                "risk_score": self._calculate_risk_score(position)
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate position risk: {e}")
            return self._get_empty_risk_metrics()

    async def _calculate_portfolio_risk(self, user_id: int) -> Dict:
        """Calculate overall portfolio risk metrics"""
        try:
            if user_id not in self._positions:
                return self._get_empty_risk_metrics()

            positions = self._positions[user_id]
            total_value = sum(p.amount * p.current_price for p in positions.values())
            total_exposure = sum(max(Decimal("0"), (p.amount * p.current_price) - (p.amount * p.entry_price)) 
                               for p in positions.values())

            return {
                "total_exposure": str(total_exposure),
                "exposure_percentage": str((total_exposure / total_value) * Decimal("100")) if total_value else "0",
                "largest_position_percentage": self._get_largest_position_percentage(user_id),
                "risk_distribution": self._calculate_risk_distribution(user_id)
            }

        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio risk: {e}")
            return self._get_empty_risk_metrics()

    def _get_empty_risk_metrics(self) -> Dict:
        """Get empty risk metrics template"""
        return {
            "value_at_risk": "0",
            "max_drawdown": "0",
            "exposure_percentage": "0",
            "risk_score": "0"
        }

    def _calculate_risk_score(self, position: Position) -> str:
        """Calculate risk score (0-100) for position"""
        # Implement your risk scoring logic here
        return "50"

    def _get_largest_position_percentage(self, user_id: int) -> str:
        """Get largest position as percentage of portfolio"""
        try:
            positions = self._positions[user_id]
            if not positions:
                return "0"

            total_value = sum(p.amount * p.current_price for p in positions.values())
            largest_value = max(p.amount * p.current_price for p in positions.values())

            return str((largest_value / total_value) * Decimal("100")) if total_value else "0"

        except Exception:
            return "0"

    def _calculate_risk_distribution(self, user_id: int) -> Dict:
        """Calculate risk distribution across positions"""
        try:
            positions = self._positions[user_id]
            total_value = sum(p.amount * p.current_price for p in positions.values())
            
            risk_distribution = {
                "high_risk": Decimal("0"),
                "medium_risk": Decimal("0"),
                "low_risk": Decimal("0")
            }

            for position in positions.values():
                position_value = position.amount * position.current_price
                risk_score = Decimal(self._calculate_risk_score(position))

                if risk_score >= Decimal("70"):
                    risk_distribution["high_risk"] += position_value
                elif risk_score >= Decimal("30"):
                    risk_distribution["medium_risk"] += position_value
                else:
                    risk_distribution["low_risk"] += position_value

            # Convert to percentages
            if total_value:
                for key in risk_distribution:
                    risk_distribution[key] = str((risk_distribution[key] / total_value) * Decimal("100"))

            return risk_distribution

        except Exception as e:
            self.logger.error(f"Failed to calculate risk distribution: {e}")
            return {"high_risk": "0", "medium_risk": "0", "low_risk": "0"}

    def _get_total_portfolio_value(self, position: Position) -> Decimal:
        """Get total portfolio value for position's user"""
        try:
            for user_id, positions in self._positions.items():
                if position.token_address in positions:
                    return sum(p.amount * p.current_price for p in positions.values())
            return Decimal("0")
        except Exception:
            return Decimal("0")