from typing import Dict, List, Optional
from decimal import Decimal
import logging
from datetime import datetime
from enum import Enum

class ExposureLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class ExposureMonitor:
    def __init__(self, config: Dict):
        """Initialize exposure monitor"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize exposure limits
        self._exposure_limits = {
            "total_portfolio": Decimal("0.75"),  # 75% max total exposure
            "single_token": Decimal("0.25"),     # 25% max single token
            "token_category": Decimal("0.40"),   # 40% max per category
        }
        
        # Track exposures
        self._positions = {}
        self._category_exposure = {}
        self._alerts = []

    async def add_position(self, position_data: Dict) -> Dict:
        """Add new position to monitoring"""
        try:
            position_id = position_data["position_id"]
            self._positions[position_id] = {
                "token_address": position_data["token_address"],
                "amount": Decimal(str(position_data["amount"])),
                "current_price": Decimal(str(position_data["current_price"])),
                "category": position_data.get("category", "uncategorized"),
                "timestamp": datetime.utcnow()
            }
            
            # Update category exposure
            await self._update_category_exposure()
            
            # Check exposure limits
            exposure_status = await self.check_exposure_limits()
            
            return {
                "status": "success",
                "position_added": True,
                "exposure_status": exposure_status
            }
            
        except Exception as e:
            self.logger.error(f"Failed to add position: {e}")
            return {"status": "error", "message": str(e)}

    async def update_position(self, position_id: str, updates: Dict) -> Dict:
        """Update existing position"""
        try:
            if position_id not in self._positions:
                return {"status": "error", "message": "Position not found"}
                
            position = self._positions[position_id]
            
            # Update position data
            for key, value in updates.items():
                if key in position:
                    if isinstance(value, (int, float)):
                        position[key] = Decimal(str(value))
                    else:
                        position[key] = value
            
            # Recalculate exposures
            await self._update_category_exposure()
            exposure_status = await self.check_exposure_limits()
            
            return {
                "status": "success",
                "exposure_status": exposure_status
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update position: {e}")
            return {"status": "error", "message": str(e)}

    async def check_exposure_limits(self) -> Dict:
        """Check all exposure limits"""
        try:
            total_value = self._calculate_total_value()
            exposures = {
                "total": await self._check_total_exposure(total_value),
                "token": await self._check_token_exposure(total_value),
                "category": await self._check_category_exposure(total_value)
            }
            
            # Determine overall exposure level
            overall_level = self._determine_exposure_level(exposures)
            
            return {
                "status": "success",
                "exposure_level": overall_level.value,
                "details": exposures
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check exposure limits: {e}")
            return {"status": "error", "message": str(e)}

    async def get_exposure_metrics(self) -> Dict:
        """Get detailed exposure metrics"""
        try:
            total_value = self._calculate_total_value()
            
            metrics = {
                "total_exposure": str(total_value),
                "position_exposure": self._calculate_position_exposures(total_value),
                "category_exposure": self._calculate_category_exposures(total_value),
                "exposure_distribution": self._calculate_exposure_distribution()
            }
            
            return {
                "status": "success",
                "metrics": metrics
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get exposure metrics: {e}")
            return {"status": "error", "message": str(e)}

    def _calculate_total_value(self) -> Decimal:
        """Calculate total portfolio value"""
        return sum(
            pos["amount"] * pos["current_price"]
            for pos in self._positions.values()
        )

    async def _update_category_exposure(self) -> None:
        """Update category exposure tracking"""
        self._category_exposure = {}
        
        for position in self._positions.values():
            category = position["category"]
            value = position["amount"] * position["current_price"]
            
            if category not in self._category_exposure:
                self._category_exposure[category] = Decimal("0")
            
            self._category_exposure[category] += value

    async def _check_total_exposure(self, total_value: Decimal) -> Dict:
        """Check total portfolio exposure"""
        max_exposure = self._exposure_limits["total_portfolio"]
        current_ratio = total_value / self.config.get("portfolio_value", total_value)
        
        return {
            "current": str(current_ratio),
            "limit": str(max_exposure),
            "exceeded": current_ratio > max_exposure
        }

    async def _check_token_exposure(self, total_value: Decimal) -> Dict:
        """Check individual token exposures"""
        max_exposure = self._exposure_limits["single_token"]
        exceeded_tokens = []
        
        for position in self._positions.values():
            value = position["amount"] * position["current_price"]
            ratio = value / total_value if total_value else Decimal("0")
            
            if ratio > max_exposure:
                exceeded_tokens.append({
                    "token": position["token_address"],
                    "exposure": str(ratio)
                })
        
        return {
            "limit": str(max_exposure),
            "exceeded_tokens": exceeded_tokens
        }

    async def _check_category_exposure(self, total_value: Decimal) -> Dict:
        """Check category exposures"""
        max_exposure = self._exposure_limits["token_category"]
        exceeded_categories = []
        
        for category, value in self._category_exposure.items():
            ratio = value / total_value if total_value else Decimal("0")
            
            if ratio > max_exposure:
                exceeded_categories.append({
                    "category": category,
                    "exposure": str(ratio)
                })
        
        return {
            "limit": str(max_exposure),
            "exceeded_categories": exceeded_categories
        }

    def _determine_exposure_level(self, exposures: Dict) -> ExposureLevel:
        """Determine overall exposure level"""
        if exposures["total"]["exceeded"]:
            return ExposureLevel.CRITICAL
            
        exceeded_tokens = len(exposures["token"]["exceeded_tokens"])
        exceeded_categories = len(exposures["category"]["exceeded_categories"])
        
        if exceeded_tokens > 0 or exceeded_categories > 0:
            return ExposureLevel.HIGH
        
        total_ratio = Decimal(exposures["total"]["current"])
        if total_ratio > Decimal("0.6"):
            return ExposureLevel.MODERATE
            
        return ExposureLevel.LOW

    def _calculate_position_exposures(self, total_value: Decimal) -> Dict:
        """Calculate exposure for each position"""
        exposures = {}
        
        for pos_id, position in self._positions.items():
            value = position["amount"] * position["current_price"]
            ratio = value / total_value if total_value else Decimal("0")
            
            exposures[pos_id] = {
                "token": position["token_address"],
                "value": str(value),
                "ratio": str(ratio)
            }
            
        return exposures

    def _calculate_category_exposures(self, total_value: Decimal) -> Dict:
        """Calculate exposure for each category"""
        return {
            category: {
                "value": str(value),
                "ratio": str(value / total_value if total_value else Decimal("0"))
            }
            for category, value in self._category_exposure.items()
        }

    def _calculate_exposure_distribution(self) -> Dict:
        """Calculate exposure distribution"""
        total_positions = len(self._positions)
        if total_positions == 0:
            return {"low": "0", "moderate": "0", "high": "0"}
            
        exposures = []
        total_value = self._calculate_total_value()
        
        for position in self._positions.values():
            value = position["amount"] * position["current_price"]
            ratio = value / total_value if total_value else Decimal("0")
            exposures.append(ratio)
        
        return {
            "low": str(len([e for e in exposures if e <= Decimal("0.1")]) / total_positions),
            "moderate": str(len([e for e in exposures if Decimal("0.1") < e <= Decimal("0.2")]) / total_positions),
            "high": str(len([e for e in exposures if e > Decimal("0.2")]) / total_positions)
        }