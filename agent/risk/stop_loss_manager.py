from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
import logging
import asyncio
from enum import Enum

class StopType(Enum):
    FIXED = "fixed"
    TRAILING = "trailing"
    DYNAMIC = "dynamic"
    TIME = "time"

class StopLossManager:
    def __init__(self, config: Dict):
        """Initialize stop loss manager"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._active_stops = {}
        self._stop_monitoring_tasks = {}
        self._execution_callback = None

    async def set_stop_loss(self, position_id: str, stop_config: Dict) -> Dict:
        """Set stop loss for a position"""
        try:
            stop_type = StopType(stop_config.get("type", "fixed"))
            
            stop_data = {
                "position_id": position_id,
                "type": stop_type,
                "entry_price": Decimal(str(stop_config["entry_price"])),
                "current_price": Decimal(str(stop_config["current_price"])),
                "stop_price": Decimal(str(stop_config["stop_price"])),
                "quantity": Decimal(str(stop_config["quantity"])),
                "trailing_distance": Decimal(str(stop_config.get("trailing_distance", "0"))),
                "created_at": datetime.utcnow(),
                "status": "active"
            }
            
            # Add to active stops
            self._active_stops[position_id] = stop_data
            
            # Start monitoring task
            self._stop_monitoring_tasks[position_id] = asyncio.create_task(
                self._monitor_stop_loss(position_id)
            )
            
            return {
                "status": "success",
                "stop_data": stop_data
            }
            
        except Exception as e:
            self.logger.error(f"Failed to set stop loss: {e}")
            return {"status": "error", "message": str(e)}

    async def update_stop_loss(self, position_id: str, updates: Dict) -> Dict:
        """Update existing stop loss"""
        try:
            if position_id not in self._active_stops:
                return {"status": "error", "message": "Stop loss not found"}
                
            stop_data = self._active_stops[position_id]
            
            # Update fields
            for key, value in updates.items():
                if key in stop_data:
                    stop_data[key] = Decimal(str(value)) if isinstance(value, (int, float)) else value
            
            return {
                "status": "success",
                "stop_data": stop_data
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update stop loss: {e}")
            return {"status": "error", "message": str(e)}

    def set_execution_callback(self, callback) -> None:
        """Set callback for stop loss execution"""
        self._execution_callback = callback

    async def cancel_stop_loss(self, position_id: str) -> Dict:
        """Cancel stop loss order"""
        try:
            if position_id in self._stop_monitoring_tasks:
                self._stop_monitoring_tasks[position_id].cancel()
                del self._stop_monitoring_tasks[position_id]
            
            if position_id in self._active_stops:
                del self._active_stops[position_id]
            
            return {
                "status": "success",
                "message": "Stop loss cancelled"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to cancel stop loss: {e}")
            return {"status": "error", "message": str(e)}

    async def update_price(self, position_id: str, current_price: Decimal) -> Dict:
        """Update current price and check stop loss"""
        try:
            if position_id not in self._active_stops:
                return {"status": "error", "message": "Stop loss not found"}
            
            stop_data = self._active_stops[position_id]
            stop_data["current_price"] = current_price
            
            # Update trailing stop if applicable
            if stop_data["type"] == StopType.TRAILING:
                await self._update_trailing_stop(position_id, current_price)
            
            # Check if stop loss triggered
            if await self._is_stop_triggered(position_id):
                await self._execute_stop_loss(position_id)
                return {
                    "status": "success",
                    "message": "Stop loss triggered",
                    "execution_price": str(current_price)
                }
            
            return {
                "status": "success",
                "message": "Price updated"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update price: {e}")
            return {"status": "error", "message": str(e)}

    async def _monitor_stop_loss(self, position_id: str) -> None:
        """Monitor stop loss conditions"""
        try:
            while position_id in self._active_stops:
                stop_data = self._active_stops[position_id]
                
                if await self._is_stop_triggered(position_id):
                    await self._execute_stop_loss(position_id)
                    break
                
                await asyncio.sleep(self.config.get("stop_check_interval", 1))
                
        except asyncio.CancelledError:
            self.logger.info(f"Stop loss monitoring cancelled for {position_id}")
        except Exception as e:
            self.logger.error(f"Stop loss monitoring error: {e}")

    async def _update_trailing_stop(self, position_id: str, current_price: Decimal) -> None:
        """Update trailing stop price"""
        stop_data = self._active_stops[position_id]
        trailing_distance = stop_data["trailing_distance"]
        
        # Update stop price if price moved in favorable direction
        if stop_data["entry_price"] < current_price:  # Long position
            new_stop = current_price - trailing_distance
            if new_stop > stop_data["stop_price"]:
                stop_data["stop_price"] = new_stop
        else:  # Short position
            new_stop = current_price + trailing_distance
            if new_stop < stop_data["stop_price"]:
                stop_data["stop_price"] = new_stop

    async def _is_stop_triggered(self, position_id: str) -> bool:
        """Check if stop loss is triggered"""
        stop_data = self._active_stops[position_id]
        current_price = stop_data["current_price"]
        stop_price = stop_data["stop_price"]
        
        if stop_data["type"] == StopType.TIME:
            # Check time-based stop
            time_limit = stop_data.get("time_limit")
            if time_limit and datetime.utcnow() >= time_limit:
                return True
                
        elif stop_data["type"] == StopType.DYNAMIC:
            # Check dynamic stop conditions
            return await self._check_dynamic_stop(stop_data)
            
        else:  # Fixed or Trailing stop
            # Check if price crossed stop level
            if stop_data["entry_price"] < current_price:  # Long position
                return current_price <= stop_price
            else:  # Short position
                return current_price >= stop_price
        
        return False

    async def _execute_stop_loss(self, position_id: str) -> None:
        """Execute stop loss order"""
        try:
            stop_data = self._active_stops[position_id]
            execution_price = stop_data["current_price"]
            
            # Call execution callback if set
            if self._execution_callback:
                await self._execution_callback(
                    position_id=position_id,
                    price=execution_price,
                    quantity=stop_data["quantity"]
                )
            
            # Clean up
            await self.cancel_stop_loss(position_id)
            
        except Exception as e:
            self.logger.error(f"Failed to execute stop loss: {e}")

    async def _check_dynamic_stop(self, stop_data: Dict) -> bool:
        """Check dynamic stop loss conditions"""
        try:
            current_price = stop_data["current_price"]
            entry_price = stop_data["entry_price"]
            
            # Calculate profit/loss percentage
            pnl_percentage = ((current_price - entry_price) / entry_price) * Decimal("100")
            
            # Get dynamic thresholds
            profit_threshold = Decimal(str(stop_data.get("profit_threshold", "5")))
            loss_threshold = Decimal(str(stop_data.get("loss_threshold", "2")))
            
            # Check conditions
            if pnl_percentage >= profit_threshold:
                # Move stop to breakeven
                stop_data["stop_price"] = entry_price
                return False
            elif pnl_percentage <= -loss_threshold:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Dynamic stop check failed: {e}")
            return False