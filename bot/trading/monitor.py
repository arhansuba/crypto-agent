from typing import Dict, List, Optional, Callable
import asyncio
import logging
from datetime import datetime
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper

class PriceMonitor:
    def __init__(self, config: Dict):
        """Initialize price monitor with CDP integration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        self.tools = self.toolkit.get_tools()
        
        # Track monitored tokens and alerts
        self._monitored_tokens = {}
        self._alert_callbacks = {}
        self._monitoring_tasks = {}

    async def start_monitoring(self, token_address: str, user_id: int, alert_conditions: Dict) -> Dict:
        """Start monitoring a token for a user"""
        try:
            if token_address not in self._monitored_tokens:
                self._monitored_tokens[token_address] = {}
            
            self._monitored_tokens[token_address][user_id] = {
                "conditions": alert_conditions,
                "last_price": await self.tools.get_token_price(token_address),
                "last_check": datetime.utcnow(),
                "alerts_triggered": 0
            }
            
            # Start monitoring task if not already running
            if token_address not in self._monitoring_tasks:
                task = asyncio.create_task(self._monitor_token_price(token_address))
                self._monitoring_tasks[token_address] = task
            
            return {
                "status": "success",
                "message": "Price monitoring started",
                "current_price": self._monitored_tokens[token_address][user_id]["last_price"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            return {"status": "error", "message": str(e)}

    async def stop_monitoring(self, token_address: str, user_id: int) -> Dict:
        """Stop monitoring a token for a user"""
        try:
            if token_address in self._monitored_tokens:
                if user_id in self._monitored_tokens[token_address]:
                    del self._monitored_tokens[token_address][user_id]
                    
                    # If no users are monitoring this token, stop the task
                    if not self._monitored_tokens[token_address]:
                        if token_address in self._monitoring_tasks:
                            self._monitoring_tasks[token_address].cancel()
                            del self._monitoring_tasks[token_address]
                        del self._monitored_tokens[token_address]
            
            return {
                "status": "success",
                "message": "Price monitoring stopped"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
            return {"status": "error", "message": str(e)}

    def register_alert_callback(self, callback: Callable) -> None:
        """Register callback for price alerts"""
        self._alert_callbacks['default'] = callback

    async def get_monitoring_status(self, user_id: int) -> Dict:
        """Get monitoring status for a user"""
        try:
            user_monitors = {}
            for token_address, monitors in self._monitored_tokens.items():
                if user_id in monitors:
                    user_monitors[token_address] = {
                        "conditions": monitors[user_id]["conditions"],
                        "last_price": monitors[user_id]["last_price"],
                        "last_check": monitors[user_id]["last_check"],
                        "alerts_triggered": monitors[user_id]["alerts_triggered"]
                    }
            
            return {
                "status": "success",
                "monitors": user_monitors
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get monitoring status: {e}")
            return {"status": "error", "message": str(e)}

    async def update_alert_conditions(self, token_address: str, user_id: int, new_conditions: Dict) -> Dict:
        """Update alert conditions for a monitored token"""
        try:
            if (token_address in self._monitored_tokens and 
                user_id in self._monitored_tokens[token_address]):
                self._monitored_tokens[token_address][user_id]["conditions"] = new_conditions
                return {
                    "status": "success",
                    "message": "Alert conditions updated"
                }
            return {
                "status": "error",
                "message": "Token not being monitored"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update alert conditions: {e}")
            return {"status": "error", "message": str(e)}

    async def _monitor_token_price(self, token_address: str) -> None:
        """Monitor token price and trigger alerts"""
        try:
            while token_address in self._monitored_tokens:
                current_price = await self.tools.get_token_price(token_address)
                
                # Check alerts for each user monitoring this token
                for user_id, monitor_data in self._monitored_tokens[token_address].items():
                    await self._check_alert_conditions(
                        token_address,
                        user_id,
                        current_price,
                        monitor_data
                    )
                    
                    # Update last price and check time
                    monitor_data["last_price"] = current_price
                    monitor_data["last_check"] = datetime.utcnow()
                
                # Sleep before next check
                await asyncio.sleep(self.config.get("price_check_interval", 60))
                
        except asyncio.CancelledError:
            self.logger.info(f"Monitoring stopped for {token_address}")
        except Exception as e:
            self.logger.error(f"Error monitoring {token_address}: {e}")

    async def _check_alert_conditions(
        self,
        token_address: str,
        user_id: int,
        current_price: float,
        monitor_data: Dict
    ) -> None:
        """Check if alert conditions are met"""
        try:
            conditions = monitor_data["conditions"]
            last_price = monitor_data["last_price"]
            
            # Calculate price change
            price_change = ((current_price - last_price) / last_price) * 100
            
            # Check different alert conditions
            alerts_triggered = []
            
            # Price change threshold
            if abs(price_change) >= conditions.get("price_change_threshold", 5.0):
                alerts_triggered.append({
                    "type": "price_change",
                    "change": price_change,
                    "current_price": current_price
                })
            
            # Price threshold alerts
            if conditions.get("price_above") and current_price >= conditions["price_above"]:
                alerts_triggered.append({
                    "type": "price_above",
                    "threshold": conditions["price_above"],
                    "current_price": current_price
                })
                
            if conditions.get("price_below") and current_price <= conditions["price_below"]:
                alerts_triggered.append({
                    "type": "price_below",
                    "threshold": conditions["price_below"],
                    "current_price": current_price
                })
            
            # Trigger alerts
            if alerts_triggered and self._alert_callbacks.get('default'):
                for alert in alerts_triggered:
                    await self._alert_callbacks['default'](
                        user_id,
                        token_address,
                        alert
                    )
                    monitor_data["alerts_triggered"] += 1
                    
        except Exception as e:
            self.logger.error(f"Error checking alert conditions: {e}")