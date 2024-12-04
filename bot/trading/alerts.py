from typing import Dict, List, Optional, Callable
import logging
from datetime import datetime
from enum import Enum

class AlertType(Enum):
    PRICE_CHANGE = "price_change"
    PRICE_TARGET = "price_target"
    VOLUME_SPIKE = "volume_spike"
    LIQUIDITY_CHANGE = "liquidity_change"
    TREND_CHANGE = "trend_change"
    PATTERN_DETECTED = "pattern_detected"
    ERROR = "error"

class AlertPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertSystem:
    def __init__(self, config: Dict):
        """Initialize alert system"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alert storage
        self._active_alerts = {}
        self._alert_history = []
        self._alert_handlers = {}
        
        # Initialize default handlers
        self._setup_default_handlers()

    async def create_alert(self, user_id: int, alert_config: Dict) -> Dict:
        """Create a new alert"""
        try:
            alert_id = self._generate_alert_id()
            alert = {
                "id": alert_id,
                "user_id": user_id,
                "type": alert_config["type"],
                "conditions": alert_config["conditions"],
                "priority": alert_config.get("priority", AlertPriority.MEDIUM.value),
                "status": "active",
                "created_at": datetime.utcnow().isoformat(),
                "triggered_count": 0
            }
            
            self._active_alerts[alert_id] = alert
            return {"status": "success", "alert_id": alert_id}
            
        except Exception as e:
            self.logger.error(f"Failed to create alert: {e}")
            return {"status": "error", "message": str(e)}

    async def trigger_alert(self, alert_id: str, trigger_data: Dict) -> Dict:
        """Trigger an alert and notify user"""
        try:
            if alert_id not in self._active_alerts:
                return {"status": "error", "message": "Alert not found"}

            alert = self._active_alerts[alert_id]
            alert["triggered_count"] += 1
            alert["last_triggered"] = datetime.utcnow().isoformat()
            
            # Format alert message
            message = self._format_alert_message(alert, trigger_data)
            
            # Send alert through appropriate handlers
            await self._send_alert(alert, message)
            
            # Archive if needed
            if self._should_archive_alert(alert):
                await self._archive_alert(alert_id)
            
            return {"status": "success", "message": "Alert triggered"}
            
        except Exception as e:
            self.logger.error(f"Failed to trigger alert: {e}")
            return {"status": "error", "message": str(e)}

    async def get_user_alerts(self, user_id: int) -> Dict:
        """Get all active alerts for a user"""
        try:
            user_alerts = {
                alert_id: alert for alert_id, alert in self._active_alerts.items()
                if alert["user_id"] == user_id
            }
            
            return {
                "status": "success",
                "alerts": user_alerts
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get user alerts: {e}")
            return {"status": "error", "message": str(e)}

    def register_handler(self, alert_type: AlertType, handler: Callable) -> None:
        """Register a new alert handler"""
        self._alert_handlers[alert_type] = handler

    async def update_alert(self, alert_id: str, updates: Dict) -> Dict:
        """Update an existing alert"""
        try:
            if alert_id not in self._active_alerts:
                return {"status": "error", "message": "Alert not found"}
            
            alert = self._active_alerts[alert_id]
            alert.update(updates)
            
            return {
                "status": "success",
                "alert": alert
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update alert: {e}")
            return {"status": "error", "message": str(e)}

    async def delete_alert(self, alert_id: str) -> Dict:
        """Delete an alert"""
        try:
            if alert_id in self._active_alerts:
                alert = self._active_alerts.pop(alert_id)
                await self._archive_alert(alert_id)
                
                return {
                    "status": "success",
                    "message": "Alert deleted"
                }
            
            return {
                "status": "error",
                "message": "Alert not found"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to delete alert: {e}")
            return {"status": "error", "message": str(e)}

    def _setup_default_handlers(self) -> None:
        """Setup default alert handlers"""
        self._alert_handlers = {
            AlertType.PRICE_CHANGE: self._handle_price_alert,
            AlertType.VOLUME_SPIKE: self._handle_volume_alert,
            AlertType.ERROR: self._handle_error_alert
        }

    async def _send_alert(self, alert: Dict, message: str) -> None:
        """Send alert through appropriate handler"""
        try:
            alert_type = AlertType(alert["type"])
            handler = self._alert_handlers.get(alert_type)
            
            if handler:
                await handler(alert, message)
            else:
                self.logger.warning(f"No handler found for alert type: {alert_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")

    async def _handle_price_alert(self, alert: Dict, message: str) -> None:
        """Handle price-related alerts"""
        # Example implementation - extend based on your notification system
        priority = AlertPriority(alert["priority"])
        
        if priority == AlertPriority.HIGH:
            # Send immediate notification
            await self._send_urgent_notification(alert["user_id"], message)
        else:
            # Queue for batch processing
            await self._queue_notification(alert["user_id"], message)

    async def _handle_volume_alert(self, alert: Dict, message: str) -> None:
        """Handle volume-related alerts"""
        # Implement volume alert handling
        await self._send_notification(alert["user_id"], message)

    async def _handle_error_alert(self, alert: Dict, message: str) -> None:
        """Handle error alerts"""
        # Log error and notify if critical
        self.logger.error(f"System alert: {message}")
        if alert["priority"] == AlertPriority.CRITICAL.value:
            await self._send_urgent_notification(alert["user_id"], message)

    def _format_alert_message(self, alert: Dict, trigger_data: Dict) -> str:
        """Format alert message"""
        template = self._get_alert_template(alert["type"])
        return template.format(**trigger_data)

    def _get_alert_template(self, alert_type: str) -> str:
        """Get message template for alert type"""
        templates = {
            AlertType.PRICE_CHANGE.value: "🚨 Price Alert!\nToken: {token}\nPrice Change: {change}%\nCurrent Price: ${price}",
            AlertType.VOLUME_SPIKE.value: "📈 Volume Spike!\nToken: {token}\nVolume Increase: {change}%\n24h Volume: ${volume}",
            AlertType.PATTERN_DETECTED.value: "🎯 Pattern Detected!\nToken: {token}\nPattern: {pattern}\nConfidence: {confidence}%",
            AlertType.ERROR.value: "⚠️ Error Alert!\nType: {error_type}\nMessage: {message}"
        }
        return templates.get(alert_type, "Alert: {message}")

    def _should_archive_alert(self, alert: Dict) -> bool:
        """Determine if alert should be archived"""
        # Archive if max triggers reached or alert is old
        max_triggers = self.config.get("max_alert_triggers", 10)
        return alert["triggered_count"] >= max_triggers

    async def _archive_alert(self, alert_id: str) -> None:
        """Archive an alert"""
        if alert_id in self._active_alerts:
            alert = self._active_alerts.pop(alert_id)
            alert["archived_at"] = datetime.utcnow().isoformat()
            self._alert_history.append(alert)

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        import uuid
        return str(uuid.uuid4())