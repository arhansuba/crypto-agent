from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from enum import Enum

class EventType(Enum):
    ANALYSIS = "analysis"
    TRADE = "trade"
    ALERT = "alert"
    LOGIN = "login"
    SUBSCRIPTION = "subscription"
    ERROR = "error"
    COMMAND = "command"

class AnalyticsService:
    def __init__(self):
        """Initialize analytics service"""
        self.logger = logging.getLogger(__name__)
        # In-memory storage (replace with database in production)
        self._events = []
        self._user_stats = {}
        self._system_stats = {
            "total_users": 0,
            "total_analyses": 0,
            "total_trades": 0,
            "total_alerts": 0
        }

    async def track_event(self, event_type: EventType, user_id: int, data: Dict) -> Dict:
        """Track a new event"""
        try:
            event = {
                "type": event_type.value,
                "user_id": user_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }
            
            self._events.append(event)
            self._update_stats(event)
            
            return {"status": "success", "event_id": len(self._events) - 1}
        except Exception as e:
            self.logger.error(f"Error tracking event: {e}")
            return {"status": "error", "message": str(e)}

    async def get_user_stats(self, user_id: int, period: Optional[str] = "24h") -> Dict:
        """Get statistics for a specific user"""
        try:
            stats = self._user_stats.get(user_id, self._create_empty_stats())
            
            return {
                "status": "success",
                "stats": self._filter_stats_by_period(stats, period)
            }
        except Exception as e:
            self.logger.error(f"Error getting user stats: {e}")
            return {"status": "error", "message": str(e)}

    async def get_system_stats(self) -> Dict:
        """Get overall system statistics"""
        try:
            return {
                "status": "success",
                "stats": self._system_stats
            }
        except Exception as e:
            self.logger.error(f"Error getting system stats: {e}")
            return {"status": "error", "message": str(e)}

    def _update_stats(self, event: Dict) -> None:
        """Update user and system statistics"""
        try:
            user_id = event["user_id"]
            event_type = event["type"]
            
            # Initialize user stats if not exists
            if user_id not in self._user_stats:
                self._user_stats[user_id] = self._create_empty_stats()
            
            # Update user stats
            user_stats = self._user_stats[user_id]
            self._increment_stats(user_stats, event_type)
            
            # Update system stats
            self._increment_stats(self._system_stats, event_type)
            
        except Exception as e:
            self.logger.error(f"Error updating stats: {e}")

    def _create_empty_stats(self) -> Dict:
        """Create empty statistics template"""
        return {
            "total_analyses": 0,
            "total_trades": 0,
            "total_alerts": 0,
            "successful_trades": 0,
            "failed_trades": 0,
            "alerts_triggered": 0,
            "last_active": datetime.utcnow().isoformat(),
            "hourly_stats": {},
            "daily_stats": {},
            "weekly_stats": {}
        }

    def _increment_stats(self, stats: Dict, event_type: str) -> None:
        """Increment specific statistics"""
        if event_type == EventType.ANALYSIS.value:
            stats["total_analyses"] += 1
        elif event_type == EventType.TRADE.value:
            stats["total_trades"] += 1
        elif event_type == EventType.ALERT.value:
            stats["total_alerts"] += 1

    def _filter_stats_by_period(self, stats: Dict, period: str) -> Dict:
        """Filter statistics by time period"""
        now = datetime.utcnow()
        
        if period == "24h":
            start_time = now - timedelta(hours=24)
        elif period == "7d":
            start_time = now - timedelta(days=7)
        elif period == "30d":
            start_time = now - timedelta(days=30)
        else:
            return stats
            
        filtered_stats = self._create_empty_stats()
        
        for event in self._events:
            event_time = datetime.fromisoformat(event["timestamp"])
            if event_time >= start_time and event["user_id"] == stats.get("user_id"):
                self._increment_stats(filtered_stats, event["type"])
                
        return filtered_stats

    async def get_usage_metrics(self, user_id: Optional[int] = None) -> Dict:
        """Get detailed usage metrics"""
        try:
            if user_id:
                events = [e for e in self._events if e["user_id"] == user_id]
            else:
                events = self._events
                
            metrics = {
                "total_events": len(events),
                "event_types": {},
                "hourly_distribution": {},
                "daily_distribution": {},
                "error_rate": 0.0
            }
            
            for event in events:
                # Count event types
                event_type = event["type"]
                metrics["event_types"][event_type] = metrics["event_types"].get(event_type, 0) + 1
                
                # Track time distributions
                timestamp = datetime.fromisoformat(event["timestamp"])
                hour = timestamp.hour
                day = timestamp.strftime("%Y-%m-%d")
                
                metrics["hourly_distribution"][hour] = metrics["hourly_distribution"].get(hour, 0) + 1
                metrics["daily_distribution"][day] = metrics["daily_distribution"].get(day, 0) + 1
            
            # Calculate error rate
            error_count = metrics["event_types"].get(EventType.ERROR.value, 0)
            metrics["error_rate"] = (error_count / len(events)) * 100 if events else 0
            
            return {
                "status": "success",
                "metrics": metrics
            }
        except Exception as e:
            self.logger.error(f"Error getting usage metrics: {e}")
            return {"status": "error", "message": str(e)}

    async def clean_old_events(self, days: int = 30) -> Dict:
        """Clean events older than specified days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            original_count = len(self._events)
            self._events = [
                event for event in self._events
                if datetime.fromisoformat(event["timestamp"]) >= cutoff_date
            ]
            
            cleaned_count = original_count - len(self._events)
            
            return {
                "status": "success",
                "message": f"Cleaned {cleaned_count} old events",
                "remaining_events": len(self._events)
            }
        except Exception as e:
            self.logger.error(f"Error cleaning old events: {e}")
            return {"status": "error", "message": str(e)}