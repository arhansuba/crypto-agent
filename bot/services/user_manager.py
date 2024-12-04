from typing import Dict, Optional, List
import logging
from datetime import datetime
import json

class UserManager:
    def __init__(self):
        """Initialize user manager"""
        self.logger = logging.getLogger(__name__)
        # In-memory storage (replace with database in production)
        self._users = {}
        self._settings = {}
        self._preferences = {}
        
    async def register_user(self, user_id: int, telegram_data: Optional[Dict] = None) -> Dict:
        """Register a new user"""
        try:
            if user_id in self._users:
                return {
                    "status": "error",
                    "message": "User already registered"
                }
            
            # Create user profile
            user_data = {
                "user_id": user_id,
                "registration_date": datetime.utcnow().isoformat(),
                "telegram_data": telegram_data or {},
                "active": True,
                "last_active": datetime.utcnow().isoformat()
            }
            
            self._users[user_id] = user_data
            
            # Initialize user settings with defaults
            self._settings[user_id] = self._get_default_settings()
            
            # Initialize user preferences
            self._preferences[user_id] = self._get_default_preferences()
            
            return {
                "status": "success",
                "message": "User registered successfully",
                "user_data": user_data
            }
            
        except Exception as e:
            self.logger.error(f"Error registering user: {e}")
            return {"status": "error", "message": str(e)}

    async def get_user(self, user_id: int) -> Dict:
        """Get user data"""
        try:
            user_data = self._users.get(user_id)
            if not user_data:
                return {
                    "status": "error",
                    "message": "User not found"
                }
            
            return {
                "status": "success",
                "user_data": user_data
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user data: {e}")
            return {"status": "error", "message": str(e)}

    async def update_user_settings(self, user_id: int, settings: Dict) -> Dict:
        """Update user settings"""
        try:
            if user_id not in self._settings:
                return {
                    "status": "error",
                    "message": "User not found"
                }
            
            # Update only provided settings
            current_settings = self._settings[user_id]
            current_settings.update(settings)
            self._settings[user_id] = current_settings
            
            return {
                "status": "success",
                "message": "Settings updated successfully",
                "settings": current_settings
            }
            
        except Exception as e:
            self.logger.error(f"Error updating settings: {e}")
            return {"status": "error", "message": str(e)}

    async def get_user_settings(self, user_id: int) -> Dict:
        """Get user settings"""
        try:
            settings = self._settings.get(user_id)
            if not settings:
                return {
                    "status": "error",
                    "message": "Settings not found"
                }
            
            return {
                "status": "success",
                "settings": settings
            }
            
        except Exception as e:
            self.logger.error(f"Error getting settings: {e}")
            return {"status": "error", "message": str(e)}

    async def update_last_active(self, user_id: int) -> Dict:
        """Update user's last active timestamp"""
        try:
            if user_id not in self._users:
                return {
                    "status": "error",
                    "message": "User not found"
                }
            
            self._users[user_id]["last_active"] = datetime.utcnow().isoformat()
            
            return {
                "status": "success",
                "message": "Last active timestamp updated"
            }
            
        except Exception as e:
            self.logger.error(f"Error updating last active: {e}")
            return {"status": "error", "message": str(e)}

    async def set_user_preference(self, user_id: int, key: str, value: any) -> Dict:
        """Set a user preference"""
        try:
            if user_id not in self._preferences:
                self._preferences[user_id] = self._get_default_preferences()
            
            self._preferences[user_id][key] = value
            
            return {
                "status": "success",
                "message": f"Preference {key} updated successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Error setting preference: {e}")
            return {"status": "error", "message": str(e)}

    async def get_user_preferences(self, user_id: int) -> Dict:
        """Get all user preferences"""
        try:
            preferences = self._preferences.get(user_id, self._get_default_preferences())
            
            return {
                "status": "success",
                "preferences": preferences
            }
            
        except Exception as e:
            self.logger.error(f"Error getting preferences: {e}")
            return {"status": "error", "message": str(e)}

    def _get_default_settings(self) -> Dict:
        """Get default user settings"""
        return {
            "notification_enabled": True,
            "alert_threshold": 5.0,  # 5% price change
            "alert_interval": 300,   # 5 minutes
            "max_alerts": 10,
            "default_slippage": 0.5, # 0.5% slippage
            "gas_price_alert": 100,  # Gwei
            "theme": "light",
            "language": "en"
        }

    def _get_default_preferences(self) -> Dict:
        """Get default user preferences"""
        return {
            "analysis_display": "detailed",    # detailed/simple
            "price_format": "standard",        # standard/compact
            "chart_interval": "1h",            # 1h/4h/1d
            "trading_mode": "manual",          # manual/semi-auto
            "risk_level": "medium",           # low/medium/high
            "alert_channels": ["telegram"],
            "timezone": "UTC"
        }

    async def deactivate_user(self, user_id: int) -> Dict:
        """Deactivate a user"""
        try:
            if user_id not in self._users:
                return {
                    "status": "error",
                    "message": "User not found"
                }
            
            self._users[user_id]["active"] = False
            
            return {
                "status": "success",
                "message": "User deactivated successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Error deactivating user: {e}")
            return {"status": "error", "message": str(e)}

    async def export_user_data(self, user_id: int) -> Dict:
        """Export all user data (for GDPR compliance)"""
        try:
            if user_id not in self._users:
                return {
                    "status": "error",
                    "message": "User not found"
                }
            
            user_data = {
                "profile": self._users[user_id],
                "settings": self._settings.get(user_id, {}),
                "preferences": self._preferences.get(user_id, {})
            }
            
            return {
                "status": "success",
                "data": user_data
            }
            
        except Exception as e:
            self.logger.error(f"Error exporting user data: {e}")
            return {"status": "error", "message": str(e)}