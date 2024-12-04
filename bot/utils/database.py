import sqlite3
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
import asyncio
from contextlib import asynccontextmanager

class Database:
    def __init__(self, db_path: str = "data/bot.db"):
        """Initialize database connection"""
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._initialize_database()

    async def initialize(self) -> None:
        """Initialize database tables"""
        await self._create_tables()

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    async def _create_tables(self) -> None:
        """Create database tables"""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    telegram_id INTEGER UNIQUE,
                    subscription_tier TEXT DEFAULT 'free',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_active TIMESTAMP,
                    settings JSON
                )
            """)
            
            # Subscriptions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    tier TEXT,
                    start_date TIMESTAMP,
                    end_date TIMESTAMP,
                    status TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # Monitored tokens table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS monitored_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    token_address TEXT,
                    alert_conditions JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_check TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    token_address TEXT,
                    alert_type TEXT,
                    conditions JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    triggered_at TIMESTAMP,
                    status TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # Trading history table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    token_address TEXT,
                    action TEXT,
                    amount REAL,
                    price REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    tx_hash TEXT,
                    status TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            conn.commit()

    async def create_user(self, user_data: Dict) -> int:
        """Create new user"""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO users (telegram_id, settings)
                    VALUES (?, ?)
                """, (user_data["telegram_id"], json.dumps(user_data.get("settings", {}))))
                
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                self.logger.error(f"Failed to create user: {e}")
                raise

    async def get_user(self, user_id: int) -> Optional[Dict]:
        """Get user by ID"""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            
            if row:
                user_data = dict(row)
                user_data["settings"] = json.loads(user_data["settings"])
                return user_data
            return None

    async def update_user(self, user_id: int, updates: Dict) -> bool:
        """Update user data"""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                updates_str = ", ".join([f"{k} = ?" for k in updates.keys()])
                values = list(updates.values()) + [user_id]
                
                cursor.execute(f"""
                    UPDATE users 
                    SET {updates_str}
                    WHERE user_id = ?
                """, values)
                
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                self.logger.error(f"Failed to update user: {e}")
                raise

    async def create_alert(self, alert_data: Dict) -> int:
        """Create new alert"""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO alerts (
                        user_id, token_address, alert_type, 
                        conditions, status
                    )
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    alert_data["user_id"],
                    alert_data["token_address"],
                    alert_data["alert_type"],
                    json.dumps(alert_data["conditions"]),
                    "active"
                ))
                
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                self.logger.error(f"Failed to create alert: {e}")
                raise

    async def get_user_alerts(self, user_id: int) -> List[Dict]:
        """Get all alerts for user"""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM alerts 
                WHERE user_id = ? AND status = 'active'
            """, (user_id,))
            
            alerts = []
            for row in cursor.fetchall():
                alert = dict(row)
                alert["conditions"] = json.loads(alert["conditions"])
                alerts.append(alert)
                
            return alerts

    async def log_trade(self, trade_data: Dict) -> int:
        """Log trading activity"""
        async with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    INSERT INTO trading_history (
                        user_id, token_address, action, 
                        amount, price, tx_hash, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data["user_id"],
                    trade_data["token_address"],
                    trade_data["action"],
                    trade_data["amount"],
                    trade_data["price"],
                    trade_data["tx_hash"],
                    trade_data["status"]
                ))
                
                conn.commit()
                return cursor.lastrowid
            except Exception as e:
                self.logger.error(f"Failed to log trade: {e}")
                raise

    def _initialize_database(self) -> None:
        """Initialize database connection and tables"""
        asyncio.run(self.initialize())