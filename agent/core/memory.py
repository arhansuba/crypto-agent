"""
State management implementation for the AI Crypto Agent.
Handles persistent storage, state tracking, and historical data management.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
import os
import asyncio
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from contextlib import contextmanager

class StateType(Enum):
    """Defines different types of state data"""
    TRANSACTION = "transaction"
    MARKET = "market"
    PORTFOLIO = "portfolio"
    NETWORK = "network"
    AGENT = "agent"

@dataclass
class TransactionState:
    """Represents transaction-related state"""
    tx_hash: str
    timestamp: float
    status: str
    value: float
    gas_used: float
    success: bool
    contract_address: Optional[str]
    error_message: Optional[str]

@dataclass
class MarketState:
    """Represents market-related state"""
    timestamp: float
    gas_price: float
    network_congestion: float
    sentiment_score: float
    token_prices: Dict[str, float]
    liquidity_pools: Dict[str, Dict[str, float]]

@dataclass
class PortfolioState:
    """Represents portfolio-related state"""
    timestamp: float
    eth_balance: float
    token_balances: Dict[str, float]
    deployed_contracts: List[str]
    total_value_locked: float
    pending_transactions: List[str]

@dataclass
class AgentState:
    """Represents agent-specific state"""
    last_action_timestamp: float
    daily_transaction_count: int
    mode: str
    active_strategies: List[str]
    performance_metrics: Dict[str, float]

class StateManager:
    """
    Manages state persistence and retrieval for the AI Crypto Agent.
    Implements both in-memory and persistent storage mechanisms.
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """Initialize the state management system"""
        self.config = config
        self.logger = logger
        
        # Initialize storage
        self.db_path = config['memory']['storage_path']
        self._initialize_database()
        
        # Initialize in-memory state
        self.current_state = {
            StateType.TRANSACTION: {},
            StateType.MARKET: None,
            StateType.PORTFOLIO: None,
            StateType.AGENT: None
        }
        
        # Load initial state
        self._load_initial_state()
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Create tables for different state types
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    tx_hash TEXT PRIMARY KEY,
                    timestamp REAL,
                    status TEXT,
                    value REAL,
                    gas_used REAL,
                    success INTEGER,
                    contract_address TEXT,
                    error_message TEXT
                )
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_states (
                    timestamp REAL PRIMARY KEY,
                    gas_price REAL,
                    network_congestion REAL,
                    sentiment_score REAL,
                    token_prices TEXT,
                    liquidity_pools TEXT
                )
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_states (
                    timestamp REAL PRIMARY KEY,
                    eth_balance REAL,
                    token_balances TEXT,
                    deployed_contracts TEXT,
                    total_value_locked REAL,
                    pending_transactions TEXT
                )
                """)
                
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    timestamp REAL PRIMARY KEY,
                    daily_transaction_count INTEGER,
                    mode TEXT,
                    active_strategies TEXT,
                    performance_metrics TEXT
                )
                """)
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise
    
    @contextmanager
    def _get_db_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def _load_initial_state(self):
        """Load the most recent state from persistent storage"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Load most recent market state
                cursor.execute(
                    "SELECT * FROM market_states ORDER BY timestamp DESC LIMIT 1"
                )
                market_data = cursor.fetchone()
                if market_data:
                    self.current_state[StateType.MARKET] = MarketState(
                        timestamp=market_data[0],
                        gas_price=market_data[1],
                        network_congestion=market_data[2],
                        sentiment_score=market_data[3],
                        token_prices=json.loads(market_data[4]),
                        liquidity_pools=json.loads(market_data[5])
                    )
                
                # Load most recent portfolio state
                cursor.execute(
                    "SELECT * FROM portfolio_states ORDER BY timestamp DESC LIMIT 1"
                )
                portfolio_data = cursor.fetchone()
                if portfolio_data:
                    self.current_state[StateType.PORTFOLIO] = PortfolioState(
                        timestamp=portfolio_data[0],
                        eth_balance=portfolio_data[1],
                        token_balances=json.loads(portfolio_data[2]),
                        deployed_contracts=json.loads(portfolio_data[3]),
                        total_value_locked=portfolio_data[4],
                        pending_transactions=json.loads(portfolio_data[5])
                    )
                
                # Load most recent agent state
                cursor.execute(
                    "SELECT * FROM agent_states ORDER BY timestamp DESC LIMIT 1"
                )
                agent_data = cursor.fetchone()
                if agent_data:
                    self.current_state[StateType.AGENT] = AgentState(
                        last_action_timestamp=agent_data[0],
                        daily_transaction_count=agent_data[1],
                        mode=agent_data[2],
                        active_strategies=json.loads(agent_data[3]),
                        performance_metrics=json.loads(agent_data[4])
                    )
                
            self.logger.info("Initial state loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load initial state: {str(e)}")
            raise
    
    async def update_transaction_state(self, transaction: TransactionState):
        """Update the state of a transaction"""
        try:
            # Update in-memory state
            self.current_state[StateType.TRANSACTION][transaction.tx_hash] = transaction
            
            # Update persistent storage
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO transactions
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    transaction.tx_hash,
                    transaction.timestamp,
                    transaction.status,
                    transaction.value,
                    transaction.gas_used,
                    transaction.success,
                    transaction.contract_address,
                    transaction.error_message
                ))
                conn.commit()
            
            self.logger.info(f"Transaction state updated: {transaction.tx_hash}")
            
        except Exception as e:
            self.logger.error(f"Failed to update transaction state: {str(e)}")
            raise
    
    async def update_market_state(self, state: MarketState):
        """Update market-related state"""
        try:
            # Update in-memory state
            self.current_state[StateType.MARKET] = state
            
            # Update persistent storage
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO market_states
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    state.timestamp,
                    state.gas_price,
                    state.network_congestion,
                    state.sentiment_score,
                    json.dumps(state.token_prices),
                    json.dumps(state.liquidity_pools)
                ))
                conn.commit()
            
            # Clean up old market states
            await self._cleanup_old_states('market_states')
            
        except Exception as e:
            self.logger.error(f"Failed to update market state: {str(e)}")
            raise
    
    async def update_portfolio_state(self, state: PortfolioState):
        """Update portfolio-related state"""
        try:
            # Update in-memory state
            self.current_state[StateType.PORTFOLIO] = state
            
            # Update persistent storage
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO portfolio_states
                VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    state.timestamp,
                    state.eth_balance,
                    json.dumps(state.token_balances),
                    json.dumps(state.deployed_contracts),
                    state.total_value_locked,
                    json.dumps(state.pending_transactions)
                ))
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to update portfolio state: {str(e)}")
            raise
    
    async def update_agent_state(self, state: AgentState):
        """Update agent-specific state"""
        try:
            # Update in-memory state
            self.current_state[StateType.AGENT] = state
            
            # Update persistent storage
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO agent_states
                VALUES (?, ?, ?, ?, ?)
                """, (
                    state.last_action_timestamp,
                    state.daily_transaction_count,
                    state.mode,
                    json.dumps(state.active_strategies),
                    json.dumps(state.performance_metrics)
                ))
                conn.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to update agent state: {str(e)}")
            raise
    
    def get_transaction_state(self, tx_hash: str) -> Optional[TransactionState]:
        """Retrieve the state of a specific transaction"""
        try:
            # Check in-memory state first
            if tx_hash in self.current_state[StateType.TRANSACTION]:
                return self.current_state[StateType.TRANSACTION][tx_hash]
            
            # Check persistent storage
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT * FROM transactions WHERE tx_hash = ?",
                    (tx_hash,)
                )
                data = cursor.fetchone()
                
                if data:
                    return TransactionState(
                        tx_hash=data[0],
                        timestamp=data[1],
                        status=data[2],
                        value=data[3],
                        gas_used=data[4],
                        success=bool(data[5]),
                        contract_address=data[6],
                        error_message=data[7]
                    )
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve transaction state: {str(e)}")
            return None
    
    def get_current_state(self, state_type: StateType) -> Optional[Any]:
        """Retrieve current state for a specific type"""
        return self.current_state.get(state_type)
    
    async def get_historical_states(
        self,
        state_type: StateType,
        start_time: float,
        end_time: float
    ) -> List[Any]:
        """Retrieve historical states for a specific type and time range"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                table_name = f"{state_type.value}_states"
                
                cursor.execute(f"""
                SELECT * FROM {table_name}
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
                """, (start_time, end_time))
                
                data = cursor.fetchall()
                
                if state_type == StateType.MARKET:
                    return [MarketState(
                        timestamp=row[0],
                        gas_price=row[1],
                        network_congestion=row[2],
                        sentiment_score=row[3],
                        token_prices=json.loads(row[4]),
                        liquidity_pools=json.loads(row[5])
                    ) for row in data]
                    
                elif state_type == StateType.PORTFOLIO:
                    return [PortfolioState(
                        timestamp=row[0],
                        eth_balance=row[1],
                        token_balances=json.loads(row[2]),
                        deployed_contracts=json.loads(row[3]),
                        total_value_locked=row[4],
                        pending_transactions=json.loads(row[5])
                    ) for row in data]
                    
                elif state_type == StateType.AGENT:
                    return [AgentState(
                        last_action_timestamp=row[0],
                        daily_transaction_count=row[1],
                        mode=row[2],
                        active_strategies=json.loads(row[3]),
                        performance_metrics=json.loads(row[4])
                    ) for row in data]
                
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve historical states: {str(e)}")
            return []
    
    async def _cleanup_old_states(self, table_name: str):
        """Remove states older than retention period"""
        try:
            retention_days = self.config['memory']['retention_period']
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(f"""
                DELETE FROM {table_name}
                WHERE timestamp < ?
                """, (cutoff_time.timestamp(),))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"State cleanup failed: {str(e)}")
            raise