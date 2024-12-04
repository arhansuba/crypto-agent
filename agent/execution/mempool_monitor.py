from typing import Dict, List, Optional, Set, Callable, Awaitable
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit

class MempoolMonitor:
    """
    Monitors the mempool for relevant transactions and opportunities
    using CDP's toolkit while implementing efficient filtering and analysis.
    """
    
    def __init__(self, config: Dict):
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        # Monitoring parameters
        self.poll_interval = config.get('poll_interval', 0.1)  # 100ms
        self.max_tracked_txs = config.get('max_tracked_txs', 1000)
        self.min_gas_price = Decimal(str(config.get('min_gas_price', '1')))
        
        # State tracking
        self.tracked_transactions: Dict[str, Dict] = {}
        self.watched_addresses: Set[str] = set()
        self.transaction_callbacks: Dict[str, List[Callable]] = {}
        self.processed_txs: Set[str] = set()

    async def start_monitoring(
        self,
        callback: Optional[Callable[[Dict], Awaitable[None]]] = None
    ) -> None:
        """
        Start monitoring mempool for transactions.
        
        Args:
            callback: Optional callback for new transactions
        """
        while True:
            try:
                # Get pending transactions
                pending_txs = await self._get_pending_transactions()
                
                # Process new transactions
                for tx in pending_txs:
                    if tx['hash'] not in self.processed_txs:
                        await self._process_transaction(tx)
                        if callback:
                            await callback(tx)
                            
                # Cleanup old transactions
                await self._cleanup_old_transactions()
                
                await asyncio.sleep(self.poll_interval)
                
            except Exception as e:
                self._log_error("Mempool monitoring error", e)
                await asyncio.sleep(1)  # Longer interval on error

    async def add_address_watch(self, address: str, callback: Callable) -> None:
        """Add address to watch list with callback."""
        self.watched_addresses.add(address)
        if address not in self.transaction_callbacks:
            self.transaction_callbacks[address] = []
        self.transaction_callbacks[address].append(callback)

    async def analyze_transaction(self, tx: Dict) -> Dict:
        """
        Analyze a transaction for relevant information.
        
        Args:
            tx: Transaction to analyze
            
        Returns:
            Transaction analysis results
        """
        tools = self.toolkit.get_tools()
        
        # Decode transaction input
        decoded_input = await tools.decode_transaction_input(tx)
        
        # Get gas metrics
        gas_metrics = await self._analyze_gas_metrics(tx)
        
        # Analyze value transfer
        value_analysis = await self._analyze_value_transfer(tx)
        
        return {
            'hash': tx['hash'],
            'decoded_input': decoded_input,
            'gas_metrics': gas_metrics,
            'value_analysis': value_analysis,
            'type': await self._determine_transaction_type(tx, decoded_input),
            'priority': self._calculate_transaction_priority(tx, gas_metrics),
            'timestamp': datetime.utcnow()
        }

    async def get_transaction_status(self, tx_hash: str) -> Optional[Dict]:
        """Get current status of a tracked transaction."""
        return self.tracked_transactions.get(tx_hash)

    async def _get_pending_transactions(self) -> List[Dict]:
        """Get pending transactions from CDP toolkit."""
        tools = self.toolkit.get_tools()
        return await tools.get_pending_transactions()

    async def _process_transaction(self, tx: Dict) -> None:
        """Process a new transaction."""
        # Add to tracked transactions
        self.tracked_transactions[tx['hash']] = {
            'transaction': tx,
            'analysis': await self.analyze_transaction(tx),
            'first_seen': datetime.utcnow(),
            'last_updated': datetime.utcnow(),
            'status': 'pending'
        }
        
        # Check watched addresses
        if tx['to'] in self.watched_addresses:
            callbacks = self.transaction_callbacks.get(tx['to'], [])
            for callback in callbacks:
                try:
                    await callback(tx)
                except Exception as e:
                    self._log_error(f"Callback error for {tx['hash']}", e)
                    
        self.processed_txs.add(tx['hash'])

    async def _analyze_gas_metrics(self, tx: Dict) -> Dict:
        """Analyze transaction gas metrics."""
        return {
            'gas_price': Decimal(str(tx.get('gasPrice', '0'))),
            'gas_limit': int(tx.get('gas', '0')),
            'max_fee_per_gas': Decimal(str(tx.get('maxFeePerGas', '0'))),
            'max_priority_fee': Decimal(str(tx.get('maxPriorityFeePerGas', '0')))
        }

    async def _analyze_value_transfer(self, tx: Dict) -> Dict:
        """Analyze value transfer in transaction."""
        value = Decimal(str(tx.get('value', '0')))
        return {
            'value': value,
            'value_usd': await self._get_value_in_usd(value),
            'token_transfer': await self._check_token_transfer(tx)
        }

    async def _cleanup_old_transactions(self) -> None:
        """Cleanup old tracked transactions."""
        current_time = datetime.utcnow()
        removal_keys = []
        
        for tx_hash, tx_data in self.tracked_transactions.items():
            age = current_time - tx_data['first_seen']
            if age > timedelta(hours=1):  # Remove after 1 hour
                removal_keys.append(tx_hash)
                
        for key in removal_keys:
            del self.tracked_transactions[key]
            self.processed_txs.discard(key)

    def _log_error(self, message: str, error: Exception) -> None:
        """Log error with details."""
        error_details = {
            'message': message,
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        print(f"Mempool Monitor Error: {error_details}")  # Replace with proper logging