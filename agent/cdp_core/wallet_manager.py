from cdp_langchain.utils import CdpAgentkitWrapper
from typing import Dict, Optional, List, Union
from eth_typing import Address, HexStr
import asyncio
from datetime import datetime

class CDPWalletManager:
    def __init__(self, config: Dict):
        self.cdp = CdpAgentkitWrapper()
        self.config = config
        self.transaction_queue = asyncio.Queue()
        self.active_transactions = {}
        self.wallet_data = None
        self.balance_cache = {}
        self.last_balance_check = {}

    async def initialize_wallet(self) -> bool:
        """Initialize CDP MPC wallet"""
        try:
            self.wallet_data = self.cdp.export_wallet()
            await self._verify_wallet()
            return True
        except Exception as e:
            self._log_error("Wallet initialization failed", e)
            return False

    async def get_wallet_details(self) -> Dict:
        """Get CDP wallet details and status"""
        tools = self.cdp.toolkit.get_tools()
        details = await tools.get_wallet_details()
        return {
            'address': details['address'],
            'network': details['network'],
            'baseName': details.get('baseName'),
            'status': 'active' if details['address'] else 'inactive'
        }

    async def get_balance(self, token_address: Optional[str] = None) -> Dict[str, float]:
        """Get wallet balance for specified token or all tokens"""
        tools = self.cdp.toolkit.get_tools()
        cached = self._check_balance_cache(token_address)
        if cached:
            return cached

        balances = await tools.get_balance(token_address)
        self._update_balance_cache(token_address, balances)
        return balances

    async def execute_transaction(self, params: Dict) -> Dict:
        """Execute transaction via CDP wallet"""
        try:
            await self._validate_transaction(params)
            await self.transaction_queue.put(params)
            return await self._process_transaction(params)
        except Exception as e:
            self._log_error("Transaction execution failed", e)
            raise

    async def register_basename(self, basename: str) -> Dict:
        """Register CDP basename for the wallet"""
        tools = self.cdp.toolkit.get_tools()
        return await tools.register_basename(basename)

    async def monitor_transactions(self):
        """Monitor and process queued transactions"""
        while True:
            try:
                params = await self.transaction_queue.get()
                await self._process_transaction(params)
                await asyncio.sleep(1)
            except Exception as e:
                self._log_error("Transaction monitoring error", e)

    async def _verify_wallet(self) -> bool:
        """Verify wallet initialization and connectivity"""
        tools = self.cdp.toolkit.get_tools()
        details = await tools.get_wallet_details()
        return bool(details and details.get('address'))

    async def _validate_transaction(self, params: Dict) -> bool:
        """Validate transaction parameters and balances"""
        required = ['type', 'amount', 'recipient']
        if not all(key in params for key in required):
            raise ValueError("Missing required transaction parameters")

        balance = await self.get_balance()
        if float(params['amount']) > float(balance.get('eth', 0)):
            raise ValueError("Insufficient balance")

        return True

    async def _process_transaction(self, params: Dict) -> Dict:
        """Process and monitor transaction execution"""
        tools = self.cdp.toolkit.get_tools()
        tx = await tools.execute_transaction(params)
        self.active_transactions[tx['hash']] = {
            'status': 'pending',
            'timestamp': datetime.utcnow(),
            'params': params
        }
        
        await self._monitor_transaction(tx['hash'])
        return tx

    async def _monitor_transaction(self, tx_hash: str):
        """Monitor transaction status"""
        tools = self.cdp.toolkit.get_tools()
        while True:
            status = await tools.get_transaction_status(tx_hash)
            self.active_transactions[tx_hash]['status'] = status
            if status in ['confirmed', 'failed']:
                break
            await asyncio.sleep(2)

    def _check_balance_cache(self, token_address: Optional[str]) -> Optional[Dict]:
        """Check cached balance if recent"""
        key = token_address or 'eth'
        last_check = self.last_balance_check.get(key)
        if not last_check:
            return None

        cache_duration = self.config.get('balance_cache_duration', 30)
        if (datetime.utcnow() - last_check).seconds < cache_duration:
            return self.balance_cache.get(key)
        return None

    def _update_balance_cache(self, token_address: Optional[str], balance: Dict):
        """Update balance cache"""
        key = token_address or 'eth'
        self.balance_cache[key] = balance
        self.last_balance_check[key] = datetime.utcnow()

    def _log_error(self, message: str, error: Exception):
        """Log wallet errors"""
        error_data = {
            'message': message,
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        print(f"Wallet Error: {error_data}")  # Replace with proper logging