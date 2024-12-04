from typing import Dict, List, Optional, Union
from decimal import Decimal
from datetime import datetime
import asyncio
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit

class CDPExecutor:
    """
    Handles transaction execution through CDP's toolkit with advanced features
    for monitoring, retrying, and optimizing transactions.
    """
    
    def __init__(self, config: Dict):
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        # Execution parameters
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1)
        self.confirmation_blocks = config.get('confirmation_blocks', 2)
        
        # Transaction tracking
        self.pending_transactions: Dict[str, Dict] = {}
        self.transaction_history: List[Dict] = []
        self.nonce_tracker: Dict[str, int] = {}

    async def execute_transaction(
        self,
        transaction_params: Dict,
        priority: str = 'medium'
    ) -> Dict:
        """
        Execute a transaction through CDP with monitoring and optimization.
        
        Args:
            transaction_params: Transaction parameters
            priority: Transaction priority (low, medium, high)
            
        Returns:
            Transaction execution results
        """
        try:
            # Validate parameters
            await self._validate_transaction(transaction_params)
            
            # Prepare transaction
            prepared_tx = await self._prepare_transaction(
                transaction_params,
                priority
            )
            
            # Execute with retry logic
            result = await self._execute_with_retry(prepared_tx)
            
            # Monitor confirmation
            confirmed_tx = await self._monitor_confirmation(result)
            
            # Update tracking
            self._update_transaction_history(confirmed_tx)
            
            return confirmed_tx
            
        except Exception as e:
            self._log_error("Transaction execution failed", e)
            raise

    async def get_transaction_status(self, tx_hash: str) -> Dict:
        """Get current status of a transaction."""
        tools = self.toolkit.get_tools()
        
        try:
            # Get transaction receipt
            receipt = await tools.get_transaction_receipt(tx_hash)
            
            # Get block confirmations
            current_block = await tools.get_block_number()
            confirmations = current_block - receipt['blockNumber']
            
            return {
                'hash': tx_hash,
                'status': receipt['status'],
                'confirmations': confirmations,
                'gas_used': receipt['gasUsed'],
                'block_number': receipt['blockNumber'],
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            self._log_error(f"Error getting transaction status for {tx_hash}", e)
            return {'hash': tx_hash, 'status': 'unknown'}

    async def _prepare_transaction(self, params: Dict, priority: str) -> Dict:
        """Prepare transaction for execution with optimized parameters."""
        tools = self.toolkit.get_tools()
        
        # Get nonce
        nonce = await self._get_next_nonce()
        
        # Get gas parameters
        gas_params = await self._optimize_gas_params(priority)
        
        # Estimate gas limit
        gas_limit = await tools.estimate_gas(params)
        gas_limit = int(gas_limit * 1.2)  # Add 20% buffer
        
        return {
            **params,
            'nonce': nonce,
            'gas_limit': gas_limit,
            **gas_params,
            'priority': priority
        }

    async def _execute_with_retry(self, prepared_tx: Dict) -> Dict:
        """Execute transaction with retry logic."""
        tools = self.toolkit.get_tools()
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            try:
                # Execute transaction
                result = await tools.execute_transaction(prepared_tx)
                
                if result['success']:
                    return result
                    
                # Update gas price on failure
                if attempt < self.max_retries - 1:
                    prepared_tx = await self._update_gas_params(prepared_tx)
                    
            except Exception as e:
                last_error = e
                
            attempt += 1
            await asyncio.sleep(self.retry_delay)
            
        raise Exception(f"Transaction failed after {attempt} attempts: {str(last_error)}")

    async def _monitor_confirmation(self, tx_result: Dict) -> Dict:
        """Monitor transaction confirmation."""
        tools = self.toolkit.get_tools()
        
        while True:
            status = await self.get_transaction_status(tx_result['hash'])
            
            if status['confirmations'] >= self.confirmation_blocks:
                return {
                    **tx_result,
                    **status,
                    'confirmed': True
                }
                
            await asyncio.sleep(1)

    async def _get_next_nonce(self) -> int:
        """Get next nonce for transaction."""
        tools = self.toolkit.get_tools()
        
        wallet_address = await tools.get_wallet_address()
        current_nonce = await tools.get_nonce(wallet_address)
        
        if wallet_address not in self.nonce_tracker:
            self.nonce_tracker[wallet_address] = current_nonce
            
        tracked_nonce = self.nonce_tracker[wallet_address]
        next_nonce = max(current_nonce, tracked_nonce + 1)
        
        self.nonce_tracker[wallet_address] = next_nonce
        return next_nonce

    def _log_error(self, message: str, error: Exception) -> None:
        """Log error with details."""
        error_details = {
            'message': message,
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        print(f"CDP Executor Error: {error_details}")  # Replace with proper logging