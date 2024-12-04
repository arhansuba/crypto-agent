from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime
import asyncio
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit
from eth_typing import Address, HexStr

class FlashLoanExecutor:
    """
    Flash loan execution system using CDP toolkit.
    Handles flash loan borrowing, arbitrage execution, and repayment in a single atomic transaction.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        # Configuration and limits
        self.max_loan_amount = Decimal(str(config.get('max_loan_amount', '100.0')))
        self.min_profit_threshold = Decimal(str(config.get('min_profit_threshold', '0.01')))
        self.gas_buffer = Decimal(str(config.get('gas_buffer', '0.002')))
        
        # Transaction tracking
        self.active_loans: Dict[str, Dict] = {}
        self.execution_history: List[Dict] = []
        self.last_execution_time: Optional[datetime] = None

    async def execute_flash_arbitrage(
        self,
        token_address: str,
        loan_amount: Decimal,
        trade_params: Dict
    ) -> Dict:
        """
        Execute flash loan arbitrage with safety checks and monitoring.
        
        Args:
            token_address: Address of token to borrow
            loan_amount: Amount to borrow
            trade_params: Trading parameters for arbitrage execution
        
        Returns:
            Transaction result including profit/loss and details
        """
        try:
            await self._validate_execution_params(token_address, loan_amount, trade_params)
            
            tools = self.toolkit.get_tools()
            
            # Prepare flash loan transaction
            flash_params = await self._prepare_flash_loan(token_address, loan_amount)
            
            # Execute flash loan with trading logic
            tx = await tools.execute_flash_loan(
                flash_params,
                callback=lambda: self._execute_arbitrage_logic(trade_params)
            )
            
            # Process and validate result
            result = await self._process_execution_result(tx)
            
            self._record_execution(result)
            return result
            
        except Exception as e:
            self._log_error("Flash loan execution failed", e)
            raise

    async def _validate_execution_params(
        self,
        token_address: str,
        loan_amount: Decimal,
        trade_params: Dict
    ) -> None:
        """
        Validate flash loan parameters against safety limits.
        
        Args:
            token_address: Token address to validate
            loan_amount: Loan amount to validate
            trade_params: Trading parameters to validate
        
        Raises:
            ValueError: If parameters don't meet safety requirements
        """
        if loan_amount > self.max_loan_amount:
            raise ValueError(f"Loan amount exceeds maximum allowed: {self.max_loan_amount}")
            
        # Verify token is supported for flash loans
        tools = self.toolkit.get_tools()
        supported_tokens = await tools.get_supported_flash_tokens()
        if token_address not in supported_tokens:
            raise ValueError(f"Token not supported for flash loans: {token_address}")
            
        # Verify sufficient liquidity
        liquidity = await self._check_token_liquidity(token_address)
        if loan_amount > liquidity:
            raise ValueError(f"Insufficient liquidity for loan amount: {liquidity}")
            
        # Validate trade parameters
        await self._validate_trade_params(trade_params)

    async def _prepare_flash_loan(
        self,
        token_address: str,
        loan_amount: Decimal
    ) -> Dict:
        """
        Prepare flash loan parameters and validation data.
        
        Args:
            token_address: Token to borrow
            loan_amount: Amount to borrow
            
        Returns:
            Prepared flash loan parameters
        """
        tools = self.toolkit.get_tools()
        
        # Get current lending rates and fees
        lending_data = await tools.get_flash_loan_data(token_address)
        
        # Calculate required repayment amount
        repayment_amount = loan_amount + (loan_amount * lending_data['fee_rate'])
        
        return {
            'token': token_address,
            'amount': loan_amount,
            'repayment': repayment_amount,
            'fee_rate': lending_data['fee_rate'],
            'provider': lending_data['provider']
        }

    async def _execute_arbitrage_logic(self, trade_params: Dict) -> bool:
        """
        Execute arbitrage trading logic within flash loan transaction.
        
        Args:
            trade_params: Parameters for arbitrage execution
            
        Returns:
            True if execution successful, False otherwise
        """
        tools = self.toolkit.get_tools()
        
        # Execute buy transaction
        buy_tx = await tools.execute_trade({
            'dex': trade_params['buy_dex'],
            'token': trade_params['token'],
            'amount': trade_params['amount'],
            'side': 'buy'
        })
        
        if not buy_tx['success']:
            return False
            
        # Execute sell transaction
        sell_tx = await tools.execute_trade({
            'dex': trade_params['sell_dex'],
            'token': trade_params['token'],
            'amount': trade_params['amount'],
            'side': 'sell'
        })
        
        return sell_tx['success']

    async def _process_execution_result(self, tx: Dict) -> Dict:
        """
        Process and validate flash loan execution result.
        
        Args:
            tx: Transaction result to process
            
        Returns:
            Processed execution result with profit/loss calculation
        """
        tools = self.toolkit.get_tools()
        
        # Get transaction receipt and details
        receipt = await tools.get_transaction_receipt(tx['hash'])
        
        # Calculate actual profit/loss
        profit = await self._calculate_profit(receipt)
        
        return {
            'transaction_hash': tx['hash'],
            'success': receipt['status'],
            'profit': profit,
            'gas_used': receipt['gasUsed'],
            'timestamp': datetime.utcnow(),
            'details': receipt
        }

    def _record_execution(self, result: Dict) -> None:
        """Record flash loan execution result for analysis."""
        self.execution_history.append(result)
        self.last_execution_time = datetime.utcnow()

    def _log_error(self, message: str, error: Exception) -> None:
        """Log error with details."""
        error_details = {
            'message': message,
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        print(f"Flash Loan Error: {error_details}")  # Replace with proper logging