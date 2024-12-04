from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime
import asyncio
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit

class BackrunExecutor:
    """Executes backrun strategies using CDP's toolkit."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        self.min_profit = Decimal(str(config.get('min_profit', '0.05')))
        self.max_gas_price = Decimal(str(config.get('max_gas_price', '500')))
        self.execution_timeout = config.get('execution_timeout', 2)
        
        self.active_backruns: Dict[str, Dict] = {}
        self.execution_history: List[Dict] = []

    async def execute_backrun(self, target_tx: Dict) -> Dict:
        """Execute backrun strategy for target transaction."""
        try:
            tools = self.toolkit.get_tools()
            
            if not await self._validate_target(target_tx):
                raise ValueError("Invalid target transaction")

            strategy = await self._build_backrun_strategy(target_tx)
            
            tx = await tools.execute_transaction(
                strategy,
                priority='high',
                timeout=self.execution_timeout
            )
            
            result = await self._process_result(tx, target_tx)
            self.execution_history.append(result)
            
            return result
            
        except Exception as e:
            self._log_error("Backrun execution failed", e)
            raise

    async def _build_backrun_strategy(self, target_tx: Dict) -> Dict:
        """Build optimal backrun strategy for target transaction."""
        tools = self.toolkit.get_tools()
        
        gas_price = await self._calculate_optimal_gas_price(target_tx)
        route = await tools.find_optimal_route(target_tx['route_params'])
        
        return {
            'target_hash': target_tx['hash'],
            'route': route,
            'gas_price': gas_price,
            'gas_limit': self._estimate_gas_limit(route),
            'deadline': int(datetime.utcnow().timestamp() + self.execution_timeout)
        }

    async def _validate_target(self, target_tx: Dict) -> bool:
        """Validate target transaction for backrunning."""
        tools = self.toolkit.get_tools()
        
        # Check profitability
        profit = await tools.calculate_backrun_profit(target_tx)
        if profit < self.min_profit:
            return False
            
        # Verify gas costs
        gas_estimate = await tools.estimate_gas(target_tx)
        if gas_estimate > self.max_gas_price:
            return False
            
        return True

    def _log_error(self, message: str, error: Exception) -> None:
        """Log error with details."""
        error_details = {
            'message': message,
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        print(f"Backrun Error: {error_details}")