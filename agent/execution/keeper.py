from typing import Dict, List, Optional, Set
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit

class LiquidationKeeper:
    """
    Monitors and executes liquidation opportunities using CDP's toolkit.
    Focuses on profitable liquidations while maintaining efficient execution.
    """
    
    def __init__(self, config: Dict):
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        # Liquidation parameters
        self.min_profit_threshold = Decimal(str(config.get('min_profit_threshold', '0.05')))
        self.max_gas_price = Decimal(str(config.get('max_gas_price', '500')))
        self.min_health_factor = Decimal(str(config.get('min_health_factor', '1.0')))
        
        # Tracking state
        self.monitored_positions: Dict[str, Dict] = {}
        self.active_liquidations: Dict[str, Dict] = {}
        self.liquidation_history: List[Dict] = []
        self.bundle_opportunities: List[Dict] = []

    async def start_monitoring(self) -> None:
        """Start monitoring for liquidation opportunities."""
        try:
            await asyncio.gather(
                self._monitor_health_factors(),
                self._monitor_price_updates(),
                self._execute_liquidations()
            )
        except Exception as e:
            self._log_error("Monitoring initialization failed", e)
            raise

    async def analyze_opportunity(self, position_data: Dict) -> Dict:
        """
        Analyze a potential liquidation opportunity.
        
        Args:
            position_data: Position data to analyze
            
        Returns:
            Liquidation opportunity analysis
        """
        tools = self.toolkit.get_tools()
        
        # Get current price data
        price_data = await tools.get_price_data(position_data['collateral_token'])
        
        # Calculate liquidation values
        liquidation_values = await self._calculate_liquidation_values(
            position_data,
            price_data
        )
        
        # Estimate execution costs
        execution_costs = await self._estimate_execution_costs(
            position_data,
            liquidation_values
        )
        
        # Calculate potential profit
        profit = await self._calculate_profit(
            liquidation_values,
            execution_costs
        )
        
        return {
            'position_id': position_data['id'],
            'health_factor': position_data['health_factor'],
            'liquidation_values': liquidation_values,
            'execution_costs': execution_costs,
            'potential_profit': profit,
            'is_profitable': profit > self.min_profit_threshold,
            'timestamp': datetime.utcnow()
        }

    async def execute_liquidation(self, opportunity: Dict) -> Dict:
        """
        Execute a liquidation opportunity.
        
        Args:
            opportunity: Liquidation opportunity details
            
        Returns:
            Liquidation execution results
        """
        try:
            tools = self.toolkit.get_tools()
            
            # Prepare liquidation transaction
            tx_params = await self._prepare_liquidation(opportunity)
            
            # Execute liquidation
            result = await tools.execute_transaction(tx_params, priority='high')
            
            if result['success']:
                await self._process_successful_liquidation(
                    opportunity,
                    result
                )
            
            return {
                'status': 'success' if result['success'] else 'failed',
                'transaction': result,
                'profit': await self._calculate_actual_profit(result),
                'gas_used': result.get('gasUsed'),
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            self._log_error(f"Liquidation execution failed for {opportunity['position_id']}", e)
            raise

    async def _monitor_health_factors(self) -> None:
        """Monitor position health factors continuously."""
        tools = self.toolkit.get_tools()
        
        while True:
            try:
                # Get positions near liquidation
                risky_positions = await tools.get_risky_positions(
                    max_health=self.min_health_factor
                )
                
                for position in risky_positions:
                    if position['health_factor'] < self.min_health_factor:
                        opportunity = await self.analyze_opportunity(position)
                        if opportunity['is_profitable']:
                            await self._queue_liquidation(opportunity)
                            
                await asyncio.sleep(1)  # Fast polling interval
                
            except Exception as e:
                self._log_error("Health factor monitoring error", e)
                await asyncio.sleep(5)  # Longer interval on error

    def _log_error(self, message: str, error: Exception) -> None:
        """Log error with details."""
        error_details = {
            'message': message,
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        print(f"Liquidation Keeper Error: {error_details}")  # Replace with proper logging