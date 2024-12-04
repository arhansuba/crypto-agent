from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_DOWN
from datetime import datetime
import asyncio
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit

class ProfitScanner:
    """
    Advanced profit calculation system that analyzes trading opportunities
    and tracks realized profits while considering all costs and risks.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        # Profit thresholds and cost factors
        self.min_profit_threshold = Decimal(str(config.get('min_profit_threshold', '0.005')))
        self.gas_buffer = Decimal(str(config.get('gas_buffer', '0.002')))
        self.slippage_tolerance = Decimal(str(config.get('slippage_tolerance', '0.003')))
        
        # Historical data
        self.profit_history: List[Dict] = []
        self.gas_price_history: List[Tuple[datetime, Decimal]] = []
        self.execution_costs: Dict[str, Decimal] = {}

    async def calculate_opportunity_profit(
        self,
        buy_price: Decimal,
        sell_price: Decimal,
        amount: Decimal,
        trade_params: Dict
    ) -> Dict:
        """
        Calculate potential profit for a trading opportunity including all costs.
        
        Args:
            buy_price: Entry price for the trade
            sell_price: Exit price for the trade
            amount: Trading amount
            trade_params: Additional trading parameters
            
        Returns:
            Dictionary containing profit calculations and analysis
        """
        tools = self.toolkit.get_tools()
        
        # Get current gas prices and network conditions
        gas_data = await tools.get_gas_price()
        network_params = await self._get_network_parameters()
        
        # Calculate raw profit
        raw_profit = (sell_price - buy_price) * amount
        
        # Calculate costs
        gas_cost = await self._calculate_gas_cost(trade_params, gas_data)
        slippage_cost = self._estimate_slippage_cost(amount, trade_params)
        protocol_fees = self._calculate_protocol_fees(amount, trade_params)
        
        # Calculate net profit
        total_costs = gas_cost + slippage_cost + protocol_fees
        net_profit = raw_profit - total_costs
        
        return {
            'raw_profit': raw_profit,
            'net_profit': net_profit,
            'profit_margin': net_profit / (buy_price * amount),
            'costs': {
                'gas': gas_cost,
                'slippage': slippage_cost,
                'protocol_fees': protocol_fees,
                'total_costs': total_costs
            },
            'metrics': {
                'roi': (net_profit / (buy_price * amount)) * Decimal('100'),
                'profit_factor': raw_profit / total_costs if total_costs > 0 else Decimal('0'),
                'break_even_price': buy_price + (total_costs / amount)
            },
            'is_profitable': net_profit > self.min_profit_threshold,
            'timestamp': datetime.utcnow()
        }

    async def analyze_historical_performance(
        self,
        timeframe: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict:
        """
        Analyze historical trading performance and profitability metrics.
        
        Args:
            timeframe: Optional tuple of start and end datetime for analysis
            
        Returns:
            Dictionary containing performance metrics and analysis
        """
        filtered_history = self._filter_history_by_timeframe(timeframe)
        
        total_profit = sum(trade['net_profit'] for trade in filtered_history)
        total_trades = len(filtered_history)
        profitable_trades = sum(1 for trade in filtered_history if trade['net_profit'] > 0)
        
        return {
            'total_profit': total_profit,
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': Decimal(profitable_trades) / Decimal(total_trades) if total_trades > 0 else Decimal('0'),
            'average_profit': total_profit / Decimal(total_trades) if total_trades > 0 else Decimal('0'),
            'largest_profit': max((trade['net_profit'] for trade in filtered_history), default=Decimal('0')),
            'largest_loss': min((trade['net_profit'] for trade in filtered_history), default=Decimal('0')),
            'cost_metrics': await self._analyze_cost_metrics(filtered_history),
            'timeframe': timeframe
        }

    async def _calculate_gas_cost(self, trade_params: Dict, gas_data: Dict) -> Decimal:
        """
        Calculate total gas cost for a trade including current network conditions.
        
        Args:
            trade_params: Trading parameters
            gas_data: Current gas price data
            
        Returns:
            Calculated gas cost in base currency
        """
        base_gas = Decimal(str(trade_params.get('estimated_gas', 150000)))
        gas_price = Decimal(str(gas_data['fast_gas_price']))
        
        # Add safety buffer for gas price volatility
        gas_price_with_buffer = gas_price * (Decimal('1') + self.gas_buffer)
        
        return (base_gas * gas_price_with_buffer) / Decimal('1e9')

    def _estimate_slippage_cost(self, amount: Decimal, trade_params: Dict) -> Decimal:
        """
        Estimate slippage cost based on trade size and market conditions.
        
        Args:
            amount: Trading amount
            trade_params: Trading parameters
            
        Returns:
            Estimated slippage cost
        """
        base_slippage = self.slippage_tolerance
        
        # Adjust slippage based on trade size
        size_factor = amount / Decimal(str(self.config.get('reference_trade_size', '1.0')))
        adjusted_slippage = base_slippage * (Decimal('1') + size_factor)
        
        return amount * adjusted_slippage

    def _calculate_protocol_fees(self, amount: Decimal, trade_params: Dict) -> Decimal:
        """
        Calculate protocol fees for the trade.
        
        Args:
            amount: Trading amount
            trade_params: Trading parameters
            
        Returns:
            Total protocol fees
        """
        fee_rate = Decimal(str(trade_params.get('fee_rate', '0.003')))
        return amount * fee_rate

    def _filter_history_by_timeframe(
        self,
        timeframe: Optional[Tuple[datetime, datetime]]
    ) -> List[Dict]:
        """Filter profit history by specified timeframe."""
        if not timeframe:
            return self.profit_history
            
        start_time, end_time = timeframe
        return [
            trade for trade in self.profit_history
            if start_time <= trade['timestamp'] <= end_time
        ]

    async def _analyze_cost_metrics(self, history: List[Dict]) -> Dict:
        """Analyze cost metrics from trading history."""
        if not history:
            return {}
            
        total_gas = sum(trade['costs']['gas'] for trade in history)
        total_slippage = sum(trade['costs']['slippage'] for trade in history)
        total_fees = sum(trade['costs']['protocol_fees'] for trade in history)
        
        return {
            'average_gas_cost': total_gas / Decimal(len(history)),
            'average_slippage': total_slippage / Decimal(len(history)),
            'average_protocol_fees': total_fees / Decimal(len(history)),
            'cost_breakdown_percentage': {
                'gas': (total_gas / (total_gas + total_slippage + total_fees)) * Decimal('100'),
                'slippage': (total_slippage / (total_gas + total_slippage + total_fees)) * Decimal('100'),
                'protocol_fees': (total_fees / (total_gas + total_slippage + total_fees)) * Decimal('100')
            }
        }

    async def _get_network_parameters(self) -> Dict:
        """Get current network parameters from CDP toolkit."""
        tools = self.toolkit.get_tools()
        return await tools.get_network_parameters()