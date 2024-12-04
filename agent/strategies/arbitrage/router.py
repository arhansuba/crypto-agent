from typing import Dict, List, Optional, Set, Tuple
from decimal import Decimal
from datetime import datetime
import asyncio
import heapq
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit

class RouteOptimizer:
    """
    Optimizes trading routes across DEXs for maximum profitability while
    considering execution costs and market impact.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        # Optimization parameters
        self.max_route_depth = config.get('max_route_depth', 3)
        self.min_liquidity_threshold = Decimal(str(config.get('min_liquidity', '1000.0')))
        self.max_price_impact = Decimal(str(config.get('max_price_impact', '0.01')))
        
        # Cache for optimization
        self.liquidity_cache: Dict[str, Dict] = {}
        self.route_cache: Dict[str, List[Dict]] = {}
        self.last_update: Dict[str, datetime] = {}

    async def find_optimal_route(
        self,
        token_in: str,
        token_out: str,
        amount: Decimal,
        max_hops: Optional[int] = None
    ) -> Dict:
        """
        Find the optimal trading route between two tokens.
        
        Args:
            token_in: Input token address
            token_out: Output token address
            amount: Trade amount
            max_hops: Maximum number of intermediate hops
            
        Returns:
            Optimal route details including execution steps and expected output
        """
        tools = self.toolkit.get_tools()
        
        # Get current market state
        await self._update_market_state([token_in, token_out])
        
        # Find all possible routes
        routes = await self._find_possible_routes(
            token_in,
            token_out,
            max_hops or self.max_route_depth
        )
        
        # Calculate metrics for each route
        route_metrics = await asyncio.gather(*[
            self._calculate_route_metrics(route, amount)
            for route in routes
        ])
        
        # Select optimal route
        optimal_route = await self._select_optimal_route(route_metrics)
        
        # Validate and prepare execution plan
        execution_plan = await self._prepare_execution_plan(optimal_route, amount)
        
        return {
            'route': optimal_route,
            'execution_plan': execution_plan,
            'metrics': await self._calculate_final_metrics(execution_plan),
            'timestamp': datetime.utcnow()
        }

    async def _find_possible_routes(
        self,
        token_in: str,
        token_out: str,
        max_hops: int
    ) -> List[List[Dict]]:
        """
        Find all possible trading routes between tokens.
        
        Args:
            token_in: Input token address
            token_out: Output token address
            max_hops: Maximum number of hops
            
        Returns:
            List of possible trading routes
        """
        tools = self.toolkit.get_tools()
        
        # Get supported DEXs and pairs
        dexs = await tools.get_supported_dexes()
        pairs = await tools.get_token_pairs()
        
        routes = []
        visited = set()
        
        async def find_routes(current_token: str, current_route: List[Dict], depth: int):
            if depth > max_hops:
                return
                
            if current_token == token_out:
                routes.append(current_route.copy())
                return
                
            for dex in dexs:
                for pair in pairs:
                    if self._is_valid_next_hop(
                        current_token,
                        pair,
                        dex,
                        visited,
                        current_route
                    ):
                        next_token = self._get_other_token(current_token, pair)
                        visited.add(next_token)
                        current_route.append({
                            'dex': dex,
                            'pair': pair,
                            'token_in': current_token,
                            'token_out': next_token
                        })
                        
                        await find_routes(next_token, current_route, depth + 1)
                        
                        current_route.pop()
                        visited.remove(next_token)
        
        visited.add(token_in)
        await find_routes(token_in, [], 0)
        
        return routes

    async def _calculate_route_metrics(
        self,
        route: List[Dict],
        amount: Decimal
    ) -> Dict:
        """
        Calculate execution metrics for a potential route.
        
        Args:
            route: Trading route to analyze
            amount: Trade amount
            
        Returns:
            Route metrics including costs and expected output
        """
        tools = self.toolkit.get_tools()
        
        # Initialize metrics
        remaining_amount = amount
        total_gas_cost = Decimal('0')
        total_price_impact = Decimal('0')
        min_liquidity = Decimal('inf')
        
        # Calculate metrics for each hop
        for hop in route:
            # Get liquidity information
            liquidity = await tools.get_pair_liquidity(hop['pair'])
            min_liquidity = min(min_liquidity, Decimal(str(liquidity['token_in'])))
            
            # Calculate expected output and price impact
            output_amount = await tools.get_expected_output(
                hop['pair'],
                remaining_amount,
                hop['dex']
            )
            
            price_impact = (remaining_amount - output_amount) / remaining_amount
            total_price_impact += price_impact
            
            # Estimate gas cost
            gas_cost = await self._estimate_hop_gas_cost(hop)
            total_gas_cost += gas_cost
            
            remaining_amount = output_amount
        
        return {
            'route': route,
            'expected_output': remaining_amount,
            'total_gas_cost': total_gas_cost,
            'total_price_impact': total_price_impact,
            'min_liquidity': min_liquidity,
            'score': self._calculate_route_score(
                remaining_amount,
                total_gas_cost,
                total_price_impact,
                min_liquidity
            )
        }

    async def _select_optimal_route(self, route_metrics: List[Dict]) -> Dict:
        """Select the optimal route based on calculated metrics."""
        if not route_metrics:
            raise ValueError("No valid routes found")
            
        # Sort routes by score
        sorted_routes = sorted(
            route_metrics,
            key=lambda x: x['score'],
            reverse=True
        )
        
        # Validate best route meets thresholds
        best_route = sorted_routes[0]
        if (
            best_route['total_price_impact'] > self.max_price_impact or
            best_route['min_liquidity'] < self.min_liquidity_threshold
        ):
            raise ValueError("No routes meet execution criteria")
            
        return best_route

    async def _prepare_execution_plan(self, route: Dict, amount: Decimal) -> List[Dict]:
        """Prepare detailed execution plan for optimal route."""
        tools = self.toolkit.get_tools()
        
        execution_steps = []
        remaining_amount = amount
        
        for hop in route['route']:
            # Get exact quotation
            quotation = await tools.get_exact_quotation(
                hop['pair'],
                remaining_amount,
                hop['dex']
            )
            
            execution_steps.append({
                'dex': hop['dex'],
                'pair': hop['pair'],
                'token_in': hop['token_in'],
                'token_out': hop['token_out'],
                'amount_in': remaining_amount,
                'expected_out': quotation['amount_out'],
                'min_out': quotation['min_amount_out'],
                'gas_estimate': quotation['gas_estimate']
            })
            
            remaining_amount = quotation['amount_out']
        
        return execution_steps

    def _calculate_route_score(
        self,
        output_amount: Decimal,
        gas_cost: Decimal,
        price_impact: Decimal,
        min_liquidity: Decimal
    ) -> Decimal:
        """Calculate overall route score for comparison."""
        return (
            output_amount *
            (Decimal('1') - price_impact) *
            (min_liquidity / self.min_liquidity_threshold) -
            gas_cost
        )