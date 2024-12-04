from typing import Dict, List, Optional, Set, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit

class CDPDEXMonitor:
    """
    DEX monitoring system for arbitrage opportunities using CDP toolkit.
    Monitors price differences across multiple DEXs and identifies profitable trades.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        # Monitoring state
        self.monitored_pairs: Set[str] = set()
        self.price_cache: Dict[str, Dict] = {}
        self.last_update: Dict[str, datetime] = {}
        self.active_opportunities: List[Dict] = []
        
        # Configuration
        self.min_profit_threshold = Decimal(str(config.get('min_profit_threshold', '0.005')))
        self.max_price_age = timedelta(seconds=config.get('max_price_age', 30))
        self.update_interval = config.get('update_interval', 1.0)
        self.gas_buffer = Decimal(str(config.get('gas_buffer', '0.002')))

    async def start_monitoring(self, token_pairs: List[str]) -> None:
        """
        Start monitoring specified token pairs across DEXs.
        
        Args:
            token_pairs: List of token pairs to monitor (e.g., ['ETH-USDC', 'WBTC-ETH'])
        """
        self.monitored_pairs.update(token_pairs)
        monitoring_tasks = [
            self._monitor_pair(pair)
            for pair in token_pairs
        ]
        await asyncio.gather(*monitoring_tasks)

    async def get_arbitrage_opportunities(self) -> List[Dict]:
        """
        Get current arbitrage opportunities that meet profit threshold.
        
        Returns:
            List of profitable arbitrage opportunities with execution details
        """
        opportunities = []
        
        for pair in self.monitored_pairs:
            if not self._is_price_fresh(pair):
                continue
                
            prices = self.price_cache.get(pair, {})
            opportunity = await self._analyze_opportunity(pair, prices)
            
            if opportunity and self._meets_profit_threshold(opportunity):
                opportunities.append(opportunity)
                
        return opportunities

    async def _monitor_pair(self, pair: str) -> None:
        """
        Monitor a specific trading pair across multiple DEXs.
        
        Args:
            pair: Trading pair identifier (e.g., 'ETH-USDC')
        """
        while True:
            try:
                tools = self.toolkit.get_tools()
                prices = {}
                
                # Get prices from different DEXs using CDP tools
                for dex in await self._get_supported_dexes():
                    price = await tools.get_token_price(
                        pair,
                        exchange=dex
                    )
                    prices[dex] = Decimal(str(price))
                
                self._update_price_cache(pair, prices)
                await self._check_opportunities(pair, prices)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self._log_error(f"Error monitoring {pair}", e)
                await asyncio.sleep(self.update_interval * 2)

    async def _analyze_opportunity(self, pair: str, prices: Dict[str, Decimal]) -> Optional[Dict]:
        """
        Analyze price differences for arbitrage opportunities.
        
        Args:
            pair: Trading pair identifier
            prices: Dictionary of prices by DEX
            
        Returns:
            Arbitrage opportunity details if profitable
        """
        if len(prices) < 2:
            return None
            
        lowest_price = min(prices.values())
        highest_price = max(prices.values())
        
        profit_ratio = (highest_price - lowest_price) / lowest_price
        
        if profit_ratio <= self.min_profit_threshold:
            return None
            
        buy_dex = min(prices.items(), key=lambda x: x[1])[0]
        sell_dex = max(prices.items(), key=lambda x: x[1])[0]
        
        # Calculate optimal trade size using CDP's position sizing
        trade_size = await self._calculate_optimal_trade_size(
            pair, profit_ratio, lowest_price
        )
        
        return {
            'pair': pair,
            'buy_dex': buy_dex,
            'sell_dex': sell_dex,
            'buy_price': lowest_price,
            'sell_price': highest_price,
            'profit_ratio': profit_ratio,
            'trade_size': trade_size,
            'timestamp': datetime.utcnow(),
            'estimated_profit': trade_size * (highest_price - lowest_price) - self.gas_buffer
        }

    async def _calculate_optimal_trade_size(
        self,
        pair: str,
        profit_ratio: Decimal,
        entry_price: Decimal
    ) -> Decimal:
        """
        Calculate optimal trade size considering liquidity and risk.
        
        Args:
            pair: Trading pair identifier
            profit_ratio: Expected profit ratio
            entry_price: Entry price for the trade
        
        Returns:
            Optimal trade size in base currency
        """
        tools = self.toolkit.get_tools()
        
        # Get liquidity information from CDP
        liquidity = await tools.get_pair_liquidity(pair)
        max_trade_size = min(
            Decimal(str(liquidity['base_token'])) * Decimal('0.1'),  # Use 10% of liquidity
            Decimal(str(self.config.get('max_trade_size', '1.0')))
        )
        
        # Calculate size based on profit ratio
        risk_adjusted_size = max_trade_size * (profit_ratio / Decimal('0.1'))
        
        return min(risk_adjusted_size, max_trade_size)

    def _update_price_cache(self, pair: str, prices: Dict[str, Decimal]) -> None:
        """Update price cache with latest prices."""
        self.price_cache[pair] = prices
        self.last_update[pair] = datetime.utcnow()

    def _is_price_fresh(self, pair: str) -> bool:
        """Check if cached price is within freshness threshold."""
        last_update = self.last_update.get(pair)
        if not last_update:
            return False
        return datetime.utcnow() - last_update <= self.max_price_age

    def _meets_profit_threshold(self, opportunity: Dict) -> bool:
        """Check if opportunity meets minimum profit threshold."""
        return opportunity['profit_ratio'] > self.min_profit_threshold

    async def _get_supported_dexes(self) -> List[str]:
        """Get list of supported DEXs from CDP toolkit."""
        tools = self.toolkit.get_tools()
        return await tools.get_supported_dexes()

    def _log_error(self, message: str, error: Exception) -> None:
        """Log error with details."""
        error_details = {
            'message': message,
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        print(f"DEX Monitor Error: {error_details}")  # Replace with proper logging