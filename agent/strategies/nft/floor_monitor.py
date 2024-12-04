from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit

class NFTFloorMonitor:
    """
    Advanced NFT floor price monitoring system that tracks and analyzes floor prices
    across multiple marketplaces while identifying potential trading opportunities.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        # Monitoring parameters
        self.update_interval = config.get('update_interval', 60)
        self.price_threshold = Decimal(str(config.get('price_threshold', '0.01')))
        self.volatility_window = config.get('volatility_window', 24)
        
        # State tracking
        self.floor_prices: Dict[str, Dict] = {}
        self.price_history: Dict[str, List[Tuple[datetime, Decimal]]] = {}
        self.active_alerts: Dict[str, List[Dict]] = {}
        self.marketplace_data: Dict[str, Dict] = {}

    async def start_monitoring(self, collection_addresses: List[str]) -> None:
        """
        Begin monitoring specified NFT collections for floor price changes.
        
        Args:
            collection_addresses: List of NFT collection addresses to monitor
        """
        monitoring_tasks = [
            self._monitor_collection(address)
            for address in collection_addresses
        ]
        await asyncio.gather(*monitoring_tasks)

    async def analyze_collection(self, collection_address: str) -> Dict:
        """
        Perform comprehensive analysis of collection floor price metrics.
        
        Args:
            collection_address: NFT collection address to analyze
            
        Returns:
            Analysis results including price trends and volatility metrics
        """
        history = self.price_history.get(collection_address, [])
        if not history:
            return {}
        
        current_floor = self.floor_prices.get(collection_address, {}).get('floor_price')
        if not current_floor:
            return {}
            
        prices = [price for _, price in history]
        timestamps = [ts for ts, _ in history]
        
        return {
            'current_floor': current_floor,
            'metrics': {
                'daily_change': self._calculate_price_change(prices, 24),
                'weekly_change': self._calculate_price_change(prices, 168),
                'volatility': self._calculate_volatility(prices),
                'volume': await self._get_volume_metrics(collection_address),
                'liquidity': await self._analyze_liquidity(collection_address)
            },
            'trends': self._analyze_price_trends(prices, timestamps),
            'opportunities': await self._identify_opportunities(collection_address),
            'timestamp': datetime.utcnow()
        }

    async def get_floor_prices(
        self,
        collection_address: str,
        marketplaces: Optional[List[str]] = None
    ) -> Dict[str, Decimal]:
        """
        Get current floor prices across different marketplaces.
        
        Args:
            collection_address: NFT collection address
            marketplaces: Optional list of specific marketplaces to check
            
        Returns:
            Dictionary of floor prices by marketplace
        """
        tools = self.toolkit.get_tools()
        
        if not marketplaces:
            marketplaces = await tools.get_supported_nft_marketplaces()
            
        prices = {}
        for marketplace in marketplaces:
            try:
                price = await tools.get_nft_floor_price(
                    collection_address,
                    marketplace
                )
                prices[marketplace] = Decimal(str(price))
            except Exception as e:
                self._log_error(f"Error getting price from {marketplace}", e)
                
        return prices

    async def _monitor_collection(self, collection_address: str) -> None:
        """Monitor a specific NFT collection continuously."""
        while True:
            try:
                # Get current floor prices
                prices = await self.get_floor_prices(collection_address)
                
                # Update tracking data
                await self._update_collection_data(collection_address, prices)
                
                # Check for significant changes
                if await self._check_price_changes(collection_address, prices):
                    await self._handle_price_alert(collection_address)
                    
                # Update market metrics
                await self._update_market_metrics(collection_address)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                self._log_error(f"Error monitoring collection {collection_address}", e)
                await asyncio.sleep(self.update_interval * 2)

    async def _update_collection_data(
        self,
        collection_address: str,
        prices: Dict[str, Decimal]
    ) -> None:
        """Update stored collection data with new prices."""
        timestamp = datetime.utcnow()
        
        # Update current floor prices
        self.floor_prices[collection_address] = {
            'floor_price': min(prices.values()),
            'prices_by_marketplace': prices,
            'timestamp': timestamp
        }
        
        # Update price history
        if collection_address not in self.price_history:
            self.price_history[collection_address] = []
            
        self.price_history[collection_address].append(
            (timestamp, min(prices.values()))
        )
        
        # Trim history to window size
        window = timedelta(hours=self.volatility_window)
        self.price_history[collection_address] = [
            (ts, price) for ts, price in self.price_history[collection_address]
            if timestamp - ts <= window
        ]

    async def _check_price_changes(
        self,
        collection_address: str,
        current_prices: Dict[str, Decimal]
    ) -> bool:
        """Check for significant price changes that warrant alerts."""
        history = self.price_history.get(collection_address, [])
        if not history:
            return False
            
        prev_price = history[-1][1]
        current_price = min(current_prices.values())
        
        price_change = abs(current_price - prev_price) / prev_price
        return price_change > self.price_threshold

    def _calculate_volatility(self, prices: List[Decimal]) -> Decimal:
        """Calculate price volatility over the specified window."""
        if len(prices) < 2:
            return Decimal('0')
            
        returns = [
            (prices[i] - prices[i-1]) / prices[i-1]
            for i in range(1, len(prices))
        ]
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        return Decimal(str(variance)).sqrt()

    def _log_error(self, message: str, error: Exception) -> None:
        """Log error with details."""
        error_details = {
            'message': message,
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        print(f"Floor Monitor Error: {error_details}")  # Replace with proper logging