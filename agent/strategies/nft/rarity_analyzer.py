from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from datetime import datetime
import asyncio
from collections import Counter
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit

class RarityAnalyzer:
    """
    Advanced NFT rarity analysis system that calculates trait rarity scores
    and identifies valuable NFTs based on their unique characteristics.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        # Analysis parameters
        self.trait_weights = config.get('trait_weights', {})
        self.min_rarity_score = Decimal(str(config.get('min_rarity_score', '0.01')))
        self.statistical_significance = Decimal(str(config.get('statistical_significance', '0.05')))
        
        # Cache and state
        self.collection_metadata: Dict[str, Dict] = {}
        self.trait_statistics: Dict[str, Dict] = {}
        self.rarity_scores: Dict[str, Dict] = {}
        self.last_update: Dict[str, datetime] = {}

    async def analyze_collection(self, collection_address: str) -> Dict:
        """
        Perform comprehensive rarity analysis for an NFT collection.
        
        Args:
            collection_address: Address of the NFT collection to analyze
            
        Returns:
            Collection analysis including trait distribution and rarity scores
        """
        await self._update_collection_data(collection_address)
        
        collection_stats = await self._calculate_collection_statistics(collection_address)
        trait_analysis = await self._analyze_trait_distribution(collection_address)
        rarity_rankings = self._calculate_rarity_rankings(collection_address)
        
        return {
            'collection_stats': collection_stats,
            'trait_analysis': trait_analysis,
            'rarity_rankings': rarity_rankings,
            'timestamp': datetime.utcnow()
        }

    async def calculate_nft_rarity(
        self,
        collection_address: str,
        token_id: int
    ) -> Dict:
        """
        Calculate comprehensive rarity score for a specific NFT.
        
        Args:
            collection_address: Collection address
            token_id: Token ID of the NFT
            
        Returns:
            Detailed rarity analysis for the NFT
        """
        tools = self.toolkit.get_tools()
        
        # Get NFT metadata
        metadata = await tools.get_nft_metadata(collection_address, token_id)
        
        # Ensure collection data is up to date
        await self._update_collection_data(collection_address)
        
        # Calculate trait scores
        trait_scores = await self._calculate_trait_scores(
            collection_address,
            metadata['attributes']
        )
        
        # Calculate overall rarity score
        total_score = sum(trait_scores.values())
        
        # Get relative rarity ranking
        ranking = self._get_rarity_ranking(collection_address, total_score)
        
        return {
            'token_id': token_id,
            'total_score': total_score,
            'trait_scores': trait_scores,
            'ranking': ranking,
            'percentile': self._calculate_percentile(ranking, collection_address),
            'unique_traits': self._identify_unique_traits(metadata['attributes']),
            'analysis': self._generate_rarity_analysis(trait_scores, ranking)
        }

    async def find_rare_nfts(
        self,
        collection_address: str,
        min_score: Optional[Decimal] = None
    ) -> List[Dict]:
        """
        Find NFTs that meet specified rarity criteria.
        
        Args:
            collection_address: Collection address
            min_score: Minimum rarity score threshold
            
        Returns:
            List of NFTs meeting rarity criteria
        """
        min_score = min_score or self.min_rarity_score
        
        tools = self.toolkit.get_tools()
        collection_size = await tools.get_collection_size(collection_address)
        
        rare_nfts = []
        for token_id in range(collection_size):
            try:
                rarity = await self.calculate_nft_rarity(collection_address, token_id)
                if rarity['total_score'] >= min_score:
                    rare_nfts.append(rarity)
            except Exception as e:
                self._log_error(f"Error analyzing token {token_id}", e)
                
        return sorted(rare_nfts, key=lambda x: x['total_score'], reverse=True)

    async def _calculate_collection_statistics(self, collection_address: str) -> Dict:
        """Calculate statistical metrics for the collection."""
        tools = self.toolkit.get_tools()
        metadata = self.collection_metadata[collection_address]
        
        traits_count = Counter()
        trait_combinations = Counter()
        
        for token_metadata in metadata.values():
            trait_set = frozenset((t['trait_type'], t['value']) 
                                for t in token_metadata['attributes'])
            traits_count.update(trait_set)
            trait_combinations[trait_set] += 1
        
        return {
            'total_supply': len(metadata),
            'unique_traits': len(traits_count),
            'unique_combinations': len(trait_combinations),
            'trait_frequency': dict(traits_count),
            'statistical_rarity': self._calculate_statistical_rarity(traits_count)
        }

    async def _analyze_trait_distribution(self, collection_address: str) -> Dict:
        """Analyze trait distribution and identify significant patterns."""
        metadata = self.collection_metadata[collection_address]
        
        trait_types = {}
        for token_metadata in metadata.values():
            for trait in token_metadata['attributes']:
                trait_type = trait['trait_type']
                trait_value = trait['value']
                
                if trait_type not in trait_types:
                    trait_types[trait_type] = Counter()
                trait_types[trait_type][trait_value] += 1
        
        return {
            'trait_types': {
                trait_type: {
                    'count': len(counts),
                    'distribution': dict(counts),
                    'entropy': self._calculate_entropy(counts)
                }
                for trait_type, counts in trait_types.items()
            }
        }

    def _calculate_entropy(self, counts: Counter) -> Decimal:
        """Calculate Shannon entropy for trait distribution."""
        total = sum(counts.values())
        probabilities = [count/total for count in counts.values()]
        
        entropy = Decimal('0')
        for p in probabilities:
            if p > 0:
                entropy -= Decimal(str(p)) * Decimal(str(p)).ln()
        
        return entropy

    def _log_error(self, message: str, error: Exception) -> None:
        """Log error with details."""
        error_details = {
            'message': message,
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        print(f"Rarity Analyzer Error: {error_details}")  # Replace with proper logging