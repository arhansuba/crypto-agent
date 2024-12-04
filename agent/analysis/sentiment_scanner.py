from typing import Dict, List, Optional, Tuple, Union
from decimal import Decimal
from datetime import datetime, timedelta
import asyncio
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.agent_toolkits import CdpToolkit
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentScanner:
    """
    Advanced sentiment analysis system that processes social media and market data
    to generate comprehensive sentiment insights.
    """
    
    def __init__(self, config: Dict):
        self.cdp = CdpAgentkitWrapper()
        self.toolkit = CdpToolkit.from_cdp_agentkit_wrapper(self.cdp)
        
        # Initialize sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        
        # Analysis parameters
        self.min_confidence = Decimal(str(config.get('min_confidence', '0.6')))
        self.sentiment_window = timedelta(hours=config.get('sentiment_window', 24))
        self.volume_weight = Decimal(str(config.get('volume_weight', '0.3')))
        
        # State tracking
        self.sentiment_history: Dict[str, List[Dict]] = {}
        self.keyword_trends: Dict[str, Dict] = {}
        self.source_weights: Dict[str, Decimal] = {
            'twitter': Decimal('0.4'),
            'telegram': Decimal('0.3'),
            'discord': Decimal('0.3')
        }

    async def analyze_token_sentiment(self, token_address: str) -> Dict:
        """
        Analyze sentiment for a specific token across multiple sources.
        
        Args:
            token_address: Token address to analyze
            
        Returns:
            Comprehensive sentiment analysis
        """
        tools = self.toolkit.get_tools()
        
        # Get social data
        twitter_data = await tools.get_twitter_mentions(token_address)
        telegram_data = await tools.get_telegram_mentions(token_address)
        discord_data = await tools.get_discord_mentions(token_address)
        
        # Analyze each source
        twitter_sentiment = await self._analyze_source_sentiment(twitter_data)
        telegram_sentiment = await self._analyze_source_sentiment(telegram_data)
        discord_sentiment = await self._analyze_source_sentiment(discord_data)
        
        # Combine sentiment scores
        combined_sentiment = self._combine_sentiment_scores({
            'twitter': twitter_sentiment,
            'telegram': telegram_sentiment,
            'discord': discord_sentiment
        })
        
        # Analyze patterns and trends
        sentiment_trends = self._analyze_sentiment_trends(token_address)
        
        return {
            'current_sentiment': combined_sentiment,
            'source_sentiment': {
                'twitter': twitter_sentiment,
                'telegram': telegram_sentiment,
                'discord': discord_sentiment
            },
            'trends': sentiment_trends,
            'keyword_analysis': await self._analyze_keywords(token_address),
            'confidence_score': self._calculate_confidence(combined_sentiment),
            'timestamp': datetime.utcnow()
        }

    async def monitor_sentiment_changes(
        self,
        token_address: str,
        callback: Optional[callable] = None
    ) -> None:
        """
        Monitor sentiment changes in real-time.
        
        Args:
            token_address: Token to monitor
            callback: Optional callback for significant changes
        """
        previous_sentiment = None
        
        while True:
            try:
                current_analysis = await self.analyze_token_sentiment(token_address)
                current_sentiment = current_analysis['current_sentiment']
                
                if previous_sentiment is not None:
                    change = self._calculate_sentiment_change(
                        previous_sentiment,
                        current_sentiment
                    )
                    
                    if self._is_significant_change(change) and callback:
                        await callback(current_analysis)
                        
                previous_sentiment = current_sentiment
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self._log_error("Sentiment monitoring error", e)
                await asyncio.sleep(300)  # Longer interval on error

    async def _analyze_source_sentiment(self, data: List[Dict]) -> Dict:
        """Analyze sentiment for a specific data source."""
        if not data:
            return {'score': Decimal('0'), 'magnitude': Decimal('0')}
            
        scores = []
        magnitudes = []
        
        for item in data:
            # Analyze with VADER
            vader_scores = self.vader.polarity_scores(item['text'])
            
            # Analyze with TextBlob for comparison
            blob_analysis = TextBlob(item['text'])
            
            # Combine scores with weights
            combined_score = (
                Decimal(str(vader_scores['compound'])) * Decimal('0.7') +
                Decimal(str(blob_analysis.sentiment.polarity)) * Decimal('0.3')
            )
            
            scores.append(combined_score)
            magnitudes.append(Decimal(str(abs(combined_score))))
            
        return {
            'score': sum(scores) / len(scores) if scores else Decimal('0'),
            'magnitude': sum(magnitudes) / len(magnitudes) if magnitudes else Decimal('0'),
            'sample_size': len(scores)
        }

    async def _analyze_keywords(self, token_address: str) -> Dict:
        """Analyze keyword trends and significance."""
        tools = self.toolkit.get_tools()
        
        # Get recent mentions
        mentions = await tools.get_all_mentions(token_address)
        
        # Extract and analyze keywords
        keywords = {}
        for mention in mentions:
            words = self._extract_keywords(mention['text'])
            for word in words:
                if word not in keywords:
                    keywords[word] = {
                        'count': 0,
                        'sentiment_sum': Decimal('0'),
                        'contexts': []
                    }
                    
                keywords[word]['count'] += 1
                keywords[word]['sentiment_sum'] += self._get_word_sentiment(word, mention['text'])
                
                if len(keywords[word]['contexts']) < 5:  # Keep up to 5 example contexts
                    keywords[word]['contexts'].append(mention['text'])
                    
        return {
            'trending_keywords': self._get_trending_keywords(keywords),
            'sentiment_by_keyword': self._calculate_keyword_sentiment(keywords),
            'keyword_correlations': self._analyze_keyword_correlations(keywords)
        }

    def _log_error(self, message: str, error: Exception) -> None:
        """Log error with details."""
        error_details = {
            'message': message,
            'error': str(error),
            'timestamp': datetime.utcnow().isoformat()
        }
        print(f"Sentiment Scanner Error: {error_details}")