"""
Twitter integration module for AI crypto agent.
Handles tweet posting, interaction monitoring, and sentiment analysis.
"""

import tweepy
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import asyncio
import json
from dataclasses import dataclass
from textblob import TextBlob
import nltk
from collections import deque
import re
import numpy as np

@dataclass
class TweetData:
    """Represents processed tweet information"""
    tweet_id: str
    text: str
    created_at: datetime
    user: str
    sentiment_score: float
    engagement_metrics: Dict[str, int]
    hashtags: List[str]
    mentions: List[str]
    urls: List[str]
    is_reply: bool
    reply_to: Optional[str]

class TwitterManager:
    """
    Manages Twitter interactions and sentiment analysis for the AI agent.
    Handles automated posting, engagement monitoring, and market sentiment tracking.
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """Initialize Twitter manager with API credentials"""
        self.config = config
        self.logger = logger
        
        # Initialize Twitter API client
        self.client = self._initialize_twitter_client()
        
        # Initialize sentiment analyzer
        nltk.download('vader_lexicon', quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize tweet history
        self.tweet_history = deque(maxlen=config['twitter']['history_size'])
        self.interaction_history = {}
        
        # Load tweet templates
        self.tweet_templates = self._load_tweet_templates()
        
        self.logger.info("Twitter manager initialized successfully")

    def _initialize_twitter_client(self) -> tweepy.Client:
        """Initialize Twitter API client with credentials"""
        try:
            client = tweepy.Client(
                bearer_token=self.config['twitter']['bearer_token'],
                consumer_key=self.config['twitter']['api_key'],
                consumer_secret=self.config['twitter']['api_secret'],
                access_token=self.config['twitter']['access_token'],
                access_token_secret=self.config['twitter']['access_token_secret']
            )
            return client
            
        except Exception as e:
            self.logger.error(f"Twitter client initialization failed: {str(e)}")
            raise

    def _load_tweet_templates(self) -> Dict[str, List[str]]:
        """Load tweet templates for different scenarios"""
        try:
            with open(self.config['twitter']['templates_path'], 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load tweet templates: {str(e)}")
            return {}

    async def post_market_update(
        self,
        prediction_data: Dict[str, Any],
        risk_metrics: Dict[str, float]
    ) -> str:
        """Post automated market analysis update"""
        try:
            # Select appropriate template
            template = self._select_tweet_template(
                'market_update',
                prediction_data['trend_direction']
            )
            
            # Format tweet content
            tweet_text = template.format(
                symbol=prediction_data.get('symbol', 'BTC'),
                price=f"${prediction_data['price_prediction']:,.2f}",
                change_percent=f"{prediction_data['price_change']*100:.1f}%",
                confidence=f"{prediction_data['confidence_score']*100:.0f}%",
                risk_level=self._get_risk_level(risk_metrics['overall_risk'])
            )
            
            # Add cashtags and hashtags
            tweet_text = self._add_tags(tweet_text)
            
            # Post tweet
            response = await self._post_tweet(tweet_text)
            
            # Store in history
            self.tweet_history.append(TweetData(
                tweet_id=response.data['id'],
                text=tweet_text,
                created_at=datetime.now(),
                user=self.config['twitter']['username'],
                sentiment_score=self._analyze_sentiment(tweet_text),
                engagement_metrics={'likes': 0, 'retweets': 0, 'replies': 0},
                hashtags=self._extract_hashtags(tweet_text),
                mentions=self._extract_mentions(tweet_text),
                urls=self._extract_urls(tweet_text),
                is_reply=False,
                reply_to=None
            ))
            
            return response.data['id']
            
        except Exception as e:
            self.logger.error(f"Failed to post market update: {str(e)}")
            raise

    def _select_tweet_template(
        self,
        category: str,
        condition: str
    ) -> str:
        """Select appropriate tweet template based on conditions"""
        templates = self.tweet_templates.get(category, {}).get(condition, [])
        if not templates:
            return self.tweet_templates['default'][0]
        return templates[hash(str(datetime.now())) % len(templates)]

    async def monitor_mentions(self) -> List[TweetData]:
        """Monitor and process mentions of the bot"""
        try:
            # Get recent mentions
            mentions = self.client.get_users_mentions(
                self.config['twitter']['user_id'],
                max_results=100,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            processed_mentions = []
            
            for tweet in mentions.data or []:
                # Process each mention
                tweet_data = TweetData(
                    tweet_id=tweet.id,
                    text=tweet.text,
                    created_at=tweet.created_at,
                    user=tweet.author_id,
                    sentiment_score=self._analyze_sentiment(tweet.text),
                    engagement_metrics=tweet.public_metrics,
                    hashtags=self._extract_hashtags(tweet.text),
                    mentions=self._extract_mentions(tweet.text),
                    urls=self._extract_urls(tweet.text),
                    is_reply=tweet.in_reply_to_user_id is not None,
                    reply_to=tweet.in_reply_to_user_id
                )
                
                processed_mentions.append(tweet_data)
                
                # Handle interaction if needed
                await self._handle_mention(tweet_data)
            
            return processed_mentions
            
        except Exception as e:
            self.logger.error(f"Failed to monitor mentions: {str(e)}")
            return []

    async def analyze_market_sentiment(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> Dict[str, float]:
        """Analyze market sentiment from crypto-related tweets"""
        try:
            # Search for relevant tweets
            query = f"({symbol} OR #{symbol}) lang:en -is:retweet"
            
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            sentiment_scores = []
            weighted_sentiment = 0
            total_weight = 0
            
            for tweet in tweets.data or []:
                # Calculate sentiment
                sentiment_score = self._analyze_sentiment(tweet.text)
                engagement_score = self._calculate_engagement_score(
                    tweet.public_metrics
                )
                
                sentiment_scores.append(sentiment_score)
                weighted_sentiment += sentiment_score * engagement_score
                total_weight += engagement_score
            
            return {
                'average_sentiment': sum(sentiment_scores) / len(sentiment_scores)
                if sentiment_scores else 0,
                'weighted_sentiment': weighted_sentiment / total_weight
                if total_weight > 0 else 0,
                'sentiment_volatility': np.std(sentiment_scores)
                if sentiment_scores else 0,
                'sample_size': len(sentiment_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze market sentiment: {str(e)}")
            return {}

    async def _handle_mention(self, tweet_data: TweetData):
        """Handle mentions and generate appropriate responses"""
        try:
            # Check if we should respond
            if not self._should_respond(tweet_data):
                return
            
            # Generate response based on mention content
            response_text = await self._generate_response(tweet_data)
            
            if response_text:
                # Post reply
                await self._post_tweet(
                    response_text,
                    reply_to=tweet_data.tweet_id
                )
                
                # Update interaction history
                self.interaction_history[tweet_data.tweet_id] = {
                    'original_tweet': tweet_data,
                    'response': response_text,
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            self.logger.error(f"Failed to handle mention: {str(e)}")

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using VADER"""
        try:
            # Clean text
            cleaned_text = self._clean_text(text)
            
            # Get sentiment scores
            sentiment_scores = self.sentiment_analyzer.polarity_scores(cleaned_text)
            
            # Return compound score
            return sentiment_scores['compound']
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {str(e)}")
            return 0.0

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean tweet text for analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()

    @staticmethod
    def _calculate_engagement_score(metrics: Dict[str, int]) -> float:
        """Calculate engagement score for a tweet"""
        return (
            metrics.get('like_count', 0) * 1.0 +
            metrics.get('retweet_count', 0) * 2.0 +
            metrics.get('reply_count', 0) * 1.5 +
            metrics.get('quote_count', 0) * 1.8
        )

    @staticmethod
    def _extract_hashtags(text: str) -> List[str]:
        """Extract hashtags from tweet text"""
        return re.findall(r'#(\w+)', text)

    @staticmethod
    def _extract_mentions(text: str) -> List[str]:
        """Extract mentions from tweet text"""
        return re.findall(r'@(\w+)', text)

    @staticmethod
    def _extract_urls(text: str) -> List[str]:
        """Extract URLs from tweet text"""
        return re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)

    @staticmethod
    def _get_risk_level(risk_score: float) -> str:
        """Convert risk score to text level"""
        if risk_score >= 0.7:
            return "High"
        elif risk_score >= 0.3:
            return "Moderate"
        return "Low"

    def _should_respond(self, tweet_data: TweetData) -> bool:
        """Determine if bot should respond to mention"""
        # Check if we've already responded
        if tweet_data.tweet_id in self.interaction_history:
            return False
        
        # Check if it's a reply to our own tweet
        if tweet_data.reply_to == self.config['twitter']['user_id']:
            return False
        
        # Check interaction rate limiting
        recent_interactions = sum(
            1 for interaction in self.interaction_history.values()
            if (datetime.now() - interaction['timestamp']).seconds < 3600
        )
        
        if recent_interactions >= self.config['twitter']['hourly_interaction_limit']:
            return False
        
        return True

    async def _generate_response(self, tweet_data: TweetData) -> Optional[str]:
        """Generate appropriate response to mention"""
        # Implement response generation logic based on mention content
        # This could integrate with a language model or use template-based responses
        pass

    async def _post_tweet(
        self,
        text: str,
        reply_to: Optional[str] = None
    ) -> Any:
        """Post tweet with rate limiting and error handling"""
        try:
            # Check rate limits
            if not self._check_rate_limits():
                raise Exception("Rate limit exceeded")
            
            # Post tweet
            if reply_to:
                response = self.client.create_tweet(
                    text=text,
                    in_reply_to_tweet_id=reply_to
                )
            else:
                response = self.client.create_tweet(text=text)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to post tweet: {str(e)}")
            raise

    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits"""
        recent_tweets = sum(
            1 for tweet in self.tweet_history
            if (datetime.now() - tweet.created_at).seconds < 3600
        )
        
        return recent_tweets < self.config['twitter']['hourly_tweet_limit']