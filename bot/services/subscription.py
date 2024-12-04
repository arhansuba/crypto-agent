from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
from enum import Enum

class SubscriptionTier(Enum):
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    PREMIUM = "premium"

class SubscriptionService:
    def __init__(self):
        """Initialize subscription service"""
        self.logger = logging.getLogger(__name__)
        self._subscription_plans = {
            SubscriptionTier.FREE: {
                'price': 0,
                'limits': {
                    'daily_analyses': 3,
                    'active_alerts': 1,
                    'tokens_tracked': 5
                },
                'features': [
                    'Basic price checking',
                    'Simple market updates',
                    'Community access'
                ]
            },
            SubscriptionTier.BASIC: {
                'price': 29.99,
                'limits': {
                    'daily_analyses': 20,
                    'active_alerts': 10,
                    'tokens_tracked': 15
                },
                'features': [
                    'Technical analysis',
                    'Pattern detection',
                    'Basic AI insights',
                    'Email alerts'
                ]
            },
            SubscriptionTier.PRO: {
                'price': 79.99,
                'limits': {
                    'daily_analyses': 100,
                    'active_alerts': 50,
                    'tokens_tracked': 50
                },
                'features': [
                    'Advanced AI analysis',
                    'Custom strategies',
                    'Portfolio tracking',
                    'Premium signals'
                ]
            },
            SubscriptionTier.PREMIUM: {
                'price': 199.99,
                'limits': {
                    'daily_analyses': float('inf'),
                    'active_alerts': float('inf'),
                    'tokens_tracked': float('inf')
                },
                'features': [
                    'Unlimited analysis',
                    'Priority support',
                    'Custom strategies',
                    'Advanced AI features',
                    'VIP trading signals'
                ]
            }
        }
        
        # Store user subscriptions in memory (replace with database in production)
        self._user_subscriptions = {}
        self._usage_tracking = {}

    async def get_subscription_plans(self) -> Dict:
        """Get all available subscription plans"""
        return self._subscription_plans

    async def get_user_subscription(self, user_id: int) -> Dict:
        """Get user's current subscription details"""
        try:
            subscription = self._user_subscriptions.get(user_id, {
                'tier': SubscriptionTier.FREE,
                'start_date': datetime.utcnow(),
                'end_date': None,
                'auto_renew': False
            })
            
            return {
                'status': 'success',
                'subscription': subscription
            }
        except Exception as e:
            self.logger.error(f"Error getting user subscription: {e}")
            return {'status': 'error', 'message': str(e)}

    async def subscribe_user(self, user_id: int, tier: SubscriptionTier, months: int = 1) -> Dict:
        """Subscribe user to a plan"""
        try:
            current_time = datetime.utcnow()
            
            # Calculate subscription period
            start_date = current_time
            end_date = current_time + timedelta(days=30 * months)
            
            # Create subscription
            self._user_subscriptions[user_id] = {
                'tier': tier,
                'start_date': start_date,
                'end_date': end_date,
                'auto_renew': True
            }
            
            # Initialize usage tracking
            self._reset_usage_tracking(user_id)
            
            return {
                'status': 'success',
                'message': f'Successfully subscribed to {tier.value} plan',
                'subscription': self._user_subscriptions[user_id]
            }
        except Exception as e:
            self.logger.error(f"Error subscribing user: {e}")
            return {'status': 'error', 'message': str(e)}

    async def check_feature_access(self, user_id: int, feature: str) -> bool:
        """Check if user has access to a specific feature"""
        try:
            subscription = await self.get_user_subscription(user_id)
            if subscription['status'] != 'success':
                return False
                
            tier = subscription['subscription']['tier']
            return feature in self._subscription_plans[tier]['features']
        except Exception as e:
            self.logger.error(f"Error checking feature access: {e}")
            return False

    async def check_limit(self, user_id: int, limit_type: str) -> Dict:
        """Check if user has reached their usage limit"""
        try:
            subscription = await self.get_user_subscription(user_id)
            if subscription['status'] != 'success':
                return {'status': 'error', 'message': 'Subscription not found'}
                
            tier = subscription['subscription']['tier']
            current_usage = self._get_usage(user_id, limit_type)
            limit = self._subscription_plans[tier]['limits'].get(limit_type, 0)
            
            return {
                'status': 'success',
                'has_access': current_usage < limit,
                'current_usage': current_usage,
                'limit': limit
            }
        except Exception as e:
            self.logger.error(f"Error checking limit: {e}")
            return {'status': 'error', 'message': str(e)}

    async def record_usage(self, user_id: int, usage_type: str) -> Dict:
        """Record feature usage"""
        try:
            if user_id not in self._usage_tracking:
                self._reset_usage_tracking(user_id)
                
            self._usage_tracking[user_id][usage_type] += 1
            
            return {
                'status': 'success',
                'current_usage': self._usage_tracking[user_id][usage_type]
            }
        except Exception as e:
            self.logger.error(f"Error recording usage: {e}")
            return {'status': 'error', 'message': str(e)}

    async def cancel_subscription(self, user_id: int) -> Dict:
        """Cancel user's subscription"""
        try:
            if user_id not in self._user_subscriptions:
                return {'status': 'error', 'message': 'No active subscription found'}
                
            subscription = self._user_subscriptions[user_id]
            subscription['auto_renew'] = False
            
            return {
                'status': 'success',
                'message': 'Subscription will be cancelled at the end of the billing period',
                'end_date': subscription['end_date']
            }
        except Exception as e:
            self.logger.error(f"Error cancelling subscription: {e}")
            return {'status': 'error', 'message': str(e)}

    def _reset_usage_tracking(self, user_id: int) -> None:
        """Reset usage tracking for a user"""
        self._usage_tracking[user_id] = {
            'daily_analyses': 0,
            'active_alerts': 0,
            'tokens_tracked': 0
        }

    def _get_usage(self, user_id: int, usage_type: str) -> int:
        """Get current usage count for a specific feature"""
        if user_id not in self._usage_tracking:
            self._reset_usage_tracking(user_id)
        return self._usage_tracking[user_id].get(usage_type, 0)