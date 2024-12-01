"""
Monitoring script for AI crypto agent.
Handles performance monitoring, alerts, and system health checks.
"""

import asyncio
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List
import aiohttp

from src.agent.core import CryptoAgent
from src.utils.logger import CryptoAgentLogger
from src.utils.config import ConfigManager

class AgentMonitor:
    """Monitors agent performance and system health"""
    
    def __init__(
        self,
        config: Dict,
        logger: logging.Logger
    ):
        self.config = config
        self.logger = logger
        self.metrics_history = []
        self.alert_thresholds = config['monitoring']['thresholds']
    
    async def collect_metrics(self, agent: CryptoAgent) -> Dict:
        """Collect current agent metrics"""
        try:
            metrics = {
                'timestamp': datetime.now(),
                'state': await agent.get_state(),
                'performance': await agent.get_performance_metrics(),
                'system': await self._get_system_metrics(),
                'market': await self._get_market_metrics()
            }
            
            self.metrics_history.append(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics collection failed: {str(e)}")
            raise

    async def check_alerts(self, metrics: Dict) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        
        try:
            # Check performance alerts
            if metrics['performance']['loss_rate'] > self.alert_thresholds['max_loss_rate']:
                alerts.append({
                    'type': 'performance',
                    'severity': 'high',
                    'message': f"Loss rate exceeded threshold: {metrics['performance']['loss_rate']:.2f}%"
                })
            
            # Check system alerts
            if metrics['system']['memory_usage'] > self.alert_thresholds['max_memory_usage']:
                alerts.append({
                    'type': 'system',
                    'severity': 'medium',
                    'message': f"High memory usage: {metrics['system']['memory_usage']:.2f}%"
                })
            
            # Send alerts
            if alerts:
                await self._send_alerts(alerts)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Alert check failed: {str(e)}")
            raise

    async def _get_system_metrics(self) -> Dict:
        """Collect system metrics"""
        import psutil
        
        return {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }

    async def _get_market_metrics(self) -> Dict:
        """Collect market metrics"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config['monitoring']['market_api']) as response:
                    data = await response.json()
                    return {
                        'price': data['price'],
                        'volume': data['volume'],
                        'volatility': data['volatility']
                    }
        except Exception:
            return {}

    async def _send_alerts(self, alerts: List[Dict]):
        """Send alerts through configured channels"""
        for alert in alerts:
            # Log alert
            self.logger.warning(f"Alert: {alert['message']}")
            
            # Send notifications if configured
            if self.config['monitoring'].get('notifications'):
                await self._send_notifications(alert)

    async def _send_notifications(self, alert: Dict):
        """Send notifications through configured channels"""
        if 'slack' in self.config['monitoring']['notifications']:
            await self._send_slack_notification(alert)
        
        if 'email' in self.config['monitoring']['notifications']:
            await self._send_email_notification(alert)

    async def generate_report(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> Dict:
        """Generate monitoring report"""
        try:
            # Filter metrics for time period
            period_metrics = [
                m for m in self.metrics_history
                if start_time <= m['timestamp'] <= end_time
            ]
            
            # Calculate statistics
            df = pd.DataFrame(period_metrics)
            
            report = {
                'period': {
                    'start': start_time,
                    'end': end_time
                },
                'performance': {
                    'avg_return': df['performance'].apply(
                        lambda x: x['return']
                    ).mean(),
                    'success_rate': df['performance'].apply(
                        lambda x: x['success_rate']
                    ).mean(),
                    'trade_count': df['performance'].apply(
                        lambda x: x['trade_count']
                    ).sum()
                },
                'system': {
                    'avg_cpu': df['system'].apply(
                        lambda x: x['cpu_usage']
                    ).mean(),
                    'avg_memory': df['system'].apply(
                        lambda x: x['memory_usage']
                    ).mean()
                },
                'alerts': df['alerts'].explode().value_counts().to_dict()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            raise

async def main(config_path: str):
    """Main monitoring function"""
    # Load configuration
    config_manager = ConfigManager(config_path)
    config = config_manager.get_all()
    
    # Setup logging
    logger = CryptoAgentLogger(config)
    
    try:
        # Initialize monitor
        monitor = AgentMonitor(config, logger)
        
        # Initialize agent connection
        agent = CryptoAgent(config, logger)
        
        while True:
            # Collect metrics
            metrics = await monitor.collect_metrics(agent)
            
            # Check alerts
            await monitor.check_alerts(metrics)
            
            # Generate periodic report
            if datetime.now().minute == 0:  # Every hour
                report = await monitor.generate_report(
                    datetime.now() - timedelta(hours=1),
                    datetime.now()
                )
                logger.info(f"Hourly Report: {report}")
            
            # Wait for next iteration
            await asyncio.sleep(config['monitoring']['interval'])
            
    except Exception as e:
        logger.error(f"Monitoring script failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    args = parser.parse_args()
    
    asyncio.run(main(args.config))