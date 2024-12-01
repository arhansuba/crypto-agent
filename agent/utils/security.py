"""
Security module for AI crypto agent.
Implements encryption, key management, transaction validation, and security monitoring.
"""

import os
import hashlib
import hmac
import base64
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from web3.types import TxParams
import json
import re
from dataclasses import dataclass

@dataclass
class SecurityAlert:
    """Represents a security alert"""
    timestamp: datetime
    alert_type: str
    severity: str
    description: str
    affected_component: str
    transaction_hash: Optional[str]
    metadata: Dict[str, Any]

class SecurityManager:
    """
    Manages security operations including encryption, transaction validation,
    and security monitoring for the AI crypto agent.
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """Initialize the security manager"""
        self.config = config
        self.logger = logger
        
        # Initialize encryption
        self.encryption_key = self._initialize_encryption()
        self.fernet = Fernet(self.encryption_key)
        
        # Initialize security state
        self.transaction_history = {}
        self.alert_history = []
        self.blocked_addresses = set()
        
        # Load security rules
        self.security_rules = self._load_security_rules()
        
        self.logger.info("Security manager initialized")

    def _initialize_encryption(self) -> bytes:
        """Initialize encryption key"""
        try:
            key_path = self.config['security']['key_path']
            
            if os.path.exists(key_path):
                with open(key_path, 'rb') as f:
                    return base64.urlsafe_b64decode(f.read())
            
            # Generate new key
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=480000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(os.urandom(32)))
            
            # Save key securely
            with open(key_path, 'wb') as f:
                f.write(key)
            
            return base64.urlsafe_b64decode(key)
            
        except Exception as e:
            self.logger.error(f"Encryption initialization failed: {str(e)}")
            raise

    def _load_security_rules(self) -> Dict:
        """Load security rules from configuration"""
        try:
            with open(self.config['security']['rules_path'], 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load security rules: {str(e)}")
            return {}

    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            return self.fernet.encrypt(data.encode()).decode()
        except Exception as e:
            self.logger.error(f"Encryption failed: {str(e)}")
            raise

    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt encrypted data"""
        try:
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            self.logger.error(f"Decryption failed: {str(e)}")
            raise

    def validate_transaction(
        self,
        transaction: TxParams,
        wallet_balance: float
    ) -> Tuple[bool, Optional[str]]:
        """Validate transaction against security rules"""
        try:
            # Check transaction value limits
            if float(transaction.get('value', 0)) > self.security_rules['max_transaction_value']:
                return False, "Transaction value exceeds maximum allowed"
            
            # Check gas price limits
            if float(transaction.get('gasPrice', 0)) > self.security_rules['max_gas_price']:
                return False, "Gas price exceeds maximum allowed"
            
            # Check sufficient balance
            total_cost = float(transaction.get('value', 0)) + \
                        float(transaction.get('gasPrice', 0)) * \
                        float(transaction.get('gas', 21000))
            
            if total_cost > wallet_balance:
                return False, "Insufficient balance for transaction"
            
            # Check recipient address
            to_address = transaction.get('to')
            if to_address in self.blocked_addresses:
                return False, "Recipient address is blocked"
            
            if not self._validate_address_pattern(to_address):
                return False, "Invalid recipient address format"
            
            # Check transaction frequency
            if not self._check_transaction_frequency(to_address):
                return False, "Transaction frequency limit exceeded"
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"Transaction validation failed: {str(e)}")
            return False, f"Validation error: {str(e)}"

    def _validate_address_pattern(self, address: str) -> bool:
        """Validate Ethereum address format"""
        if not address:
            return False
        
        # Check basic format
        if not re.match(r'^0x[a-fA-F0-9]{40}$', address):
            return False
        
        # Additional checks could be added here
        return True

    def _check_transaction_frequency(self, address: str) -> bool:
        """Check if transaction frequency is within limits"""
        recent_time = datetime.now() - timedelta(
            hours=self.security_rules['frequency_check_hours']
        )
        
        recent_transactions = sum(
            1 for tx in self.transaction_history.values()
            if tx['timestamp'] > recent_time and tx['to'] == address
        )
        
        return recent_transactions < self.security_rules['max_transactions_per_period']

    def record_transaction(self, transaction_data: Dict):
        """Record transaction for monitoring"""
        try:
            tx_hash = transaction_data['hash']
            self.transaction_history[tx_hash] = {
                'timestamp': datetime.now(),
                'to': transaction_data['to'],
                'value': float(transaction_data['value']),
                'status': transaction_data['status']
            }
            
            # Clean old transaction history
            self._cleanup_transaction_history()
            
        except Exception as e:
            self.logger.error(f"Transaction recording failed: {str(e)}")

    def _cleanup_transaction_history(self):
        """Clean up old transaction records"""
        cutoff_time = datetime.now() - timedelta(
            days=self.security_rules['transaction_history_days']
        )
        
        self.transaction_history = {
            tx_hash: data
            for tx_hash, data in self.transaction_history.items()
            if data['timestamp'] > cutoff_time
        }

    def monitor_security_events(self, event_data: Dict) -> Optional[SecurityAlert]:
        """Monitor and analyze security events"""
        try:
            alert = None
            
            # Check security rules violations
            if event_data['type'] == 'transaction':
                alert = self._check_transaction_security(event_data)
            elif event_data['type'] == 'api_access':
                alert = self._check_api_security(event_data)
            elif event_data['type'] == 'system':
                alert = self._check_system_security(event_data)
            
            if alert:
                self.alert_history.append(alert)
                self._handle_security_alert(alert)
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Security monitoring failed: {str(e)}")
            return None

    def _check_transaction_security(self, event_data: Dict) -> Optional[SecurityAlert]:
        """Check transaction-related security"""
        if event_data['value'] > self.security_rules['alert_threshold_value']:
            return SecurityAlert(
                timestamp=datetime.now(),
                alert_type='high_value_transaction',
                severity='warning',
                description=f"High value transaction detected: {event_data['value']}",
                affected_component='transaction',
                transaction_hash=event_data.get('hash'),
                metadata=event_data
            )
        
        return None

    def _check_api_security(self, event_data: Dict) -> Optional[SecurityAlert]:
        """Check API-related security"""
        if event_data.get('failed_attempts', 0) > self.security_rules['max_api_failures']:
            return SecurityAlert(
                timestamp=datetime.now(),
                alert_type='api_security',
                severity='critical',
                description="Excessive API failures detected",
                affected_component='api',
                transaction_hash=None,
                metadata=event_data
            )
        
        return None

    def _check_system_security(self, event_data: Dict) -> Optional[SecurityAlert]:
        """Check system-related security"""
        if event_data.get('cpu_usage', 0) > self.security_rules['max_cpu_usage']:
            return SecurityAlert(
                timestamp=datetime.now(),
                alert_type='system_resource',
                severity='warning',
                description="High CPU usage detected",
                affected_component='system',
                transaction_hash=None,
                metadata=event_data
            )
        
        return None

    def _handle_security_alert(self, alert: SecurityAlert):
        """Handle security alerts"""
        try:
            # Log alert
            self.logger.warning(
                f"Security Alert: {alert.alert_type} - {alert.description}"
            )
            
            # Take action based on severity
            if alert.severity == 'critical':
                self._handle_critical_alert(alert)
            elif alert.severity == 'warning':
                self._handle_warning_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Alert handling failed: {str(e)}")

    def _handle_critical_alert(self, alert: SecurityAlert):
        """Handle critical security alerts"""
        try:
            # Send emergency notification
            self._send_emergency_notification(alert)
            
            # Block affected addresses if applicable
            if alert.transaction_hash and alert.metadata.get('to'):
                self.blocked_addresses.add(alert.metadata['to'])
            
            # Additional emergency actions could be added here
            
        except Exception as e:
            self.logger.error(f"Critical alert handling failed: {str(e)}")

    def _handle_warning_alert(self, alert: SecurityAlert):
        """Handle warning-level security alerts"""
        try:
            # Update monitoring thresholds
            self._update_monitoring_thresholds(alert)
            
            # Log extended alert information
            self._log_extended_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Warning alert handling failed: {str(e)}")

    def _send_emergency_notification(self, alert: SecurityAlert):
        """Send emergency notification for critical alerts"""
        # Implementation would depend on notification system
        pass

    def _update_monitoring_thresholds(self, alert: SecurityAlert):
        """Update security monitoring thresholds based on alerts"""
        # Implementation would depend on specific monitoring needs
        pass

    def _log_extended_alert(self, alert: SecurityAlert):
        """Log extended alert information for analysis"""
        try:
            log_data = {
                'timestamp': alert.timestamp.isoformat(),
                'type': alert.alert_type,
                'severity': alert.severity,
                'description': alert.description,
                'component': alert.affected_component,
                'metadata': alert.metadata
            }
            
            self.logger.warning(f"Extended Alert Log: {json.dumps(log_data)}")
            
        except Exception as e:
            self.logger.error(f"Extended alert logging failed: {str(e)}")