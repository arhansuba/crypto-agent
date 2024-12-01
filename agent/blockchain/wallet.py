"""
Comprehensive wallet management implementation for the AI crypto agent.
Handles wallet creation, transaction signing, key management, and state tracking.
"""

from typing import Dict, List, Optional, Union, Tuple
from eth_account import Account
from eth_account.messages import encode_defunct
from eth_keys import KeyAPI
from web3 import Web3
from web3.contract import Contract
import json
import logging
import asyncio
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

@dataclass
class WalletBalance:
    """Represents wallet balance information"""
    eth_balance: float
    tokens: Dict[str, Dict[str, Union[float, str]]]
    nfts: List[Dict[str, Union[int, str]]]
    last_updated: datetime

@dataclass
class TransactionRecord:
    """Represents a transaction record"""
    tx_hash: str
    timestamp: datetime
    network_id: str
    type: str
    status: str
    value: float
    gas_used: Optional[float]
    gas_price: Optional[float]
    error: Optional[str]

class WalletManager:
    """
    Manages cryptocurrency wallets, including creation, transaction signing,
    balance tracking, and secure key storage.
    """
    
    def __init__(self, config: Dict, web3_provider: Web3, logger: logging.Logger):
        """Initialize the wallet manager"""
        self.config = config
        self.web3 = web3_provider
        self.logger = logger
        
        self.accounts: Dict[str, Account] = {}
        self.balances: Dict[str, WalletBalance] = {}
        self.transactions: Dict[str, List[TransactionRecord]] = {}
        
        # Initialize encryption key
        self.encryption_key = self._initialize_encryption()
        
        # Load existing wallets
        self._load_wallets()

    def _initialize_encryption(self) -> bytes:
        """Initialize encryption for secure wallet storage"""
        try:
            key_path = Path(self.config['wallet']['key_path'])
            
            if key_path.exists():
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
            self.logger.error(f"Failed to initialize encryption: {str(e)}")
            raise

    def _load_wallets(self):
        """Load existing wallets from secure storage"""
        try:
            wallet_path = Path(self.config['wallet']['storage_path'])
            if not wallet_path.exists():
                return
            
            fernet = Fernet(self.encryption_key)
            
            with open(wallet_path, 'rb') as f:
                encrypted_data = f.read()
                wallet_data = json.loads(fernet.decrypt(encrypted_data))
            
            for address, data in wallet_data.items():
                private_key = data['private_key']
                self.accounts[address] = Account.from_key(private_key)
                
                if 'transactions' in data:
                    self.transactions[address] = [
                        TransactionRecord(**tx) for tx in data['transactions']
                    ]
            
            self.logger.info(f"Loaded {len(self.accounts)} wallets")
            
        except Exception as e:
            self.logger.error(f"Failed to load wallets: {str(e)}")
            raise

    def _save_wallets(self):
        """Save wallets to secure storage"""
        try:
            wallet_data = {}
            
            for address, account in self.accounts.items():
                wallet_data[address] = {
                    'private_key': account.key.hex(),
                    'transactions': [
                        tx.__dict__ for tx in self.transactions.get(address, [])
                    ]
                }
            
            fernet = Fernet(self.encryption_key)
            encrypted_data = fernet.encrypt(json.dumps(wallet_data).encode())
            
            wallet_path = Path(self.config['wallet']['storage_path'])
            with open(wallet_path, 'wb') as f:
                f.write(encrypted_data)
            
            self.logger.info("Wallets saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save wallets: {str(e)}")
            raise

    async def create_wallet(self) -> Tuple[str, str]:
        """Create a new wallet"""
        try:
            # Generate new account
            account = Account.create()
            address = account.address
            
            # Store account
            self.accounts[address] = account
            self.transactions[address] = []
            
            # Save updated wallet data
            self._save_wallets()
            
            self.logger.info(f"Created new wallet: {address}")
            return address, account.key.hex()
            
        except Exception as e:
            self.logger.error(f"Failed to create wallet: {str(e)}")
            raise

    async def import_wallet(self, private_key: str) -> str:
        """Import existing wallet using private key"""
        try:
            account = Account.from_key(private_key)
            address = account.address
            
            if address in self.accounts:
                raise ValueError("Wallet already exists")
            
            self.accounts[address] = account
            self.transactions[address] = []
            
            self._save_wallets()
            
            self.logger.info(f"Imported wallet: {address}")
            return address
            
        except Exception as e:
            self.logger.error(f"Failed to import wallet: {str(e)}")
            raise

    async def update_balance(self, address: str, network_id: str):
        """Update wallet balance information"""
        try:
            if address not in self.accounts:
                raise ValueError("Wallet not found")

            # Get ETH balance
            eth_balance = self.web3.eth.get_balance(address)
            
            # Get token balances
            tokens = {}
            for token_address in self.config['tokens']:
                token_contract = self.web3.eth.contract(
                    address=token_address,
                    abi=self.config['token_abi']
                )
                balance = token_contract.functions.balanceOf(address).call()
                symbol = token_contract.functions.symbol().call()
                decimals = token_contract.functions.decimals().call()
                
                tokens[token_address] = {
                    'symbol': symbol,
                    'balance': balance / (10 ** decimals),
                    'decimals': decimals
                }
            
            # Get NFT holdings
            nfts = []
            for nft_address in self.config['nft_contracts']:
                nft_contract = self.web3.eth.contract(
                    address=nft_address,
                    abi=self.config['nft_abi']
                )
                balance = nft_contract.functions.balanceOf(address).call()
                
                if balance > 0:
                    for i in range(balance):
                        token_id = nft_contract.functions.tokenOfOwnerByIndex(
                            address, i
                        ).call()
                        token_uri = nft_contract.functions.tokenURI(token_id).call()
                        
                        nfts.append({
                            'contract': nft_address,
                            'token_id': token_id,
                            'token_uri': token_uri
                        })
            
            self.balances[address] = WalletBalance(
                eth_balance=self.web3.from_wei(eth_balance, 'ether'),
                tokens=tokens,
                nfts=nfts,
                last_updated=datetime.now()
            )
            
            self.logger.info(f"Updated balance for wallet: {address}")
            
        except Exception as e:
            self.logger.error(f"Failed to update balance: {str(e)}")
            raise

    async def sign_transaction(
        self,
        address: str,
        transaction: Dict
    ) -> Tuple[str, bytes]:
        """Sign a transaction"""
        try:
            if address not in self.accounts:
                raise ValueError("Wallet not found")
            
            account = self.accounts[address]
            
            # Sign transaction
            signed_txn = account.sign_transaction(transaction)
            
            return signed_txn.hash.hex(), signed_txn.rawTransaction
            
        except Exception as e:
            self.logger.error(f"Failed to sign transaction: {str(e)}")
            raise

    async def sign_message(self, address: str, message: str) -> str:
        """Sign a message using the wallet"""
        try:
            if address not in self.accounts:
                raise ValueError("Wallet not found")
            
            account = self.accounts[address]
            message_hash = encode_defunct(text=message)
            signed_message = account.sign_message(message_hash)
            
            return signed_message.signature.hex()
            
        except Exception as e:
            self.logger.error(f"Failed to sign message: {str(e)}")
            raise

    async def record_transaction(
        self,
        address: str,
        tx_hash: str,
        network_id: str,
        tx_type: str,
        value: float,
        **kwargs
    ):
        """Record transaction details"""
        try:
            if address not in self.transactions:
                self.transactions[address] = []
            
            record = TransactionRecord(
                tx_hash=tx_hash,
                timestamp=datetime.now(),
                network_id=network_id,
                type=tx_type,
                status='pending',
                value=value,
                gas_used=kwargs.get('gas_used'),
                gas_price=kwargs.get('gas_price'),
                error=None
            )
            
            self.transactions[address].append(record)
            self._save_wallets()
            
        except Exception as e:
            self.logger.error(f"Failed to record transaction: {str(e)}")
            raise

    async def update_transaction_status(
        self,
        address: str,
        tx_hash: str,
        status: str,
        **kwargs
    ):
        """Update transaction status and details"""
        try:
            if address not in self.transactions:
                return
            
            for tx in self.transactions[address]:
                if tx.tx_hash == tx_hash:
                    tx.status = status
                    tx.gas_used = kwargs.get('gas_used', tx.gas_used)
                    tx.error = kwargs.get('error')
                    break
            
            self._save_wallets()
            
        except Exception as e:
            self.logger.error(f"Failed to update transaction status: {str(e)}")
            raise

    def get_transaction_history(
        self,
        address: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[TransactionRecord]:
        """Get transaction history for a wallet"""
        try:
            if address not in self.transactions:
                return []
            
            transactions = self.transactions[address]
            
            if start_time:
                transactions = [tx for tx in transactions if tx.timestamp >= start_time]
            if end_time:
                transactions = [tx for tx in transactions if tx.timestamp <= end_time]
            
            return sorted(transactions, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Failed to get transaction history: {str(e)}")
            raise

    def export_wallet(self, address: str, password: str) -> str:
        """Export wallet data in encrypted format"""
        try:
            if address not in self.accounts:
                raise ValueError("Wallet not found")
            
            account = self.accounts[address]
            
            # Create encryption key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=os.urandom(16),
                iterations=480000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            fernet = Fernet(key)
            
            # Encrypt wallet data
            wallet_data = {
                'private_key': account.key.hex(),
                'address': address
            }
            
            encrypted_data = fernet.encrypt(json.dumps(wallet_data).encode())
            return base64.b64encode(encrypted_data).decode()
            
        except Exception as e:
            self.logger.error(f"Failed to export wallet: {str(e)}")
            raise