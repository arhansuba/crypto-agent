"""
Network interaction module for blockchain communication.
Handles multiple network connections, transaction management, and network state monitoring.
"""

from typing import Dict, List, Optional, Union, Tuple
from web3 import Web3, HTTPProvider, WebsocketProvider
from web3.middleware import geth_poa_middleware, construct_sign_and_send_raw_middleware
from web3.types import TxParams, TxReceipt
from eth_account.account import Account
import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class NetworkType(Enum):
    """Supported network types"""
    MAINNET = "mainnet"
    TESTNET = "testnet"
    LOCAL = "local"

@dataclass
class NetworkConfig:
    """Network configuration parameters"""
    chain_id: int
    rpc_url: str
    explorer_url: str
    network_type: NetworkType
    is_poa: bool
    gas_limit: int
    priority_fee: int
    base_fee_multiplier: float
    confirmation_blocks: int
    retry_count: int
    timeout: int

@dataclass
class NetworkState:
    """Current network state information"""
    current_block: int
    gas_price: int
    base_fee: Optional[int]
    priority_fee: int
    network_congestion: float
    peer_count: int
    is_syncing: bool
    last_updated: datetime

class NetworkManager:
    """
    Manages blockchain network interactions and state monitoring.
    Supports multiple networks, transaction management, and network health monitoring.
    """
    
    def __init__(self, config_path: str, logger: logging.Logger):
        """Initialize network manager with configuration"""
        self.logger = logger
        self.networks: Dict[str, NetworkConfig] = self._load_network_configs(config_path)
        self.web3_connections: Dict[str, Web3] = {}
        self.network_states: Dict[str, NetworkState] = {}
        self._initialize_connections()

    def _load_network_configs(self, config_path: str) -> Dict[str, NetworkConfig]:
        """Load network configurations from file"""
        try:
            with open(config_path, 'r') as file:
                config_data = json.load(file)
            
            networks = {}
            for network_id, config in config_data['networks'].items():
                networks[network_id] = NetworkConfig(
                    chain_id=config['chain_id'],
                    rpc_url=config['rpc_url'],
                    explorer_url=config['explorer_url'],
                    network_type=NetworkType(config['type']),
                    is_poa=config.get('is_poa', False),
                    gas_limit=config.get('gas_limit', 2000000),
                    priority_fee=config.get('priority_fee', 1),
                    base_fee_multiplier=config.get('base_fee_multiplier', 1.5),
                    confirmation_blocks=config.get('confirmation_blocks', 1),
                    retry_count=config.get('retry_count', 3),
                    timeout=config.get('timeout', 30)
                )
            
            return networks
            
        except Exception as e:
            self.logger.error(f"Failed to load network configurations: {str(e)}")
            raise

    def _initialize_connections(self):
        """Initialize Web3 connections for all configured networks"""
        try:
            for network_id, config in self.networks.items():
                # Create Web3 instance
                if config.rpc_url.startswith('ws'):
                    provider = WebsocketProvider(config.rpc_url)
                else:
                    provider = HTTPProvider(config.rpc_url)
                
                web3 = Web3(provider)
                
                # Add PoA middleware if needed
                if config.is_poa:
                    web3.middleware_onion.inject(geth_poa_middleware, layer=0)
                
                self.web3_connections[network_id] = web3
                self.logger.info(f"Initialized connection to network: {network_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize network connections: {str(e)}")
            raise

    async def update_network_state(self, network_id: str):
        """Update network state information"""
        try:
            web3 = self.web3_connections[network_id]
            
            # Get network state information
            current_block = web3.eth.block_number
            gas_price = web3.eth.gas_price
            base_fee = None
            priority_fee = None
            
            try:
                # Get base fee and priority fee for EIP-1559 networks
                latest_block = web3.eth.get_block('latest')
                base_fee = latest_block.get('baseFeePerGas', None)
                priority_fee = web3.eth.max_priority_fee
            except Exception:
                pass
            
            # Calculate network congestion
            recent_blocks = [web3.eth.get_block(current_block - i) for i in range(10)]
            avg_gas_used = sum(block.gasUsed for block in recent_blocks) / len(recent_blocks)
            network_congestion = avg_gas_used / sum(block.gasLimit for block in recent_blocks)
            
            self.network_states[network_id] = NetworkState(
                current_block=current_block,
                gas_price=gas_price,
                base_fee=base_fee,
                priority_fee=priority_fee or self.networks[network_id].priority_fee,
                network_congestion=network_congestion,
                peer_count=web3.net.peer_count,
                is_syncing=web3.eth.syncing != False,
                last_updated=datetime.now()
            )
            
            self.logger.debug(f"Updated network state for {network_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update network state for {network_id}: {str(e)}")
            raise

    async def send_transaction(
        self,
        network_id: str,
        transaction: TxParams,
        account: Account
    ) -> Tuple[str, TxReceipt]:
        """Send a transaction to the network with retry logic"""
        try:
            web3 = self.web3_connections[network_id]
            config = self.networks[network_id]
            
            for attempt in range(config.retry_count):
                try:
                    # Update gas price and nonce
                    transaction['nonce'] = web3.eth.get_transaction_count(
                        account.address,
                        'pending'
                    )
                    
                    if 'gasPrice' not in transaction and 'maxFeePerGas' not in transaction:
                        transaction['gasPrice'] = await self._get_optimal_gas_price(
                            network_id
                        )
                    
                    # Sign and send transaction
                    signed_txn = web3.eth.account.sign_transaction(
                        transaction,
                        account.key
                    )
                    
                    tx_hash = web3.eth.send_raw_transaction(signed_txn.rawTransaction)
                    
                    # Wait for confirmation
                    receipt = web3.eth.wait_for_transaction_receipt(
                        tx_hash,
                        timeout=config.timeout,
                        poll_latency=2
                    )
                    
                    if receipt.status == 1:
                        self.logger.info(
                            f"Transaction confirmed on {network_id}: {tx_hash.hex()}"
                        )
                        return tx_hash.hex(), receipt
                    else:
                        raise Exception("Transaction failed")
                        
                except Exception as e:
                    if attempt == config.retry_count - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        except Exception as e:
            self.logger.error(f"Transaction failed on {network_id}: {str(e)}")
            raise

    async def _get_optimal_gas_price(self, network_id: str) -> int:
        """Calculate optimal gas price based on network conditions"""
        try:
            await self.update_network_state(network_id)
            state = self.network_states[network_id]
            config = self.networks[network_id]
            
            if state.base_fee is not None:
                # EIP-1559 transaction
                max_fee_per_gas = int(state.base_fee * config.base_fee_multiplier)
                return max_fee_per_gas + state.priority_fee
            else:
                # Legacy transaction
                return int(state.gas_price * config.base_fee_multiplier)
                
        except Exception as e:
            self.logger.error(f"Failed to calculate optimal gas price: {str(e)}")
            raise

    async def monitor_transaction(
        self,
        network_id: str,
        tx_hash: str,
        confirmation_blocks: Optional[int] = None
    ) -> TxReceipt:
        """Monitor transaction status and wait for confirmations"""
        try:
            web3 = self.web3_connections[network_id]
            confirmations = confirmation_blocks or self.networks[network_id].confirmation_blocks
            
            receipt = await asyncio.wait_for(
                self._wait_for_transaction(web3, tx_hash, confirmations),
                timeout=self.networks[network_id].timeout
            )
            
            return receipt
            
        except Exception as e:
            self.logger.error(f"Transaction monitoring failed: {str(e)}")
            raise

    async def _wait_for_transaction(
        self,
        web3: Web3,
        tx_hash: str,
        confirmations: int
    ) -> TxReceipt:
        """Wait for transaction confirmations"""
        while True:
            try:
                receipt = web3.eth.get_transaction_receipt(tx_hash)
                if receipt is not None:
                    confirmation_block = receipt.blockNumber + confirmations
                    current_block = web3.eth.block_number
                    
                    if current_block >= confirmation_block:
                        return receipt
                    
            except Exception as e:
                self.logger.warning(f"Error checking transaction: {str(e)}")
                
            await asyncio.sleep(2)

    def get_explorer_url(self, network_id: str, tx_hash: str) -> str:
        """Get block explorer URL for transaction"""
        return f"{self.networks[network_id].explorer_url}/tx/{tx_hash}"

    async def estimate_gas(
        self,
        network_id: str,
        transaction: TxParams
    ) -> Tuple[int, int]:
        """Estimate gas cost for transaction"""
        try:
            web3 = self.web3_connections[network_id]
            gas_estimate = web3.eth.estimate_gas(transaction)
            gas_price = await self._get_optimal_gas_price(network_id)
            
            return gas_estimate, gas_price
            
        except Exception as e:
            self.logger.error(f"Gas estimation failed: {str(e)}")
            raise