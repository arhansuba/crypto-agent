"""
Deployment script for AI crypto agent.
Handles deployment configuration, smart contract deployment, and system initialization.
"""

import asyncio
import argparse
from datetime import datetime
import yaml
from pathlib import Path
from typing import Dict, Optional
import logging
from web3 import Web3
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from agent.core.core import AgentCore
from agent.blockchain.contracts.erc20 import ERC20TokenContract
from agent.blockchain.contracts.nft import NFTContract
from agent.blockchain.contracts.liquidity_pool import LiquidityPool
from agent.utils.logger import CryptoAgentLogger
from agent.utils.config import ConfigManager

async def deploy_contracts(
    config: Dict,
    web3: Web3,
    logger: logging.Logger
) -> Dict[str, str]:
    """Deploy required smart contracts"""
    try:
        contracts = {}
        
        # Deploy ERC20 token contract
        token_contract = ERC20TokenContract(web3, logger)
        token_address, _ = await token_contract.deploy(
            name=config['contracts']['token']['name'],
            symbol=config['contracts']['token']['symbol'],
            initial_supply=config['contracts']['token']['initial_supply'],
            deployer_account=config['wallet']['deployer']
        )
        contracts['token'] = token_address
        
        # Deploy NFT contract
        nft_contract = NFTContract(web3, logger)
        nft_address, _ = await nft_contract.deploy(
            name=config['contracts']['nft']['name'],
            symbol=config['contracts']['nft']['symbol'],
            base_uri=config['contracts']['nft']['base_uri'],
            deployer_account=config['wallet']['deployer']
        )
        contracts['nft'] = nft_address
        
        # Deploy Liquidity Pool
        pool_contract = LiquidityPool(web3, logger)
        pool_address, _ = await pool_contract.deploy(
            token0_address=token_address,
            token1_address=config['contracts']['pool']['token1'],
            deployer_account=config['wallet']['deployer']
        )
        contracts['pool'] = pool_address
        
        return contracts
        
    except Exception as e:
        logger.error(f"Contract deployment failed: {str(e)}")
        raise

async def initialize_agent(
    config: Dict,
    contracts: Dict[str, str],
    logger: logging.Logger
) -> AgentCore:
    """Initialize the AI crypto agent"""
    try:
        # Update config with deployed contract addresses
        config['contracts']['addresses'] = contracts
        
        # Initialize agent
        agent = AgentCore(config, logger)
        await agent.initialize()
        
        return agent
        
    except Exception as e:
        logger.error(f"Agent initialization failed: {str(e)}")
        raise

async def main(config_path: str, environment: str):
    """Main deployment function"""
    # Load configuration
    config_manager = ConfigManager(config_path, environment)
    config = config_manager.get_all()
    
    # Setup logging
    logger = CryptoAgentLogger(config)
    
    try:
        # Initialize Web3
        web3 = Web3(Web3.HTTPProvider(config['network']['rpc_url']))
        
        # Deploy contracts
        logger.info("Deploying smart contracts...")
        contracts = await deploy_contracts(config, web3, logger)
        
        # Initialize agent
        logger.info("Initializing agent...")
        agent = await initialize_agent(config, contracts, logger)
        
        # Save deployment info
        deployment_info = {
            'environment': environment,
            'contracts': contracts,
            'timestamp': datetime.now().isoformat(),
            'version': config['version']
        }
        
        with open('deployment_info.yaml', 'w') as f:
            yaml.dump(deployment_info, f)
        
        logger.info("Deployment completed successfully")
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--env', default='development')
    args = parser.parse_args()
    
    asyncio.run(main(args.config, args.env))
