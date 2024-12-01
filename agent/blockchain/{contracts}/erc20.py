"""
ERC20 token contract template implementation.
Provides functionalities for creating and deploying standard ERC20 tokens with additional features.
"""

from typing import Dict, Optional, Tuple
from eth_typing import Address
from web3 import Web3
from web3.contract import Contract
from eth_account.account import Account
import json
import logging

class ERC20TokenContract:
    """
    Implementation of ERC20 token contract deployment and management.
    Includes standard ERC20 functionalities with additional security features.
    """
    
    # Standard ERC20 contract template with additional security features
    CONTRACT_TEMPLATE = """
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.19;

    import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
    import "@openzeppelin/contracts/security/Pausable.sol";
    import "@openzeppelin/contracts/access/AccessControl.sol";
    import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

    contract CustomToken is ERC20, Pausable, AccessControl, ReentrancyGuard {
        bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
        bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
        
        uint256 public constant MAX_SUPPLY = 1000000000 * 10**18;  // 1 billion tokens
        uint256 public immutable deploymentTime;
        
        mapping(address => uint256) private _lastTransferTime;
        uint256 public constant TRANSFER_COOLDOWN = 1 minutes;
        
        event TokensMinted(address indexed to, uint256 amount);
        event TokensBurned(address indexed from, uint256 amount);
        
        constructor(
            string memory name,
            string memory symbol,
            uint256 initialSupply,
            address owner
        ) ERC20(name, symbol) {
            require(initialSupply <= MAX_SUPPLY, "Initial supply exceeds maximum");
            
            _grantRole(DEFAULT_ADMIN_ROLE, owner);
            _grantRole(MINTER_ROLE, owner);
            _grantRole(PAUSER_ROLE, owner);
            
            deploymentTime = block.timestamp;
            _mint(owner, initialSupply);
        }
        
        function mint(address to, uint256 amount) 
            public 
            onlyRole(MINTER_ROLE) 
            nonReentrant 
        {
            require(totalSupply() + amount <= MAX_SUPPLY, "Would exceed max supply");
            _mint(to, amount);
            emit TokensMinted(to, amount);
        }
        
        function burn(uint256 amount) 
            public 
            nonReentrant 
        {
            _burn(_msgSender(), amount);
            emit TokensBurned(_msgSender(), amount);
        }
        
        function pause() public onlyRole(PAUSER_ROLE) {
            _pause();
        }
        
        function unpause() public onlyRole(PAUSER_ROLE) {
            _unpause();
        }
        
        function _beforeTokenTransfer(
            address from,
            address to,
            uint256 amount
        ) internal virtual override whenNotPaused {
            require(
                block.timestamp >= _lastTransferTime[from] + TRANSFER_COOLDOWN,
                "Transfer cooldown active"
            );
            _lastTransferTime[from] = block.timestamp;
            
            super._beforeTokenTransfer(from, to, amount);
        }
        
        function getTransferCooldown(address account) 
            public 
            view 
            returns (uint256) 
        {
            uint256 timeSinceLastTransfer = block.timestamp - _lastTransferTime[account];
            if (timeSinceLastTransfer >= TRANSFER_COOLDOWN) {
                return 0;
            }
            return TRANSFER_COOLDOWN - timeSinceLastTransfer;
        }
        
        function recoverERC20(
            address tokenAddress,
            address recipient,
            uint256 amount
        ) external onlyRole(DEFAULT_ADMIN_ROLE) {
            require(tokenAddress != address(this), "Cannot recover native token");
            IERC20(tokenAddress).transfer(recipient, amount);
        }
    }
    """

    def __init__(self, web3_provider: Web3, logger: logging.Logger):
        """Initialize the ERC20 contract template"""
        self.web3 = web3_provider
        self.logger = logger
        self.contract_abi = None
        self.contract_bytecode = None
        self._compile_contract()

    def _compile_contract(self):
        """Compile the contract and prepare ABI and bytecode"""
        try:
            # This would typically use solc to compile the contract
            # For this implementation, we assume the contract is pre-compiled
            # and ABI/bytecode are loaded from build files
            with open('build/contracts/CustomToken.json', 'r') as file:
                compiled_contract = json.load(file)
                self.contract_abi = compiled_contract['abi']
                self.contract_bytecode = compiled_contract['bytecode']
                
            self.logger.info("Contract compilation successful")
            
        except Exception as e:
            self.logger.error(f"Contract compilation failed: {str(e)}")
            raise

    async def deploy(
        self,
        name: str,
        symbol: str,
        initial_supply: int,
        deployer_account: Account,
        gas_price_gwei: Optional[int] = None
    ) -> Tuple[str, Contract]:
        """Deploy a new ERC20 token contract"""
        try:
            # Validate parameters
            if not name or not symbol:
                raise ValueError("Name and symbol cannot be empty")
                
            if initial_supply <= 0:
                raise ValueError("Initial supply must be greater than 0")
                
            if initial_supply > 1000000000:
                raise ValueError("Initial supply exceeds maximum allowed")

            # Prepare contract deployment
            contract = self.web3.eth.contract(
                abi=self.contract_abi,
                bytecode=self.contract_bytecode
            )

            # Calculate deployment parameters
            nonce = self.web3.eth.get_transaction_count(deployer_account.address)
            gas_price = self.web3.eth.gas_price if not gas_price_gwei else Web3.to_wei(gas_price_gwei, 'gwei')

            # Prepare constructor arguments
            constructor_args = (
                name,
                symbol,
                initial_supply * (10 ** 18),  # Convert to token units
                deployer_account.address
            )

            # Estimate gas for deployment
            gas_estimate = contract.constructor(*constructor_args).estimate_gas({
                'from': deployer_account.address
            })

            # Prepare transaction
            transaction = contract.constructor(*constructor_args).build_transaction({
                'from': deployer_account.address,
                'nonce': nonce,
                'gas': int(gas_estimate * 1.2),  # Add 20% buffer
                'gasPrice': gas_price
            })

            # Sign and send transaction
            signed_txn = self.web3.eth.account.sign_transaction(
                transaction,
                deployer_account.key
            )
            
            # Send transaction and wait for receipt
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

            if receipt.status != 1:
                raise Exception("Contract deployment failed")

            # Create contract instance
            deployed_contract = self.web3.eth.contract(
                address=receipt.contractAddress,
                abi=self.contract_abi
            )

            self.logger.info(
                f"Token contract deployed at {receipt.contractAddress}"
            )
            
            return receipt.contractAddress, deployed_contract

        except Exception as e:
            self.logger.error(f"Contract deployment failed: {str(e)}")
            raise

    def get_token_info(self, contract_address: str) -> Dict:
        """Retrieve token information from a deployed contract"""
        try:
            contract = self.web3.eth.contract(
                address=contract_address,
                abi=self.contract_abi
            )

            info = {
                'name': contract.functions.name().call(),
                'symbol': contract.functions.symbol().call(),
                'totalSupply': contract.functions.totalSupply().call(),
                'decimals': contract.functions.decimals().call(),
                'maxSupply': contract.functions.MAX_SUPPLY().call(),
                'deploymentTime': contract.functions.deploymentTime().call(),
                'paused': contract.functions.paused().call()
            }

            return info

        except Exception as e:
            self.logger.error(f"Failed to retrieve token info: {str(e)}")
            raise

    async def estimate_deployment_cost(
        self,
        name: str,
        symbol: str,
        initial_supply: int,
        deployer_address: str
    ) -> Dict:
        """Estimate the cost of deploying a new token contract"""
        try:
            contract = self.web3.eth.contract(
                abi=self.contract_abi,
                bytecode=self.contract_bytecode
            )

            gas_estimate = contract.constructor(
                name,
                symbol,
                initial_supply * (10 ** 18),
                deployer_address
            ).estimate_gas({
                'from': deployer_address
            })

            gas_price = self.web3.eth.gas_price
            total_cost_wei = gas_estimate * gas_price
            total_cost_eth = self.web3.from_wei(total_cost_wei, 'ether')

            return {
                'gas_estimate': gas_estimate,
                'gas_price_gwei': self.web3.from_wei(gas_price, 'gwei'),
                'total_cost_wei': total_cost_wei,
                'total_cost_eth': total_cost_eth
            }

        except Exception as e:
            self.logger.error(f"Failed to estimate deployment cost: {str(e)}")
            raise