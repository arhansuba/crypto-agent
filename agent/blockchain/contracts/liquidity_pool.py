"""
Liquidity pool implementation for automated market making.
Supports token pair creation, liquidity provision, and swapping with configurable fee structures.
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from web3 import Web3
from web3.contract import Contract
from eth_account.account import Account
import json
import logging
import asyncio
from dataclasses import dataclass

@dataclass
class PoolState:
    """Represents the current state of a liquidity pool"""
    token0_reserve: int
    token1_reserve: int
    total_supply: int
    fee_rate: int
    k_last: int
    unlocked: bool

class LiquidityPool:
    """
    Implementation of automated market maker liquidity pool.
    Supports constant product formula with configurable fees and flash swap capabilities.
    """
    
    CONTRACT_TEMPLATE = """
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.19;

    import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
    import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
    import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
    import "@openzeppelin/contracts/access/Ownable.sol";
    import "@openzeppelin/contracts/utils/math/Math.sol";

    contract LiquidityPool is ERC20, ReentrancyGuard, Ownable {
        using Math for uint256;

        IERC20 public immutable token0;
        IERC20 public immutable token1;
        
        uint256 private reserve0;
        uint256 private reserve1;
        uint256 private constant MINIMUM_LIQUIDITY = 1000;
        uint256 public fee_rate = 30; // 0.3% fee
        uint256 private unlocked = 1;
        uint256 private k_last;
        
        event Mint(address indexed sender, uint256 amount0, uint256 amount1);
        event Burn(address indexed sender, uint256 amount0, uint256 amount1);
        event Swap(
            address indexed sender,
            uint256 amount0In,
            uint256 amount1In,
            uint256 amount0Out,
            uint256 amount1Out
        );
        event Sync(uint256 reserve0, uint256 reserve1);
        
        modifier lock() {
            require(unlocked == 1, 'LOCKED');
            unlocked = 0;
            _;
            unlocked = 1;
        }
        
        constructor(
            address _token0,
            address _token1,
            string memory name,
            string memory symbol
        ) ERC20(name, symbol) {
            token0 = IERC20(_token0);
            token1 = IERC20(_token1);
        }
        
        function getReserves() public view returns (uint256, uint256) {
            return (reserve0, reserve1);
        }
        
        function mint(address to) external lock nonReentrant returns (uint256 liquidity) {
            (uint256 _reserve0, uint256 _reserve1) = getReserves();
            uint256 balance0 = token0.balanceOf(address(this));
            uint256 balance1 = token1.balanceOf(address(this));
            uint256 amount0 = balance0 - _reserve0;
            uint256 amount1 = balance1 - _reserve1;
            
            uint256 _totalSupply = totalSupply();
            if (_totalSupply == 0) {
                liquidity = Math.sqrt(amount0 * amount1) - MINIMUM_LIQUIDITY;
                _mint(address(0), MINIMUM_LIQUIDITY);
            } else {
                liquidity = Math.min(
                    (amount0 * _totalSupply) / _reserve0,
                    (amount1 * _totalSupply) / _reserve1
                );
            }
            
            require(liquidity > 0, 'INSUFFICIENT_LIQUIDITY_MINTED');
            _mint(to, liquidity);
            
            _update(balance0, balance1);
            k_last = reserve0 * reserve1;
            emit Mint(to, amount0, amount1);
        }
        
        function burn(
            address to
        ) external lock nonReentrant returns (uint256 amount0, uint256 amount1) {
            uint256 balance0 = token0.balanceOf(address(this));
            uint256 balance1 = token1.balanceOf(address(this));
            uint256 liquidity = balanceOf(address(this));
            uint256 _totalSupply = totalSupply();
            
            amount0 = (liquidity * balance0) / _totalSupply;
            amount1 = (liquidity * balance1) / _totalSupply;
            require(amount0 > 0 && amount1 > 0, 'INSUFFICIENT_LIQUIDITY_BURNED');
            
            _burn(address(this), liquidity);
            token0.transfer(to, amount0);
            token1.transfer(to, amount1);
            
            balance0 = token0.balanceOf(address(this));
            balance1 = token1.balanceOf(address(this));
            _update(balance0, balance1);
            k_last = reserve0 * reserve1;
            emit Burn(to, amount0, amount1);
        }
        
        function swap(
            uint256 amount0Out,
            uint256 amount1Out,
            address to
        ) external lock nonReentrant {
            require(amount0Out > 0 || amount1Out > 0, 'INSUFFICIENT_OUTPUT_AMOUNT');
            (uint256 _reserve0, uint256 _reserve1) = getReserves();
            require(amount0Out < _reserve0 && amount1Out < _reserve1, 'INSUFFICIENT_LIQUIDITY');
            
            if (amount0Out > 0) token0.transfer(to, amount0Out);
            if (amount1Out > 0) token1.transfer(to, amount1Out);
            
            uint256 balance0 = token0.balanceOf(address(this));
            uint256 balance1 = token1.balanceOf(address(this));
            uint256 amount0In = balance0 > _reserve0 - amount0Out ? 
                balance0 - (_reserve0 - amount0Out) : 0;
            uint256 amount1In = balance1 > _reserve1 - amount1Out ? 
                balance1 - (_reserve1 - amount1Out) : 0;
            require(amount0In > 0 || amount1In > 0, 'INSUFFICIENT_INPUT_AMOUNT');
            
            uint256 balance0Adjusted = balance0 * 1000 - amount0In * fee_rate;
            uint256 balance1Adjusted = balance1 * 1000 - amount1In * fee_rate;
            require(
                balance0Adjusted * balance1Adjusted >= _reserve0 * _reserve1 * 1000**2,
                'K'
            );
            
            _update(balance0, balance1);
            emit Swap(msg.sender, amount0In, amount1In, amount0Out, amount1Out);
        }
        
        function _update(uint256 balance0, uint256 balance1) private {
            reserve0 = balance0;
            reserve1 = balance1;
            emit Sync(reserve0, reserve1);
        }
        
        function setFeeRate(uint256 _feeRate) external onlyOwner {
            require(_feeRate <= 100, 'FEE_TOO_HIGH');  // Max 1% fee
            fee_rate = _feeRate;
        }
    }
    """

    def __init__(self, web3_provider: Web3, logger: logging.Logger):
        """Initialize the liquidity pool contract template"""
        self.web3 = web3_provider
        self.logger = logger
        self.contract_abi = None
        self.contract_bytecode = None
        self._compile_contract()

    def _compile_contract(self):
        """Compile the contract and prepare ABI and bytecode"""
        try:
            with open('build/contracts/LiquidityPool.json', 'r') as file:
                compiled_contract = json.load(file)
                self.contract_abi = compiled_contract['abi']
                self.contract_bytecode = compiled_contract['bytecode']
            
            self.logger.info("Liquidity pool contract compilation successful")
            
        except Exception as e:
            self.logger.error(f"Contract compilation failed: {str(e)}")
            raise

    async def deploy(
        self,
        token0_address: str,
        token1_address: str,
        name: str,
        symbol: str,
        deployer_account: Account,
        gas_price_gwei: Optional[int] = None
    ) -> Tuple[str, Contract]:
        """Deploy a new liquidity pool contract"""
        try:
            # Validate parameters
            if not Web3.is_address(token0_address) or not Web3.is_address(token1_address):
                raise ValueError("Invalid token addresses")
            
            if token0_address >= token1_address:
                raise ValueError("Token addresses must be sorted")

            # Prepare contract deployment
            contract = self.web3.eth.contract(
                abi=self.contract_abi,
                bytecode=self.contract_bytecode
            )

            # Prepare constructor arguments
            constructor_args = (token0_address, token1_address, name, symbol)

            # Get deployment parameters
            nonce = self.web3.eth.get_transaction_count(deployer_account.address)
            gas_price = self.web3.eth.gas_price if not gas_price_gwei else Web3.to_wei(gas_price_gwei, 'gwei')

            # Estimate gas
            gas_estimate = contract.constructor(*constructor_args).estimate_gas({
                'from': deployer_account.address
            })

            # Prepare transaction
            transaction = contract.constructor(*constructor_args).build_transaction({
                'from': deployer_account.address,
                'nonce': nonce,
                'gas': int(gas_estimate * 1.2),
                'gasPrice': gas_price
            })

            # Sign and send transaction
            signed_txn = self.web3.eth.account.sign_transaction(
                transaction,
                deployer_account.key
            )
            
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

            if receipt.status != 1:
                raise Exception("Pool deployment failed")

            # Create contract instance
            deployed_contract = self.web3.eth.contract(
                address=receipt.contractAddress,
                abi=self.contract_abi
            )

            self.logger.info(f"Liquidity pool deployed at {receipt.contractAddress}")
            return receipt.contractAddress, deployed_contract

        except Exception as e:
            self.logger.error(f"Pool deployment failed: {str(e)}")
            raise

    async def add_liquidity(
        self,
        pool_contract: Contract,
        token0_amount: int,
        token1_amount: int,
        provider_account: Account
    ) -> Tuple[str, int]:
        """Add liquidity to the pool"""
        try:
            # Approve token transfers
            token0 = self.web3.eth.contract(
                address=await pool_contract.functions.token0().call(),
                abi=self.contract_abi  # Using standard ERC20 ABI
            )
            token1 = self.web3.eth.contract(
                address=await pool_contract.functions.token1().call(),
                abi=self.contract_abi
            )

            # Approve tokens
            await self._approve_tokens(
                token0,
                token1,
                pool_contract.address,
                token0_amount,
                token1_amount,
                provider_account
            )

            # Add liquidity
            nonce = self.web3.eth.get_transaction_count(provider_account.address)
            transaction = pool_contract.functions.mint(
                provider_account.address
            ).build_transaction({
                'from': provider_account.address,
                'nonce': nonce,
                'gas': 300000,
                'gasPrice': self.web3.eth.gas_price
            })

            # Sign and send transaction
            signed_txn = self.web3.eth.account.sign_transaction(
                transaction,
                provider_account.key
            )
            
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

            if receipt.status != 1:
                raise Exception("Liquidity addition failed")

            # Get liquidity minted from event logs
            mint_event = pool_contract.events.Mint().process_receipt(receipt)[0]
            liquidity_minted = mint_event.args.liquidity

            return receipt.transactionHash.hex(), liquidity_minted

        except Exception as e:
            self.logger.error(f"Liquidity addition failed: {str(e)}")
            raise

    async def _approve_tokens(
        self,
        token0: Contract,
        token1: Contract,
        spender: str,
        amount0: int,
        amount1: int,
        owner_account: Account
    ):
        """Approve token transfers"""
        try:
            # Approve token0
            nonce = self.web3.eth.get_transaction_count(owner_account.address)
            tx0 = token0.functions.approve(spender, amount0).build_transaction({
                'from': owner_account.address,
                'nonce': nonce,
                'gas': 100000,
                'gasPrice': self.web3.eth.gas_price
            })
            signed_tx0 = self.web3.eth.account.sign_transaction(tx0, owner_account.key)
            tx_hash0 = self.web3.eth.send_raw_transaction(signed_tx0.rawTransaction)
            await self.web3.eth.wait_for_transaction_receipt(tx_hash0)

            # Approve token1
            nonce = self.web3.eth.get_transaction_count(owner_account.address)
            tx1 = token1.functions.approve(spender, amount1).build_transaction({
                'from': owner_account.address,
                'nonce': nonce,
                'gas': 100000,
                'gasPrice': self.web3.eth.gas_price
            })
            signed_tx1 = self.web3.eth.account.sign_transaction(tx1, owner_account.key)
            tx_hash1 = self.web3.eth.send_raw_transaction(signed_tx1.rawTransaction)
            await self.web3.eth.wait_for_transaction_receipt(tx_hash1)

        except Exception as e:
            self.logger.error(f"Token approval failed: {str(e)}")
            raise

    def get_pool_state(self, pool_address: str) -> PoolState:
        """Get current state of the liquidity pool"""
        try:
            contract = self.web3.eth.contract(
                address=pool_address,
                abi=self.contract_abi
            )

            reserves = contract.functions.getReserves().call()
            total_supply = contract.functions.totalSupply().call()
            fee_rate = contract.functions.fee_rate().call()
            k_last = contract.functions.k_last().call()
            unlocked = contract.functions.unlocked().call()

            return PoolState(
                token0_reserve=reserves[0],
                token1_reserve=reserves[1],
                total_supply=total_supply,
                fee_rate=fee_rate,
                k_last=k_last,
                unlocked=unlocked
            )
        except Exception as e:
            self.logger.error(f"Failed to get pool state: {str(e)}")
            raise