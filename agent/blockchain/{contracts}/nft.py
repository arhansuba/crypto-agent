"""
NFT (ERC721) contract implementation with advanced features for AI agent integration.
Includes metadata management, marketplace compatibility, and dynamic minting capabilities.
"""

from typing import Dict, List, Optional, Tuple, Union
from web3 import Web3
from web3.contract import Contract
from eth_account.account import Account
import json
import logging
from datetime import datetime
import aiohttp
import base64

class NFTContract:
    """
    Implementation of NFT (ERC721) contract deployment and management.
    Supports dynamic metadata, multiple minting strategies, and marketplace integration.
    """
    
    CONTRACT_TEMPLATE = """
    // SPDX-License-Identifier: MIT
    pragma solidity ^0.8.19;

    import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
    import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
    import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
    import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Burnable.sol";
    import "@openzeppelin/contracts/access/AccessControl.sol";
    import "@openzeppelin/contracts/security/Pausable.sol";
    import "@openzeppelin/contracts/utils/Counters.sol";
    import "@openzeppelin/contracts/utils/Strings.sol";

    contract CustomNFT is 
        ERC721, 
        ERC721Enumerable, 
        ERC721URIStorage, 
        ERC721Burnable,
        AccessControl,
        Pausable 
    {
        using Counters for Counters.Counter;
        using Strings for uint256;

        bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
        bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");
        
        Counters.Counter private _tokenIdCounter;
        
        string private _baseTokenURI;
        uint256 public maxSupply;
        uint256 public mintPrice;
        bool public publicMintEnabled;
        
        mapping(uint256 => string) private _tokenAttributes;
        mapping(address => uint256) private _mintedCount;
        uint256 public maxMintsPerAddress;
        
        event TokenMinted(address indexed to, uint256 tokenId, string uri);
        event BaseURIUpdated(string newBaseURI);
        event MintPriceUpdated(uint256 newPrice);
        event PublicMintToggled(bool enabled);
        
        constructor(
            string memory name,
            string memory symbol,
            string memory baseURI,
            uint256 _maxSupply,
            uint256 _mintPrice,
            uint256 _maxMintsPerAddress
        ) ERC721(name, symbol) {
            _baseTokenURI = baseURI;
            maxSupply = _maxSupply;
            mintPrice = _mintPrice;
            maxMintsPerAddress = _maxMintsPerAddress;
            publicMintEnabled = false;
            
            _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
            _grantRole(MINTER_ROLE, msg.sender);
            _grantRole(PAUSER_ROLE, msg.sender);
        }
        
        function _baseURI() internal view virtual override returns (string memory) {
            return _baseTokenURI;
        }
        
        function pause() public onlyRole(PAUSER_ROLE) {
            _pause();
        }
        
        function unpause() public onlyRole(PAUSER_ROLE) {
            _unpause();
        }
        
        function setBaseURI(string memory newBaseURI) 
            public 
            onlyRole(DEFAULT_ADMIN_ROLE) 
        {
            _baseTokenURI = newBaseURI;
            emit BaseURIUpdated(newBaseURI);
        }
        
        function setMintPrice(uint256 newPrice) 
            public 
            onlyRole(DEFAULT_ADMIN_ROLE) 
        {
            mintPrice = newPrice;
            emit MintPriceUpdated(newPrice);
        }
        
        function togglePublicMint() 
            public 
            onlyRole(DEFAULT_ADMIN_ROLE) 
        {
            publicMintEnabled = !publicMintEnabled;
            emit PublicMintToggled(publicMintEnabled);
        }
        
        function mint(address to) 
            public 
            payable 
            whenNotPaused 
            returns (uint256) 
        {
            require(
                hasRole(MINTER_ROLE, msg.sender) || publicMintEnabled,
                "Minting not allowed"
            );
            require(
                _tokenIdCounter.current() < maxSupply,
                "Max supply reached"
            );
            require(
                _mintedCount[to] < maxMintsPerAddress,
                "Max mints per address reached"
            );
            
            if (!hasRole(MINTER_ROLE, msg.sender)) {
                require(msg.value >= mintPrice, "Insufficient payment");
            }
            
            _tokenIdCounter.increment();
            uint256 tokenId = _tokenIdCounter.current();
            _safeMint(to, tokenId);
            _mintedCount[to]++;
            
            emit TokenMinted(to, tokenId, tokenURI(tokenId));
            return tokenId;
        }
        
        function mintWithURI(
            address to, 
            string memory uri
        ) public onlyRole(MINTER_ROLE) returns (uint256) {
            uint256 tokenId = mint(to);
            _setTokenURI(tokenId, uri);
            return tokenId;
        }
        
        function mintBatch(
            address to,
            uint256 amount
        ) public onlyRole(MINTER_ROLE) returns (uint256[] memory) {
            uint256[] memory tokenIds = new uint256[](amount);
            for (uint256 i = 0; i < amount; i++) {
                tokenIds[i] = mint(to);
            }
            return tokenIds;
        }
        
        function setTokenAttributes(
            uint256 tokenId,
            string memory attributes
        ) public onlyRole(MINTER_ROLE) {
            require(_exists(tokenId), "Token does not exist");
            _tokenAttributes[tokenId] = attributes;
        }
        
        function getTokenAttributes(
            uint256 tokenId
        ) public view returns (string memory) {
            require(_exists(tokenId), "Token does not exist");
            return _tokenAttributes[tokenId];
        }
        
        function withdraw() public onlyRole(DEFAULT_ADMIN_ROLE) {
            uint256 balance = address(this).balance;
            payable(msg.sender).transfer(balance);
        }
        
        // Required overrides
        function _beforeTokenTransfer(
            address from,
            address to,
            uint256 tokenId,
            uint256 batchSize
        ) internal override(ERC721, ERC721Enumerable) whenNotPaused {
            super._beforeTokenTransfer(from, to, tokenId, batchSize);
        }
        
        function _burn(uint256 tokenId) 
            internal 
            override(ERC721, ERC721URIStorage) 
        {
            super._burn(tokenId);
        }
        
        function tokenURI(uint256 tokenId)
            public
            view
            override(ERC721, ERC721URIStorage)
            returns (string memory)
        {
            return super.tokenURI(tokenId);
        }
        
        function supportsInterface(bytes4 interfaceId)
            public
            view
            override(ERC721, ERC721Enumerable, AccessControl)
            returns (bool)
        {
            return super.supportsInterface(interfaceId);
        }
    }
    """

    def __init__(self, web3_provider: Web3, logger: logging.Logger):
        """Initialize the NFT contract template"""
        self.web3 = web3_provider
        self.logger = logger
        self.contract_abi = None
        self.contract_bytecode = None
        self._compile_contract()
        
    def _compile_contract(self):
        """Compile the contract and prepare ABI and bytecode"""
        try:
            # Assuming pre-compiled contract data
            with open('build/contracts/CustomNFT.json', 'r') as file:
                compiled_contract = json.load(file)
                self.contract_abi = compiled_contract['abi']
                self.contract_bytecode = compiled_contract['bytecode']
            
            self.logger.info("NFT contract compilation successful")
            
        except Exception as e:
            self.logger.error(f"NFT contract compilation failed: {str(e)}")
            raise

    async def deploy(
        self,
        name: str,
        symbol: str,
        base_uri: str,
        max_supply: int,
        mint_price: float,
        max_mints_per_address: int,
        deployer_account: Account,
        gas_price_gwei: Optional[int] = None
    ) -> Tuple[str, Contract]:
        """Deploy a new NFT contract"""
        try:
            # Validate parameters
            if not name or not symbol:
                raise ValueError("Name and symbol cannot be empty")
            
            if max_supply <= 0:
                raise ValueError("Max supply must be greater than 0")
            
            if max_mints_per_address <= 0:
                raise ValueError("Max mints per address must be greater than 0")

            # Prepare contract deployment
            contract = self.web3.eth.contract(
                abi=self.contract_abi,
                bytecode=self.contract_bytecode
            )

            # Convert mint price to wei
            mint_price_wei = self.web3.to_wei(mint_price, 'ether')

            # Prepare constructor arguments
            constructor_args = (
                name,
                symbol,
                base_uri,
                max_supply,
                mint_price_wei,
                max_mints_per_address
            )

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
                raise Exception("NFT contract deployment failed")

            # Create contract instance
            deployed_contract = self.web3.eth.contract(
                address=receipt.contractAddress,
                abi=self.contract_abi
            )

            self.logger.info(f"NFT contract deployed at {receipt.contractAddress}")
            return receipt.contractAddress, deployed_contract

        except Exception as e:
            self.logger.error(f"NFT contract deployment failed: {str(e)}")
            raise

    async def prepare_metadata(
        self,
        name: str,
        description: str,
        attributes: List[Dict],
        image_url: Optional[str] = None,
        external_url: Optional[str] = None
    ) -> Dict:
        """Prepare NFT metadata following OpenSea standards"""
        try:
            metadata = {
                "name": name,
                "description": description,
                "attributes": attributes,
                "image": image_url,
                "external_url": external_url,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Remove None values
            return {k: v for k, v in metadata.items() if v is not None}
            
        except Exception as e:
            self.logger.error(f"Metadata preparation failed: {str(e)}")
            raise

    async def upload_metadata(
        self,
        metadata: Dict,
        ipfs_gateway: str
    ) -> str:
        """Upload metadata to IPFS"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{ipfs_gateway}/api/v0/add",
                    data=json.dumps(metadata)
                ) as response:
                    if response.status != 200:
                        raise Exception("IPFS upload failed")
                    
                    result = await response.json()
                    return f"ipfs://{result['Hash']}"
                    
        except Exception as e:
            self.logger.error(f"Metadata upload failed: {str(e)}")
            raise

    async def mint_nft(
        self,
        contract: Contract,
        to_address: str,
        metadata_uri: str,
        minter_account: Account,
        mint_price: Optional[float] = None
    ) -> Tuple[int, str]:
        """Mint a new NFT with the specified metadata"""
        try:
            # Prepare transaction data
            nonce = self.web3.eth.get_transaction_count(minter_account.address)
            
            # Get mint price if not provided
            if mint_price is None:
                mint_price = contract.functions.mintPrice().call()
            
            # Prepare transaction
            transaction = contract.functions.mintWithURI(
                to_address,
                metadata_uri
            ).build_transaction({
                'from': minter_account.address,
                'nonce': nonce,
                'gas': 500000,  # Estimated gas limit
                'gasPrice': self.web3.eth.gas_price,
                'value': mint_price
            })

            # Sign and send transaction
            signed_txn = self.web3.eth.account.sign_transaction(
                transaction,
                minter_account.key
            )
            
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)

            if receipt.status != 1:
                raise Exception("NFT minting failed")

            # Get token ID from event logs
            mint_event = contract.events.TokenMinted().process_receipt(receipt)[0]
            token_id = mint_event.args.tokenId

            self.logger.info(f"NFT minted with token ID: {token_id}")
            return token_id, receipt.transactionHash.hex()

        except Exception as e:
            self.logger.error(f"NFT minting failed: {str(e)}")
            raise

    def get_collection_info(self, contract_address: str) -> Dict:
        """Retrieve collection information from a deployed contract"""
        try:
            contract = self.web3.eth.contract(
                address=contract_address,
                abi=self.contract_abi
            )

            info = {
                'name': contract.functions.name().call(),
                'symbol': contract.functions.symbol().call(),
                'totalSupply': contract.functions.totalSupply().call(),
                'maxSupply': contract.functions.maxSupply().call(),
                'mintPrice': self.web3.from_wei(
            contract.functions.maxSupply().call()
            )
            }
            return info

        except Exception as e:
            self.logger.error(f"Failed to retrieve collection info: {str(e)}")
            raise e