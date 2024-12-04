from typing import Dict, Optional, List, Tuple
from langchain_openai import ChatOpenAI
import re
import logging
from datetime import datetime

class IntentParser:
    """Parses and understands user messages to determine their intent"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.logger = logging.getLogger(__name__)
        
        # Common patterns
        self.address_pattern = re.compile(r'^0x[a-fA-F0-9]{40}$')
        self.amount_pattern = re.compile(r'(\d+\.?\d*)\s*(eth|usdt|usdc)', re.IGNORECASE)
        
    async def parse_message(self, message: str) -> Dict:
        """Parse user message to determine intent and extract parameters"""
        try:
            # Clean and normalize message
            cleaned_message = self._clean_message(message)
            
            # Try pattern matching first
            pattern_match = await self._pattern_match(cleaned_message)
            if pattern_match["confidence"] > 0.8:
                return pattern_match
            
            # Use LLM for more complex understanding
            return await self._llm_parse(cleaned_message)
            
        except Exception as e:
            self.logger.error(f"Failed to parse message: {e}")
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "parameters": {},
                "error": str(e)
            }
    
    async def _pattern_match(self, message: str) -> Dict:
        """Match message against common patterns"""
        # Check for token address analysis
        if "analyze" in message.lower():
            addresses = self._extract_addresses(message)
            if addresses:
                return {
                    "intent": "analyze_token",
                    "confidence": 0.9,
                    "parameters": {
                        "token_address": addresses[0],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
        
        # Check for monitoring requests
        if "monitor" in message.lower() or "track" in message.lower():
            addresses = self._extract_addresses(message)
            if addresses:
                return {
                    "intent": "monitor_token",
                    "confidence": 0.9,
                    "parameters": {
                        "token_address": addresses[0],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
        
        # Check for trading intents
        if any(word in message.lower() for word in ["buy", "sell", "trade"]):
            amount = self._extract_amount(message)
            addresses = self._extract_addresses(message)
            if amount and addresses:
                return {
                    "intent": "trade_token",
                    "confidence": 0.85,
                    "parameters": {
                        "token_address": addresses[0],
                        "amount": amount["value"],
                        "currency": amount["currency"],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
        
        return {"intent": "unknown", "confidence": 0.0, "parameters": {}}
    
    async def _llm_parse(self, message: str) -> Dict:
        """Use LLM to understand more complex messages"""
        try:
            prompt = self._create_parsing_prompt(message)
            response = await self.llm.agenerate([prompt])
            
            # Parse LLM response
            parsed = self._parse_llm_response(response.text)
            
            # Extract any additional parameters
            params = self._extract_parameters(message, parsed["intent"])
            parsed["parameters"].update(params)
            
            return parsed
            
        except Exception as e:
            self.logger.error(f"LLM parsing failed: {e}")
            return {"intent": "unknown", "confidence": 0.0, "parameters": {}}
    
    def _clean_message(self, message: str) -> str:
        """Clean and normalize user message"""
        # Remove extra whitespace
        cleaned = " ".join(message.split())
        # Convert to lowercase
        cleaned = cleaned.lower()
        # Remove special characters except addresses
        cleaned = re.sub(r'[^\w\s0x]', '', cleaned)
        return cleaned
    
    def _extract_addresses(self, message: str) -> List[str]:
        """Extract Ethereum addresses from message"""
        addresses = self.address_pattern.findall(message)
        return [addr for addr in addresses if self._is_valid_address(addr)]
    
    def _extract_amount(self, message: str) -> Optional[Dict]:
        """Extract amount and currency from message"""
        match = self.amount_pattern.search(message)
        if match:
            return {
                "value": float(match.group(1)),
                "currency": match.group(2).lower()
            }
        return None
    
    def _is_valid_address(self, address: str) -> bool:
        """Validate Ethereum address"""
        if not self.address_pattern.match(address):
            return False
        # Add additional validation if needed
        return True
    
    def _create_parsing_prompt(self, message: str) -> str:
        """Create prompt for LLM parsing"""
        return (
            "Parse the following message for trading bot intent and parameters:\n"
            f"Message: {message}\n"
            "Respond in JSON format with intent, confidence, and parameters."
        )
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response into structured format"""
        try:
            # Basic parsing of LLM JSON response
            import json
            parsed = json.loads(response)
            
            # Ensure required fields
            return {
                "intent": parsed.get("intent", "unknown"),
                "confidence": float(parsed.get("confidence", 0.0)),
                "parameters": parsed.get("parameters", {})
            }
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return {"intent": "unknown", "confidence": 0.0, "parameters": {}}
    
    def _extract_parameters(self, message: str, intent: str) -> Dict:
        """Extract additional parameters based on intent"""
        params = {}
        
        if intent == "monitor_token":
            # Extract monitoring parameters
            params["threshold"] = self._extract_threshold(message)
            params["duration"] = self._extract_duration(message)
        
        elif intent == "trade_token":
            # Extract trading parameters
            params["slippage"] = self._extract_slippage(message)
            params["deadline"] = self._extract_deadline(message)
        
        return params
    
    def _extract_threshold(self, message: str) -> Optional[float]:
        """Extract price threshold from message"""
        pattern = r'(\d+(?:\.\d+)?)\s*%'
        match = re.search(pattern, message)
        return float(match.group(1)) if match else None
    
    def _extract_duration(self, message: str) -> Optional[int]:
        """Extract monitoring duration from message"""
        pattern = r'(\d+)\s*(hour|day|week|month)'
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            unit = match.group(2).lower()
            multipliers = {
                'hour': 1,
                'day': 24,
                'week': 24 * 7,
                'month': 24 * 30
            }
            return value * multipliers.get(unit, 1)
        return None
    
    def _extract_slippage(self, message: str) -> Optional[float]:
        """Extract slippage tolerance from message"""
        pattern = r'slippage\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*%'
        match = re.search(pattern, message, re.IGNORECASE)
        return float(match.group(1)) if match else None
    
    def _extract_deadline(self, message: str) -> Optional[int]:
        """Extract transaction deadline from message"""
        pattern = r'deadline\s*(?:of|=|:)?\s*(\d+)\s*(minute|hour)'
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            unit = match.group(2).lower()
            multipliers = {'minute': 60, 'hour': 3600}
            return value * multipliers.get(unit, 60)
        return None