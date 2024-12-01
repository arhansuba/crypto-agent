# AI Crypto Agent API Reference

## Overview

The AI Crypto Agent provides a comprehensive API for interacting with the trading system, managing configurations, and monitoring performance.

## Authentication

All API requests require authentication using API keys:

```python
headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}
```

## Core API Endpoints

### Agent Control

#### Start Agent
```http
POST /api/v1/agent/start
```

Request body:
```json
{
    "mode": "autonomous",
    "config_path": "config/production.yaml",
    "initial_state": {}
}
```

#### Stop Agent
```http
POST /api/v1/agent/stop
```

#### Get Agent Status
```http
GET /api/v1/agent/status
```

Response:
```json
{
    "status": "running",
    "uptime": "2h 15m",
    "current_state": {},
    "performance_metrics": {}
}
```

### Trading Operations

#### Create Token
```http
POST /api/v1/trading/token/create
```

Request body:
```json
{
    "name": "Example Token",
    "symbol": "EXT",
    "initial_supply": 1000000,
    "decimals": 18
}
```

#### Create Liquidity Pool
```http
POST /api/v1/trading/pool/create
```

Request body:
```json
{
    "token_address": "0x...",
    "initial_liquidity": 1.0,
    "fee_tier": 0.3
}
```

### Market Analysis

#### Get Market Prediction
```http
GET /api/v1/analysis/prediction
```

Response:
```json
{
    "price_prediction": 1500.50,
    "confidence": 0.85,
    "trend": "bullish",
    "timeframe": "4h",
    "supporting_metrics": {}
}
```

#### Get Risk Analysis
```http
GET /api/v1/analysis/risk
```

Response:
```json
{
    "overall_risk": 0.35,
    "volatility_risk": 0.4,
    "liquidity_risk": 0.3,
    "market_risk": 0.35,
    "recommendations": []
}
```

## WebSocket API

### Market Data Stream
```javascript
ws://api/v1/stream/market
```

Message format:
```json
{
    "event": "price_update",
    "data": {
        "price": 1500.50,
        "timestamp": "2024-01-01T12:00:00Z",
        "volume": 1000000
    }
}
```

### Agent Events Stream
```javascript
ws://api/v1/stream/events
```

Message format:
```json
{
    "event": "trade_executed",
    "data": {
        "transaction_hash": "0x...",
        "status": "confirmed",
        "details": {}
    }
}
```

## Error Handling

### Error Responses
```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Error description",
        "details": {}
    }
}
```

Common error codes:
- `INVALID_PARAMETERS`
- `INSUFFICIENT_FUNDS`
- `UNAUTHORIZED`
- `RATE_LIMITED`
- `INTERNAL_ERROR`

## Rate Limits

- Standard tier: 60 requests per minute
- Premium tier: 300 requests per minute
- Websocket connections: 5 per client

## SDK Examples

### Python SDK
```python
from crypto_agent_sdk import CryptoAgent

agent = CryptoAgent(api_key="YOUR_API_KEY")

# Start agent
await agent.start(mode="autonomous")

# Create token
token_address = await agent.create_token(
    name="Example Token",
    symbol="EXT",
    initial_supply=1000000
)

# Get market prediction
prediction = await agent.get_market_prediction()
```

### JavaScript SDK
```javascript
const { CryptoAgent } = require('crypto-agent-sdk');

const agent = new CryptoAgent('YOUR_API_KEY');

// Start agent
await agent.start({ mode: 'autonomous' });

// Subscribe to events
agent.events.on('trade_executed', (data) => {
    console.log('Trade executed:', data);
});
```

## Webhooks

### Configuration
```http
POST /api/v1/webhooks/configure
```

Request body:
```json
{
    "url": "https://your-server.com/webhook",
    "events": ["trade_executed", "error_occurred"],
    "secret": "your_webhook_secret"
}
```

### Event Format
```json
{
    "event": "trade_executed",
    "timestamp": "2024-01-01T12:00:00Z",
    "data": {},
    "signature": "..."
}
```

## API Versioning

The API uses semantic versioning (v1, v2, etc.). Breaking changes are introduced in new major versions.

## Support

- API Status: https://status.example.com
- Documentation: https://docs.example.com
- Support Email: api-support@example.com