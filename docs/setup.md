# AI Crypto Agent Setup Guide

## Prerequisites

- Python 3.10 or higher
- Node.js 16+ (for web3 dependencies)
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/ai-crypto-agent.git
cd ai-crypto-agent
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
```bash
cp config/config.example.yaml config/config.yaml
```

## Configuration

### Required Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key
CDP_API_KEY=your_cdp_api_key
CDP_API_SECRET=your_cdp_api_secret
NETWORK_ID=base-sepolia
```

### Network Configuration

Edit `config/network_config.yaml`:

```yaml
networks:
  base_sepolia:
    rpc_url: "https://sepolia.base.org"
    chain_id: 84532
    explorer_url: "https://sepolia.basescan.org"
```

## Running Tests

```bash
pytest tests/
```

## Development Setup

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Set up pre-commit hooks:
```bash
pre-commit install
```

## Deployment

### Local Development
```bash
python scripts/run_agent.py --mode=development
```

### Production Deployment
```bash
python scripts/run_agent.py --mode=production --config=config/production.yaml
```

## Security Considerations

1. Secure your API keys and private keys
2. Use hardware wallet for production deployments
3. Set appropriate transaction limits in config
4. Enable monitoring and alerts

## Troubleshooting

Common issues and solutions:

1. Web3 Connection Issues:
   - Check network configuration
   - Verify RPC endpoint status

2. API Rate Limits:
   - Implement exponential backoff
   - Check API quota

3. Memory Issues:
   - Adjust cache settings
   - Monitor resource usage

## Updates and Maintenance

1. Regular updates:
```bash
git pull origin main
pip install -r requirements.txt
```

2. Database migrations:
```bash
python scripts/migrate.py
```

## Support

For support and questions:
- Create an issue on GitHub
- Join our Discord community
- Email: support@example.com