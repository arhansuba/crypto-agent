class CDPAgentError(Exception):
    """Base class for all CDP agent errors."""
    pass

class InvalidConfigError(CDPAgentError):
    """Raised when the agent configuration is invalid."""
    pass

class WalletInitializationError(CDPAgentError):
    """Raised when there is an error initializing the wallet."""
    pass