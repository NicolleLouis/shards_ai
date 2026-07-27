class InvalidActionError(ValueError):
    """Raised when an action is not legal in the current game state."""


class InvalidGameStateError(RuntimeError):
    """Raised when a requested state transition cannot be completed."""
