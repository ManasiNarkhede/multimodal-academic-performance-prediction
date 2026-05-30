"""Custom exceptions for the application."""


class ModelNotFoundError(Exception):
    """Raised when a requested model or resource is not found."""

    message: str

    def __init__(self, message: str = "Resource not found") -> None:
        self.message = message
        super().__init__(self.message)


class PredictionError(Exception):
    """Raised when a prediction operation fails."""

    message: str

    def __init__(self, message: str = "Prediction failed") -> None:
        self.message = message
        super().__init__(self.message)


class AuthenticationError(Exception):
    """Raised when authentication or authorization fails."""

    message: str

    def __init__(self, message: str = "Authentication failed") -> None:
        self.message = message
        super().__init__(self.message)
