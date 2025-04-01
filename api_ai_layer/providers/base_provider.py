from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from api_ai_layer.exceptions import ProviderError, RateLimitError

class BaseTransformerProvider(ABC):
    """
    Abstract base class for transformer model providers.
    Defines the interface for different AI model providers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the provider with configuration.
        
        Args:
            config (Dict[str, Any]): Provider-specific configuration
        """
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        """
        Validate provider-specific configuration.
        
        Raises:
            ProviderError: If configuration is invalid
        """
        pass

    @abstractmethod
    def load_model(self, model_name: Optional[str] = None) -> Any:
        """
        Load a transformer model.
        
        Args:
            model_name (Optional[str]): Name or path of the model to load
        
        Returns:
            Loaded model instance
        
        Raises:
            ProviderError: If model loading fails
        """
        pass

    @abstractmethod
    def predict(
        self, 
        inputs: List[str], 
        model: Any, 
        max_tokens: int = 2048, 
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions for given inputs.
        
        Args:
            inputs (List[str]): Input texts to generate predictions for
            model (Any): Loaded model instance
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature for generation
        
        Returns:
            List of prediction dictionaries with results and metadata
        
        Raises:
            RateLimitError: If rate limits are exceeded
            ProviderError: For other prediction errors
        """
        pass

    def handle_rate_limit(self, func):
        """
        Decorator to handle rate limiting for provider methods.
        
        Args:
            func (Callable): Method to wrap with rate limit handling
        
        Returns:
            Wrapped method with rate limit protection
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                # Implement exponential backoff or fallback strategy
                # This is a placeholder - actual implementation would be more sophisticated
                raise e
        return wrapper

    def calculate_confidence(self, predictions: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for predictions.
        
        Args:
            predictions (List[Dict[str, Any]]): List of prediction results
        
        Returns:
            float: Aggregate confidence score
        """
        # Basic implementation - can be overridden by specific providers
        if not predictions:
            return 0.0
        
        # Simple average of individual prediction confidences
        confidences = [
            pred.get('confidence', 0.0) for pred in predictions
        ]
        return sum(confidences) / len(confidences) if confidences else 0.0