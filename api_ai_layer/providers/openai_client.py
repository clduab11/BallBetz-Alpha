import openai
from typing import List, Dict, Any, Optional
import logging
import asyncio
import time

from api_ai_layer.providers.base_provider import BaseTransformerProvider
from api_ai_layer.exceptions import ProviderError, RateLimitError

class OpenAIProvider(BaseTransformerProvider):
    """
    Provider for OpenAI transformer models using the OpenAI API.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI provider.
        
        Args:
            config (Dict[str, Any]): Configuration for OpenAI provider
        """
        super().__init__(config)
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Configure OpenAI client
        openai.api_key = config['providers']['openai']['api_key']
        openai.base_url = config['providers']['openai'].get('base_url', 'https://api.openai.com/v1')
        
        # Provider-specific configuration
        self._timeout = config['providers']['openai'].get('timeout', 30)
        self._max_retries = config['providers']['openai'].get('max_retries', 3)
        
        # Rate limiting tracking
        self._request_timestamps = []
        self._token_usage = 0

    def _validate_config(self) -> None:
        """
        Validate OpenAI provider configuration.
        
        Raises:
            ProviderError: If configuration is invalid
        """
        config = self.config['providers']['openai']
        
        if not config.get('api_key'):
            raise ProviderError("OpenAI API key is required")
        
        if not config.get('base_url', '').startswith(('http://', 'https://')):
            raise ProviderError("Invalid OpenAI base URL")

    def load_model(self, model_name: Optional[str] = None) -> Any:
        """
        Load OpenAI model configuration.
        
        Args:
            model_name (Optional[str]): Specific model to load
        
        Returns:
            Model configuration dictionary
        """
        default_model = self.config['transformer_model']['default_model']
        model = model_name or default_model
        
        return {
            'model': model,
            'max_tokens': self.config['transformer_model']['max_tokens'],
            'temperature': self.config['transformer_model']['temperature']
        }

    def predict(
        self, 
        inputs: List[str], 
        model: Any, 
        max_tokens: int = 2048, 
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Generate predictions using OpenAI API.
        
        Args:
            inputs (List[str]): Input texts to generate predictions for
            model (Any): Model configuration
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature for generation
        
        Returns:
            List of prediction dictionaries
        
        Raises:
            RateLimitError: If rate limits are exceeded
            ProviderError: For other prediction errors
        """
        self._check_rate_limits()
        
        try:
            predictions = []
            for input_text in inputs:
                prediction = self._generate_single_prediction(
                    input_text, 
                    model['model'], 
                    max_tokens, 
                    temperature
                )
                predictions.append(prediction)
            
            return predictions
        
        except openai.RateLimitError as e:
            self._logger.warning(f"OpenAI rate limit exceeded: {e}")
            raise RateLimitError("OpenAI API rate limit exceeded") from e
        except Exception as e:
            self._logger.error(f"OpenAI prediction error: {e}")
            raise ProviderError(f"Prediction failed: {e}") from e

    def _generate_single_prediction(
        self, 
        input_text: str, 
        model: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """
        Generate a single prediction using OpenAI API.
        
        Args:
            input_text (str): Input text for prediction
            model (str): Model name
            max_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
        
        Returns:
            Prediction dictionary
        """
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Extract prediction details
        prediction = {
            'input': input_text,
            'model': model,
            'prediction': response.choices[0].message.content,
            'confidence': self._calculate_confidence(response),
            'tokens_used': response.usage.total_tokens,
            'raw_response': response
        }
        
        # Update token usage tracking
        self._token_usage += response.usage.total_tokens
        
        return prediction

    def _calculate_confidence(self, response: Any) -> float:
        """
        Calculate confidence based on OpenAI response.
        
        Args:
            response (Any): OpenAI API response
        
        Returns:
            Confidence score
        """
        # Basic confidence calculation based on logprobs or other metrics
        # This is a placeholder and should be customized
        return 0.7  # Default confidence

    def _check_rate_limits(self) -> None:
        """
        Check and enforce rate limits for OpenAI API.
        
        Raises:
            RateLimitError: If rate limits are exceeded
        """
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self._request_timestamps = [
            ts for ts in self._request_timestamps 
            if current_time - ts < 60
        ]
        
        # Check request frequency limit
        max_requests = self.config['rate_limit']['max_requests_per_minute']
        if len(self._request_timestamps) >= max_requests:
            raise RateLimitError("Exceeded maximum requests per minute")
        
        # Check token usage limit
        max_tokens = self.config['rate_limit']['max_tokens_per_minute']
        if self._token_usage >= max_tokens:
            raise RateLimitError("Exceeded maximum tokens per minute")
        
        # Record current request timestamp
        self._request_timestamps.append(current_time)

    def calculate_confidence(self, predictions: List[Dict[str, Any]]) -> float:
        """
        Calculate aggregate confidence for predictions.
        
        Args:
            predictions (List[Dict[str, Any]]): List of prediction results
        
        Returns:
            float: Aggregate confidence score
        """
        if not predictions:
            return 0.0
        
        # Average confidence across predictions
        confidences = [pred.get('confidence', 0.0) for pred in predictions]
        return sum(confidences) / len(confidences)