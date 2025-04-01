from typing import List, Dict, Any, Optional
from functools import lru_cache
import logging

from api_ai_layer.config import APIAILayerConfig
from api_ai_layer.exceptions import ModelLoadError, PredictionError
from api_ai_layer.providers.base_provider import BaseTransformerProvider

class TransformerModelInterface:
    """
    Interface for managing and interacting with transformer models.
    Provides a unified approach to working with different AI providers.
    """

    def __init__(
        self, 
        config: Optional[Dict[str, Any]] = None, 
        providers: Optional[List[BaseTransformerProvider]] = None
    ):
        """
        Initialize the transformer model interface.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration for the interface
            providers (Optional[List[BaseTransformerProvider]]): List of AI providers
        """
        self.config = config or APIAILayerConfig.load_config()
        self.providers = providers or []
        self._models = {}
        self._cache = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def register_provider(self, provider: BaseTransformerProvider) -> None:
        """
        Register a new AI provider with the interface.
        
        Args:
            provider (BaseTransformerProvider): Provider to register
        """
        self.providers.append(provider)

    @lru_cache(maxsize=128)
    def load_model(
        self, 
        model_name: Optional[str] = None, 
        provider_index: Optional[int] = None
    ) -> Any:
        """
        Load a transformer model from a specific or default provider.
        
        Args:
            model_name (Optional[str]): Specific model to load
            provider_index (Optional[int]): Index of provider to use
        
        Returns:
            Loaded model instance
        
        Raises:
            ModelLoadError: If no providers are available or model loading fails
        """
        if not self.providers:
            raise ModelLoadError("No AI providers registered")
        
        # Use specified provider or default to first provider
        provider = (
            self.providers[provider_index] if provider_index is not None 
            else self.providers[0]
        )
        
        try:
            model = provider.load_model(model_name)
            self._models[model_name or 'default'] = model
            return model
        except Exception as e:
            self._logger.error(f"Model loading failed: {e}")
            raise ModelLoadError(f"Failed to load model: {e}")

    def predict_batch(
        self, 
        inputs: List[str], 
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate batch predictions using the specified or default model.
        
        Args:
            inputs (List[str]): Input texts to generate predictions for
            model_name (Optional[str]): Name of model to use
            max_tokens (Optional[int]): Override for max tokens
            temperature (Optional[float]): Override for sampling temperature
        
        Returns:
            List of prediction dictionaries
        
        Raises:
            PredictionError: If prediction fails
        """
        # Use configuration defaults if not specified
        max_tokens = max_tokens or self.config['transformer_model']['max_tokens']
        temperature = temperature or self.config['transformer_model']['temperature']
        
        # Retrieve or load model
        try:
            model = self._models.get(model_name or 'default')
            if not model:
                model = self.load_model(model_name)
            
            # Attempt prediction with first available provider
            for provider in self.providers:
                try:
                    predictions = provider.predict(
                        inputs, 
                        model, 
                        max_tokens=max_tokens, 
                        temperature=temperature
                    )
                    
                    # Calculate confidence
                    confidence = provider.calculate_confidence(predictions)
                    
                    # Optional caching if enabled
                    if self.config['cache']['enabled']:
                        self._cache_predictions(inputs, predictions)
                    
                    return predictions
                except Exception as provider_error:
                    self._logger.warning(f"Provider prediction failed: {provider_error}")
            
            raise PredictionError("All providers failed to generate predictions")
        
        except Exception as e:
            self._logger.error(f"Batch prediction failed: {e}")
            raise PredictionError(f"Prediction error: {e}")

    def _cache_predictions(
        self, 
        inputs: List[str], 
        predictions: List[Dict[str, Any]]
    ) -> None:
        """
        Cache predictions if caching is enabled.
        
        Args:
            inputs (List[str]): Input texts
            predictions (List[Dict[str, Any]]): Prediction results
        """
        cache_config = self.config['cache']
        if not cache_config['enabled']:
            return
        
        for input_text, prediction in zip(inputs, predictions):
            # Use a simple hash as cache key
            cache_key = hash(input_text)
            self._cache[cache_key] = {
                'prediction': prediction,
                'timestamp': self._get_current_timestamp()
            }
        
        # Implement basic cache size management
        if len(self._cache) > cache_config['max_size']:
            self._prune_cache()

    def _prune_cache(self) -> None:
        """
        Prune the cache based on size and TTL.
        """
        current_time = self._get_current_timestamp()
        cache_config = self.config['cache']
        
        # Remove expired entries
        self._cache = {
            k: v for k, v in self._cache.items()
            if current_time - v['timestamp'] < cache_config['ttl']
        }

    @staticmethod
    def _get_current_timestamp() -> float:
        """
        Get current timestamp.
        
        Returns:
            Current timestamp as float
        """
        import time
        return time.time()