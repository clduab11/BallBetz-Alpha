import os
from typing import Dict, Any

class APIAILayerConfig:
    """Configuration management for API/Local AI Layer."""

    @classmethod
    def load_config(cls) -> Dict[str, Any]:
        """
        Load configuration from environment variables with sensible defaults.
        
        Returns:
            Dict containing configuration parameters
        """
        return {
            # Transformer Model Configuration
            'transformer_model': {
                'default_model': os.getenv('TRANSFORMER_DEFAULT_MODEL', 'gpt-3.5-turbo'),
                'max_tokens': int(os.getenv('TRANSFORMER_MAX_TOKENS', 2048)),
                'temperature': float(os.getenv('TRANSFORMER_TEMPERATURE', 0.7)),
                'batch_size': int(os.getenv('TRANSFORMER_BATCH_SIZE', 16)),
            },
            
            # Provider Configuration
            'providers': {
                'openai': {
                    'api_key': os.getenv('OPENAI_API_KEY', ''),
                    'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                    'timeout': int(os.getenv('OPENAI_TIMEOUT', 30)),
                    'max_retries': int(os.getenv('OPENAI_MAX_RETRIES', 3)),
                },
                'local_model': {
                    'model_path': os.getenv('LOCAL_MODEL_PATH', './models/local_transformer'),
                    'device': os.getenv('LOCAL_MODEL_DEVICE', 'cpu'),
                }
            },
            
            # Caching Configuration
            'cache': {
                'enabled': os.getenv('TRANSFORMER_CACHE_ENABLED', 'true').lower() == 'true',
                'max_size': int(os.getenv('TRANSFORMER_CACHE_MAX_SIZE', 1000)),
                'ttl': int(os.getenv('TRANSFORMER_CACHE_TTL', 3600)),  # 1 hour default
            },
            
            # Rate Limiting Configuration
            'rate_limit': {
                'max_requests_per_minute': int(os.getenv('TRANSFORMER_MAX_REQUESTS_PER_MINUTE', 60)),
                'max_tokens_per_minute': int(os.getenv('TRANSFORMER_MAX_TOKENS_PER_MINUTE', 90000)),
            },
            
            # Prediction Orchestration
            'orchestration': {
                'confidence_threshold': float(os.getenv('PREDICTION_CONFIDENCE_THRESHOLD', 0.7)),
                'fallback_strategy': os.getenv('PREDICTION_FALLBACK_STRATEGY', 'local_model'),
            }
        }

    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> None:
        """
        Validate the configuration parameters.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary to validate
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate OpenAI configuration
        if not config['providers']['openai']['api_key']:
            raise ValueError("OpenAI API key must be provided")
        
        # Validate temperature range
        temp = config['transformer_model']['temperature']
        if temp < 0 or temp > 2:
            raise ValueError("Temperature must be between 0 and 2")
        
        # Validate batch size
        batch_size = config['transformer_model']['batch_size']
        if batch_size < 1 or batch_size > 128:
            raise ValueError("Batch size must be between 1 and 128")