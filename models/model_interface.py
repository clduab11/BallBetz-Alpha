"""
Model interface module for BallBetz.

This module defines the interface for all model providers and implementations
for sklearn, Ollama, and OpenAI models.
"""

import abc
import json
import logging
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from .config import ModelConfig, ModelProvider

logger = logging.getLogger(__name__)

class ModelInterface(abc.ABC):
    """Abstract base class for all model providers."""
    
    @abc.abstractmethod
    def predict(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fantasy points for players.
        
        Args:
            player_data: DataFrame containing player statistics
            
        Returns:
            pd.DataFrame: DataFrame with predictions added
        """
        pass
    
    @abc.abstractmethod
    def get_prediction_intervals(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calculate prediction intervals.
        
        Args:
            predictions: Array of predictions
            
        Returns:
            np.ndarray: Array of prediction intervals (lower, upper)
        """
        pass
    
    @classmethod
    def create(cls, provider: ModelProvider, **kwargs) -> 'ModelInterface':
        """
        Factory method to create a model interface based on provider.
        
        Args:
            provider: The model provider to use
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            ModelInterface: An instance of the appropriate model interface
        """
        if provider == ModelProvider.SKLEARN:
            return SklearnModel(**kwargs)
        elif provider == ModelProvider.OLLAMA:
            return OllamaModel(**kwargs)
        elif provider == ModelProvider.OPENAI:
            return OpenAIModel(**kwargs)
        else:
            raise ValueError(f"Unsupported model provider: {provider}")


class SklearnModel(ModelInterface):
    """Implementation of the model interface for scikit-learn models."""
    
    def __init__(self, model=None, scaler=None, feature_columns=None, **kwargs):
        """
        Initialize the sklearn model.
        
        Args:
            model: The scikit-learn model to use
            scaler: The scaler to use for feature scaling
            feature_columns: The feature columns to use for prediction
            **kwargs: Additional arguments
        """
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        
    def predict(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fantasy points using the scikit-learn model.
        
        Args:
            player_data: DataFrame containing player statistics
            
        Returns:
            pd.DataFrame: DataFrame with predictions added
        """
        try:
            if self.model is None:
                logger.error("CRITICAL: Model not trained or loaded - cannot make predictions")
                return None
            
            logger.info(f"Making predictions for {len(player_data)} players using sklearn model")
                
            # Prepare features
            X = self._prepare_features(player_data)
            logger.info(f"Features prepared: {X.shape}")
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            logger.info("Features scaled successfully")
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            logger.info(f"Predictions generated: {len(predictions)} values")
            
            # Calculate prediction intervals
            prediction_intervals = self.get_prediction_intervals(X_scaled)
            logger.info("Prediction intervals calculated")
            
            # Add predictions to player data
            result = player_data.copy()
            result['predicted_points'] = predictions.round(2)
            result['prediction_lower'] = prediction_intervals[:, 0].round(2)
            result['prediction_upper'] = prediction_intervals[:, 1].round(2)
            result['prediction_confidence'] = (
                (prediction_intervals[:, 1] - prediction_intervals[:, 0]) / predictions
            ).round(4)
            result['model_provider'] = ModelProvider.SKLEARN
            
            logger.info("Predictions successfully added to player data")
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions with sklearn model: {str(e)}")
            return None
    
    def get_prediction_intervals(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Calculate prediction intervals using the Random Forest's tree variance.
        
        Args:
            X_scaled: Scaled input features
            
        Returns:
            np.ndarray: Array of prediction intervals (lower, upper)
        """
        predictions = []
        for estimator in self.model.estimators_:
            predictions.append(estimator.predict(X_scaled))
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # 95% confidence intervals
        return np.column_stack([
            mean_pred - 1.96 * std_pred,
            mean_pred + 1.96 * std_pred
        ])
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer features from player data.
        
        Args:
            df: DataFrame containing player statistics
            
        Returns:
            pd.DataFrame: Engineered features for prediction
        """
        try:
            logger.info(f"Preparing features from data with {len(df)} records")
            
            # Check for required columns
            required_cols = ['passing_yards', 'rushing_yards', 'receiving_yards']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                logger.info(f"Available columns: {df.columns.tolist()}")
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Base features
            features = df[['passing_yards', 'rushing_yards', 'receiving_yards']].copy()
            
            # Split touchdown types if available
            if 'passing_touchdowns' in df.columns:
                features['passing_touchdowns'] = df['passing_touchdowns']
                features['rushing_touchdowns'] = df['rushing_touchdowns']
                features['receiving_touchdowns'] = df['receiving_touchdowns']
            elif 'touchdowns' in df.columns:
                features['touchdowns'] = df['touchdowns']
            
            # Calculate rolling averages if week information is available
            if 'week' in df.columns and 'name' in df.columns:
                for col in features.columns:
                    rolling_avg = (
                        df.groupby('name')[col]
                        .rolling(window=3, min_periods=1)
                        .mean()
                        .reset_index(0, drop=True)
                    )
                    features[f'{col}_rolling_avg'] = rolling_avg
            
            # Add position encoding if available
            if 'position' in df.columns:
                position_dummies = pd.get_dummies(df['position'], prefix='pos')
                features = pd.concat([features, position_dummies], axis=1)
            
            # Ensure all feature columns from training are present
            if self.feature_columns:
                for col in self.feature_columns:
                    if col not in features.columns:
                        features[col] = 0  # Add missing columns with default value
                
                # Reorder columns to match training data
                features = features[self.feature_columns]
            
            logger.info(f"Feature preparation complete: {len(features.columns)} features created")
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise


class OllamaModel(ModelInterface):
    """Implementation of the model interface for Ollama models."""
    
    def __init__(self, **kwargs):
        """
        Initialize the Ollama model.
        
        Args:
            **kwargs: Additional arguments
        """
        self.base_url = ModelConfig.OLLAMA_BASE_URL
        self.timeout = ModelConfig.OLLAMA_TIMEOUT
        self.max_retries = ModelConfig.MAX_RETRIES
        self.retry_backoff = ModelConfig.RETRY_BACKOFF
        self.available_models = self._get_available_models()
        
    def _get_available_models(self) -> List[str]:
        """
        Get the list of available models from Ollama.
        
        Returns:
            List[str]: List of available model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            models = response.json().get('models', [])
            model_names = [model.get('name') for model in models]
            logger.info(f"Available Ollama models: {model_names}")
            return model_names
        except Exception as e:
            logger.error(f"Error getting available Ollama models: {str(e)}")
            return []
    
    def _select_model(self, player_position: str) -> str:
        """
        Select the appropriate model based on player position.
        
        Args:
            player_position: The player's position
            
        Returns:
            str: The selected model name
        """
        # For now, use a simple selection strategy
        # In a real implementation, you might want to use different models for different positions
        
        # Prefer smaller models for faster inference
        small_models = [m for m in self.available_models if '3b' in m.lower()]
        if small_models:
            return small_models[0]
        
        # Fall back to any available model
        if self.available_models:
            return self.available_models[0]
        
        # Default fallback
        return "llama3.2-3b-instruct"
    
    @retry(
        stop=stop_after_attempt(ModelConfig.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=ModelConfig.RETRY_BACKOFF),
        retry=retry_if_exception_type((requests.RequestException, json.JSONDecodeError)),
        reraise=True
    )
    def _call_ollama_api(self, model: str, prompt: str) -> str:
        """
        Call the Ollama API to generate a prediction.
        
        Args:
            model: The model to use
            prompt: The prompt to send to the model
            
        Returns:
            str: The model's response
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": ModelConfig.TEMPERATURE,
                    "num_predict": ModelConfig.MAX_TOKENS
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
            
        except requests.RequestException as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Ollama API response: {str(e)}")
            raise
    
    def _format_player_stats(self, player: pd.Series) -> str:
        """
        Format player statistics for the prompt.
        
        Args:
            player: Series containing player statistics
            
        Returns:
            str: Formatted player statistics
        """
        stats = []
        
        # Add passing stats
        if 'passing_yards' in player and player['passing_yards'] > 0:
            stats.append(f"Passing: {player['passing_yards']} yards")
            if 'passing_touchdowns' in player:
                stats.append(f"{player['passing_touchdowns']} TDs")
            if 'interceptions' in player:
                stats.append(f"{player['interceptions']} INTs")
        
        # Add rushing stats
        if 'rushing_yards' in player and player['rushing_yards'] > 0:
            stats.append(f"Rushing: {player['rushing_yards']} yards")
            if 'rushing_touchdowns' in player:
                stats.append(f"{player['rushing_touchdowns']} TDs")
        
        # Add receiving stats
        if 'receiving_yards' in player and player['receiving_yards'] > 0:
            stats.append(f"Receiving: {player['receiving_yards']} yards")
            if 'receptions' in player:
                stats.append(f"{player['receptions']} receptions")
            if 'receiving_touchdowns' in player:
                stats.append(f"{player['receiving_touchdowns']} TDs")
        
        # Add additional stats if available
        if 'games_played' in player:
            stats.append(f"Games played: {player['games_played']}")
        
        return ", ".join(stats)
    
    def _extract_prediction(self, response: str) -> float:
        """
        Extract the prediction value from the model's response.
        
        Args:
            response: The model's response
            
        Returns:
            float: The extracted prediction value
        """
        try:
            # Try to find a number in the response
            import re
            numbers = re.findall(r'\d+\.\d+|\d+', response)
            if numbers:
                return float(numbers[0])
            
            # If no number is found, return a default value
            logger.warning(f"Could not extract prediction from response: {response}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error extracting prediction: {str(e)}")
            return 0.0
    
    def predict(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fantasy points using the Ollama model.
        
        Args:
            player_data: DataFrame containing player statistics
            
        Returns:
            pd.DataFrame: DataFrame with predictions added
        """
        try:
            logger.info(f"Making predictions for {len(player_data)} players using Ollama model")
            
            # Create a copy of the input data
            result = player_data.copy()
            
            # Initialize prediction columns
            result['predicted_points'] = 0.0
            result['prediction_lower'] = 0.0
            result['prediction_upper'] = 0.0
            result['prediction_confidence'] = 0.0
            result['model_provider'] = ModelProvider.OLLAMA
            
            # Process each player
            for idx, player in result.iterrows():
                # Select model based on player position
                position = player.get('position', 'UNKNOWN')
                model = self._select_model(position)
                
                # Format player stats
                stats = self._format_player_stats(player)
                
                # Create prompt
                model_size = "small"  # Assume small model for Ollama
                prompt_template = ModelConfig.get_prompt_template(model_size)
                prompt = prompt_template.format(
                    player_name=player.get('name', 'Unknown'),
                    position=position,
                    team=player.get('team', 'Unknown'),
                    stats=stats
                )
                
                # Call Ollama API
                response = self._call_ollama_api(model, prompt)
                
                # Extract prediction
                prediction = self._extract_prediction(response)
                
                # Update result
                result.at[idx, 'predicted_points'] = round(prediction, 2)
                
                # Simple prediction intervals (±20%)
                result.at[idx, 'prediction_lower'] = round(prediction * 0.8, 2)
                result.at[idx, 'prediction_upper'] = round(prediction * 1.2, 2)
                result.at[idx, 'prediction_confidence'] = 0.2  # Fixed confidence for now
            
            logger.info("Predictions successfully added to player data")
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions with Ollama model: {str(e)}")
            return None
    
    def get_prediction_intervals(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calculate prediction intervals.
        
        Args:
            predictions: Array of predictions
            
        Returns:
            np.ndarray: Array of prediction intervals (lower, upper)
        """
        # Simple prediction intervals (±20%)
        return np.column_stack([
            predictions * 0.8,
            predictions * 1.2
        ])


class OpenAIModel(ModelInterface):
    """Implementation of the model interface for OpenAI models."""
    
    def __init__(self, **kwargs):
        """
        Initialize the OpenAI model.
        
        Args:
            **kwargs: Additional arguments
        """
        self.api_key = ModelConfig.OPENAI_API_KEY
        self.primary_model = ModelConfig.OPENAI_PRIMARY_MODEL
        self.fallback_model = ModelConfig.OPENAI_FALLBACK_MODEL
        self.timeout = ModelConfig.OPENAI_TIMEOUT
        self.max_retries = ModelConfig.MAX_RETRIES
        self.retry_backoff = ModelConfig.RETRY_BACKOFF
        
        # Import OpenAI library
        try:
            import openai
            self.openai = openai
            self.openai.api_key = self.api_key
        except ImportError:
            logger.error("OpenAI library not installed")
            self.openai = None
    
    @retry(
        stop=stop_after_attempt(ModelConfig.MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=ModelConfig.RETRY_BACKOFF),
        retry=retry_if_exception_type((requests.RequestException, json.JSONDecodeError)),
        reraise=True
    )
    def _call_openai_api(self, model: str, prompt: str, use_fallback: bool = False) -> str:
        """
        Call the OpenAI API to generate a prediction.
        
        Args:
            model: The model to use
            prompt: The prompt to send to the model
            use_fallback: Whether to use the fallback model
            
        Returns:
            str: The model's response
        """
        if self.openai is None:
            logger.error("OpenAI library not available")
            return ""
        
        try:
            # Use the specified model or fallback if requested
            current_model = self.fallback_model if use_fallback else model
            
            response = self.openai.chat.completions.create(
                model=current_model,
                messages=[
                    {"role": "system", "content": "You are a fantasy football analyst specializing in UFL player performance predictions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=ModelConfig.TEMPERATURE,
                max_tokens=ModelConfig.MAX_TOKENS,
                timeout=self.timeout
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API with model {model}: {str(e)}")
            
            # Try fallback model if not already using it
            if not use_fallback and ModelConfig.ENABLE_FALLBACK:
                logger.info(f"Trying fallback model: {self.fallback_model}")
                return self._call_openai_api(model, prompt, use_fallback=True)
            
            raise
    
    def _format_player_stats(self, player: pd.Series) -> str:
        """
        Format player statistics for the prompt.
        
        Args:
            player: Series containing player statistics
            
        Returns:
            str: Formatted player statistics
        """
        stats = []
        
        # Add passing stats
        if 'passing_yards' in player and player['passing_yards'] > 0:
            stats.append(f"Passing: {player['passing_yards']} yards")
            if 'passing_touchdowns' in player:
                stats.append(f"{player['passing_touchdowns']} TDs")
            if 'interceptions' in player:
                stats.append(f"{player['interceptions']} INTs")
        
        # Add rushing stats
        if 'rushing_yards' in player and player['rushing_yards'] > 0:
            stats.append(f"Rushing: {player['rushing_yards']} yards")
            if 'rushing_touchdowns' in player:
                stats.append(f"{player['rushing_touchdowns']} TDs")
        
        # Add receiving stats
        if 'receiving_yards' in player and player['receiving_yards'] > 0:
            stats.append(f"Receiving: {player['receiving_yards']} yards")
            if 'receptions' in player:
                stats.append(f"{player['receptions']} receptions")
            if 'receiving_touchdowns' in player:
                stats.append(f"{player['receiving_touchdowns']} TDs")
        
        # Add additional stats if available
        if 'games_played' in player:
            stats.append(f"Games played: {player['games_played']}")
        
        return ", ".join(stats)
    
    def _extract_prediction(self, response: str) -> float:
        """
        Extract the prediction value from the model's response.
        
        Args:
            response: The model's response
            
        Returns:
            float: The extracted prediction value
        """
        try:
            # Try to find a number in the response
            import re
            numbers = re.findall(r'\d+\.\d+|\d+', response)
            if numbers:
                return float(numbers[0])
            
            # If no number is found, return a default value
            logger.warning(f"Could not extract prediction from response: {response}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error extracting prediction: {str(e)}")
            return 0.0
    
    def predict(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fantasy points using the OpenAI model.
        
        Args:
            player_data: DataFrame containing player statistics
            
        Returns:
            pd.DataFrame: DataFrame with predictions added
        """
        try:
            if self.openai is None:
                logger.error("OpenAI library not available - cannot make predictions")
                return None
            
            logger.info(f"Making predictions for {len(player_data)} players using OpenAI model")
            
            # Create a copy of the input data
            result = player_data.copy()
            
            # Initialize prediction columns
            result['predicted_points'] = 0.0
            result['prediction_lower'] = 0.0
            result['prediction_upper'] = 0.0
            result['prediction_confidence'] = 0.0
            result['model_provider'] = ModelProvider.OPENAI
            
            # Process each player
            for idx, player in result.iterrows():
                # Format player stats
                stats = self._format_player_stats(player)
                
                # Create prompt
                model_size = ModelConfig.get_model_size(ModelProvider.OPENAI)
                prompt_template = ModelConfig.get_prompt_template(model_size)
                prompt = prompt_template.format(
                    player_name=player.get('name', 'Unknown'),
                    position=player.get('position', 'UNKNOWN'),
                    team=player.get('team', 'Unknown'),
                    stats=stats
                )
                
                # Call OpenAI API
                response = self._call_openai_api(self.primary_model, prompt)
                
                # Extract prediction
                prediction = self._extract_prediction(response)
                
                # Update result
                result.at[idx, 'predicted_points'] = round(prediction, 2)
                
                # Simple prediction intervals (±15%)
                result.at[idx, 'prediction_lower'] = round(prediction * 0.85, 2)
                result.at[idx, 'prediction_upper'] = round(prediction * 1.15, 2)
                result.at[idx, 'prediction_confidence'] = 0.15  # Fixed confidence for now
            
            logger.info("Predictions successfully added to player data")
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions with OpenAI model: {str(e)}")
            return None
    
    def get_prediction_intervals(self, predictions: np.ndarray) -> np.ndarray:
        """
        Calculate prediction intervals.
        
        Args:
            predictions: Array of predictions
            
        Returns:
            np.ndarray: Array of prediction intervals (lower, upper)
        """
        # Simple prediction intervals (±15%)
        return np.column_stack([
            predictions * 0.85,
            predictions * 1.15
        ])