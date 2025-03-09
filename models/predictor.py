"""
Player Performance Predictor module for BallBetz.

This module provides functionality for predicting player fantasy performance
using multiple model providers: scikit-learn, Ollama, and OpenAI.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path
import logging
from typing import Optional, Dict, List, Tuple, Union, Any
from datetime import datetime
import os

from .config import ModelConfig, ModelProvider
from .model_interface import ModelInterface, SklearnModel, OllamaModel, OpenAIModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('predictor_diagnostics.log')
    ]
)
logger = logging.getLogger(__name__)

logger.info("=== PlayerPerformancePredictor Diagnostic Logs ===")

class PlayerPerformancePredictor:
    """
    Predicts player fantasy performance using multiple model providers.
    
    This class supports multiple model providers:
    - scikit-learn: Traditional machine learning model (Random Forest)
    - Ollama: Local LLM inference via Ollama API
    - OpenAI: Cloud-based LLM inference via OpenAI API
    
    The class provides seamless switching between providers and fallback mechanisms.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            model_path: Optional path to a saved model file
        """
        self.model_path = model_path
        self.sklearn_model: Optional[RandomForestRegressor] = None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.model_metadata: Dict = {}
        
        # Default provider from config
        self.default_provider = ModelConfig.DEFAULT_PROVIDER
        
        # Initialize model interfaces
        self.model_interfaces: Dict[ModelProvider, ModelInterface] = {}
        
        # Diagnostic: Check model path and existence
        if model_path:
            model_file = Path(model_path)
            logger.info(f"Model path provided: {model_path}")
            logger.info(f"Model file exists: {model_file.exists()}")
            
            if model_file.exists():
                logger.info(f"Loading existing model from {model_path}")
                self.load_model(model_path)
            else:
                logger.warning(f"Model file not found at {model_path}")
                logger.info("Initializing new model instead")
                self._initialize_model()
        else:
            logger.info("No model path provided, initializing new model")
            self._initialize_model()
            
    def _initialize_model(self) -> None:
        """Initialize a new Random Forest model with optimized parameters."""
        logger.info("Initializing new Random Forest model")
        self.sklearn_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        # Initialize the sklearn model interface
        self.model_interfaces[ModelProvider.SKLEARN] = SklearnModel(
            model=self.sklearn_model,
            scaler=self.scaler,
            feature_columns=self.feature_columns
        )
        
        # Initialize other model interfaces if enabled
        if ModelConfig.DEFAULT_PROVIDER == ModelProvider.OLLAMA or ModelConfig.ENABLE_FALLBACK:
            self.model_interfaces[ModelProvider.OLLAMA] = OllamaModel()
            
        if ModelConfig.DEFAULT_PROVIDER == ModelProvider.OPENAI or ModelConfig.ENABLE_FALLBACK:
            if ModelConfig.OPENAI_API_KEY:
                self.model_interfaces[ModelProvider.OPENAI] = OpenAIModel()
            else:
                logger.warning("OpenAI API key not provided - OpenAI model will not be available")
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer features from player data.
        
        Args:
            df: DataFrame containing player statistics
            
        Returns:
            pd.DataFrame: Engineered features for prediction
        """
        try:
            logger.info(f"Preparing features from data with {len(df)} records")
            
            # Diagnostic: Check for required columns
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
            
            logger.info("Calculating rolling averages if week information is available")
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
            
            logger.info("Adding position encoding if available")
            # Add position encoding if available
            if 'position' in df.columns:
                position_dummies = pd.get_dummies(df['position'], prefix='pos')
                features = pd.concat([features, position_dummies], axis=1)
            
            # Store feature columns for future use
            self.feature_columns = features.columns.tolist()
            logger.info(f"Feature preparation complete: {len(features.columns)} features created")
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
            
    def train(self, historical_data: pd.DataFrame, target_col: str = 'fantasy_points') -> bool:
        """
        Train the prediction model using historical data.
        
        Args:
            historical_data: DataFrame containing historical player data
            target_col: Column name for the target variable
            
        Returns:
            bool: True if training was successful
        """
        try:
            if historical_data.empty:
                logger.warning("No data available for training")
                return False
                
            logger.info("Preparing features for training")
            X = self.prepare_features(historical_data)
            y = historical_data[target_col]
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                self.sklearn_model, X_train_scaled, y_train, 
                cv=5, scoring='r2'
            )
            
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            # Train final model
            logger.info("Training final model")
            self.sklearn_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.sklearn_model.score(X_train_scaled, y_train)
            val_score = self.sklearn_model.score(X_val_scaled, y_val)
            
            # Make predictions for validation set
            val_pred = self.sklearn_model.predict(X_val_scaled)
            mse = mean_squared_error(y_val, val_pred)
            rmse = np.sqrt(mse)
            
            # Store metadata about the model
            self.model_metadata = {
                'training_date': datetime.now().isoformat(),
                'train_score': train_score,
                'val_score': val_score,
                'rmse': rmse,
                'cv_scores_mean': cv_scores.mean(),
                'cv_scores_std': cv_scores.std(),
                'feature_importance': dict(zip(self.feature_columns, 
                                            self.sklearn_model.feature_importances_))
            }
            
            logger.info(f"Training score: {train_score:.4f}")
            logger.info(f"Validation score: {val_score:.4f}")
            logger.info(f"RMSE: {rmse:.4f}")
            
            # Update the sklearn model interface
            self.model_interfaces[ModelProvider.SKLEARN] = SklearnModel(
                model=self.sklearn_model,
                scaler=self.scaler,
                feature_columns=self.feature_columns
            )
            
            # Save model checkpoint
            self._save_checkpoint()
            
            return True
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            return False
            
    def predict(self, player_data: pd.DataFrame, provider: Optional[ModelProvider] = None) -> Optional[pd.DataFrame]:
        """
        Predict fantasy points for players using the specified provider.
        
        Args:
            player_data: DataFrame containing player statistics
            provider: Optional model provider to use (defaults to the default provider)
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with predictions added
        """
        try:
            # Use specified provider or default
            current_provider = provider or self.default_provider
            logger.info(f"Making predictions using provider: {current_provider}")
            
            # Check if provider is available
            if current_provider not in self.model_interfaces:
                logger.error(f"Provider {current_provider} not available")
                
                # Try fallback if enabled
                if ModelConfig.ENABLE_FALLBACK:
                    for fallback_provider in ModelConfig.FALLBACK_ORDER:
                        if fallback_provider in self.model_interfaces:
                            logger.info(f"Falling back to provider: {fallback_provider}")
                            current_provider = fallback_provider
                            break
                    else:
                        logger.error("No fallback provider available")
                        return None
                else:
                    return None
            
            # Get the model interface for the provider
            model_interface = self.model_interfaces[current_provider]
            
            # Make predictions using the model interface
            result = model_interface.predict(player_data)
            
            logger.info(f"Predictions successfully generated using provider: {current_provider}")
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            
            # Try fallback if enabled
            if ModelConfig.ENABLE_FALLBACK and provider is not None:
                logger.info("Trying fallback providers")
                return self.predict(player_data, None)  # Use default fallback logic
                
            return None
            
    def _calculate_prediction_intervals(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Calculate prediction intervals using the Random Forest's tree variance.
        
        Args:
            X_scaled: Scaled input features
            
        Returns:
            np.ndarray: Array of prediction intervals (lower, upper)
        """
        predictions = []
        for estimator in self.sklearn_model.estimators_:
            predictions.append(estimator.predict(X_scaled))
        
        predictions = np.array(predictions)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # 95% confidence intervals
        return np.column_stack([
            mean_pred - 1.96 * std_pred,
            mean_pred + 1.96 * std_pred
        ])
        
    def _save_checkpoint(self) -> None:
        """Save a training checkpoint with metadata."""
        try:
            checkpoint_dir = Path('models/checkpoints')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = checkpoint_dir / f'model_checkpoint_{timestamp}.joblib'
            
            self.save_model(str(checkpoint_path))
            logger.info(f"Saved model checkpoint to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            
    def save_model(self, path: str) -> bool:
        """
        Save the trained model and metadata to disk.
        
        Args:
            path: Path where to save the model
            
        Returns:
            bool: True if save was successful
        """
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                'model': self.sklearn_model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'metadata': self.model_metadata
            }, path)
            logger.info(f"Model saved to {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
            
    def load_model(self, path: str) -> bool:
        """
        Load a trained model and metadata from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            bool: True if load was successful
        """
        try:
            logger.info(f"Loading model from {path}")
            
            if not Path(path).exists():
                logger.error(f"Model file does not exist: {path}")
                return False
                
            saved = joblib.load(path)
            
            # Diagnostic: Check saved model structure
            expected_keys = ['model', 'scaler', 'feature_columns']
            missing_keys = [key for key in expected_keys if key not in saved]
            if missing_keys:
                logger.error(f"Loaded model is missing required components: {missing_keys}")
                return False
                
            self.sklearn_model = saved['model']
            self.scaler = saved['scaler']
            self.feature_columns = saved['feature_columns']
            self.model_metadata = saved.get('metadata', {})
            
            # Update the sklearn model interface
            self.model_interfaces[ModelProvider.SKLEARN] = SklearnModel(
                model=self.sklearn_model,
                scaler=self.scaler,
                feature_columns=self.feature_columns
            )
            
            logger.info(f"Model successfully loaded from {path}")
            logger.info(f"Model metadata: {list(self.model_metadata.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def set_provider(self, provider: ModelProvider) -> bool:
        """
        Set the default model provider.
        
        Args:
            provider: The model provider to use
            
        Returns:
            bool: True if provider was set successfully
        """
        if provider not in self.model_interfaces:
            logger.error(f"Provider {provider} not available")
            return False
            
        self.default_provider = provider
        logger.info(f"Default provider set to: {provider}")
        return True
    
    def get_available_providers(self) -> List[ModelProvider]:
        """
        Get the list of available model providers.
        
        Returns:
            List[ModelProvider]: List of available providers
        """
        return list(self.model_interfaces.keys())