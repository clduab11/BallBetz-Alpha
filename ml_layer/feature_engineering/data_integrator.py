import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

from scrapers.ufl_scraper import UFLScraper
from .config import FeatureEngineeringConfig
from .exceptions import DataPreprocessingError

class UFLDataIntegrator:
    """
    Integrates and preprocesses UFL player data for machine learning
    
    Combines multiple data sources, handles feature engineering,
    and prepares data for predictive modeling.
    """
    
    def __init__(
        self, 
        config: Optional[FeatureEngineeringConfig] = None,
        scraper: Optional[UFLScraper] = None
    ):
        """
        Initialize the data integrator
        
        :param config: Feature engineering configuration
        :param scraper: Optional UFLScraper instance
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or FeatureEngineeringConfig.from_env()
        self.scraper = scraper or UFLScraper()
        
        # Label encoders for categorical features
        self.team_encoder = LabelEncoder()
        self.position_encoder = LabelEncoder()
    
    def _preprocess_player_data(self, data: List[Dict]) -> pd.DataFrame:
        """
        Convert raw player data to a preprocessed DataFrame
        
        :param data: Raw player statistics
        :return: Preprocessed DataFrame
        """
        try:
            df = pd.DataFrame(data)
            
            # Encode categorical features
            df['team_encoded'] = self.team_encoder.fit_transform(df['team'])
            df['position_encoded'] = self.position_encoder.fit_transform(df['position'])
            
            # Feature engineering
            df['total_touchdowns'] = (
                df.get('passing_touchdowns', 0) + 
                df.get('rushing_touchdowns', 0) + 
                df.get('receiving_touchdowns', 0)
            )
            
            df['yards_per_game'] = (
                df.get('passing_yards', 0) / df['games_played'] +
                df.get('rushing_yards', 0) / df['games_played'] +
                df.get('receiving_yards', 0) / df['games_played']
            )
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error preprocessing player data: {e}")
            raise DataPreprocessingError(f"Player data preprocessing failed: {e}")
    
    def _select_features(
        self, 
        X: pd.DataFrame, 
        y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Select the most relevant features
        
        :param X: Feature matrix
        :param y: Optional target variable
        :return: DataFrame with selected features
        """
        # Numeric columns for feature selection
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        if y is not None:
            selector = SelectKBest(
                score_func=f_classif, 
                k=min(self.config.max_features_to_select, len(numeric_columns))
            )
            X_selected = selector.fit_transform(X[numeric_columns], y)
            selected_columns = numeric_columns[selector.get_support()]
            return X[selected_columns]
        
        return X[numeric_columns]
    
    def prepare_prediction_dataset(
        self, 
        prediction_type: str = 'win_loss'
    ) -> Dict[str, np.ndarray]:
        """
        Prepare a dataset for machine learning predictions
        
        :param prediction_type: Type of prediction (win_loss, point_spread)
        :return: Dictionary with features and optional target
        """
        try:
            # Scrape player data
            raw_data = self.scraper.scrape_player_data()
            
            # Preprocess data
            df = self._preprocess_player_data(raw_data)
            
            # Create target variable based on prediction type
            if prediction_type == 'win_loss':
                # Example: Binary classification based on total touchdowns
                df['target'] = (df['total_touchdowns'] > df['total_touchdowns'].median()).astype(int)
            elif prediction_type == 'point_spread':
                # Example: Regression target based on yards per game
                df['target'] = df['yards_per_game']
            else:
                raise ValueError(f"Unsupported prediction type: {prediction_type}")
            
            # Select features
            X = self._select_features(df.drop('target', axis=1), df['target'])
            y = df['target'].values
            
            return {
                'features': X.values,
                'target': y,
                'feature_names': X.columns.tolist(),
                'team_mapping': dict(zip(
                    self.team_encoder.classes_, 
                    self.team_encoder.transform(self.team_encoder.classes_)
                )),
                'position_mapping': dict(zip(
                    self.position_encoder.classes_, 
                    self.position_encoder.transform(self.position_encoder.classes_)
                ))
            }
        
        except Exception as e:
            self.logger.error(f"Error preparing prediction dataset: {e}")
            raise DataPreprocessingError(f"Dataset preparation failed: {e}")