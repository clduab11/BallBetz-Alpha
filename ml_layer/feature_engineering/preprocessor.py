import os
import pandas as pd
import numpy as np
from typing import Union, Dict, Any, List, Tuple
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

from .config import FeatureEngineeringConfig
from .exceptions import DataPreprocessingError, FeatureSelectionError

class UFLFeaturePreprocessor:
    """
    Preprocessor for UFL (University Football League) statistical data
    
    Handles data cleaning, missing value imputation, 
    feature scaling, and encoding.
    """
    
    def __init__(self, config: FeatureEngineeringConfig = None):
        """
        Initialize the preprocessor with optional configuration
        
        :param config: Feature engineering configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or FeatureEngineeringConfig.from_env()
        self.logger.info("UFLFeaturePreprocessor initialized")
    
    def _identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numeric and categorical columns in the dataset
        
        :param df: Input DataFrame
        :return: Tuple of numeric and categorical column lists
        """
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        return numeric_columns, categorical_columns
    
    def _create_preprocessing_pipeline(
        self, 
        numeric_columns: List[str], 
        categorical_columns: List[str]
    ) -> Pipeline:
        """
        Create a preprocessing pipeline for the dataset
        
        :param numeric_columns: List of numeric column names
        :param categorical_columns: List of categorical column names
        :return: Scikit-learn preprocessing pipeline
        """
        try:
            # Numeric feature preprocessing
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.config.numeric_imputation_strategy)),
                ('scaler', StandardScaler() if self.config.scale_numeric_features else 'passthrough')
            ])
            
            # Categorical feature preprocessing
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=self.config.categorical_imputation_strategy)),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_columns),
                    ('cat', categorical_transformer, categorical_columns)
                ])
            
            return preprocessor
        
        except Exception as e:
            self.logger.error(f"Error creating preprocessing pipeline: {e}")
            raise DataPreprocessingError(f"Pipeline creation failed: {e}")
    
    def preprocess(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        target_column: str = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input data
        
        :param data: Input data as DataFrame or NumPy array
        :param target_column: Optional target column name for feature selection
        :return: Preprocessed features and optional target
        """
        try:
            # Convert to DataFrame if not already
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
            # Identify column types
            numeric_columns, categorical_columns = self._identify_column_types(data)
            
            # Create preprocessing pipeline
            preprocessor = self._create_preprocessing_pipeline(
                numeric_columns, 
                categorical_columns
            )
            
            # Separate features and target if target_column provided
            if target_column:
                X = data.drop(columns=[target_column])
                y = data[target_column]
            else:
                X = data
                y = None
            
            # Fit and transform data
            X_transformed = preprocessor.fit_transform(X)
            
            # Optional feature selection
            if target_column and y is not None:
                selector = SelectKBest(
                    score_func=f_classif, 
                    k=min(self.config.max_features_to_select, X_transformed.shape[1])
                )
                X_transformed = selector.fit_transform(X_transformed, y)
            
            return X_transformed, y
        
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise DataPreprocessingError(f"Data preprocessing error: {e}")
    
    def log_preprocessing_details(self, X_transformed: np.ndarray):
        """
        Log details about the preprocessed data
        
        :param X_transformed: Preprocessed feature matrix
        """
        if self.config.verbose:
            self.logger.info(f"Preprocessed data shape: {X_transformed.shape}")
            self.logger.info(f"Preprocessed data type: {X_transformed.dtype}")