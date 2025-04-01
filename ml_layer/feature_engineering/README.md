# UFL Feature Engineering Module

## Overview
This module provides robust feature preprocessing capabilities for University Football League (UFL) statistical data. It handles data cleaning, imputation, scaling, and feature selection with high configurability.

## Features
- Automatic column type detection
- Configurable imputation strategies
- Optional feature scaling
- One-hot encoding for categorical variables
- Flexible feature selection
- Environment variable configuration

## Configuration

The module uses environment variables for configuration. Available options:

| Environment Variable           | Description                           | Default Value   |
|--------------------------------|---------------------------------------|-----------------|
| `UFL_NUMERIC_IMPUTATION`       | Numeric column imputation strategy    | 'mean'          |
| `UFL_CATEGORICAL_IMPUTATION`   | Categorical column imputation strategy| 'most_frequent' |
| `UFL_SCALE_NUMERIC`            | Scale numeric features                | 'true'          |
| `UFL_MAX_FEATURES`             | Maximum number of features to select  | 50              |
| `UFL_VERBOSE_PREPROCESSING`    | Enable verbose logging                | 'false'         |

## Usage Example

```python
import pandas as pd
from ml_layer.feature_engineering import UFLFeaturePreprocessor, FeatureEngineeringConfig

# Load your UFL dataset
data = pd.read_csv('ufl_stats.csv')

# Create preprocessor with default configuration
preprocessor = UFLFeaturePreprocessor()

# Preprocess data with target column
X_processed, y = preprocessor.preprocess(
    data, 
    target_column='game_outcome'
)

# Log preprocessing details
preprocessor.log_preprocessing_details(X_processed)
```

## Error Handling

The module provides custom exceptions for different preprocessing scenarios:
- `FeatureEngineeringError`: Base exception
- `ConfigurationError`: Configuration validation errors
- `DataPreprocessingError`: Data preprocessing failures
- `FeatureSelectionError`: Feature selection issues

## Best Practices

1. Always validate your input data before preprocessing
2. Use environment variables for flexible configuration
3. Handle potential exceptions during preprocessing
4. Review preprocessing logs for insights

## Dependencies
- pandas
- numpy
- scikit-learn

## License
[Specify your project's license]