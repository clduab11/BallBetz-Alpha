import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from models.model_interface import SklearnModel

@pytest.fixture
def mock_sklearn_model():
    mock = MagicMock()
    mock.predict.return_value = np.array([15.0, 20.0, 25.0])  # Sample predictions
    mock.estimators_ = [MagicMock()]  # Required for scikit-learn style models
    return mock

def test_sklearn_model_predict(mock_sklearn_model):
    # Mock scaler and dependencies
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.array([[0.1], [0.2], [0.3]])
    
    # Instantiate model with mocked components
    model = SklearnModel(
        model=mock_sklearn_model,
        scaler=mock_scaler,
        feature_columns=['feature1', 'passing_yards', 'rushing_yards', 'receiving_yards']
    )
    
    # Mock prediction intervals to match test data length
    model.get_prediction_intervals = MagicMock(
        return_value=np.array([[14.5, 15.5], [19.5, 20.5], [24.5, 25.5]])
    )
    
    # Test input data
    test_data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'passing_yards': [100, 200, 300],
        'rushing_yards': [10, 20, 30],
        'receiving_yards': [1, 2, 3]
    })
    
    # Execute prediction
    predictions = model.predict(test_data)
    
    # Validate output
    assert isinstance(predictions, pd.DataFrame)
    assert predictions.shape == (3, 3)
    assert list(predictions.columns) == ['predicted_points', 'lower_bound', 'upper_bound']
    assert np.allclose(predictions['predicted_points'], [15.0, 20.0, 25.0])