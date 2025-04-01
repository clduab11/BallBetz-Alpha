class DataPreprocessingError(Exception):
    """
    Exception raised for errors during data preprocessing
    
    This exception is raised when there are issues with data cleaning,
    transformation, or preparation for machine learning.
    """
    pass

class FeatureSelectionError(Exception):
    """
    Exception raised for errors during feature selection
    
    This exception is raised when there are issues with selecting
    relevant features for machine learning models.
    """
    pass

class DataIntegrationError(Exception):
    """
    Exception raised for errors during data integration
    
    This exception is raised when there are issues with combining
    data from multiple sources or formats.
    """
    pass

class FeatureEngineeringConfigError(Exception):
    """
    Exception raised for errors in feature engineering configuration
    
    This exception is raised when there are issues with the configuration
    parameters for feature engineering processes.
    """
    pass