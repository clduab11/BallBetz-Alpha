class ModelLoadError(Exception):
    """
    Exception raised for errors when loading models
    
    This exception is raised when there are issues with loading
    a model from a checkpoint or other storage.
    """
    pass

class InferenceFailed(Exception):
    """
    Exception raised for errors during inference
    
    This exception is raised when there are issues with generating
    predictions using a loaded model.
    """
    pass

class ConfidenceThresholdError(Exception):
    """
    Exception raised for errors with confidence thresholds
    
    This exception is raised when no predictions meet the
    configured confidence threshold.
    """
    pass

class PredictionConfigError(Exception):
    """
    Exception raised for errors in prediction configuration
    
    This exception is raised when there are issues with the configuration
    parameters for prediction processes.
    """
    pass

class EnsemblePredictionError(Exception):
    """
    Exception raised for errors with ensemble predictions
    
    This exception is raised when there are issues with combining
    predictions from multiple models.
    """
    pass

class OutputFormatError(Exception):
    """
    Exception raised for errors with output formatting
    
    This exception is raised when there are issues with formatting
    prediction outputs in the requested format.
    """
    pass