class ModelTrainingError(Exception):
    """
    Exception raised for errors during model training
    
    This exception is raised when there are issues with the training
    process, such as convergence problems or incompatible data.
    """
    pass

class DataPreparationError(Exception):
    """
    Exception raised for errors during data preparation
    
    This exception is raised when there are issues with preparing
    data for training, such as splitting or scaling problems.
    """
    pass

class CheckpointError(Exception):
    """
    Exception raised for errors with model checkpoints
    
    This exception is raised when there are issues with saving,
    loading, or managing model checkpoints.
    """
    pass

class TrainingConfigError(Exception):
    """
    Exception raised for errors in training configuration
    
    This exception is raised when there are issues with the configuration
    parameters for training processes.
    """
    pass

class EarlyStoppingError(Exception):
    """
    Exception raised for errors with early stopping
    
    This exception is raised when there are issues with the early
    stopping mechanism during training.
    """
    pass