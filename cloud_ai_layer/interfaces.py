"""Interfaces for Cloud AI Layer integration.

Defines abstract base classes and interfaces for interaction
with ML Layer, API/Local AI Layer, and external systems.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import logging
from enum import Enum, auto

class PredictionSourceType(Enum):
    """Enumeration of prediction source types."""
    ML_MODEL = auto()
    API_AI = auto()
    EXTERNAL_FACTOR = auto()
    CROSS_LEAGUE_PATTERN = auto()

@dataclass
class PredictionRequest:
    """
    Standardized prediction request structure for cross-layer communication.
    
    Attributes:
        context: Contextual information for the prediction
        metadata: Additional metadata about the prediction request
        source_type: Type of prediction source
    """
    context: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    source_type: PredictionSourceType = PredictionSourceType.ML_MODEL

@dataclass
class PredictionResult:
    """
    Standardized prediction result structure.
    
    Attributes:
        prediction: Numerical prediction value
        confidence: Confidence score of the prediction
        explanation: Detailed explanation of the prediction
        source_type: Source type of the prediction
        raw_data: Raw prediction data for further analysis
    """
    prediction: float
    confidence: float
    explanation: Optional[Dict[str, Any]] = None
    source_type: PredictionSourceType = PredictionSourceType.ML_MODEL
    raw_data: Optional[Dict[str, Any]] = None

class CloudAILayerInterface(ABC):
    """
    Abstract base class defining the interface for the Cloud AI Layer.
    
    Provides a standardized contract for prediction generation,
    cross-layer communication, and result processing.
    """

    def __init__(self):
        """Initialize logging for the interface."""
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def process_prediction_request(self, request: PredictionRequest) -> PredictionResult:
        """
        Process a prediction request from another layer.
        
        Args:
            request: Standardized prediction request
        
        Returns:
            Standardized prediction result
        """
        pass

    @abstractmethod
    def integrate_layer_predictions(self, 
                                    predictions: List[PredictionResult]) -> PredictionResult:
        """
        Integrate predictions from multiple sources.
        
        Args:
            predictions: List of prediction results from different sources
        
        Returns:
            Integrated and weighted final prediction
        """
        pass

    @abstractmethod
    def validate_prediction_request(self, request: PredictionRequest) -> bool:
        """
        Validate the incoming prediction request.
        
        Args:
            request: Prediction request to validate
        
        Returns:
            Boolean indicating request validity
        """
        pass

class LayerCommunicationProtocol(ABC):
    """
    Abstract base class defining communication protocols between layers.
    
    Provides methods for secure and standardized inter-layer communication.
    """

    @abstractmethod
    def serialize_prediction_request(self, request: PredictionRequest) -> str:
        """
        Serialize a prediction request for transmission.
        
        Args:
            request: Prediction request to serialize
        
        Returns:
            Serialized representation of the request
        """
        pass

    @abstractmethod
    def deserialize_prediction_request(self, serialized_request: str) -> PredictionRequest:
        """
        Deserialize a prediction request from transmission format.
        
        Args:
            serialized_request: Serialized prediction request
        
        Returns:
            Deserialized prediction request
        """
        pass

    @abstractmethod
    def encrypt_prediction_data(self, data: Union[PredictionRequest, PredictionResult]) -> str:
        """
        Encrypt prediction-related data for secure transmission.
        
        Args:
            data: Prediction request or result to encrypt
        
        Returns:
            Encrypted data string
        """
        pass

    @abstractmethod
    def decrypt_prediction_data(self, encrypted_data: str) -> Union[PredictionRequest, PredictionResult]:
        """
        Decrypt prediction-related data received from another layer.
        
        Args:
            encrypted_data: Encrypted prediction data
        
        Returns:
            Decrypted prediction request or result
        """
        pass

def create_prediction_request(
    context: Dict[str, Any], 
    source_type: PredictionSourceType = PredictionSourceType.ML_MODEL,
    metadata: Optional[Dict[str, Any]] = None
) -> PredictionRequest:
    """
    Factory function to create a standardized prediction request.
    
    Args:
        context: Contextual information for the prediction
        source_type: Type of prediction source
        metadata: Additional metadata about the prediction request
    
    Returns:
        Standardized PredictionRequest instance
    """
    return PredictionRequest(
        context=context,
        source_type=source_type,
        metadata=metadata or {}
    )

def create_prediction_result(
    prediction: float,
    confidence: float,
    source_type: PredictionSourceType = PredictionSourceType.ML_MODEL,
    explanation: Optional[Dict[str, Any]] = None,
    raw_data: Optional[Dict[str, Any]] = None
) -> PredictionResult:
    """
    Factory function to create a standardized prediction result.
    
    Args:
        prediction: Numerical prediction value
        confidence: Confidence score of the prediction
        source_type: Source type of the prediction
        explanation: Detailed explanation of the prediction
        raw_data: Raw prediction data for further analysis
    
    Returns:
        Standardized PredictionResult instance
    """
    return PredictionResult(
        prediction=prediction,
        confidence=confidence,
        source_type=source_type,
        explanation=explanation or {},
        raw_data=raw_data or {}
    )