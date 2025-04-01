"""Cross-League Pattern Analysis Module.

This module provides functionality for recognizing and scoring patterns
across different leagues and historical data.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .config import CloudAILayerConfig
from .exceptions import PatternAnalysisError

class CrossLeaguePatternAnalyzer:
    """
    Analyzes and scores patterns across different leagues and historical data.
    
    Provides methods for pattern recognition, similarity scoring, 
    and historical pattern matching.
    """

    def __init__(self, config: CloudAILayerConfig = None):
        """
        Initialize the Cross-League Pattern Analyzer.
        
        Args:
            config: Configuration for pattern analysis
        """
        self.config = config or CloudAILayerConfig
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.LOGGING_LEVEL)
        
        # In-memory pattern cache
        self._pattern_cache: Dict[str, Dict[str, Any]] = {}

    def vectorize_league_data(self, league_data: Dict[str, Any]) -> np.ndarray:
        """
        Convert league data into a numerical vector for comparison.
        
        Args:
            league_data: Dictionary of league-specific features
        
        Returns:
            Numerical vector representation of league data
        """
        try:
            # Extract key numerical features
            features = [
                league_data.get('win_rate', 0),
                league_data.get('avg_score', 0),
                league_data.get('home_advantage', 0),
                league_data.get('team_strength', 0)
            ]
            return np.array(features)
        except Exception as e:
            self.logger.error(f"Vectorization failed: {e}")
            raise PatternAnalysisError("Failed to vectorize league data") from e

    def calculate_league_similarity(self, 
                                    league1_data: Dict[str, Any], 
                                    league2_data: Dict[str, Any]) -> float:
        """
        Calculate similarity score between two leagues.
        
        Args:
            league1_data: Data for the first league
            league2_data: Data for the second league
        
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Vectorize league data
            vec1 = self.vectorize_league_data(league1_data)
            vec2 = self.vectorize_league_data(league2_data)
            
            # Compute cosine similarity
            similarity = cosine_similarity([vec1], [vec2])[0][0]
            
            # Apply configured threshold
            return max(0, min(1, similarity)) if similarity > self.config.CROSS_LEAGUE_SIMILARITY_THRESHOLD else 0
        except Exception as e:
            self.logger.warning(f"Similarity calculation failed: {e}")
            return 0

    def find_historical_patterns(self, 
                                 current_league_data: Dict[str, Any], 
                                 historical_leagues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find historical leagues with similar patterns.
        
        Args:
            current_league_data: Data for the current league
            historical_leagues: List of historical league data
        
        Returns:
            List of similar historical league patterns
        """
        similar_patterns = []
        
        for league in historical_leagues[:self.config.MAX_HISTORICAL_PATTERNS]:
            similarity_score = self.calculate_league_similarity(current_league_data, league)
            
            if similarity_score > 0:
                similar_patterns.append({
                    'league_data': league,
                    'similarity_score': similarity_score
                })
        
        # Sort patterns by similarity score in descending order
        return sorted(similar_patterns, key=lambda x: x['similarity_score'], reverse=True)

    def cache_league_pattern(self, league_id: str, pattern_data: Dict[str, Any]) -> None:
        """
        Cache league pattern for future reference.
        
        Args:
            league_id: Unique identifier for the league
            pattern_data: Pattern data to cache
        """
        try:
            self._pattern_cache[league_id] = {
                'data': pattern_data,
                'timestamp': np.datetime64('now')
            }
            
            # Prune cache if it exceeds max size
            if len(self._pattern_cache) > self.config.MAX_HISTORICAL_PATTERNS:
                oldest_key = min(self._pattern_cache, key=lambda k: self._pattern_cache[k]['timestamp'])
                del self._pattern_cache[oldest_key]
        except Exception as e:
            self.logger.warning(f"Pattern caching failed: {e}")

    def get_cached_pattern(self, league_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached pattern for a league.
        
        Args:
            league_id: Unique identifier for the league
        
        Returns:
            Cached pattern data or None if not found
        """
        return self._pattern_cache.get(league_id, {}).get('data')

    def analyze_cross_league_patterns(self, 
                                      current_league_data: Dict[str, Any], 
                                      historical_leagues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive cross-league pattern analysis.
        
        Args:
            current_league_data: Data for the current league
            historical_leagues: List of historical league data
        
        Returns:
            Analysis results with patterns and insights
        """
        try:
            # Find similar historical patterns
            similar_patterns = self.find_historical_patterns(current_league_data, historical_leagues)
            
            # Cache current league pattern
            self.cache_league_pattern(current_league_data.get('league_id', 'unknown'), current_league_data)
            
            return {
                'current_league': current_league_data,
                'similar_patterns': similar_patterns,
                'total_similar_leagues': len(similar_patterns)
            }
        except Exception as e:
            self.logger.error(f"Cross-league pattern analysis failed: {e}")
            raise PatternAnalysisError("Comprehensive pattern analysis failed") from e