# Usage Documentation

This guide provides detailed instructions on how to use the BallBetz-Alpha Triple-Layer Prediction Engine in various contexts, including API endpoints, code examples, and integration patterns.

## Basic Usage

The Triple-Layer Prediction Engine can be used in several ways:

1. Through the BallBetz-Alpha web application
2. Via direct API calls
3. Through programmatic integration in Python code

### Web Application Usage

The BallBetz-Alpha web application provides a user-friendly interface for generating predictions:

1. Navigate to the BallBetz-Alpha web application
2. Log in with your credentials
3. Select "Generate Lineup" from the main menu
4. Choose your platform, maximum lineups, and minimum salary
5. Click "Generate" to receive predictions and optimized lineups

### API Endpoints

The Triple-Layer Prediction Engine exposes several API endpoints:

#### Generate Lineup Endpoint

```
POST /generate_lineup
```

**Request Parameters:**
- `platform` (string): The fantasy sports platform (e.g., "draftkings", "fanduel")
- `max_lineups` (integer): Maximum number of lineups to generate
- `min_salary` (integer, optional): Minimum salary to use (default: 0)

**Example Request:**
```json
{
  "platform": "draftkings",
  "max_lineups": 5,
  "min_salary": 45000
}
```

**Example Response:**
```json
{
  "lineups": [
    {
      "players": [
        {"name": "Player 1", "position": "QB", "team": "Team A", "salary": 8000, "projected_points": 25.7},
        {"name": "Player 2", "position": "RB", "team": "Team B", "salary": 7500, "projected_points": 18.3},
        // Additional players...
      ],
      "total_salary": 49500,
      "projected_points": 145.2
    },
    // Additional lineups...
  ],
  "prediction_details": {
    "ml_layer_contribution": 0.4,
    "transformer_layer_contribution": 0.3,
    "cloud_layer_contribution": 0.3
  }
}
```

#### Player Prediction Endpoint

```
POST /player_prediction
```

**Request Parameters:**
- `player_id` (string): The unique identifier for the player
- `prediction_type` (string): Type of prediction (e.g., "fantasy_points", "passing_yards")
- `include_details` (boolean, optional): Whether to include detailed breakdown (default: false)

**Example Request:**
```json
{
  "player_id": "UFL12345",
  "prediction_type": "fantasy_points",
  "include_details": true
}
```

**Example Response:**
```json
{
  "player_id": "UFL12345",
  "name": "John Smith",
  "position": "QB",
  "team": "Birmingham Stallions",
  "prediction": {
    "value": 22.5,
    "confidence": 0.85,
    "range": [18.7, 26.3]
  },
  "details": {
    "ml_prediction": 21.8,
    "transformer_prediction": 23.2,
    "external_factors": {
      "weather_impact": -0.5,
      "injury_impact": 0.0,
      "matchup_impact": 1.2
    },
    "cross_league_patterns": {
      "nfl_similarity": 0.7,
      "ncaa_similarity": 0.8
    }
  }
}
```

## Programmatic Usage

### Python Integration

To use the Triple-Layer Prediction Engine in your Python code:

```python
from triple_layer_prediction_engine import PredictionEngine
from triple_layer_prediction_engine.config import load_config

# Load configuration
config = load_config()

# Initialize the prediction engine
engine = PredictionEngine(config)

# Generate predictions for a player
player_data = {
    "player_id": "UFL12345",
    "name": "John Smith",
    "position": "QB",
    "team": "Birmingham Stallions",
    "opponent": "Michigan Panthers",
    "home_game": True,
    "stats": {
        "passing_yards_avg": 285.3,
        "passing_tds_avg": 2.1,
        "interceptions_avg": 0.7,
        "rushing_yards_avg": 32.5,
        "rushing_tds_avg": 0.3
    }
}

# Get prediction
prediction = engine.predict(player_data, prediction_type="fantasy_points")

print(f"Predicted fantasy points: {prediction['value']}")
print(f"Confidence: {prediction['confidence']}")
print(f"Range: {prediction['range']}")

# Get detailed breakdown
details = engine.get_prediction_details(player_data, prediction_type="fantasy_points")
print(f"ML Layer contribution: {details['ml_contribution']}")
print(f"Transformer Layer contribution: {details['transformer_contribution']}")
print(f"External factors contribution: {details['external_contribution']}")
print(f"Cross-league contribution: {details['cross_league_contribution']}")
```

### Batch Predictions

For batch predictions of multiple players:

```python
import pandas as pd
from triple_layer_prediction_engine import PredictionEngine
from triple_layer_prediction_engine.config import load_config

# Load player data
player_data = pd.read_csv("player_data.csv")

# Initialize the prediction engine
config = load_config()
engine = PredictionEngine(config)

# Generate batch predictions
predictions = engine.predict_batch(player_data, prediction_type="fantasy_points")

# Save predictions to CSV
predictions.to_csv("player_predictions.csv", index=False)
```

## Advanced Usage

### Customizing Prediction Weights

You can customize the weights used for combining predictions from different layers:

```python
from triple_layer_prediction_engine import PredictionEngine
from triple_layer_prediction_engine.config import load_config

# Load configuration
config = load_config()

# Customize weights
custom_weights = {
    "ml": 0.5,            # Increase ML layer weight
    "transformer": 0.3,    # Keep transformer layer weight
    "external": 0.1,       # Decrease external factors weight
    "cross_league": 0.1    # Keep cross-league weight
}

# Initialize the prediction engine with custom weights
engine = PredictionEngine(config, weights=custom_weights)

# Generate predictions with custom weights
prediction = engine.predict(player_data, prediction_type="fantasy_points")
```

### Using Different Transformer Models

You can specify which transformer model to use for predictions:

```python
from triple_layer_prediction_engine import PredictionEngine
from triple_layer_prediction_engine.config import load_config

# Load configuration
config = load_config()

# Initialize the prediction engine
engine = PredictionEngine(config)

# Generate predictions with a specific transformer model
prediction = engine.predict(
    player_data, 
    prediction_type="fantasy_points",
    transformer_model="gpt-4"  # Specify a different model
)
```

### Accessing Layer-Specific Predictions

You can access predictions from individual layers:

```python
from triple_layer_prediction_engine import PredictionEngine
from triple_layer_prediction_engine.config import load_config

# Load configuration
config = load_config()

# Initialize the prediction engine
engine = PredictionEngine(config)

# Get ML Layer prediction only
ml_prediction = engine.predict_ml_layer(player_data, prediction_type="fantasy_points")

# Get Transformer Layer prediction only
transformer_prediction = engine.predict_transformer_layer(player_data, prediction_type="fantasy_points")

# Get Cloud Layer prediction only
cloud_prediction = engine.predict_cloud_layer(player_data, prediction_type="fantasy_points")
```

## Error Handling

The Triple-Layer Prediction Engine provides comprehensive error handling:

```python
from triple_layer_prediction_engine import PredictionEngine
from triple_layer_prediction_engine.exceptions import (
    MLLayerError,
    TransformerLayerError,
    CloudLayerError,
    PredictionEngineError
)

# Initialize the prediction engine
engine = PredictionEngine(config)

try:
    prediction = engine.predict(player_data, prediction_type="fantasy_points")
except MLLayerError as e:
    print(f"ML Layer error: {str(e)}")
    # Handle ML Layer error
except TransformerLayerError as e:
    print(f"Transformer Layer error: {str(e)}")
    # Handle Transformer Layer error
except CloudLayerError as e:
    print(f"Cloud Layer error: {str(e)}")
    # Handle Cloud Layer error
except PredictionEngineError as e:
    print(f"General prediction error: {str(e)}")
    # Handle general prediction error
```

## Performance Considerations

For optimal performance when using the Triple-Layer Prediction Engine:

1. **Batch Processing**: Use batch predictions when processing multiple players
2. **Caching**: Enable caching to reduce redundant API calls
3. **Layer Selection**: Disable layers that aren't needed for your use case
4. **Local Models**: Use local transformer models when possible to reduce API costs

## Example Use Cases

### Fantasy Sports Lineup Optimization

```python
from triple_layer_prediction_engine import PredictionEngine
from lineup_optimizer import LineupOptimizer

# Initialize the prediction engine
engine = PredictionEngine(config)

# Get player data
players = get_available_players()

# Generate predictions for all players
predictions = engine.predict_batch(players, prediction_type="fantasy_points")

# Initialize the lineup optimizer
optimizer = LineupOptimizer(platform="draftkings", sport="ufl")

# Generate optimized lineups
lineups = optimizer.optimize(
    players=players,
    predictions=predictions,
    max_lineups=5,
    min_salary=45000
)

# Display lineups
for i, lineup in enumerate(lineups):
    print(f"Lineup {i+1}:")
    print(f"Projected Points: {lineup['projected_points']}")
    print(f"Total Salary: ${lineup['total_salary']}")
    for player in lineup['players']:
        print(f"  {player['position']} - {player['name']} (${player['salary']}, {player['projected_points']} pts)")
```

### Game Outcome Prediction

```python
from triple_layer_prediction_engine import PredictionEngine

# Initialize the prediction engine
engine = PredictionEngine(config)

# Game data
game_data = {
    "home_team": "Birmingham Stallions",
    "away_team": "Michigan Panthers",
    "date": "2025-04-15",
    "venue": "Protective Stadium",
    "weather": "Clear, 72°F"
}

# Predict game outcome
outcome = engine.predict_game_outcome(game_data)

print(f"Predicted winner: {outcome['winner']}")
print(f"Win probability: {outcome['win_probability']}")
print(f"Predicted score: {outcome['home_score']} - {outcome['away_score']}")
print(f"Over/under: {outcome['over_under']}")
print(f"Spread: {outcome['spread']}")
```

## Installation and Source Code

The Triple-Layer Prediction Engine is part of the BallBetz-Alpha project. You can find the source code and installation instructions at:

[https://github.com/clduab11/BallBetz-Alpha](https://github.com/clduab11/BallBetz-Alpha)

For installation instructions, see the [Installation Guide](installation.md).

## Next Steps

After mastering the basic usage of the Triple-Layer Prediction Engine:

1. Explore the [ML Layer Documentation](ml_layer.md) for details on the statistical models
2. Learn about the [API/Local AI Layer](api_ai_layer.md) for transformer model integration
3. Understand the [Cloud AI Layer](cloud_ai_layer.md) for external factor integration
4. Review the [Security Documentation](security.md) for best practices
5. Consult the [Maintenance Guide](maintenance.md) for operational considerations
