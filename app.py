from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import os
import json
from datetime import datetime
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, Optional
from functools import wraps

from scrapers.ufl_scraper import UFLScraper
from utils.data_processor import DataProcessor
from models.predictor import PlayerPerformancePredictor
from optimizers.lineup_optimizer import LineupOptimizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ballbetz_diagnostics.log')
    ]
)
logger = logging.getLogger(__name__)

# Diagnostic: Check for required dependencies
logger.info("=== BallBetz Diagnostic Logs ===")
try:
    pulp_spec = importlib.util.find_spec('pulp')
    logger.info(f"PuLP dependency check: {'FOUND' if pulp_spec else 'MISSING'}")
except ImportError:
    logger.error("PuLP dependency check: MISSING - ImportError")

# Log Python and key package versions
logger.info(f"Python version: {importlib.sys.version}")

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-secret-key')  # Change in production

# Initialize components
try:
    scraper = UFLScraper()
    processor = DataProcessor()
    predictor = PlayerPerformancePredictor('models/ufl_predictor.joblib')
    dk_optimizer = LineupOptimizer('draftkings')
    fd_optimizer = LineupOptimizer('fanduel')
    logger.info("All components initialized successfully")
    # Check if model file exists
    logger.info(f"Model file exists: {Path('models/ufl_predictor.joblib').exists()}")
    
except Exception as e:
    logger.error(f"Error initializing components: {str(e)}")
    raise

# Constants
BANKROLL_FILE = Path('data/bankroll.json')
DATA_DIR = Path('data')
REQUIRED_DIRS = ['data/raw', 'data/processed', 'models']

def setup_directories() -> None:
    """Create required directories if they don't exist."""
    for dir_path in REQUIRED_DIRS:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def handle_errors(f):
    """Decorator to handle errors in routes."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            flash(f"An error occurred: {str(e)}", 'error')
            return jsonify({'success': False, 'message': str(e)}), 500
    return wrapper

def load_bankroll() -> Dict[str, Any]:
    """
    Load bankroll data from file.
    
    Returns:
        Dict containing bankroll information
    """
    try:
        if BANKROLL_FILE.exists():
            with open(BANKROLL_FILE) as f:
                return json.load(f)
        return {
            'initial': 20,
            'current': 20,
            'history': [],
            'last_updated': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error loading bankroll: {str(e)}")
        return {
            'initial': 20,
            'current': 20,
            'history': [],
            'last_updated': datetime.now().isoformat()
        }

def save_bankroll(bankroll: Dict[str, Any]) -> bool:
    """
    Save bankroll data to file.
    
    Args:
        bankroll: Dictionary containing bankroll data
        
    Returns:
        bool: True if save was successful
    """
    try:
        BANKROLL_FILE.parent.mkdir(parents=True, exist_ok=True)
        bankroll['last_updated'] = datetime.now().isoformat()
        with open(BANKROLL_FILE, 'w') as f:
            json.dump(bankroll, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving bankroll: {str(e)}")
        return False

@app.route('/')
@handle_errors
def index():
    """Render the main dashboard page."""
    bankroll = load_bankroll()
    
    # Calculate performance metrics
    total_profit = sum(entry['profit'] for entry in bankroll['history'])
    total_contests = len(bankroll['history'])
    win_rate = sum(1 for entry in bankroll['history'] if entry['profit'] > 0) / total_contests if total_contests > 0 else 0
    
    return render_template('index.html',
                         bankroll=bankroll,
                         total_profit=total_profit,
                         total_contests=total_contests,
                         win_rate=win_rate)

@app.route('/update_data', methods=['POST'])
@handle_errors
def update_data():
    """Update player data and process it."""
    # Scrape latest data
    player_data = scraper.get_player_stats()
    if player_data.empty:
        raise ValueError("No player data retrieved")
    
    # Process data
    processed_data = processor.clean_player_data(player_data)
    if processed_data.empty:
        raise ValueError("Error processing player data")
    
    # Save to file
    save_path = DATA_DIR / 'processed/latest_player_data.csv'
    processed_data.to_csv(save_path, index=False)
    
    return jsonify({
        'success': True,
        'message': 'Data updated successfully',
        'player_count': len(processed_data),
        'last_updated': datetime.now().isoformat()
    })

@app.route('/generate_lineup', methods=['POST'])
@handle_errors
def generate_lineup():
    """Generate optimal lineup based on form parameters."""
    logger.info("=== Generating Lineup - Parameter Diagnostics ===")
    # Validate input parameters
    platform = request.form.get('platform', 'draftkings').lower()
    max_lineups = int(request.form.get('max_lineups', 1))
    min_salary = float(request.form.get('min_salary', 0))
    logger.info(f"Received parameters - platform: {platform}, max_lineups: {max_lineups}, min_salary: {min_salary}")
    
    if platform not in ['draftkings', 'fanduel']:
        raise ValueError(f"Invalid platform: {platform}")
    
    # Load and validate player data
    try:
        player_data = pd.read_csv(DATA_DIR / 'processed/latest_player_data.csv')
        logger.info(f"Loaded player data: {len(player_data)} records")
    except FileNotFoundError:
        raise ValueError("No player data available. Please update data first.")
    
    # Get fantasy prices
    prices = scraper.get_fantasy_prices(platform)
    if not prices.empty:
        player_data = player_data.merge(prices, on='name', how='left')
        logger.info(f"Merged fantasy prices: {len(prices)} records")
    else:
        logger.warning("Fantasy prices data is empty - using placeholder data")
        # Add placeholder salary for testing
        if 'salary' not in player_data.columns:
            player_data['salary'] = 5000
            logger.info("Added placeholder salary data for testing")
    
    # Generate predictions
    player_data = predictor.predict(player_data)
    logger.info(f"Generated predictions for {len(player_data)} players")
    
    # Optimize lineup
    optimizer = dk_optimizer if platform == 'draftkings' else fd_optimizer
    lineup = optimizer.optimize(  # Diagnostic: Check parameter signature
        player_data,
        max_lineups=max_lineups,
        min_salary=min_salary
    )
    
    if lineup.empty:
        logger.error("Generated lineup is empty")
        raise ValueError("Could not generate valid lineup")
    
    # Format lineup for display
    result = lineup.to_dict(orient='records')
    for player in result:
        player['salary'] = f"${player['salary']:,}"
        player['predicted_points'] = f"{player['predicted_points']:.2f}"
    
    return jsonify({
        'success': True,
        'lineup': result,
        'metadata': {
            'platform': platform,
            'generated_at': datetime.now().isoformat(),
            'lineup_count': len(lineup['lineup_number'].unique())
        }
    })

@app.route('/update_bankroll', methods=['POST'])
@handle_errors
def update_bankroll():
    """Update bankroll with contest results."""
    # Validate input
    try:
        contest_date = datetime.strptime(
            request.form.get('date', datetime.now().strftime('%Y-%m-%d')),
            '%Y-%m-%d'
        )
        platform = request.form.get('platform', 'draftkings')
        entry_fee = float(request.form.get('entry_fee', 0))
        winnings = float(request.form.get('winnings', 0))
    except ValueError as e:
        raise ValueError(f"Invalid input parameters: {str(e)}")
    
    # Calculate results
    profit = winnings - entry_fee
    roi = (profit / entry_fee * 100) if entry_fee > 0 else 0
    
    # Update bankroll
    bankroll = load_bankroll()
    bankroll['current'] += profit
    bankroll['history'].append({
        'date': contest_date.strftime('%Y-%m-%d'),
        'platform': platform,
        'entry_fee': entry_fee,
        'winnings': winnings,
        'profit': profit,
        'roi': roi,
        'balance': bankroll['current']
    })
    
    if not save_bankroll(bankroll):
        raise ValueError("Failed to save bankroll update")
    
    return jsonify({
        'success': True,
        'bankroll': bankroll,
        'update_metadata': {
            'profit': profit,
            'roi': roi,
            'new_balance': bankroll['current']
        }
    })

@app.route('/analysis')
@handle_errors
def analysis():
    """Render the analysis page with performance metrics."""
    bankroll = load_bankroll()
    
    # Calculate performance metrics
    total_invested = sum(entry['entry_fee'] for entry in bankroll['history'])
    total_won = sum(entry['winnings'] for entry in bankroll['history'])
    total_profit = total_won - total_invested
    roi = (total_profit / total_invested * 100) if total_invested > 0 else 0
    
    # Calculate platform-specific metrics
    platform_stats = {}
    for platform in ['draftkings', 'fanduel']:
        platform_entries = [e for e in bankroll['history'] if e['platform'] == platform]
        if platform_entries:
            platform_stats[platform] = {
                'total_contests': len(platform_entries),
                'total_profit': sum(e['profit'] for e in platform_entries),
                'avg_roi': sum(e['roi'] for e in platform_entries) / len(platform_entries)
            }
    
    return render_template('analysis.html',
                         bankroll=bankroll,
                         total_invested=total_invested,
                         total_won=total_won,
                         total_profit=total_profit,
                         roi=roi,
                         platform_stats=platform_stats)

if __name__ == '__main__':
    setup_directories()
    logger.info("BallBetz application starting")
    app.run(debug=True)