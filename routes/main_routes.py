"""Main routes for BallBetz-Alpha.

This module handles the main routes of the application, including the index page
and analysis page.
"""

import logging
import json
from datetime import datetime
import pandas as pd
from pathlib import Path
from flask import render_template, jsonify, flash
from flask_login import login_required

from . import main_bp
from utils.error_handlers import handle_errors

# Set up logging
logger = logging.getLogger(__name__)

# Constants
BANKROLL_FILE = Path('data/bankroll.json')


def load_bankroll():
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


def save_bankroll(bankroll):
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


@main_bp.route('/')
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


@main_bp.route('/analysis')
@handle_errors
def analysis():
    """Analyze historical performance."""
    bankroll = load_bankroll()
    
    # Convert history to DataFrame
    df = pd.DataFrame(bankroll['history'])
    if df.empty:
        return "No data available for analysis."
    
    # Calculate metrics
    df['date'] = pd.to_datetime(df['date'])
    df['profit'] = df['winnings'] - df['entry_fee']
    df['roi'] = df['profit'] / df['entry_fee'] * 100
    
    # Group by platform
    platform_summary = df.groupby('platform').agg({
        'profit': 'sum',
        'roi': 'mean',
        'entry_fee': 'sum',
        'winnings': 'sum'
    }).reset_index()
    
    # Overall metrics
    total_profit = df['profit'].sum()
    average_roi = df['roi'].mean()
    
    # Time-based analysis (e.g., monthly)
    df_monthly = df.set_index('date').resample('M').agg({
        'profit': 'sum',
        'roi': 'mean',
        'entry_fee': 'sum',
        'winnings': 'sum'
    })
    df_monthly.index.rename('month', inplace=True)
    df_monthly.reset_index(inplace=True)
  
    
    return render_template('analysis.html',
                           total_profit=f"{total_profit:.2f}",
                           average_roi=f"{average_roi:.2f}%",
                           platform_summary=platform_summary.to_dict(orient='records'),
                           monthly_summary=df_monthly.to_dict(orient='records'))


@main_bp.route('/update_bankroll', methods=['POST'])
@login_required
@handle_errors
def update_bankroll():
    """Update bankroll with contest results."""
    from flask import request
    
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
        'roi': f"{roi:.2f}%"
    })
    save_bankroll(bankroll)

    return jsonify({
        'success': True,
        'message': 'Bankroll updated successfully',
        'current_bankroll': f"${bankroll['current']:.2f}",
        'last_updated': bankroll['last_updated']
    })