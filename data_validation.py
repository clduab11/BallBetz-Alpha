import pandas as pd
import os

def validate_data(data):
    # Check for key columns and basic data integrity
    df = pd.DataFrame(data)
    if not {'name', 'team'}.issubset(df.columns):
        return False

    # Check for at least one statistical column
    stat_cols = ['passing_yards', 'rushing_yards', 'receiving_yards',
                 'kickoff_return_yards', 'punt_return_yards', 'tackles', 'sacks']
    if not any(col in df.columns for col in stat_cols):
        return False

    return True