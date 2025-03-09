import pandas as pd
from utils.data_processor import clean_data
from data_validation import validate_data
from scrapers.ufl_scraper import UFLScraper
from utils.data_processor import process_player_data
import os

def run_pipeline():
    scraper = UFLScraper()
    raw_data = scraper.scrape_player_data()
    cleaned_data = clean_data(raw_data)

    if validate_data(cleaned_data):
        # Process and store the data
        df = process_player_data(cleaned_data)
        # Create 'data' directory if it doesn't exist
        if not os.path.exists('BallBetz-Alpha/data'):
            os.makedirs('BallBetz-Alpha/data')
        df.to_csv('BallBetz-Alpha/data/processed_data.csv', index=False)
        print("Data pipeline completed successfully.")
    else:
        print("Data validation failed.")

if __name__ == "__main__":
    run_pipeline()