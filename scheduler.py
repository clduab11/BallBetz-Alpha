from apscheduler.schedulers.background import BackgroundScheduler
from data_pipeline import run_pipeline
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scheduler_diagnostics.log')  # Log to a separate file
    ]
)
logger = logging.getLogger(__name__)

def create_scheduler():
    """Creates and configures a BackgroundScheduler."""
    scheduler = BackgroundScheduler()
    # Schedule the pipeline to run daily at a specific time (e.g., 3:00 AM)
    scheduler.add_job(run_pipeline, 'cron', hour=3, minute=0)
    logger.info("Scheduler created and job added.")
    return scheduler