# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# load .env
load_dotenv()

# Environment variables
DB_TEST_RETEST_PATH = os.getenv("DB_TEST_RETEST_PATH")
DATA_SIMULATIONS_CACHE_PATH = os.getenv("DATA_SIMULATIONS_CACHE_PATH")
BASE_CACHE_PATH = os.getenv("BASE_CACHE_PATH")
BASE_RESULTS_PATH = os.getenv("BASE_RESULTS_PATH")