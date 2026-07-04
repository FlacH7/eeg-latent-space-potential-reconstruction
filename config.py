# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# load .env
load_dotenv()

# Acceso a variables de entorno
DB_TEST_RETEST_PATH = os.getenv("DB_TEST_RETEST_PATH")
DATA_SIMULATIONS_CACHE_PATH = os.getenv("DATA_SIMULATIONS_CACHE_PATH")