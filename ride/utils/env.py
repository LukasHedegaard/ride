import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

ROOT_PATH = Path(os.getenv("ROOT_PATH", default=os.getcwd()))
DATASETS_PATH = ROOT_PATH / os.getenv("DATASETS_PATH", default="datasets")
LOGS_PATH = ROOT_PATH / os.getenv("LOGS_PATH", default="logs")
RUN_LOGS_PATH = LOGS_PATH / "run_logs"
TUNE_LOGS_PATH = LOGS_PATH / "tune_logs"
CACHE_PATH = ROOT_PATH / os.getenv("CACHE_PATH", default=".cache")
LOG_LEVEL = os.getenv("LOG_LEVEL", default="INFO")

DATASETS_PATH.mkdir(exist_ok=True)
LOGS_PATH.mkdir(exist_ok=True)
RUN_LOGS_PATH.mkdir(exist_ok=True)
TUNE_LOGS_PATH.mkdir(exist_ok=True)
CACHE_PATH.mkdir(exist_ok=True)

NUM_CPU = os.cpu_count() or 1
