from joblib import Memory

from .env import CACHE_PATH

cache = Memory(CACHE_PATH, verbose=1).cache
