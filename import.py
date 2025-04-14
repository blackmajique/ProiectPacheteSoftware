import os
from kaggle.api.kaggle_api_extended import KaggleApi

# setează calea spre kaggle.json dacă nu e în locația standard
os.environ['KAGGLE_CONFIG_DIR'] = r'"C:\Users\Mina\Downloads\kaggle.json"'

api = KaggleApi()
api.authenticate()

# dataset-ul pe care îl vrei:
api.dataset_download_files('adilshamim8/student-performance-and-learning-style', path='datasets', unzip=True)
