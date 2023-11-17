import kaggle
from pathlib import Path

data_path = Path("Sports data/")

if data_path.is_dir():
    print("directory exists")
else:
    kaggle.api.authenticate()

    kaggle.api.dataset_download_files('gpiosenka/sports-classification', path='Sports data', unzip=True)