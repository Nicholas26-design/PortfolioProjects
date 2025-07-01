import pandas as pd
import requests
import json
from urllib.parse import quote
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(f'synthea_load_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def load_synthea_data(config_path=None, config_url=None, error_log_path="error_log.txt"):
    # Load config from local file or URL
    if config_url:
        config_response = requests.get(config_url)
        config = config_response.json()
    elif config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("Must provide either config_path or config_url")
    
    base_url = config['base_url']
    files = config['files']
    
    data = {}
    for file in files:
        try:
            encoded_file = quote(file)
            url = f"{base_url}/{encoded_file}"
            df = pd.read_csv(url)
            key = file.replace('.csv', '')
            data[key] = df
            print(f"Loaded {file}: {len(df)} rows")
        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            # error_message = f"Error loading {file}: {e}\n"
            # print(error_message.strip())
            # with open(error_log_path, "a") as log_file:
            #     log_file.write(error_message)
    
    return data

# Usage options:
# From local JSON file
# synthea_data = load_synthea_data(config_path='synthea_config.json')

# Or from GitHub-hosted JSON config
config_url = "https://raw.githubusercontent.com/Nicholas26-design/PorfolioProjects/main/Synthea/synthea_config"

# config_url = "https://raw.githubusercontent.com/Nicholas26-design/PorfolioProjects/main/Synthea/synthea_config.json"
synthea_data = load_synthea_data(config_url=config_url)