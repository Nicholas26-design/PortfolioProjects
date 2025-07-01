import pandas as pd
from urllib.parse import quote


def load_synthea_data():
    base_url = "https://raw.githubusercontent.com/Nicholas26-design/PorfolioProjects/main/Synthea/Data"

    files = [
        'Synthetic ML Encounter Data.csv',  # spaces in filename
        'Synthetic ML Patient Data.csv'
    ]

    data = {}
    for file in files:
        try:
            # URL encode the filename to handle spaces
            encoded_file = quote(file)
            url = f"{base_url}/{encoded_file}"
            df = pd.read_csv(url)
            key = file.replace('.csv', '')
            data[key] = df
            print(f"Loaded {file}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {file}: {e}")

    return data

# Usage
synthea_data = load_synthea_data()
# patients_df = synthea_data['patients']
# conditions_df = synthea_data['conditions']