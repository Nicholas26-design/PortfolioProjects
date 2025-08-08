'''
# House Price Prediction

**Input:**
- square footage
- location
- number of bedrooms


**Output:** price in dollars

## Models:
- Neural Network
'''

# Library imports
import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import col, to_date, datediff, unix_timestamp, lead, when, year
from pyspark.sql.types import IntegerType
from pyspark.sql import Window
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import requests
import json
from urllib.parse import quote
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import reverse_geocoder as rg
import seaborn as sns

print("TensorFlow version:", tf.__version__)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_load_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

"""
Get the Dataset
"""

# Load the California housing dataset
housing = fetch_california_housing()
df_features = pd.DataFrame(housing.data, columns=housing.feature_names)
df_target = pd.DataFrame(housing.target, columns=housing.target_names)
df_combined = pd.concat([df_features, df_target], axis=1)  # add columns

"""
Clean the Dataset
"""
# Clean the data

def clean_data(df):
    # Rename column 'MedInc' to 'MedianIncome'
    df = df.rename(columns={'MedInc': 'MedianIncome'})
    # Rename column 'AveRooms' to 'AvgRooms'
    df = df.rename(columns={'AveRooms': 'AvgRooms'})
    # Rename column 'AveBedrms' to 'AvgBedrms'
    df = df.rename(columns={'AveBedrms': 'AvgBedrms'})
    # Round down column 'AvgRooms'
    df[['AvgRooms']] = np.floor(df[['AvgRooms']])
    # Round column 'AveOccup' (Number of decimals: 1)
    df = df.round({'AveOccup': 1})
    # Round column 'AvgBedrms' (Number of decimals: 1)
    df = df.round({'AvgBedrms': 1})
    df['MedianIncome'] = df['MedianIncome'] * 10000
    df['MedHouseVal'] = df['MedHouseVal'] * 100000
    return df

df_clean = clean_data(df_combined.copy())
df_clean.head()

"""
Exploratory Data Analysis (EDA)
"""


# Assuming df_clean is a DataFrame with 'Latitude' and 'Longitude' columns
# Convert to list of tuples
coords = list(zip(df_clean['Latitude'], df_clean['Longitude']))

# Apply reverse geocoding
results = rg.search(coords)

# Turn results into DataFrame and join
results_df = pd.DataFrame(results)
df = pd.concat([df_clean, results_df], axis=1)

print(df)
# Visualize the data

sns.scatterplot(data=df, x="Longitude", y="Latitude", hue="MedHouseVal", palette="viridis")
# Visualize the correlation matrix
sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm")

