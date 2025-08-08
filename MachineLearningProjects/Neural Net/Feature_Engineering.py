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
Feature Engineering.
"""
def feature_engineering(df):
    # Categorize MedianIncome into groups
    df['IncomeCategory'] = pd.cut(df['MedianIncome'], 
                                  bins=[0, 50000, 100000, float('inf')], 
                                  labels=['Low', 'Medium', 'High'])
    # Categorize house age into bins
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
    df['HouseAgeCategory'] = pd.cut(df['HouseAge'], bins=bins, labels=labels)
     # SAFE division - replace inf with NaN, then let imputer handle it
    df['bedrooms_per_room'] = df['AvgBedrms'] / df['AvgRooms']
    df['rooms_per_person'] = df['AvgRooms'] / df['AveOccup']
    
    # Replace infinite values with NaN
    df['bedrooms_per_room'] = df['bedrooms_per_room'].replace([np.inf, -np.inf], np.nan)
    df['rooms_per_person'] = df['rooms_per_person'].replace([np.inf, -np.inf], np.nan)
    # Rename column 'name' to 'City'
    df = df.rename(columns={'name': 'City'})
    # Rename column 'admin1' to 'State'
    df = df.rename(columns={'admin1': 'State'})
    # Rename column 'admin2' to 'County'
    df = df.rename(columns={'admin2': 'County'})
    # Rename column 'cc' to 'Country'
    df = df.rename(columns={'cc': 'Country'})
    # Drop columns: 'lat', 'lon'
    df = df.drop(columns=['lat', 'lon'])
    return df

df_engineered = feature_engineering(df)
df_engineered.head()

"""
Feature and Target Selection
"""
# Define feature columns again
features_to_keep = [
    'MedianIncome',     # → Numeric pipeline (mean imputation)
    'AvgRooms',         # → Numeric pipeline (mean imputation)
    'AvgBedrms',        # → Numeric pipeline (mean imputation)
    'AveOccup',         # → Numeric pipeline (mean imputation)
    'HouseAge',         # → Numeric pipeline (mean imputation) ✓
    'IncomeCategory',   # → Categorical pipeline (most_frequent) ✓
    'HouseAgeCategory', # → Categorical pipeline (most_frequent) ✓
    'bedrooms_per_room',# → Numeric pipeline (mean imputation)
    'rooms_per_person', # → Numeric pipeline (mean imputation)
    'City',             # → Categorical pipeline (most_frequent) ✓
    'State',            # → Categorical pipeline (most_frequent) ✓
    'County'            # → Categorical pipeline (most_frequent) ✓
]

# Define target column
target_col = 'MedHouseVal'

# Select relevant columns from the DataFrame
selected_cols = features_to_keep + [target_col]
df_housing = df_engineered[selected_cols]

# Drop rows with missing values
df_housing = df_housing.dropna()

# Split into features (X) and target (y)
X = df_housing[features_to_keep]
y = df_housing[target_col]

