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

import reverse_geocoder as rg
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
import seaborn as sns
sns.scatterplot(data=df, x="Longitude", y="Latitude", hue="MedHouseVal", palette="viridis")
# Visualize the correlation matrix
sns.heatmap(df_clean.corr(), annot=True, cmap="coolwarm")

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
    df['bedrooms_per_room'] = df['AvgBedrms'] / df['AvgRooms']
    df['rooms_per_person'] = df['AvgRooms'] / df['AveOccup']
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
    'MedianIncome',
    'AvgRooms',
    'AvgBedrms',
    'AveOccup',
    'HouseAge',
    'IncomeCategory',
    'HouseAgeCategory',
    'bedrooms_per_room',
    'rooms_per_person',
    'City',
    'State',
    'County'
]

# Define target column
target_col = 'MedHouseVal'

# Select relevant columns from the DataFrame
selected_cols = features_to_keep + [target_col]
df_housing = df_engineered[selected_cols]

# Drop rows with missing values
df_housing = df_housing.dropna()

# Scale numerical features
df_housing['MedianIncome'] = df_housing['MedianIncome'] / 10000
df_housing['MedHouseVal'] = df_housing['MedHouseVal'] / 100000

# Split into features (X) and target (y)
X = df_housing[features_to_keep]
y = df_housing[target_col]

"""
Preprocessing
Step 1: Encode any categorical variables
"""

# Define categorical and numeric columns

# Check column types
print(df_housing.dtypes)

# Look for object or categorical columns
categorical_cols = df_housing.select_dtypes(include=['object', 'category']).columns
print("Categorical columns:", list(categorical_cols))
# Look for numeric columns
numeric_cols = df_housing.select_dtypes(include=['number']).columns
print("Numeric columns:", list(numeric_cols))

# One-hot encode them
df_encoded = pd.get_dummies(df_housing, columns=categorical_cols, drop_first=True, dtype=float)

"""
Second Possible Preprocessing Method
"""

# Define categorical and numeric columns

# Check column types
print(df_housing.dtypes)

# Look for object or categorical columns
categorical_cols = df_housing.select_dtypes(include=['object', 'category']).columns
print("Categorical columns:", list(categorical_cols))
# Look for numeric columns
numeric_cols = df_housing.select_dtypes(include=['number']).columns
print("Numeric columns:", list(numeric_cols))

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_cols,
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_cols,
        ),
    ]
)

# Now X and y are ready to be used in a model pipeline

"""
Modeling
"""

# Split into train/test
# Model Training
# Model Evaluation
# Calibration
