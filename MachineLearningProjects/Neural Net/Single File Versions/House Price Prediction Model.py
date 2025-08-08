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

"""
Preprocessing
"""

# Define categorical and numeric columns
# CORRECT: Define column types based on your feature matrix X
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
numeric_cols = X.select_dtypes(include=['number']).columns

print("Categorical columns:", list(categorical_cols))
print("Numeric columns:", list(numeric_cols))

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",  # Name for this transformer
            Pipeline([  # What to do with numeric columns
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,  # Which columns to apply this to
        ),
        (
            "cat",  # Name for this transformer
            Pipeline([  # What to do with categorical columns
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,  # Which columns to apply this to
        ),
    ]
)
# Now X and y are ready to be used in a model pipeline

"""
Modeling
Type: Neural Network
Step 1: Split into train/dev/test
Step 2: Model Training
Step 3: Model Evaluation
Step 4: Calibration
"""

# Step 1: Split into train/dev/test

# Enable MLflow autologging
mlflow.autolog()

# Split the data into training and testing sets
# 1. Split your data FIRST (before any preprocessing)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# This gives you 60% train, 20% dev, 20% test

# 2. Fit preprocessor only on training data
preprocessor.fit(X_train)

# 3. Transform all splits
# Don't convert to numpy - keep as DataFrame with column names
X_train_processed = preprocessor.transform(X_train)
X_dev_processed = preprocessor.transform(X_dev)
X_test_processed = preprocessor.transform(X_test)

# Convert to DataFrames to see the column names
feature_names = preprocessor.get_feature_names_out()

X_train_df = pd.DataFrame(X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed, 
                         columns=feature_names)
X_dev_df = pd.DataFrame(X_dev_processed.toarray() if hasattr(X_dev_processed, 'toarray') else X_dev_processed,
                       columns=feature_names) 
X_test_df = pd.DataFrame(X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed,
                        columns=feature_names)

print("Column names:")
print(X_train_df.columns.tolist()[:20])  # First 20 column names

# Step 2: Model Training

# Determine how many layers and their sizes
input_shape = X_train_df.shape[1]
# This is fixed - it must match your feature count
print(f"Input shape: {input_shape}")
input_size = input_shape
base = 1024  # closest power of 2
n_layers = 4

# Geometric progression: each layer halves in size
layer_sizes = [base // (2**i) for i in range(n_layers)]
print(f"Layer sizes: {layer_sizes}")

# Build the neural network model
def build_neural_network(input_shape):
    model = keras.Sequential([
        # Input layer is implicit
        
        # First hidden layer - capture main patterns
        keras.layers.Dense(512, activation='relu', input_shape=(input_shape,)),
        keras.layers.BatchNormalization(),  # Helps with 1000+ features
        keras.layers.Dropout(0.3),
        
        # Second hidden layer - refine patterns  
        keras.layers.Dense(256, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        # Third hidden layer - final abstractions
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        
        # Output layer - single value for house price
        keras.layers.Dense(1)  # No activation for regression
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    return model

model = build_neural_network(X_train_df.shape[1])

# Step 3: Model Evaluation

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

# Step 4: Calibration
