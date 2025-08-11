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
