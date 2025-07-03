# Predicting Patient Readmission Within 30 Days

This project aims to predict whether a patient will be readmitted to the hospital within 30 days based on various features such as discharge summary, prior readmissions, diagnosis, and other patient data.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Usage](#usage)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [License](#license)
- [Contact](#contact)

## Overview
Hospital readmissions within 30 days are a critical metric for healthcare providers. This project uses machine learning models to predict readmission likelihood, helping hospitals improve patient care and reduce costs.

### Input:
- Discharge Summary
- Prior Readmissions
- Diagnosis

### Output:
- Binary classification: **Yes/No** (Readmitted within 30 days)

## Dataset
The dataset is sourced from synthetic healthcare data generated using [Synthea](https://synthetichealth.github.io/synthea/). Data includes:
- Patient demographics
- Encounter details
- Conditions, claims, procedures, and observations

### Data Sources:
- CSV files located in `c:/Users/Nicholas/Documents/GitHub/PorfolioProjects/Synthea/Data/`
- Optionally, data can be loaded from Databricks Hive Metastore.

## Models
The following models were explored:
- Logistic Regression
- Random Forest
- Gradient Boosted Trees

## Usage
1. Clone the repository.
2. Install required dependencies using `pip install -r requirements.txt`.
3. Run the main script:  
   `Predicting Patient Readmission in 30 Days Standard.ipynb`
4. Adjust configuration files as needed for custom data paths or parameters.

## Features
- Predicts 30-day hospital readmission.
- Utilizes multiple machine learning models.
- Supports both CSV and Databricks Hive Metastore data sources.
- Modular and extensible codebase.

## Technologies Used
- Python 3.x
- scikit-learn
- pandas
- numpy
- Synthea (for synthetic data generation)
- Databricks (optional)

## Results
- Achieved accuracy, precision, recall, and F1-score metrics for each model.
- ROC curves and confusion matrices are available in the `results/` directory.

