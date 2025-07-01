# Predicting Patient Readmission Within 30 Days

This project aims to predict whether a patient will be readmitted to the hospital within 30 days based on various features such as discharge summary, prior readmissions, diagnosis, and other patient data.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
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

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/project-name.git

