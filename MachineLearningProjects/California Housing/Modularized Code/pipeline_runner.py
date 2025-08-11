# pipeline_runner.py

# Import shared libraries
from Imports import *

# # Optional: Load config
# import yaml
# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# Step 1: Data Engineering
import Data_Engineering
raw_data = Data_Engineering.run(config)

# Step 2: Feature Engineering
import Feature_Engineering
features = Feature_Engineering.run(raw_data, config)

# Step 3: Preprocessing
import Preprocessing
processed_data = Preprocessing.run(features, config)

# Step 4: Model Training
import Modeling
model = Modeling.run(processed_data, config)

# Step 5: Evaluation
import Evaluation
Evaluation.run(model, features, config)

# Step 6: Deployment (optional)
# import deployment
# deployment.run(model, config)