"""
Script to train machine learning model.

Created on Thu Apr 18 10:39:17 2024

@author: juergen
"""

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from pathlib import Path

from ml.data import process_data
from ml import model, data
import config as cfg
import utils

cfg.initialize_global_config()
print(cfg.CONFIG)

# Add code to load in the data.
df_data = data.read_raw_data(cfg.CONFIG["data"]["raw_file"])

# Optional enhancement, use K-fold cross validation instead of a train-test split.
df_train, df_test = train_test_split(df_data, test_size=0.20)

# write the split data frames
data_path = Path(cfg.CONFIG["data"]["data_path"])
data.write_raw_data(df_train, data_path / "train.csv")
data.write_raw_data(df_test, data_path / "test.csv")


# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    df_train, 
    categorical_features=cfg.CONFIG["preprocessing"]["categories"], 
    label="salary", 
    training=True
)

# Train and save a model.
ml_model = model.train_model(
    X_train, 
    y_train,
    model_parameters=cfg.CONFIG["model"]["parameters"],
)

utils.pickle_dump_object(
    ml_model, 
    cfg.CONFIG["model"]["file"]
)

utils.pickle_dump_object(
    encoder, 
    cfg.CONFIG["preprocessing"]["one_hot_encoder_file"]
)

utils.pickle_dump_object(
    lb, 
    cfg.CONFIG["preprocessing"]["label_encoder_file"]
)
