"""
Script to train machine learning model.


"""

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from pathlib import Path
import pandas as pd
import pickle

from ml.data import process_data
from ml import model
import config as cfg

cfg.initialize_global_config()
print(cfg.CONFIG)

# Add code to load in the data.
data = pd.read_csv(
    cfg.CONFIG["data"]["file"], 
    sep=',\s*', 
    encoding="utf-8", 
    engine="python",
    na_values={"?",}
)


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
ml_model = model.train_model(
    X_train, 
    y_train,
    model_parameters=cfg.CONFIG["model"]["parameters"],
)

with Path(cfg.CONFIG["model"]["file"]).open("wb") as fp_model:
    pickle.dump(ml_model, fp_model)
