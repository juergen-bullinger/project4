"""
Provide a RESTful API

"""
from typing import List
from pydantic import BaseModel, Field
from fastapi import FastAPI
import os

import pandas as pd

import config as cfg
from ml.data import process_data
from ml import model

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# make sure the model and the encoders are chaced
ONE_HOT_ENCODER = None
LABEL_ENCODER = None
ML_MODEL = None
CAT_COLUMNS

def initialize():
    """
    Initialize the module - load the pickled objects
    """
    global ONE_HOT_ENCODER
    global LABEL_ENCODER
    global ML_MODEL
    global CAT_COLUMNS
    ONE_HOT_ENCODER = cfg.CONFIG["preprocessing"]["one_hot_encoder_file"]
    LABEL_ENCODER = cfg.CONFIG["preprocessing"]["label_encoder_file"]
    ML_MODEL = cfg.CONFIG["model"]["file"]
    CAT_COLUMNS = cfg.CONFIG["preprocessing"]["categories"]


initialize()

# Instantiate the app.
app = FastAPI()


class DataRecord(BaseModel):
    age            : int
    workclass      : str
    fnlgt          : int
    education      : str
    education_num  : int
    marital_status : str
    occupation     : str
    relationship   : str
    race           : str
    sex            : str
    capital_gain   : int
    capital_loss   : int
    hours_per_week : int
    native_country : str


@app.get("/")
def welcome():
    """
    Show a welcome message.

    Returns
    -------
    dict.
        JSON document with some info about this endpoint.
    """
    return {
        "message": "Welcome. This is the endpoint to access the census salary category prediction",
        "additional_info": "please see ... for more details"
    }


@app.post("/inference/")
def inference(data : List[DataRecord]) -> List[bool]:
    """
    Predict the income category for the given records

    Parameters
    ----------
    data : List[DataRecord]
        The DataRecord represent the data that needs to be predicted.

    Returns
    -------
    predictions : List[bool]
        One prediction per record in data.
    """
    global ONE_HOT_ENCODER
    global LABEL_ENCODER
    global ML_MODEL
    global CAT_COLUMNS
    records = [record.dict() for record in data]
    df_data = pd.DataFrame(records)
    # restore the original column names (- instead of _)
    df_data.columns = [col.replace("_", "_") for col in df_data.columns]
    X, *_ = process_data(
        df_data,
        training=False,
        encoder=ONE_HOT_ENCODER, 
        lb=LABEL_ENCODER
    )
    y_pred = model.inference(ML_MODEL, X)
    return y_pred