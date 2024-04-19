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
import utils

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# make sure the model and the encoders are chaced
ONE_HOT_ENCODER = None
LABEL_ENCODER = None
ML_MODEL = None
CAT_COLUMNS = None

def initialize():
    """
    Initialize the module - load the pickled objects
    """
    global ONE_HOT_ENCODER
    global LABEL_ENCODER
    global ML_MODEL
    global CAT_COLUMNS
    ONE_HOT_ENCODER = utils.pickle_load_object(
        cfg.CONFIG["preprocessing"]["one_hot_encoder_file"]
    )
    LABEL_ENCODER = utils.pickle_load_object(
        cfg.CONFIG["preprocessing"]["label_encoder_file"]
    )
    ML_MODEL = utils.pickle_load_object(
        cfg.CONFIG["model"]["file"]
    )
    CAT_COLUMNS = cfg.CONFIG["preprocessing"]["categories"]

cfg.initialize_global_config()
initialize()

# Instantiate the app.
app = FastAPI()


class DataRecord(BaseModel):
    age            : int = Field(title="age", description="age of the person", )
    workclass      : str = Field(title="work class", description="work class", )
    fnlgt          : int = Field(title="fnlgt", description="read the docs", )
    education      : str = Field(title="education", description="type of education", )
    education_num  : int = Field(title="education num", description="category of education", )
    marital_status : str = Field(title="marital status", description="married, divorced, single, ...", )
    occupation     : str = Field(title="occupation", description="occupation", )
    relationship   : str = Field(title="relationship", description="relationship", )
    race           : str = Field(title="race", description="race", )
    sex            : str = Field(title="sex", description="sex", )
    capital_gain   : int = Field(title="capital gain", description="capital gain")
    capital_loss   : int = Field(title="capital loss", description="capital loss")
    hours_per_week : int = Field(title="hours per week", description="number of working hours per week")
    native_country : str = Field(title="native country", description="native country", )


@app.get("/")
async def welcome():
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


def _inference_list(record_list : List[DataRecord]) -> List[bool]:
    """
    Predict the income category for the given records

    Parameters
    ----------
    record_list : List[DataRecord]
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
    df_data = pd.DataFrame([record.dict() for record in record_list])
    # restore the original column names (- instead of _)
    df_data.columns = [col.replace("_", "-") for col in df_data.columns]
    print(df_data)
    X, *_ = process_data(
        df_data,
        training=False,
        categorical_features=CAT_COLUMNS, 
        #label="salary", 
        encoder=ONE_HOT_ENCODER, 
        lb=LABEL_ENCODER
    )
    print("The shape of X is:", X.shape)
    y_pred = model.inference(ML_MODEL, X)
    return y_pred


# example to test this with curl
# curl -X POST http://127.0.0.1:8000/inference/
@app.post("/inference_list")
async def inference_list(body : List[DataRecord]) -> List[bool]:
    """
    Predict the income category for the given records

    Parameters
    ----------
    record_list : List[DataRecord]
        The DataRecord represent the data that needs to be predicted.

    Returns
    -------
    predictions : List[bool]
        One prediction per record in data.
    """
    return _inference_list(body)


# example to test this with curl
# curl -X POST http://127.0.0.1:8000/inference/
@app.post("/inference_one")
async def inference_one(body : DataRecord) -> bool:
    """
    Predict the income category for the given records

    Parameters
    ----------
    record_list : List[DataRecord]
        The DataRecord represent the data that needs to be predicted.

    Returns
    -------
    prediction : bool
        Prediction for the record.
    """
    return _inference_list([body])


print(dir(app))
