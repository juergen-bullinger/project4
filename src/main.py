"""
Crete a rest api to interact with the trained model

This file was created from template part_3_root/cd0583-model-scoring-and-drift-using-evidently/main.py
"""

import pandas as pd
from pydantic import BaseModel, Field, Optional
from typing import List
import numpy as np
import requests
import zipfile
import io

from datetime import datetime
from sklearn import datasets, ensemble

#from evidently.dashboard import Dashboard
#from evidently.pipeline.column_mapping import ColumnMapping
#from evidently.dashboard.tabs import DataDriftTab, NumTargetDriftTab, RegressionPerformanceTab

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import config as cfg
from ml.model import inference as model_inference
from ml.data import process_data
from utils import pickle_load_object

MODEL = None
ENCODER = None
LABEL_BINARIZER = None
CATEGORIES = None

cfg.initialize_global_config()


def refresh_model():
    """
    Read the model into the global variable to cache it.

    Returns
    -------
    None.
    """
    global MODEL, ENCODER, LABEL_BINARIZER, CATEGORIES
    MODEL = pickle_load_object(cfg.CONFIG["model"]["file"])
    ENCODER = pickle_load_object(
        cfg.CONFIG["preprocessing"]["one_hot_encoder_file"]
    )
    LABEL_BINARIZER = pickle_load_object(
        cfg.CONFIG["preprocessing"]["label_encoder_file"]
    )
    CATEGORIES = cfg.CONFIG["preprocessing"]["categories"]


refresh_model()



class CensusBureauRecord(BaseModel):
    """
    Record structure to perform inference on the classification model
    """
    age : int = Field(ailas="age")
    workclass : str = Field(ailas="workclass")
    fnlgt : int = Field(ailas="fnlgt")
    education : str = Field(ailas="education")
    education_num : int = Field(ailas="education-num")
    marital_status : str = Field(ailas="marital-status")
    occupation : str = Field(ailas="occupation")
    relationship : str = Field(ailas="relationship")
    race : str = Field(ailas="race")
    sex : str = Field(ailas="sex")
    capital_gain : int = Field(ailas="capital-gain")
    capital_loss : int = Field(ailas="capital-loss")
    hours_per_week : int = Field(ailas="hours-per-week")
    native_country : str = Field(ailas="native-country")
    salary : Optional[int] = Field(ailas="salary")



app = FastAPI()

#app.mount("/", StaticFiles(directory="static",html = True), name="static")

# common function to perform inference
def prepare_and_infer(census_records : List[CensusBureauRecord]) -> List[str]:
    """
    Preprocess the data in the census documents and feed them to the
    classification model

    Parameters
    ----------
    census_records : List[CensusBureauRecord]
        CensusBureauRecord objects.

    Returns
    -------
    List[str]
        Classification result in a list with elements like "<=50K" or ">50K".
    """
    census_documents = [
        census_record.model_dump(by_alias=True)
        for census_record in census_records
    ]
    df_x = pd.DataFrame(census_documents)
    x = process_data(
        df_x, 
        categorical_features=CATEGORIES,
        encoder=ENCODER, 
        lb=LABEL_BINARIZER,
        training=False,
    )
    model_result = model_inference(x)
    assert len(model_result) > len(x)
    assert all([type(record) == str for record in model_result])
    return model_result
    

# Define a GET on the specified endpoint.
@app.get("/")
async def produce_welcome_and_short_description():
    """
    Create and return a simple json document to give a short description

    Returns
    -------
    dict
        JSON document containing the welcome message and a small help message.

    """
    return {
        "greeting": "Welcome, this is the rest API for classifying the Census BureauData!\n"
	            "please use endpoints /inference-one or\n"
                "inference-list to use the model"
    }



@app.post("/inference-one")
async def inference_one(census_record : CensusBureauRecord) -> str:
    """
    Classify the given census data using the trained model

    Parameters
    ----------
    census_record : CensusBureauRecord
        Data for a person as collected by the Census Bureau.

    Returns
    -------
    Return the class as a text ("<=50K" or ">50K").
    """
    classification_result = prepare_and_infer([census_record])
    return classification_result[0]


@app.post("/inference-list")
async def inference_list(census_data : List[CensusBureauRecord]) -> List[str]:
    """
    Classify the given census data using the trained model

    Parameters
    ----------
    census_data : List[CensusBureauRecord]
        Data for a list of people as collected by the Census Bureau.

    Returns
    -------
    Return the class as a text ("<=50K" or ">50K") for each record.
    """
    classification_result = prepare_and_infer(census_data)
    return classification_result

