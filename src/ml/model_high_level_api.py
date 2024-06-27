#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classify with the trained model

Created on Tue Jun 25 12:22:16 2024

@author: juergen
"""

from typing import List, Tuple

# from fastapi.staticfiles import StaticFiles

import config as cfg
from ml.model import inference
from ml.data import process_data
from utils import pickle_load_object, get_logger

logger = get_logger(__name__)

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
    ENCODER = pickle_load_object(cfg.CONFIG["preprocessing"]["one_hot_encoder_file"])
    LABEL_BINARIZER = pickle_load_object(
        cfg.CONFIG["preprocessing"]["label_encoder_file"]
    )
    CATEGORIES = cfg.CONFIG["preprocessing"]["categories"]


refresh_model()


def prepare_and_infer(data) -> Tuple[List[str], List[int], List[int]]:
    """
    Preprocess the data in the census documents and feed them to the
    classification model

    Parameters
    ----------
    data : DataFrame
        Raw census data to be prepared and classified.

    Returns
    -------
    Tuple of three elements containing
        List[str]
            Classification result in a list with elements
            like "<=50K" or ">50K".
        List[int]
            Raw classification result as int as returned from the model.
            E.g. for calculating metrics.
        List[int]
            Raw true lables from the data. (if provided)
    """
    logger.info("received data with columns %s", ", ".join(data.columns))
    x, y_true = process_data(
        data,
        categorical_features=CATEGORIES,
        label="salary",
        encoder=ENCODER,
        lb=LABEL_BINARIZER,
        training=False,
    )[:2]
    logger.info(f"of shape {x.shape}")
    y_pred = inference(MODEL, x)
    logger.info(f"of shape {y_pred.shape}")
    y_pred_inversed = LABEL_BINARIZER.inverse_transform(y_pred)
    return y_pred_inversed, y_pred, y_true
