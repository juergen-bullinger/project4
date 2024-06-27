#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get the slice performance on the test data.

Created on Tue Jun 25 11:48:22 2024

@author: juergen
"""

import pandas as pd

# from fastapi.staticfiles import StaticFiles

import config as cfg
from ml.model_high_level_api import prepare_and_infer
from ml.model import compute_model_metrics
from ml.data import read_raw_data
from utils import get_logger

logger = get_logger(__name__)

MODEL = None
ENCODER = None
LABEL_BINARIZER = None
CATEGORIES = None

cfg.initialize_global_config()


def evaluate_slice_performance(data: pd.DataFrame) -> pd.DataFrame:
    """
    Write the slice performance to the given file.

    Parameters
    ----------
    data : DataFrame
        Data to evaluate the slice performance on. Unporcessed data.
    file_name : str
        File to write the slice performance to.

    Returns
    -------
    DataFrame containing the evaluation results.
    """
    obj_columns = list(data.select_dtypes(object).columns)
    logger.info(
        "evaluating the slice performance on columns %s", ", ".join(obj_columns)
    )
    slice_performance = []
    for col in obj_columns:
        for col_val, df_slice in data.groupby(col, dropna=False):
            logger.info("columns in df_slice %s", ", ".join(df_slice.columns))
            y_pred_inversed, y_pred, y_true = prepare_and_infer(df_slice)
            precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
            logger.info(
                "slice for %s(%s) evaluates to pred=%s, rec=%s, fbeta=%s",
                col,
                col_val,
                precision,
                recall,
                fbeta,
            )
            slice_performance.append((col, col_val, precision, recall, fbeta))
    return pd.DataFrame.from_records(
        slice_performance, columns=["column", "value", "precision", "recall", "fbeta"]
    )


def write_slice_performance(data: pd.DataFrame, file_name: str):
    """
    Write the slice performance to the given file.

    Parameters
    ----------
    data : DataFrame
        Data to evaluate the slice performance on. Unporcessed data.
    file_name : str
        File to write the slice performance to.

    Returns
    -------
    DataFrame with the evaluation results.
    """
    df_slice_performance = evaluate_slice_performance(data)
    logger.info("writing the slice performance output to %s", file_name)
    with open(file_name, "wt") as fp:
        df_slice_performance.to_string(fp, index=False)
    return df_slice_performance


if __name__ == "__main__":
    # output the slice performance
    write_slice_performance(
        read_raw_data(cfg.CONFIG["data"]["test_file"]),
        cfg.CONFIG["model"]["slice_performance"],
    )
