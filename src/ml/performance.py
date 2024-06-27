#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the model performance on data slices.

Created on Thu Apr 18 10:39:17 2024

@author: juergen
"""
import numpy as np
import pandas as pd
from ml.data import process_data, read_raw_data
from ml import model

import config as cfg

cfg.initialize_global_config()


def get_model_performance_on_slices(ml_model, data, categorical_features, encoder, lb, unique_threshold=20):
    """
    Run model inferences on sliced data and return the measured performance.

    Inputs
    ------
    ml_model : ???
        Trained machine learning model.

    data : pandas.DataFrame
        Data used for measuring the performance.

    encoder : sklearn.OneHotEncoder
        One hot encoder trained during model training.

    lb : sklearn.LabelBinarizer
        Label binarizer trained during training of the model.

    unique_threshold : int
        Maximum number of unique values a column can have. All colums with
        more unique values are not considered.

    Returns
    -------
    df_measurments : pandas.DataFrame
        Performance measurments of the model on the slices.
    """
    df_measurments = pd.DataFrame(
        {
            "column": pd.Series([], dtype="str"),
            "value": pd.Series([], dtype="str"),
            "num_records": pd.Series([], dtype="int64"),
            "precision": pd.Series([], dtype="float64"),
            "recall": pd.Series([], dtype="float64"),
            "f1": pd.Series([], dtype="float64"),
        }
    )
    df_measurments.set_index(["column", "value"], inplace=True)

    """# add the measurements for the whole dataset before slicing
    X, y_true = process_data(
        data,
        categorical_features=categorical_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )[:2] # skip the encoders
    y_pred = model.inference(ml_model, X)
    precision, recall, f1 = model.compute_model_metrics(y_true, y_pred)
    df_measurments.loc[(np.NaN, np.NaN)] = {
        "num_records": data.shape[0],
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }"""

    for col in data.select_dtypes("object"):
        unique_values = data[col].unique()
        if len(unique_values) <= unique_threshold:
            for value in unique_values:
                value_series = data[col]
                if value is np.nan:
                    slicer = value_series.isna()
                else:
                    slicer = value_series == value
                slice_data = data[slicer]
                num_records = slice_data.shape[0]
                X, y_true = process_data(
                    slice_data,
                    categorical_features=categorical_features,
                    training=False,
                    encoder=encoder,
                    lb=lb,
                    label="salary",
                )[:2] # skip the encoders
                y_pred = model.inference(ml_model, X)
                precision, recall, f1 = model.compute_model_metrics(y_true, y_pred)
                df_measurments.loc[(col, value)] = {
                    "num_records": num_records,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
    return df_measurments


if __name__ == "__main__":
    from ml.model_high_level_api import MODEL, ENCODER, LABEL_BINARIZER, CATEGORIES, cfg
    data = read_raw_data(cfg.CONFIG["data"]["test_file"])
    df_measurements = get_model_performance_on_slices(
        MODEL,
        data,
        CATEGORIES,
        ENCODER,
        LABEL_BINARIZER,
        unique_threshold=100,
    )
