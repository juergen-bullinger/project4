#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 19:06:11 2024

@author: juergen
"""
import io
import pandas as pd
from sklearn.metrics import accuracy_score

from ml import data
from ml import model

import pytest

DATASTR = (
    "age,workclass,fnlgt,education,education-num,marital-status,"
    "occupation,relationship,race,sex,capital-gain,capital-loss,"
    "hours-per-week,native-country,salary\n"
    #
    "18,?,184101,Some-college,10,Never-married,?,Own-child,White,Male,"
    "0,0,25,United-States,<=50K\n"
    #
    "65,Private,330144,Some-college,10,Married-civ-spouse,"
    "Exec-managerial,Husband,White,Male,0,0,40,United-States,<=50K\n"
    #
    "23,Private,57827,Bachelors,13,Never-married,Farming-fishing,"
    "Not-in-family,White,Male,0,0,40,United-States,<=50K\n"
    #
    "60,Self-emp-not-inc,95445,Some-college,10,Married-civ-spouse,"
    "Transport-moving,Husband,White,Male,3137,0,46,United-States,<=50K"
    "\n"
    #
    "54, Self-emp-inc, 125417, 7th-8th, 4, Married-civ-spouse, "
    "Machine-op-inspct, Husband, White, Male, 0, 0, 40, United-States,"
    " >50K\n"
    #
    "37, Private, 635913, Bachelors, 13, Never-married, "
    "Exec-managerial, Not-in-family, Black, Male, 0, 0, 60, "
    "United-States, >50K\n"
    #
    "41, Private, 112763, Prof-school, 15, Married-civ-spouse, "
    "Prof-specialty, Wife, White, Female, 0, 0, 40, United-States, "
    ">50K\n"
    #
    "33, Private, 222205, HS-grad, 9, Married-civ-spouse, Craft-repair,"
    " Wife, White, Female, 0, 0, 40, United-States, >50K\n"
    #
    "61, Private, 69867, HS-grad, 9, Married-civ-spouse, "
    "Exec-managerial, Husband, White, Male, 0, 0, 40, United-States, "
    ">50K\n"
    #
    "35, Private, 190174, Some-college, 10, Never-married, "
    "Exec-managerial, Not-in-family, White, Female, 0, 0, 40, "
    "United-States, <=50K\n"
)

CATERGORICAL_FETURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]


@pytest.fixture
def prepared_data():
    """
    Provide some test data

    Returns
    -------
    tuple[X, y, one_hot_encoder, labelbinarizer]
    """
    sio = io.StringIO(DATASTR)
    df = pd.read_csv(
        sio,
        sep=', *',
        encoding="utf-8",
        engine="python",
        na_values={"?", }
    )
    return data.process_data(
        df,
        categorical_features=CATERGORICAL_FETURES,
        label="salary",
        training=True,
    )


@pytest.fixture
def trained_model(prepared_data):
    """
    Fixture to get a pretrained model
    """
    X, y, encoder, lb = prepared_data
    model_params = dict(
        n_estimators=15,
        max_depth=3,
        min_samples_leaf=1,
        random_state=42
    )
    ml_model = model.train_model(X, y, model_parameters=model_params)
    return ml_model


def test_train_model(prepared_data, trained_model):
    """
    Check general training.

    Parameters
    ----------
    prepared_data : fixture
        The cached return value of prepare_data including the encoders.

    trained_model : fixture
        A trained RandomForestClassifier

    Returns
    -------
    None.
    """
    X, y_true, encoder, lb = prepared_data
    y_pred = trained_model.predict(X)
    assert len(y_pred) == len(y_true)


def test_compute_model_metrics(prepared_data):
    """
    Check the computation of the model metrics.

    Parameters
    ----------
    prepared_data : fixture
        The cached return value of prepare_data including the encoders.

    Returns
    -------
    None.
    """
    X, y_true, encoder, lb = prepared_data
    y_correct = y_true
    y_wrong = 1 - y_true
    precision, recall, fbeta = model.compute_model_metrics(y_true, y_correct)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0

    precision, recall, fbeta = model.compute_model_metrics(y_true, y_wrong)
    assert precision == 0.0
    assert recall == 0.0
    assert fbeta == 0.0


def test_inference(prepared_data, trained_model):
    """
    Check the inference.

    Parameters
    ----------
    prepared_data : fixture
        The cached return value of prepare_data including the encoders.

    trained_model : fixture
        A trained RandomForestClassifier

    Returns
    -------
    None.
    """
    X, y_true, encoder, lb = prepared_data
    y_pred = model.inference(trained_model, X)
    assert len(y_pred) == len(y_true)
    acc = accuracy_score(y_true, y_pred)
    assert acc >= 0.75
