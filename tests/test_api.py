#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the rest API created by fastapi.

Created on Fri Apr 19 15:33:40 2024
@author: juergen
"""

from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app
from utils import get_logger

logger = get_logger(__name__)

# Instantiate the testing client with our app.
client = TestClient(app)

record_1 = dict(
    age=30,
    workclass="State-gov",
    fnlgt=77516,
    education="Bachelors",
    education_num=13,
    marital_status="Never-married",
    occupation="Adm-clerical",
    relationship="Not-in-family",
    race="White",
    sex="Male",
    capital_gain=2174,
    capital_loss=0,
    hours_per_week=40,
    native_country="United-States",
)

record_2 = dict(
    age=39,
    workclass="State-gov",
    fnlgt=77516,
    education="Bachelors",
    education_num=13,
    marital_status="Never-married",
    occupation="Adm-clerical",
    relationship="Not-in-family",
    race="White",
    sex="Male",
    capital_gain=2174,
    capital_loss=0,
    hours_per_week=40,
    native_country="United-States",
)

record_3 = dict(
    age=52,
    workclass="Self-emp-not-inc",
    fnlgt=209642,
    education="HS-grad",
    education_num=9,
    marital_status="Married-civ-spouse",
    occupation="Exec-managerial",
    relationship="Husband",
    race="White",
    sex="Male",
    capital_gain=0,
    capital_loss=0,
    hours_per_week=45,
    native_country="United-States",
)

record_4 = dict(
    age=52,
    workclass="State-gov",
    fnlgt=209280,
    education="Masters",
    education_num=14,
    marital_status="Married-civ-spouse",
    occupation="Prof-specialty",
    relationship="Husband",
    race="White",
    sex="Male",
    capital_gain=6514,
    capital_loss=0,
    hours_per_week=35,
    native_country="United-States",
)



# Write tests using the same syntax as with the requests module.
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "greeting" in r.json() 


def test_api_locally_inference_one_less_than_50():
    logger.info("-" * 40)
    logger.info("inference-one")
    r = client.post(
        "/inference-one",
        json=record_3,
    )
    logger.info(r)
    logger.info(r.text)
    logger.info(r.json())
    assert r.status_code == 200
    assert r.json() == "<=50K"


def test_api_locally_inference_one_more_than_50():
    logger.info("-" * 40)
    logger.info("inference-one")
    r = client.post(
        "/inference-one",
        json=record_4,
    )
    logger.info(r)
    logger.info(r.text)
    logger.info(r.json())
    assert r.status_code == 200
    assert r.json() == ">50K"


def test_api_locally_inference_list():
    logger.info("-" * 40)
    logger.info("inference-list")
    r = client.post(
        "/inference-list",
        json=[record_1, record_2, record_3]
    )
    logger.info(r)
    logger.info(r.text)
    logger.info(r.json())
    assert r.status_code == 200
