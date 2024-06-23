#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:33:40 2024

@author: juergen
"""

import requests as rq


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
    native_country="United-States"
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
    native_country="United-States"
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
    native_country="United-States"
)


def check_inference_one():
    print("-" * 40)
    print("inference-one")
    response = rq.post(
        "http://127.0.0.1:8000/inference-one",
        json=record_3,
    )

    print(response)
    print(response.text)
    print(response.json())


def check_inference_list():
    print("-" * 40)
    print("inference-list")
    response = rq.post(
        "http://127.0.0.1:8000/inference-list",
        json=[record_1, record_2, record_3]
    )

    print(response)
    print(response.text)
    print(response.json())

check_inference_one()
check_inference_list()

