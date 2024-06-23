#!/bin/sh
conda activate project_part_4
export PYTHONPATH=$(pwd)/src
uvicorn --app-dir src   main:app --reload
