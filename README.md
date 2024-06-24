Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up and used Packages
* This is using Python 3.12 with conda
* The conda environment can be built using conda_env_project4.yaml

## Repositories and connected Accounts
* The git repository can be found here: 
  github-user: juergen-bullinger
  repo: https://github.com/juergen-bullinger/project4
* I am using render to avoid licence issues.
  user: juergen-bullinger
  url: https://render.com/

## GitHub Actions
* flake8 and pytest is executed when the pipeline is run

## Data

* Download census.csv from the data folder in the starter repository.
   * Information on the dataset can be found <a href="https://archive.ics.uci.edu/ml/datasets/census+income" target="_blank">here</a>.
* Create a remote DVC remote pointing to your S3 bucket and commit the data.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.
* Commit this modified data to DVC under a new name (we often want to keep the raw data untouched but then can keep updating the cooked version).

## Model

* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
   * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

## API
### Manual API testing
* To test the API locally, you can use guvicorn executing the following command
  from the root directory of the repository (where you checked it out)
* uvicorn --app-dir src  main:app --reload

### API Endpoints
The API has three Endpoints
* GET on "/" this shows a welcome message to see if the API is up
* PUT on "/inference-one" this can be used to get the salary classification
  for one data point as in the census data resulting in one of two categories
  "<=50K" or ">50K"
* PUT on "/inference-list" which is similar to inference-one, but accepts
  a list of data points and results a list of categories (one per data point).

## API Deployment
* The API is deployed via render (see accounts) from the github pipeline.

## Testing
### Performing pytest
Before pytest is performed, the path has to be set correctly, so pytest finds
the necessary modules in the src subdirectory. This can be done either directly
from the shell using
export PYTHONPATH=$(pwd)/src

or using the script created for this purpose (which also activates the
necessary conda environment)
. setpath_pytest.sh

## Performing the sanitycheck
1. call:
python src/sanitycheck.py tests/
2. on the prompt asking for the path enter:
tests/manual_api_test.py


