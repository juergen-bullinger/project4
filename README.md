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
* See .github/workflows/python-app.yml

## Data
* The used data can be found in the data directory (offical census data)

## Model
* The model and the encoders can be found in subdirectory model in pickled form
* See the model card for more details on the model"

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
* The API documentation is reachable on render via the following url
  https://project4-fq0h.onrender.com/docs

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
tests/manual_api_test_local.py
or
tests/manual_api_test_render.py



