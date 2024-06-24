Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up
* This is using Python 3.12 with conda
* The conda environment can be built using conda_env_project4.yaml

## Repositories
* The git repository can be found here: https://github.com/juergen-bullinger/project4


## GitHub Actions

* Setup GitHub Actions on your repository. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
   * Make sure you set up the GitHub Action to have the same version of Python as you used in development.
* Add your <a href="https://github.com/marketplace/actions/configure-aws-credentials-action-for-github-actions" target="_blank">AWS credentials to the Action</a>.
* Set up <a href="https://github.com/iterative/setup-dvc" target="_blank">DVC in the action</a> and specify a command to `dvc pull`.

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

## API Creation

* Create a RESTful API using FastAPI this must implement:
   * GET on the root giving a welcome message.
   * POST that does model inference.
   * Type hinting must be used.
   * Use a Pydantic model to ingest the body from POST. This model should contain an example.
    * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

## API Deployment

* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
   * Enable automatic deployments that only deploy if your continuous integration passes.
   * Hint: think about how paths will differ in your local environment vs. on Heroku.
   * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Set up DVC on Heroku using the instructions contained in the starter directory.
* Set up access to AWS on Heroku, if using the CLI: `heroku config:set AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy`
* Write a script that uses the requests module to do one POST on your live API.

Example code
part_3_root/cd0583-model-scoring-and-drift-using-evidently/main.py

To test the API using guvicorn started from the root directory.
Option --app-dir has to be specified so guvicorn finds the python source.

uvicorn --app-dir src  main:app --reload

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


## Packages

I am using render to avoide licence issues.
https://render.com/
github-user: juergen-bullinger

