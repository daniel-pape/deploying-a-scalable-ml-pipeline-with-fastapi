# My solution to "Udacity: Deploy a scalable machine learning pipeline with FastAPI"

This is my solution to Udacity's "Deploying a Scalable ML Pipeline with FastAPI" guided project.
The solution is build upon a fork of the provided template code obtainable 
from this [link](https://github.com/udacity/Deploying-a-Scalable-ML-Pipeline-with-FastAPI).

Main tasks were:
* Setup a GitHub workflow in `.github/workflows/ci.yml` to checkout this repo, install the requirements and run tests 
and flake8 to check code quality.
* Implement code to train a machine learning model on the publicly available census data
data which can be used to generate predictions on whether a person earns more or less than 50K USD per year based on 
demographic features included in the data set.
* Document the model in `model_card.md`.
* Use the provided template code to post prediction requests to the model served by a FastAPI application.
* Implement simple unit tests for parts of the code and test the API using a simple script locally.

## Project structure

```bash
.
├── CODEOWNERS
├── LICENSE.txt                 # License provided by Udacity
├── README.md                   # This README
├── __pycache__
├── data                        # Contains the census data
├── environment.yml             # Environment file provided by Udacity
├── fastapi
├── local_api.py                # Simple script for testing the FastAPI application
├── main.py                     # Runs the FastAPI application serving prediction requests
├── ml                          # Data pre-processing and code used for model creation
├── model                       # Model artifacts
├── model_card_template.md      # Information about the served model
├── requirements.txt            # Dependencies for pip users
├── screenshots                 # Screenshots used by the Udacity reviewers
├── slice_output.txt            # Performance metrics of the model on data slices
├── test_ml.py                  # Unit tests for the model creation code
└── train_model.py              # Actual model creation
```

## How to install

For reference you can confer the GitHub workflow in `.github/workflows/ci.yml`.

Steps:

1. Install the requirements and source the virtual environment:

```bash
pip install -r requirements.txt
python3 -m venv venv
source venv/bin/activate
```

2. Verify the installation and run the tests:

```bash
python --version
pytest .
```

## How to run

