import logging

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info(f"Sending a GET using...")
r = requests.get("http://127.0.0.1:8000")

logger.info(f"Response status code: {r.status_code}")
logger.info(f"Response text: {r.text}")

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

logger.info(f"Sending a POST using data={data}...")
r = requests.post("http://127.0.0.1:8000/data/", json=data)

logger.info(f"Response status code: {r.status_code}")
logger.info(f"Response text: {r.text}")
