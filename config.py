import os
from pathlib import Path

project_path = Path(__file__).parent
model_path = os.path.join(project_path, "model", "model.pkl")
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
