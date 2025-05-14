# import mlflow
# print("Printing Tracking URI Scheme Below")
# print(mlflow.get_tracking_uri())
# print("\n")

# mlflow.set_tracking_uri("http://localhost:5000")
# print("Printing New Tracking URI Scheme Below")
# print(mlflow.get_tracking_uri())
# print("\n")


import mlflow
from packaging.version import Version

assert Version(mlflow.__version__) >= Version("2.15.1"), (
  "This feature requires MLflow version 2.15.1 or newer. "
  "Please run '%pip install -U mlflow' in a notebook cell, "
  "and restart the kernel when the command finishes."
)

from openai import OpenAI

mlflow.openai.autolog()

# Ensure that the "OPENAI_API_KEY" environment variable is set
client = OpenAI()

messages = [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello!"}
]

# Inputs and outputs of the API request will be logged in a trace
client.chat.completions.create(model="gpt-4o-mini", messages=messages)