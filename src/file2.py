# This File is for Remote / Dagshub Server

# Import Libraries
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# OPTIONAL: If using a remote MLflow server
# mlflow.set_tracking_uri("http://localhost:5000")

# Importing Dagshub
import dagshub
dagshub.init(repo_owner='Suhail-Ahmed-88', repo_name='MLflow-Experiment-Tracking-Project', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Suhail-Ahmed-88/MLflow-Experiment-Tracking-Project.mlflow")

# load dataset
wine = load_wine()
x = wine.data
y = wine.target

# Split and scale data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# define the params for RF model
max_depth = 20
n_estimators = 10

# Mention Your Experiment Below
mlflow.set_experiment("MLflow-Experiment-Tracking-Project2")

# Start run
with mlflow.start_run():
    # Add tags
    mlflow.set_tag("Author", "Suhail Ahmed")
    mlflow.set_tag("Project", "Wine Classification")
    
    
# Train Model
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(x_train, y_train)
    
    # Predict and evaluate
    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Log metrics and params 
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=wine.target_names, yticklabels=wine.target_names)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")

# Save plot
plot_path = "Confusion-matrix.png"
plt.savefig(plot_path)

# Save and log confusion matrix
mlflow.log_artifact("Confusion-matrix.png")
mlflow.log_artifact(plot_path)

# Log the Model
mlflow.sklearn.log_model(rf, "Random-Forest-Model")
    
    
print(accuracy)
