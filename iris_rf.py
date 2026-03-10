import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import dagshub 

dagshub.init(repo_owner='kush2501', repo_name='mlflow-dagshub', mlflow=True)


# Tracking uri.
mlflow.set_tracking_uri("https://dagshub.com/kush2501/mlflow-dagshub.mlflow")



# Load the iris dataset.
iris = load_iris()
X = iris.data
y = iris.target


# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Random Forest model.
max_depth = 12
n_estimators = 100


# ------- Experiment Name --------#
mlflow.set_experiment("Iris_RandomForest")


# apply mlflow.
with mlflow.start_run():

    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # log accuracy.
    mlflow.log_metric('accuracy', accuracy)

    # log parameters.
    mlflow.log_param('max_depth', max_depth)

    mlflow.log_para('n_estimators', n_estimators)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")

    # Save Plot.
    plt.savefig("Confusion_Matrix.png")

    # Log plot to MLflow.
    mlflow.log_artifact('Confusion_Matrix.png')

    # Log code.
    mlflow.log_artifact(__file__)

    # Log model.
    mlflow.sklearn.log_model(rf, name="RandomForestClassifier")

    # set tags.
    mlflow.set_tag("author", 'love')
    mlflow.set_tag("model", 'random forest')
    print('accuracy', accuracy)
