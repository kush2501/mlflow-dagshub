import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Load the iris dataset.
iris = load_iris()
X = iris.data
y = iris.target


# Split the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameters for the Random Forest model.
max_depth = 10

# apply mlflow.

# ------- Experiment Name --------#

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Iris-Decision_Tree")


with mlflow.start_run():

    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # log accuracy.
    mlflow.log_metric('accuracy', accuracy)

    # log parameters.
    mlflow.log_param('max_depth', max_depth)

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
    mlflow.sklearn.log_model(dt, name="Decision_Tree")

    # set tags.
    mlflow.set_tag("author", 'kushx')
    mlflow.set_tag("model", 'decision tree')
    print('accuracy', accuracy)
