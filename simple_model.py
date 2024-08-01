from comet_ml import Experiment, OfflineExperiment
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

api_key = os.getenv('COMET_API_KEY')
project_name = os.getenv('COMET_PROJECT_NAME')
workspace = os.getenv('COMET_WORKSPACE')

experiment = Experiment(
    api_key=api_key,
    project_name=project_name,
    workspace=workspace
)

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(knn, f)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Log the model and metrics to CometML
experiment.log_model("knn_classifier", "model.pkl")
experiment.log_metric("accuracy", accuracy)
experiment.log_metric("precision", precision)
experiment.log_metric("recall", recall)
experiment.log_metric("f1_score", f1)

# Finish the experiment
experiment.end()