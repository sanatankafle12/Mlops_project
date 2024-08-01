from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from comet_ml import Experiment, OfflineExperiment

experiment = Experiment(
    api_key="s9D1PWWDv68Hyt0N4Qkaz3Qtz",
    project_name="mlops-deploy",
    workspace="sanatankafle12"
)


iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(knn, f)

y_pred = knn.predict(X)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# Log the model and metrics to CometML
experiment.log_model("knn_classifier", knn)
experiment.log_metric("accuracy", accuracy)
experiment.log_metric("precision", precision)
experiment.log_metric("recall", recall)
experiment.log_metric("f1_score", f1)

# Finish the experiment
experiment.finish()