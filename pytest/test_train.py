# create a script, wherein you are trainign the iris ML model,
# create a test to check whether there is some file in the model


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)


# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
# model = []

# Save the model
#joblib.dump(model, 'trained_model.pkl')

# Load the model
#loaded_model = joblib.load('trained_model.pkl')

# Test 1: Model should be trained
def test_model_training():
    assert model is not None, "Model training failed!"
    assert hasattr(model, "predict"), "Trained model does not have a 'predict' method"

# Test 2: Check if model predictions match y_test
def test_model_prediction():
    predictions = model.predict(X_test)
    assert np.array_equal(predictions, y_test), "Model predictions do not match y_test"

# Test 3: Check accuracy is above 80%
def test_accuracy():
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    #accuracy = 0.70
    assert accuracy > 0.80, f"Model accuracy ({accuracy:.2f}) is not above 80%"

# Test 4: Prediction shape should match y_test
def test_prediction_shape():
    predictions = model.predict(X_test)
    assert predictions.shape == y_test.shape, f"Prediction shape {predictions.shape} does not match y_test shape {y_test.shape}"

# Test 5: Predictions must only contain valid class labels
def test_prediction_class_labels():
    predictions = model.predict(X_test)
    valid_labels = np.unique(y_train)
    invalid_preds = set(predictions) - set(valid_labels)
    assert len(invalid_preds) == 0, f"Invalid predicted class labels: {invalid_preds}"

# Test 6: Model should not predict a single class for all inputs
def test_prediction_diversity():
    predictions = model.predict(X_test)
    unique_preds = np.unique(predictions)
    assert len(unique_preds) > 1, "Model predicts a single class for all inputs — possible underfitting"

# Test 7: Accuracy should not exceed 100% (sanity check)
def test_accuracy_not_overshooting():
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy <= 1.0, f"Accuracy exceeds 100%: {accuracy}"