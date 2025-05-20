"""
test_classification.py

This test module verifies the functionality of the get_correct_classification function
using the breast cancer dataset from scikit-learn and a simple neural network classifier.

Author: Sebastian Roth
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

from covaslib.classification import get_correct_classification

def test_correct_classification_on_breast_cancer():
    """
    Tests get_correct_classification by training a basic neural network on the breast cancer dataset.
    It verifies that the returned result is a dictionary containing expected class labels
    and appropriate data structures.
    """
    # Load and prepare data
    data = load_breast_cancer()
    X = data.data
    y = data.target
    class_labels = data.target_names.tolist()
    ids = pd.Series(['id_' + str(i) for i in range(len(X))])

    X_train_scaled, X_test_scaled, y_train, y_test, id_train, id_test = train_test_split(X, y, ids, test_size=0.3, random_state=100)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_scaled)
    X_test_scaled = scaler.transform(X_test_scaled)

    # Build and train a simple neural network
    model = Sequential([
    Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=5, batch_size=16, verbose=0)

    # Run the function and check output structure
    result = get_correct_classification(model, X_test_scaled, y_test, id_test, class_labels)

    assert isinstance(result, dict)
    assert all(label in result for label in class_labels)
    for class_dict in result.values():
        assert 'index' in class_dict