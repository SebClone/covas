"""
test_covas.py

This test module verifies the functionality of the get_COVA_matrix and get_COVA_score
functions using a breast cancer classification model from test_classification.

Author: Sebastian Roth
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import shap

from covaslib.classification import get_correct_classification
from covaslib.shap_analysis import get_shap_values_for_correct_classification, get_feature_distribution
from covaslib.covas import get_COVA_matrix, get_COVA_score

def test_cova_matrix_and_score():
    """
    Trains a neural network on the breast cancer dataset and validates the structure and output
    of the COVA matrix and score computations based on SHAP values for correctly classified samples.
    """
    # Load and prepare data
    data = load_breast_cancer()
    X = data.data
    y = data.target
    class_labels = data.target_names.tolist()
    ids = pd.Series(['id_' + str(i) for i in range(len(X))])

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, ids, test_size=0.3, random_state=100)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build and train a simple neural network
    model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=16, verbose=0)

    # Generate SHAP values and compute COVA
    correct_classified = get_correct_classification(model, X_test, y_test, id_test, class_labels)
    shap_values = get_shap_values_for_correct_classification(model, X_test, correct_classified)
    feature_distributions = get_feature_distribution(shap_values, correct_classified)
    cova_matrix = get_COVA_matrix(shap_values, feature_distributions)
    cova_scores = get_COVA_score(shap_values, cova_matrix, correct_classified)

    # Basic assertions
    assert isinstance(cova_matrix, dict)
    assert isinstance(cova_scores, dict)
    assert all(label in cova_matrix for label in class_labels)
    assert all(label in cova_scores for label in class_labels)
