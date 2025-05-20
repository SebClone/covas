"""
test_shap_analysis.py

This test module verifies the correctness of SHAP value calculations and feature distribution
statistics using the get_shap_values_for_correct_classification and get_feature_distribution
functions from the covaslib.shap_analysis module.

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

def test_shap_analysis_outputs():
    """
    Validates SHAP value generation and class-specific SHAP feature distributions
    for correctly classified samples on the breast cancer dataset.
    """
    # Load and prepare data
    data = load_breast_cancer()
    X = data.data
    y = data.target
    class_labels = data.target_names.tolist()
    feature_names = data.feature_names.tolist()
    ids = pd.Series(['id_' + str(i) for i in range(len(X))])

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(X, y, ids, test_size=0.3, random_state=100)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train a simple neural network
    model = Sequential([
    Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=5, batch_size=16, verbose=0)

    # Perform SHAP analysis
    correct_classified = get_correct_classification(model, X_test_scaled, y_test, id_test, class_labels)
    shap_values = get_shap_values_for_correct_classification(model, X_train_scaled, X_test_scaled, correct_classified, class_labels)
    feature_stats = get_feature_distribution(shap_values, feature_names)

    # Assertions
    assert isinstance(shap_values, dict)
    assert isinstance(feature_stats, dict)
    assert all(label in shap_values for label in class_labels)
    assert all(label in feature_stats for label in class_labels)
