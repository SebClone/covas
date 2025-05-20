

"""
test_plotting.py

This test module verifies the functionality of the custom_decision_plot function using
SHAP values generated from a classification model trained on the breast cancer dataset.

Author: Sebastian Roth
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

from covaslib.classification import get_correct_classification
from covaslib.shap_analysis import get_shap_values_for_correct_classification
from covaslib.plotting import custom_decision_plot

def test_custom_decision_plot_execution():
    """
    Trains a neural network on the breast cancer dataset and ensures the custom_decision_plot
    function executes successfully with generated SHAP values and test data.
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

    # Compute SHAP values
    correct_classified = get_correct_classification(model, X_test, y_test, id_test, class_labels)
    shap_values = get_shap_values_for_correct_classification(model, X_test, correct_classified)

    # Call the plotting function for one class
    label = class_labels[0]
    data_ids = correct_classified[label]['index']
    if data_ids:
        sample_id = data_ids[0]
        try:
            custom_decision_plot(
                shap_values[label],
                sample_id,
                color='blue',
                show=True
            )
        except Exception as e:
            assert False, f"custom_decision_plot raised an exception: {e}"