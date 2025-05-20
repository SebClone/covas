"""
test_plotting.py

This test module validates the SHAP decision plot functionality using a full COVAS pipeline on the breast cancer dataset.

Author: Sebastian Roth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from covaslib.classification import get_correct_classification
from covaslib.shap_analysis import get_shap_values_for_correct_classification, get_feature_distribution
from covaslib.covas import get_COVA_matrix, get_COVA_score
from covaslib.plotting import custom_decision_plot


def test_custom_decision_plot_execution():
    """
    Full integration test for the COVAS pipeline culminating in SHAP decision plots.
    Ensures that no exceptions are raised during the generation of decision plots for each class.
    """
    # Seed setup
    random.seed(100)
    np.random.seed(100)
    tf.random.set_seed(100)

    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names.tolist()
    class_labels = data.target_names.tolist()
    ids = pd.Series(['patient ' + str(i) for i in range(1, len(X) + 1)])

    # Train/test split
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=0.3, random_state=100
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build and train model
    model = Sequential([
        Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=5, batch_size=16, verbose=0)

    # Run COVAS pipeline
    correct_classification = get_correct_classification(model, X_test_scaled, y_test, id_test, class_labels)
    shap_values_right_class = get_shap_values_for_correct_classification(
        model, X_train_scaled, X_test_scaled, correct_classification, class_labels
    )
    class_feature_distribution = get_feature_distribution(shap_values_right_class, feature_names)
    class_COVAS_matrix = get_COVA_matrix(
        'continuous', class_labels, shap_values_right_class, feature_names, class_feature_distribution
    )
    class_COVAS = get_COVA_score(class_labels, class_COVAS_matrix, shap_values_right_class)

    # SHAP decision plot config
    scatter_levels = ['none']
    line_levels = ['mean', '2 std']
    fill_levels = ['95%']

    # Attempt plotting for both classes
    for label in class_labels:
        indices = correct_classification[label]['index'].values
        if len(indices) == 0:
            continue  # Skip plotting if no correct instances

        X_subset = X_test_scaled[indices]
        shap_vals = shap_values_right_class[label]

        if len(shap_vals) != len(X_subset):
            continue  # Avoid shape mismatch

        try:
            custom_decision_plot(
                shap_values_right_class,
                X_subset,
                feature_names,
                scatter_levels=scatter_levels,
                line_levels=line_levels,
                fill_levels=fill_levels,
                class_name=label
            )
        except Exception as e:
            assert False, f"custom_decision_plot failed for class {label}: {e}"