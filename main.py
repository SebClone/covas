#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

This script demonstrates the complete workflow for the Classification Outlier Value Assessment (COVAS) methodology
on the breast cancer dataset. It covers:
  - Loading and preprocessing data.
  - Building and training a neural network model.
  - Identifying correctly classified instances.
  - Calculating SHAP values for interpretability.
  - Computing feature-wise SHAP distribution statistics.
  - Constructing COVA matrices in continuous or threshold modes.
  - Computing COVA scores per class.
  - Generating SHAP decision plots.

Usage:
    python main.py

Requirements:
    - pandas
    - numpy
    - matplotlib
    - tensorflow / keras
    - scikit-learn
    - shap
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap
import COVAS_functions as covas

# %%
# Setting seeds for reproducibility
random.seed(100)
np.random.seed(100)
tf.random.set_seed(100)

# %%
### Section 1: Load dataset
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data       # Features
y = data.target     # Labels

ids = pd.DataFrame()
ids['ID'] = ['patient ' + str(i) for i in range(1, len(X) + 1)]

feature_names = data.feature_names.tolist()
class_labels = data.target_names.tolist()

#%%
### Section 2: Preprocess data
# Split X und y, aber gleichzeitig auch die IDs
X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
    X, y, ids['ID'], test_size=0.3, random_state=100
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#%%
### Section 3: Build neural network model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile and train the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=20, batch_size=16, verbose=0)
# Evaluate on test data
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

# %%
### Section 4: Find correct classification
correct_classification = covas.get_correct_classification(
    model,
    X_test_scaled,
    y_test,
    test_ids,
    class_labels
)

# %%
### Section 5: Calculate SHAP values for correctly classified instances
shap_values_right_class = covas.get_shap_values_for_correct_classification(
    model,
    X_train_scaled,
    X_test_scaled,
    correct_classification,
    class_labels
)

# %%
### Section 6: Calculate SHAP value distribution per feature/class
class_feature_distribution = covas.get_feature_distribution(shap_values_right_class, feature_names)

# %%
### Section 7: Calculate COVA Matrix per class
COVA_mode = 'continuous'  # or 'threshold'
#threshold_std = 1  # Standard deviation threshold for outlier detection
# Call the function to generate the COVA matrices
class_COVAS_matrix = covas.get_COVA_matrix(
    COVA_mode,
    class_labels,
    shap_values_right_class,
    feature_names,
    class_feature_distribution,
)

# %%
### Section 8: Compute COVA Score
class_COVAS = covas.get_COVA_score(
    class_labels,
    class_COVAS_matrix,
    shap_values_right_class
)

# %%
### Section 9: SHAP Decision plot
scatter_levels = ['none']
line_levels = ['mean', '2 std']
fill_levels = ['95%']  # Options: ['68%', '95%', '99%']

# Example decision plot for the first class
class_name = class_labels[1]
# Subset feature matrix for correctly classified samples of this class
indices = correct_classification[class_name]['index'].values
X_subset = X_test_scaled[indices]
covas.custom_decision_plot(
    shap_values_right_class,
    X_subset, feature_names,
    scatter_levels=scatter_levels,
    line_levels=line_levels,
    fill_levels=fill_levels,
    class_name='benign'
)
covas.custom_decision_plot(
    shap_values_right_class,
    X_subset, feature_names,
    scatter_levels=scatter_levels,
    line_levels=line_levels,
    fill_levels=fill_levels,
    class_name= class_labels[0]
)
# %%
