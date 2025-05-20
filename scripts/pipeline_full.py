# run_full_covas_pipeline.py
# Make covaslib discoverable even in VS Code's interactive window
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Set seeds
np.random.seed(100)
tf.random.set_seed(100)

# Load data
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names.tolist()
class_labels = data.target_names.tolist()

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=10, batch_size=16)

# Run full COVAS pipeline
correct_classification = get_correct_classification(model, X_test_scaled, y_test, pd.Series(range(len(y_test))), class_labels)
shap_vals = get_shap_values_for_correct_classification(model, X_train_scaled, X_test_scaled, correct_classification, class_labels)
feat_dist = get_feature_distribution(shap_vals, feature_names)
COVA_matrices = get_COVA_matrix('continuous', class_labels, shap_vals, feature_names, feat_dist)
COVA_scores = get_COVA_score(class_labels, COVA_matrices, shap_vals)

# Plot
for label in class_labels:
    idx = correct_classification[label]['index'].values
    if len(idx) == 0:
        continue
    X_subset = X_test_scaled[idx]
    if len(shap_vals[label]) != len(X_subset):
        continue
    custom_decision_plot(
        shap_vals,
        X_subset,
        feature_names,
        scatter_levels=['none'],
        line_levels=['mean', '2 std'],
        fill_levels=['95%'],
        class_name=label
    )