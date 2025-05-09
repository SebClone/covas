#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# basic_COVAS.py
#
# This script demonstrates the COVAS (Classification Outlier Value Assessment via Shapley) methodology
# for analyzing neural network predictions on the breast cancer dataset. It builds a neural network,
# computes SHAP values for model interpretability, constructs class-wise outlier matrices (COVAS),
# and visualizes SHAP decision plots with custom overlays.
# =============================================================================
#%%
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

#%%
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


#%%
### Section 4: Identify correctly classified cases
model_predictions_probabilities = model.predict(X_test_scaled)
model_predictions = (model_predictions_probabilities.flatten() > 0.5).astype(int)

#
true_labels = np.array(y_test).reshape((y_test.shape[0],1))

right_class_indices = {}
for class_id in np.unique(model_predictions):
    # Find indices where prediction matches true label for each class
    match_indices = np.where((model_predictions == class_id) & (true_labels == class_id))[0]
    right_class_indices[class_labels[class_id]] = match_indices

for class_id, indices in right_class_indices.items():
    print(f"Number of correctly classified {class_id}: {len(indices)}")


# %%
### Section 5: Calculate SHAP values
explainer = shap.Explainer(model, X_train_scaled)
shap_values = explainer(X_test_scaled)
# Get the SHAP values and base values
shap_raw_values = shap_values.values
shap_base_values = shap_values.base_values

shap_values_right_class = {}
for class_id in class_labels:
    # Extract SHAP values for correctly classified cases of each class
    indices = right_class_indices[class_id]
    class_ids = test_ids.iloc[indices]
    shap_values_right_class[class_id] = {
        'values' : shap_raw_values[indices],
        'base value': np.mean(shap_base_values),
        'ids' : class_ids.tolist()
    }
# %%
### Section 6: Calculate SHAP value distribution per feature/class
class_feature_distribution = {}
for class_id in class_labels:
    # Select the SHAP values for the current class
    selected_shap_values = shap_values_right_class[class_id]['values']
    # Get the corresponding IDs
    selected_ids = shap_values_right_class[class_id]['ids']

    feature_distribution_info = {}
    for feature in range(len(feature_names)):
        considered_feature= {feature_names[feature] : selected_shap_values[:,feature]}

        feature_mean = np.mean(considered_feature[feature_names[feature]])
        feature_std_dev = np.std(considered_feature[feature_names[feature]])
        feature_distribution_info[feature_names[feature]] = {'fdata'   : selected_shap_values[:,feature],
                                                            'mean'   : feature_mean,
                                                            'std'    : feature_std_dev
                                                            }
    # Store distribution info for this class
    class_feature_distribution[class_id] = feature_distribution_info


# %%
# Set COVA mode: 'continuous' for z-score values, 'threshold' for binary outlier flags
COVA_mode = 'threshold'  # or 'threshold'
# Define threshold in standard deviations for threshold mode
threshold_std = 1

# %%
### Section 7: Create COVA Matrix
class_COVAS_matrix = {}
for class_id in class_labels:
    # Select the SHAP values for the current class
    current_selected_shap_vlaues = shap_values_right_class[class_id]['values']

    raw_matrix = pd.DataFrame(current_selected_shap_vlaues, columns=feature_names)
    COVAS_matrix = raw_matrix.copy()
    for feature in raw_matrix.columns:
        mean = feature_distribution_info[feature]['mean']
        std = feature_distribution_info[feature]['std']
        if COVA_mode == 'continuous':
            # continuous z-score based COVAS
            COVAS_matrix[feature] = np.abs((COVAS_matrix[feature] - mean) / std)
        elif COVA_mode == 'threshold':
            # binary threshold: 1 if abs(z-score) > threshold_std, else 0
            z_vals = (COVAS_matrix[feature] - mean) / std
            COVAS_matrix[feature] = (np.abs(z_vals) > threshold_std).astype(int)
        else:
            raise ValueError(f"Unknown COVA_mode: {COVA_mode}. Use 'continuous' or 'threshold'.")

    class_COVAS_matrix[class_id] = COVAS_matrix

# Subsection 7.2: COVA matrix with threshold method (to be implemented)

# %%
### Section 8: Compute COVA Score 
class_COVAS = {}
for class_id in class_labels:
    # Select the COVAS matrix for the current class
    current_COVAS_matrix = class_COVAS_matrix[class_id]

    number_of_CO_cases = np.sum(np.array(current_COVAS_matrix), axis=1) # CO = Classification Outliers
    number_of_features = current_COVAS_matrix.shape[1]
    COVA_Score = number_of_CO_cases/number_of_features

    # Store COVA Score and COVAS matrix as a DataFrame
    ids_for_class = shap_values_right_class[class_id]['ids']
    COVAS_scoring_df = pd.DataFrame(
        COVA_Score,
        columns=['COVAS'],
        index=ids_for_class
    ).sort_values(by='COVAS', ascending=False)
    COVAS_matrix = pd.DataFrame(current_COVAS_matrix, columns=feature_names)
    COVAS_matrix.index = ids_for_class
    class_COVAS[class_id] = {
        'COVAS Score' : COVAS_scoring_df,
        'COVAS Matrix' : COVAS_matrix,
        'IDs' : shap_values_right_class[class_id]['ids']
    }

# %%
### Section 9: SHAP Decision plot
def custom_decision_plot(shap_base, shap_vals, X_test, feature_names,
                         scatter_levels=None, line_levels=None, fill_levels=None):
    """
    Create a custom SHAP decision plot with optional overlays for mean path,
    standard deviation bounds, percentile fills, and scatter markers.

    Parameters
    ----------
    shap_base : float
        The expected value (base value) from the SHAP explainer.
    shap_vals : np.ndarray
        SHAP values for the samples to plot (shape: [samples, features]).
    X_test : np.ndarray or pd.DataFrame
        The feature values for the samples (used for SHAP plotting).
    feature_names : list of str
        List of feature names (order must match shap_vals columns).
    scatter_levels : list of str, optional
        Levels at which to plot scatter markers (e.g., ['mean', '1 std']).
    line_levels : list of str, optional
        Which lines to draw (e.g., ['mean', '1 std', '2 std']).
    fill_levels : list of str, optional
        Which percentile bands to fill (e.g., ['68%', '95%']).

    Returns
    -------
    None
        Displays a matplotlib plot.
    """
    if scatter_levels is None and line_levels is None and fill_levels is None:
        shap.decision_plot(shap_base, shap_vals, X_test, feature_names, show=True)
        return
    if scatter_levels is None:
        scatter_levels = []
    if line_levels is None:
        line_levels = []
    if fill_levels is None:
        fill_levels = []

    # std_levels = 'all'  # <-- modify this to a list like [1,2,3], 'all', or 'none'

    # Determine std_levels numeric list based on line_levels
    std_map = {'1 std': 1, '2 std': 2, '3 std': 3}
    if 'all' in line_levels:
        std_levels = [1, 2, 3]
    else:
        std_levels = [std_map[s] for s in line_levels if s in std_map]

    plot_scatter = {'mean': 'mean' in scatter_levels}
    plot_scatter.update({f'{i} std': (f'{i} std' in scatter_levels) for i in [1,2,3]})

    plot_fill = {'1': '68%' in fill_levels, '2': '95%' in fill_levels, '3': '99%' in fill_levels}

    height_in_inches = 10 #placeholder
    width_in_pixels = 2926
    DPI = 300
    # Convert width from pixels to inches
    width_in_inches = width_in_pixels / DPI

    # Prepare figure with desired size
    plt.figure(figsize=(width_in_inches, height_in_inches))
    # Generate SHAP decision plot without immediate display
    shap.decision_plot(
        shap_base,
        shap_vals,
        X_test,
        feature_names,
        show=False,
        ignore_warnings=True
    )
    ax = plt.gca()
    # Initialize cumulative sums for means and std deviations
    cumulative_mean = shap_base
    cumulative_neg_std = shap_base
    cumulative_pos_std = shap_base


    ax = plt.gca()
    # Get the actual plot order from the decision_plot
    order = [tick.get_text() for tick in ax.get_yticklabels() if tick.get_text()]
    shap_vals_ordered = shap_vals[:, [feature_names.index(f) for f in order]]

    # Compute cumulative contributions per sample
    base = shap_base
    cum_paths = base + np.cumsum(shap_vals_ordered, axis=1)

    # Compute mean and std at each feature index
    mean_path = np.mean(cum_paths, axis=0)
    std_path = np.std(cum_paths, axis=0)

    if 'mean' in line_levels:
        ax.plot(mean_path, range(len(order)), linestyle='-', linewidth=2, zorder=4, color='#333333', label='Mean Path Line')
    # Connect ±1 Std bounds
    if 1 in std_levels:
        ax.plot(mean_path - std_path, range(len(order)), linestyle='--', linewidth=2, zorder=3, color='#82A582', label='±1 Std Line')
        ax.plot(mean_path + std_path, range(len(order)), linestyle='--', linewidth=2, zorder=3, color='#82A582', label='_nolegend_')
    if 1 in std_levels and plot_fill['1']:
        ax.fill_betweenx(
            range(len(order)),
            mean_path - std_path,
            mean_path + std_path,
            color='#82A582',
            alpha=0.4,
            zorder=4,
            label='68% Perzentil'
        )
    # Connect ±2 Std bounds
    if 2 in std_levels:
        ax.plot(mean_path - 2*std_path, range(len(order)), linestyle='--', linewidth=2, zorder=2, color='#517551', label='±2 Std Line')
        ax.plot(mean_path + 2*std_path, range(len(order)), linestyle='--', linewidth=2, zorder=2, color='#517551', label='_nolegend_')
    if 2 in std_levels and plot_fill['2']:
        ax.fill_betweenx(
            range(len(order)),
            mean_path - 2*std_path,
            mean_path + 2*std_path,
            color='#517551',
            alpha=0.4,
            zorder=4,
            label='95% Perzentil'
        )
    # Connect ±3 Std bounds
    if 3 in std_levels:
        ax.plot(mean_path - 3*std_path, range(len(order)), linestyle='--', linewidth=2, zorder=1, color='#2F4F2F', label='±3 Std Line')
        ax.plot(mean_path + 3*std_path, range(len(order)), linestyle='--', linewidth=2, zorder=1, color='#2F4F2F', label='_nolegend_')
    if 3 in std_levels and plot_fill['3']:
        ax.fill_betweenx(
            range(len(order)),
            mean_path - 3*std_path,
            mean_path + 3*std_path,
            color='#2F4F2F',
            alpha=0.4,
            zorder=4,
            label='99.7% Perzentil'
        )

    for idx, feature in enumerate(order):
        # Plot mean cumulative SHAP
        if plot_scatter['mean']:
            if idx == 0:
                ax.scatter(
                    mean_path[idx],
                    idx,
                    marker='D',
                    s=50,
                    zorder=5,
                    color='#333333',
                    label='Mean Path'
                )
            else:
                ax.scatter(
                    mean_path[idx],
                    idx,
                    marker='D',
                    s=50,
                    zorder=5,
                    color='#333333'
                )
        # Plot symmetric std bounds
        if 1 in std_levels and plot_scatter['1 std']:
            if idx == 0:
                ax.scatter(
                    [mean_path[idx] - std_path[idx], mean_path[idx] + std_path[idx]],
                    [idx, idx],
                    marker='X',
                    s=50,
                    zorder=5,
                    color='#82A582',
                    label='±1 Std'
                )
            else:
                ax.scatter(
                    [mean_path[idx] - std_path[idx], mean_path[idx] + std_path[idx]],
                    [idx, idx],
                    marker='X',
                    s=50,
                    zorder=5,
                    color='#82A582'
                )
        # Plot ±2 Std bounds
        if 2 in std_levels and plot_scatter['2 std']:
            if idx == 0:
                ax.scatter(
                    [mean_path[idx] - 2*std_path[idx], mean_path[idx] + 2*std_path[idx]],
                    [idx, idx],
                    marker='s',
                    s=50,
                    zorder=5,
                    color='#517551',
                    label='±2 Std'
                )
            else:
                ax.scatter(
                    [mean_path[idx] - 2*std_path[idx], mean_path[idx] + 2*std_path[idx]],
                    [idx, idx],
                    marker='s',
                    s=50,
                    zorder=5,
                    color='#517551'
                )
        # Plot ±3 Std bounds
        if 3 in std_levels and plot_scatter['3 std']:
            if idx == 0:
                ax.scatter(
                    [mean_path[idx] - 3*std_path[idx], mean_path[idx] + 3*std_path[idx]],
                    [idx, idx],
                    marker='^',
                    s=50,
                    zorder=5,
                    color='#2F4F2F',
                    label='±3 Std'
                )
            else:
                ax.scatter(
                    [mean_path[idx] - 3*std_path[idx], mean_path[idx] + 3*std_path[idx]],
                    [idx, idx],
                    marker='^',
                    s=50,
                    zorder=5,
                    color='#2F4F2F'
                )
    # Create a legend without duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    ax.set_title(f"SHAP Decision Plot with Mean Path", fontsize=26)
    plt.show()


#%%
# Example usage:
scatter_levels = ['none']
line_levels = ['mean', '2 std']
fill_levels = ['95%']  # Options: ['68%', '95%', '99%']

custom_decision_plot(
    shap_values_right_class[class_id]['base value'],
    shap_values_right_class[class_id]['values'],
    X_test, feature_names,
    scatter_levels=scatter_levels,
    line_levels=line_levels,
    fill_levels=fill_levels
)

# %%
