#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:42:29 2024

@author: sebastian
"""
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
# Setting seeds for reproducability
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
###########################################################################

# %%

### Section 1: Load the dataset
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data       # Features
y = data.target     # Labels

ids = pd.DataFrame()
ids['ID'] = ['patient ' + str(i) for i in range(1, len(X))]

feature_names = data.feature_names.tolist()
class_labels = data.target_names.tolist()

#%%
### Section 2: Preprocess the data
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
train_ids, test_ids = train_test_split(ids, test_size=0.3, random_state=42)

# Scale the features for better performance of the neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%
### Section 3: Build the neural network model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=20, batch_size=16, verbose=0)
# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

###########################################################################


#%%
### Section 4: Identify richt classified cases
# Get model predictions on test set
model_predictions_probabilities = model.predict(X_test_scaled)
model_predictions = (model_predictions_probabilities.flatten() > 0.5).astype(int)

# Get real labels of test data (already y_test)
true_labels = np.array(y_test).reshape((y_test.shape[0],1))

# Get the indices of the correctly classified cases
# Identify matching indices for each class
right_class_indices = {}
for class_id in np.unique(model_predictions):
    # Check wether the model prediction is the same as the real label
    match_indices = np.where((model_predictions == class_id) & (true_labels == class_id))[0]

    # Store the indices of the correctly classified cases for each class
    right_class_indices[class_labels[class_id]] = match_indices

# Print the number of correctly classified cases for each class
for class_id, indices in right_class_indices.items():
    print(f"Number of correctly classified {class_id}: {len(indices)}")


# %%
### Section 5: Calculate Shapley values
# Create a SHAP explainer
explainer = shap.Explainer(model, X_train_scaled)
shap_values = explainer(X_test_scaled)
# Get the SHAP values and base values
shap_raw_values = shap_values.values
shap_base_values = shap_values.base_values

shap_values_right_class = {}
for class_id in class_labels:
    # Get the indices of the right-classified cases for the current class
    indices = right_class_indices[class_id]
    class_ids = ids.iloc[indices]
    
    # Extract the corresponding SHAP values for those cases
    shap_values_right_class[class_id] = {'values' : shap_raw_values[indices],
                                            'base value': np.mean(shap_base_values),
                                            'ids' : class_ids['ID'].tolist()
                                            }
# %%
### Section 6: Calculate distribution of SHAP values for each feature per class

# Initialize dictionary to hold feature distributions per class
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
### Section 7: Create COVA Matrix
## Subsection 7.1: Create COVA matrix
class_COVAS_matrix = {}
for class_id in class_labels:
    # Select the SHAP values for the current class
    current_selected_shap_vlaues = shap_values_right_class[class_id]['values']

    raw_matrix = pd.DataFrame(current_selected_shap_vlaues, columns=feature_names)
    COVAS_matrix = raw_matrix.copy()
    for feature in raw_matrix.columns:
        mean = feature_distribution_info[feature]['mean']
        std = feature_distribution_info[feature]['std']
        COVAS_matrix[feature] = np.abs((COVAS_matrix[feature] - mean) / std)

    # Store the COVAS matrix for this class
    class_COVAS_matrix[class_id] = COVAS_matrix

## Subsection 7.2: Create COVA matrix with a threshold method
# T.B.I

# %%
### Section 8: Create the COVA Score 
class_COVAS = {}
for class_id in class_labels:
    # Select the COVAS matrix for the current class
    current_COVAS_matrix = class_COVAS_matrix[class_id]

    number_of_CO_cases = np.sum(np.array(current_COVAS_matrix), axis=1) # CO = Classification Outliers
    number_of_features = current_COVAS_matrix.shape[1]
    COVA_Score = number_of_CO_cases/number_of_features

    # Store current COVA Score and COVAS matrix as a DataFrame
    COVAS_scoring_df = pd.DataFrame(COVA_Score,columns= ['COVAS'], index=right_class_indices[class_id]).sort_values(by='COVAS', ascending=False)
    COVAS_matrix = pd.DataFrame(current_COVAS_matrix, columns= feature_names)
    COVAS_matrix.index = right_class_indices[class_id]

    # Store the COVA Score for this class
    class_COVAS[class_id] = {'COVAS Score' : COVAS_scoring_df,
                            'COVAS Matrix' : COVAS_matrix,
                            'IDs' : shap_values_right_class[class_id]['ids']
                            }
    

# %%
### Section 9: Decision plot (SHAP)
# User can configure which std-dev lines to include: specify as list of ints [1,2,3], 'none' for none, or 'all' for all levels
std_levels = 'all'  # <-- modify this to a list like [1,2,3], 'all', or 'none'
# Option to enable or disable plotting of the mean path line and markers
plot_mean = True  # set to False to hide mean path line
if std_levels == 'all':
    std_levels = [1,2,3]
elif std_levels == 'none':
    std_levels = []

height_in_inches = 10 #placeholder
width_in_pixels = 2926
DPI = 300
# Convert width from pixels to inches
width_in_inches = width_in_pixels / DPI

class_id = 'benign' # placeholder

# Prepare figure with desired size
plt.figure(figsize=(width_in_inches, height_in_inches))
# Generate SHAP decision plot without immediate display
shap.decision_plot(
    shap_values_right_class[class_id]['base value'],
    shap_values_right_class[class_id]['values'],
    X_test,
    feature_names,
    show=False,
    ignore_warnings=True
)
ax = plt.gca()
# Initialize cumulative sums for means and std deviations
cumulative_mean = shap_values_right_class[class_id]['base value']
cumulative_neg_std = shap_values_right_class[class_id]['base value']
cumulative_pos_std = shap_values_right_class[class_id]['base value']


ax = plt.gca()
# Get the actual plot order from the decision_plot
order = [tick.get_text() for tick in ax.get_yticklabels() if tick.get_text()]
shap_vals = shap_values_right_class[class_id]['values'][:, [feature_names.index(f) for f in order]]

# Compute cumulative contributions per sample
base = shap_values_right_class[class_id]['base value']
cum_paths = base + np.cumsum(shap_vals, axis=1)

# Compute mean and std at each feature index
mean_path = np.mean(cum_paths, axis=0)
std_path = np.std(cum_paths, axis=0)

# Connect mean path points with a line
# Connect mean path points with a line
# Use new blue tone for mean path line
if plot_mean:
    ax.plot(mean_path, range(len(order)), linestyle='-', linewidth=2, zorder=4, color='#333333', label='Mean Path Line')
# Connect ±1 Std bounds
if 1 in std_levels:
    ax.plot(mean_path - std_path, range(len(order)), linestyle='--', linewidth=1, zorder=3, color='#82A582', label='±1 Std Line')
    ax.plot(mean_path + std_path, range(len(order)), linestyle='--', linewidth=1, zorder=3, color='#82A582', label='_nolegend_')
    ax.fill_betweenx(
        range(len(order)),
        mean_path - std_path,
        mean_path + std_path,
        color='#82A582',
        alpha=0.2,
        label='68% Perzentil'
    )
# Connect ±2 Std bounds
if 2 in std_levels:
    ax.plot(mean_path - 2*std_path, range(len(order)), linestyle=':', linewidth=1, zorder=2, color='#517551', label='±2 Std Line')
    ax.plot(mean_path + 2*std_path, range(len(order)), linestyle=':', linewidth=1, zorder=2, color='#517551', label='_nolegend_')
    ax.fill_betweenx(
        range(len(order)),
        mean_path - 2*std_path,
        mean_path + 2*std_path,
        color='#517551',
        alpha=0.2,
        label='95% Perzentil'
    )
# Connect ±3 Std bounds
if 3 in std_levels:
    ax.plot(mean_path - 3*std_path, range(len(order)), linestyle='-.', linewidth=1, zorder=1, color='#2F4F2F', label='±3 Std Line')
    ax.plot(mean_path + 3*std_path, range(len(order)), linestyle='-.', linewidth=1, zorder=1, color='#2F4F2F', label='_nolegend_')
    ax.fill_betweenx(
        range(len(order)),
        mean_path - 3*std_path,
        mean_path + 3*std_path,
        color='#2F4F2F',
        alpha=0.2,
        label='99.7% Perzentil'
    )

for idx, feature in enumerate(order):
    # Plot mean cumulative SHAP
    if plot_mean:
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
    if 1 in std_levels:
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
    if 2 in std_levels:
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
    if 3 in std_levels:
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
ax.set_title(f"SHAP Decision Plot for {class_id} with Mean Path", fontsize=26)
plt.show()

# %%
