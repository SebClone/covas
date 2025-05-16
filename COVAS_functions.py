#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COVAS_functions.py

This module implements the Classification Outlier Value Assessment (COVAS) methodology using SHAP values.
It provides functions to:
  - Identify correctly classified instances per class.
  - Compute SHAP values for those instances.
  - Calculate feature-wise distribution statistics of SHAP values.
  - Construct the COVA matrix in continuous or threshold modes.
  - Compute COVA scores per class.
  - Generate custom SHAP decision plots with optional overlays.

Usage:
    Import the desired function and call with appropriate model and data inputs.

Requirements:
    - pandas
    - numpy
    - tensorflow / keras
    - shap
    - matplotlib
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

# Setting seeds for reproducibility
random.seed(100)
np.random.seed(100)
tf.random.set_seed(100)

def get_correct_classification(model, X_test_scaled, y_test, test_ids, class_labels):
    """
    Identify correctly classified instances per class.

    Parameters
    ----------
    model : keras.Model
        Trained neural network model.
    X_test_scaled : numpy.ndarray
        Scaled test feature data.
    y_test : array-like
        True labels for test data.
    test_ids : array-like
        String identifiers corresponding to test samples.
    class_labels : list of str
        Names of the classes.

    Returns
    -------
    dict
        Dictionary mapping each class name to a DataFrame of correctly classified instances with
        columns:
          - 'index': position of the instance in X_test_scaled
          - 'ID': identifier from test_ids

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If input values are invalid or inconsistent.
    """
    # Input validation
    if not hasattr(model, "predict"):
        raise TypeError(f"model must have a .predict() method, got {type(model)}")
    if X_test_scaled.shape[0] != len(y_test) or len(y_test) != len(test_ids):
        raise ValueError("Lengths of X_test_scaled, y_test, and test_ids must be equal")
    if not isinstance(class_labels, list) or not all(isinstance(c, str) for c in class_labels):
        raise TypeError("class_labels must be a list of strings")
    # Predict and convert to class labels
    y_pred_raw = model.predict(X_test_scaled).flatten()
    y_pred = (y_pred_raw > 0.5).astype(int)

    # Build Series for true and predicted labels, indexed by IDs
    y_true = pd.Series(y_test, index=test_ids)
    y_pred_series = pd.Series(y_pred, index=test_ids)

    correct_classification = {}
    for class_id, class_name in enumerate(class_labels):
        # Mask correctly classified instances for this class
        mask_correct = (y_true == class_id) & (y_pred_series == class_id)
        indices = np.where(mask_correct.values)[0]
        ids_list = mask_correct[mask_correct].index.tolist()
        class_df = pd.DataFrame({
            'index': indices,
            'ID': ids_list
        })
        correct_classification[class_name] = class_df
        print(f"Class '{class_name}': {len(class_df)} correctly classified instances")
    return correct_classification


def get_shap_values_for_correct_classification(model, X_train_scaled, X_test_scaled, correct_classification, class_labels):
    """
    Calculate SHAP values for correctly classified instances per class.

    Parameters
    ----------
    model : keras.Model
        Trained neural network model.
    X_train_scaled : numpy.ndarray
        Scaled training feature data for SHAP explainer fitting.
    X_test_scaled : numpy.ndarray
        Scaled test feature data to compute SHAP values.
    correct_classification : dict
        Dictionary from get_correct_classification mapping each class name to a DataFrame 
        of correctly classified instances with 'index' positions.
    class_labels : list of str
        Names of the classes.

    Returns
    -------
    dict
        Dictionary mapping each class name to a sub-dictionary with:
          - 'values': array of SHAP values for correctly classified samples of that class
          - 'base value': float, average SHAP base value
          - 'ids': list of sample IDs corresponding to those instances

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If input values are invalid or inconsistent.
    """
    # Input validation
    if not hasattr(model, "predict"):
        raise TypeError(f"model must have a .predict() method, got {type(model)}")
    if X_train_scaled.ndim != 2 or X_test_scaled.ndim != 2:
        raise ValueError("X_train_scaled and X_test_scaled must be 2D arrays")
    if not isinstance(correct_classification, dict):
        raise TypeError("correct_classification must be a dict mapping class names to DataFrames")
    if not isinstance(class_labels, list) or not all(isinstance(c, str) for c in class_labels):
        raise TypeError("class_labels must be a list of strings")
    # Initialize SHAP explainer and compute values for all test samples
    explainer = shap.Explainer(model, X_train_scaled)
    shap_result = explainer(X_test_scaled)
    shap_raw_values = shap_result.values
    shap_base_values = shap_result.base_values

    shap_values_right_class = {}
    for class_id, class_name in enumerate(class_labels):
        # Retrieve indices and IDs for correctly classified instances
        class_df = correct_classification[class_name]
        indices = class_df['index'].values
        ids_list = class_df['ID'].tolist()
        # Store SHAP values, average base value, and IDs
        shap_values_right_class[class_name] = {
            'values': shap_raw_values[indices],
            'base value': float(np.mean(shap_base_values)),
            'ids': ids_list
        }
    return shap_values_right_class


def get_feature_distribution(shap_values_right_class, feature_names):
    """
    Calculate distribution statistics for SHAP values per feature and class.

    Parameters
    ----------
    shap_values_right_class : dict
        Dictionary mapping each class name to a sub-dictionary containing:
          - 'values': array of SHAP values for correctly classified samples.
    feature_names : list of str
        List of feature names corresponding to the columns in SHAP values.

    Returns
    -------
    dict
        Dictionary mapping each class name to a dictionary where each feature name maps to:
          - 'fdata': array of SHAP values for that feature and class
          - 'mean': float, mean of SHAP values for that feature
          - 'std': float, standard deviation of SHAP values for that feature

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If input values are invalid or inconsistent.
    """
    # Input validation
    if not isinstance(shap_values_right_class, dict):
        raise TypeError("shap_values_right_class must be a dict")
    if not isinstance(feature_names, list) or not all(isinstance(f, str) for f in feature_names):
        raise TypeError("feature_names must be a list of strings")
    class_feature_distribution = {}
    for class_name, info in shap_values_right_class.items():
        shap_vals = info['values']
        feature_distribution_info = {}
        for idx, fname in enumerate(feature_names):
            fdata = shap_vals[:, idx]
            feature_distribution_info[fname] = {
                'fdata': fdata,
                'mean': float(np.mean(fdata)),
                'std': float(np.std(fdata))
            }
        class_feature_distribution[class_name] = feature_distribution_info
    return class_feature_distribution

def get_COVA_matrix(COVA_mode, class_labels, shap_values_right_class, feature_names, class_feature_distribution, threshold_std=None):
    """
    Calculate the Classification Outlier Value Assessment (COVA) matrix for each class.

    Parameters
    ----------
    COVA_mode : str
        Mode of COVA calculation: 'continuous' for z-score values, 'threshold' for binary flags.
    class_labels : list of str
        Names of the classes.
    shap_values_right_class : dict
        Dictionary mapping each class name to a sub-dictionary containing 'values': array of SHAP values.
    feature_names : list of str
        List of feature names corresponding to columns in the SHAP arrays.
    class_feature_distribution : dict
        Dictionary mapping each class name to per-feature distribution info:
          - For each feature, a dict with keys 'mean' and 'std'.
    threshold_std : int, optional
        Threshold in standard deviations to flag an outlier when COVA_mode is 'threshold'. Required if mode is 'threshold'.

    Returns
    -------
    dict
        Dictionary mapping each class name to a pandas DataFrame representing the COVA matrix
        (rows are samples, columns are features), where:
          - In 'continuous' mode: each cell is the absolute z-score.
          - In 'threshold' mode: each cell is 1 if |z-score| > threshold_std, else 0.

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If input values are invalid or inconsistent.
    """
    # Input validation
    if COVA_mode not in ("continuous", "threshold"):
        raise ValueError(f"COVA_mode must be 'continuous' or 'threshold', got {COVA_mode!r}")
    if COVA_mode == "threshold" and threshold_std is None:
        raise ValueError("threshold_std must be provided when COVA_mode is 'threshold'")
    if not isinstance(class_labels, list) or not all(isinstance(c, str) for c in class_labels):
        raise TypeError("class_labels must be a list of strings")
    if not isinstance(feature_names, list) or not all(isinstance(f, str) for f in feature_names):
        raise TypeError("feature_names must be a list of strings")
    if not isinstance(class_feature_distribution, dict):
        raise TypeError("class_feature_distribution must be a dict")
    cova_matrix_dict = {}
    for class_name in class_labels:
        # Build raw DataFrame of SHAP values
        shap_vals = shap_values_right_class[class_name]['values']
        raw_df = pd.DataFrame(shap_vals, columns=feature_names)
        # Initialize COVA matrix
        cova_df = raw_df.copy()
        # Apply mode-specific calculation per feature
        for feature in feature_names:
            mean = class_feature_distribution[class_name][feature]['mean']
            std = class_feature_distribution[class_name][feature]['std']
            if COVA_mode == 'continuous':
                cova_df[feature] = np.abs((raw_df[feature] - mean) / std)
            elif COVA_mode == 'threshold':
                if threshold_std is None:
                    raise ValueError("threshold_std must be provided when COVA_mode is 'threshold'.")
                z_vals = (raw_df[feature] - mean) / std
                cova_df[feature] = (np.abs(z_vals) > threshold_std).astype(int)
            else:
                raise ValueError(f"Unknown COVA_mode: {COVA_mode}. Use 'continuous' or 'threshold'.")
        cova_matrix_dict[class_name] = cova_df
    return cova_matrix_dict

def get_COVA_score(class_labels, class_COVA_matrix, shap_values_right_class):
    """
    Calculate COVA scores per class.

    Parameters
    ----------
    class_labels : list of str
        Names of the classes.
    class_COVA_matrix : dict
        Dictionary mapping each class name to its pandas DataFrame COVA matrix.
    shap_values_right_class : dict
        Dictionary mapping each class name to a sub-dictionary containing:
          - 'ids': list of sample IDs for correctly classified instances.

    Returns
    -------
    dict
        Dictionary mapping each class name to a dictionary with:
          - 'COVAS Score': DataFrame of COVA scores (column 'COVAS') indexed by sample IDs, sorted descending.
          - 'COVAS Matrix': COVA matrix DataFrame with sample IDs as index.
          - 'IDs': list of sample IDs.

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If input values are invalid or inconsistent.
    """
    # Input validation
    if not isinstance(class_labels, list) or not all(isinstance(c, str) for c in class_labels):
        raise TypeError("class_labels must be a list of strings")
    if not isinstance(class_COVA_matrix, dict):
        raise TypeError("class_COVA_matrix must be a dict mapping class names to DataFrames")
    if not isinstance(shap_values_right_class, dict):
        raise TypeError("shap_values_right_class must be a dict mapping class names to info dicts")
    class_COVAS = {}
    for class_name in class_labels:
        # Retrieve the COVA matrix and IDs for this class
        current_matrix = class_COVA_matrix[class_name]
        ids_for_class = shap_values_right_class[class_name]['ids']
        # Compute number of outlier features per sample and normalize
        number_of_CO_cases = np.sum(current_matrix.values, axis=1)
        number_of_features = current_matrix.shape[1]
        COVA_score = number_of_CO_cases / number_of_features

        # Build scoring DataFrame
        COVAS_scoring_df = pd.DataFrame(
            COVA_score,
            columns=['COVAS'],
            index=ids_for_class
        ).sort_values(by='COVAS', ascending=False)

        # Ensure the matrix DataFrame is indexed by sample IDs
        COVAS_matrix_df = current_matrix.copy()
        COVAS_matrix_df.index = ids_for_class

        class_COVAS[class_name] = {
            'COVAS Score': COVAS_scoring_df,
            'COVAS Matrix': COVAS_matrix_df,
            'IDs': ids_for_class
        }
    return class_COVAS

def custom_decision_plot(shap_dictonary, X_test, feature_names,
                         scatter_levels=None, line_levels=None, fill_levels=None, class_name=None):
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

    Raises
    ------
    TypeError
        If input types are incorrect.
    ValueError
        If input values are invalid or inconsistent.
    """
    # Input validation
    if class_name not in shap_dictonary:
        raise ValueError(f"class_name {class_name!r} not found in shap_dictionary keys")
    if not hasattr(plt, "figure"):
        raise TypeError("matplotlib.pyplot must be imported as plt")
    # Retrieve the SHAP base value (expected value) and SHAP values for the given class
    shap_base = shap_dictonary[class_name]['base value'],
    shap_vals = shap_dictonary[class_name]['values']

    # If no overlays are specified, generate a standard SHAP decision plot
    if scatter_levels is None and line_levels is None and fill_levels is None:
        shap.decision_plot(shap_base, shap_vals, X_test, feature_names, show=True)
        return
    if scatter_levels is None:
        scatter_levels = []
    if line_levels is None:
        line_levels = []
    if fill_levels is None:
        fill_levels = []

    # Map string levels to numeric standard deviation values
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
    # Set figure size for plotting, converting width from pixels to inches
    width_in_inches = width_in_pixels / DPI

    # Prepare figure with desired size
    plt.figure(figsize=(width_in_inches, height_in_inches))
    # Generate the decision plot but suppress immediate display
    shap.decision_plot(
        shap_base,
        shap_vals,
        X_test,
        feature_names,
        show=False,
        ignore_warnings=True
    )
    ax = plt.gca()
    # Initialize cumulative sums for means and std deviations (not used but kept for clarity)
    cumulative_mean = shap_base
    cumulative_neg_std = shap_base
    cumulative_pos_std = shap_base

    ax = plt.gca()
    # Determine feature order used in the SHAP decision plot
    order = [tick.get_text() for tick in ax.get_yticklabels() if tick.get_text()]
    shap_vals_ordered = shap_vals[:, [feature_names.index(f) for f in order]]

    # Compute cumulative SHAP paths, mean and standard deviation for overlays
    base = shap_base
    cum_paths = base + np.cumsum(shap_vals_ordered, axis=1)
    mean_path = np.mean(cum_paths, axis=0)
    std_path = np.std(cum_paths, axis=0)

    # Plot the mean/std lines and fill regions if enabled
    if 'mean' in line_levels:
        ax.plot(mean_path, range(len(order)), linestyle='-', linewidth=2, zorder=4, color='#333333', label='Mean Path Line')
    # Plot the ±1 Std lines and fill
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
    # Plot the ±2 Std lines and fill
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
    # Plot the ±3 Std lines and fill
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

    # Plot scatter markers for mean and std bounds at each feature position
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
    # Create a clean legend with unique labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')
    # Set plot title based on the class
    if class_name is not None:
        ax.set_title(f"SHAP Decision Plot for class {class_name} with Mean Path", fontsize=26)
    else:
        ax.set_title(f"SHAP Decision Plot with Mean Path", fontsize=26)
    plt.show()

