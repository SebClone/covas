"""
shap_analysis.py

This module provides functionality to compute SHAP values and derive per-feature 
SHAP value distributions for correctly classified instances in a classification context.

Functions:
----------
- get_shap_values_for_correct_classification: Computes SHAP values for correctly classified samples per class.
- get_feature_distribution: Computes feature-level statistics from SHAP values per class.

Dependencies:
-------------
- numpy
- pandas
- shap

Author: Sebastian Roth
"""
import numpy as np
import shap
import pandas as pd

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
