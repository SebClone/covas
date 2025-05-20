"""
covas.py

This module contains core functionality for calculating Classification Outlier Value Assessment (COVA)
matrices and deriving COVA scores for model explainability and outlier detection in classification tasks.

Functions:
----------
- get_COVA_matrix: Computes the COVA matrix per class using SHAP values and class-specific feature distributions.
- get_COVA_score: Computes normalized outlier scores per sample from COVA matrices.

Dependencies:
-------------
- numpy
- pandas

Author: Sebastian Roth
"""
import numpy as np
import pandas as pd

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
