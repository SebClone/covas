"""
classification.py

This module provides functionality to identify correctly classified instances
in a test dataset based on a trained classification model.

Functions:
----------
- get_correct_classification: Returns a dictionary of correctly classified instances per class.

Dependencies:
-------------
- pandas
- numpy
- Keras-compatible model for prediction

Author: Sebastian Roth
"""
import pandas as pd
import numpy as np

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

    # --- MODEL-AGNOSTIC PREDICTION ---
    if hasattr(model, "predict_proba"):
        y_pred_raw = model.predict_proba(X_test_scaled)
        y_pred = np.argmax(y_pred_raw, axis=1)
    else:
        y_pred = model.predict(X_test_scaled)

    # Build Series indexed by IDs
    y_true = pd.Series(y_test, index=test_ids)
    y_pred_series = pd.Series(y_pred, index=test_ids)

    correct_classification = {}
    for class_id, class_name in enumerate(class_labels):
        mask_correct = (y_true == class_id) & (y_pred_series == class_id)
        indices = np.where(mask_correct.values)[0]
        ids_list = mask_correct[mask_correct].index.tolist()

        correct_classification[class_name] = pd.DataFrame({
            'index': indices,
            'ID': ids_list
        })

        print(f"Class '{class_name}': {len(ids_list)} correctly classified instances")

    return correct_classification
