# COVAS: Classification Outlier Variability Score

COVAS is a Python library for detecting explanation-level outliers in classification tasks using SHAP-based model explanations.  
Instead of measuring anomalies in the input feature space, COVAS evaluates how much an instance's SHAP decision trajectory deviates from the typical class-specific explanation pattern.  
This enables the identification of rare, atypical, or informative cases that remain hidden in standard performance metrics.

## Features

- Explanation-level outlier scoring (continuous or threshold-based)
- Class-wise SHAP value distribution analysis
- Enhanced SHAP decision plots with:
  - mean SHAP path  
  - standard deviation bands  
  - percentile shading  
  - customizable scatter markers  
- Works with all SHAP-compatible ML models  
- Reproducible pipelines and ready-to-use examples  
- Installable as a standard Python package

## Installation

### Install directly from GitHub (recommended once repository is public)

```bash
pip install git+https://github.com/SebClone/covas.git
```

### Development installation (editable mode)

```bash
pip install -e git+https://github.com/SebClone/covas.git#egg=covas
```

### Local installation

```bash
git clone https://github.com/SebClone/covas.git
cd covas
pip install .
```

Or for editable mode:

```bash
pip install -e .
```

## Example Usage

```python
from covaslib.classification import get_correct_classification
from covaslib.shap_analysis import (
    get_shap_values_for_correct_classification,
    get_feature_distribution
)
from covaslib.covas import get_COVA_matrix, get_COVA_score
from covaslib.plotting import custom_decision_plot

correct = get_correct_classification(model, X_test_scaled, y_test, ids_test, class_labels)
shap_vals = get_shap_values_for_correct_classification(
    model, X_train_scaled, X_test_scaled, correct, class_labels
)
feat_dist = get_feature_distribution(shap_vals, feature_names)

COVA_matrices = get_COVA_matrix("continuous", class_labels, shap_vals, feature_names, feat_dist)
COVA_scores = get_COVA_score(class_labels, COVA_matrices, shap_vals)

custom_decision_plot(
    shap_dictonary=shap_vals,
    X_test=X_test_scaled,
    feature_names=feature_names,
    class_name="malignant",
    save_path="decision_plot_malignant.png",
    dpi=600,
    show=True
)
```

## Project Structure

```
covaslib/
├── classification.py
├── covas.py
├── shap_analysis.py
└── plotting.py

examples/
└── pipeline_full_BC.py
```

## Dependencies

- numpy
- pandas
- scikit-learn
- tensorflow / keras
- shap
- matplotlib

## Citation

```
Comming soon...
```

## License

MIT License

## Contact

Sebastian Roth  
University of Applied Sciences Koblenz  
sroth@hs-koblenz.de
