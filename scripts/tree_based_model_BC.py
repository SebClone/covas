# run_full_covas_pipeline.py
# Make covaslib discoverable even in VS Code's interactive window
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import shap

from covaslib.shap_analysis import get_feature_distribution
from covaslib.covas import get_COVA_matrix, get_COVA_score
from covaslib.plotting import custom_decision_plot

# Set seeds
random.seed(100)
np.random.seed(100)


####################################################################
#---------------Here goes your individual Dataset code-------------#
####################################################################
# Load data
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names.tolist()
class_labels = data.target_names.tolist()

ids = pd.DataFrame()
ids['ID'] = ['patient ' + str(i) for i in range(len(X))] # Create IDs for each sample

# Example decision plot for the first class
output_dir = Path(__file__).resolve().parents[1] / 'results'
####################################################################



# Split and scale
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids, test_size=0.3, random_state=100, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train model (tree-based)
model = RandomForestClassifier(
    n_estimators=300,
    random_state=100,
    n_jobs=-1,
    class_weight=None,
)
model.fit(X_train_scaled, y_train)

# Evaluate model
accuracy = model.score(X_test_scaled, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Run full COVAS pipeline
# 1) Determine correctly classified samples per class
pred = model.predict(X_test_scaled)
correct_classification = {}

for class_idx, class_name in enumerate(class_labels):
    idx = np.where((y_test == class_idx) & (pred == class_idx))[0]
    correct_classification[class_name] = {
        'index': pd.Index(idx),
        'IDs': ids_test.iloc[idx]['ID'].tolist(),
    }

# 2) Compute SHAP values for the correctly classified samples (TreeExplainer)
explainer = shap.TreeExplainer(model)
shap_values_all = explainer.shap_values(X_test_scaled)

# For binary classification, shap_values_all is typically a list with 2 arrays (per class)
shap_vals = {}
for class_idx, class_name in enumerate(class_labels):
    idx = correct_classification[class_name]['index'].values
    shap_vals[class_name] = {
        'values': np.array(shap_values_all[class_idx])[idx]
    }

# 3) Continue with standard COVAS steps
feat_dist = get_feature_distribution(shap_vals, feature_names)
COVA_matrices = get_COVA_matrix('continuous', class_labels, shap_vals, feature_names, feat_dist)
COVA_scores = get_COVA_score(class_labels, COVA_matrices, shap_vals)



# Plot
scatter_levels = ['mean', '2 std']  # Options: ['mean', '1 std', '2 std', '3 std', 'all', 'none']   
line_levels = ['mean', '2 std']  # Options: ['mean', '1 std', '2 std', '3 std', 'all', 'none']
fill_levels = ['none']  # Options: ['68%', '95%', '99%', 'all', 'none']

for class_name in class_labels:
    # Subset feature matrix for correctly classified samples of this class
    indices = correct_classification[class_name]['index'].values
    X_subset = X_test_scaled[indices]

    custom_decision_plot(
        shap_vals,
        X_subset,
        feature_names,
        scatter_levels=scatter_levels,
        line_levels=line_levels,
        fill_levels=fill_levels,
        class_name=class_name,
        save_path=output_dir / f"rf_decision_plot_{class_name}.png",
        dpi=600,
        show=False,   # oder True, wenn du sie sehen willst
    )


# Export individual COVA components per class
for class_name in class_labels:
    # Score
    score_df = pd.DataFrame(COVA_scores[class_name]['COVAS Score'])
    score_df.index.name = 'ID'
    score_df.to_csv(output_dir / f'rf_COVA_score_{class_name}.csv')

    # Matrix
    matrix_df = pd.DataFrame(COVA_scores[class_name]['COVAS Matrix'], index=COVA_scores[class_name]['IDs'], columns=feature_names)
    matrix_df.index.name = 'ID'
    matrix_df.to_csv(output_dir / f'rf_COVA_matrix_{class_name}.csv')

    # IDs
    ids_df = pd.DataFrame(COVA_scores[class_name]['IDs'])
    ids_df.to_csv(output_dir / f'rf_COVA_IDs_{class_name}.csv', index=False)
    
    print(f"Exported COVA components for class '{class_name}'")
# %%
