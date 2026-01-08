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

from covaslib.classification import get_correct_classification
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

# Output directories
results_dir = Path(__file__).resolve().parents[1] / 'results'
plots_dir = results_dir / 'plots'
deep_output_dir = results_dir / 'deep_analysis_results'
results_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
deep_output_dir.mkdir(parents=True, exist_ok=True)
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
# # 1) Determine correctly classified samples per class
# pred = model.predict(X_test_scaled)
# correct_classification = {}

# for class_idx, class_name in enumerate(class_labels):
#     idx = np.where((y_test == class_idx) & (pred == class_idx))[0]
#     correct_classification[class_name] = {
#         'index': pd.Index(idx),
#         'IDs': ids_test.iloc[idx]['ID'].tolist(),
#     }
correct_classification = get_correct_classification(model, X_test_scaled, y_test, ids_test, class_labels)

# 2) Compute SHAP values for the correctly classified samples (TreeExplainer)
explainer = shap.TreeExplainer(model)
shap_values_all = explainer.shap_values(X_test_scaled)
expected_value = explainer.expected_value

# Normalize SHAP outputs across SHAP versions
# - Binary classification can return either a list [class0, class1] or a single array
# - expected_value can be a list/array per class or a scalar
if isinstance(shap_values_all, list):
    shap_values_per_class = shap_values_all
else:
    # If a single array is returned, treat it as the positive class and derive negative as -values
    shap_values_per_class = [ -np.array(shap_values_all), np.array(shap_values_all) ]

if isinstance(expected_value, (list, np.ndarray)):
    expected_per_class = list(expected_value)
else:
    expected_per_class = [float(expected_value), float(expected_value)]

shap_vals = {}
for class_idx, class_name in enumerate(class_labels):
    idx = correct_classification[class_name]['index'].values
    ids_for_class = correct_classification[class_name]['ID'].tolist()

    shap_vals[class_name] = {
        'values': np.array(shap_values_per_class[class_idx])[idx],
        'ids': ids_for_class,
        'base value': float(expected_per_class[class_idx]),
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
        save_path=plots_dir / f"rf_decision_plot_{class_name}.png",
        dpi=600,
        show=False,   # oder True, wenn du sie sehen willst
    )


# Export individual COVA components per class
for class_name in class_labels:
    # Score
    score_df = pd.DataFrame(COVA_scores[class_name]['COVAS Score'])
    score_df.index.name = 'ID'
    score_df.to_csv(deep_output_dir / f'rf_COVA_score_{class_name}.csv')

    # Matrix
    matrix_df = pd.DataFrame(COVA_scores[class_name]['COVAS Matrix'], index=COVA_scores[class_name]['IDs'], columns=feature_names)
    matrix_df.index.name = 'ID'
    matrix_df.to_csv(deep_output_dir / f'rf_COVA_matrix_{class_name}.csv')

    # IDs
    ids_df = pd.DataFrame(COVA_scores[class_name]['IDs'])
    ids_df.to_csv(deep_output_dir / f'rf_COVA_IDs_{class_name}.csv', index=False)
    
    print(f"Exported COVA components for class '{class_name}'")
    

# %% Comparision with fead forward neural network model

# ---------------------------------------------------------------
# Compare Random Forest vs Feedforward NN (using exported COVAS score CSVs)
# ---------------------------------------------------------------

from scipy.stats import spearmanr


def _load_scores(path):
    """Load a score CSV and return a DataFrame indexed by ID with a 'COVAS' column."""
    df = pd.read_csv(path, index_col=0)

    # Normalize column name
    if 'COVAS Score' in df.columns and 'COVAS' not in df.columns:
        df = df.rename(columns={'COVAS Score': 'COVAS'})

    # Some exports may have a single unnamed column
    if 'COVAS' not in df.columns:
        if df.shape[1] == 1:
            df.columns = ['COVAS']
        else:
            raise KeyError(f"Expected a 'COVAS' column in {path.name}, got columns={list(df.columns)}")

    df.index = df.index.astype(str)
    # Clean tuple-like IDs, e.g., ("patient 152",)
    df.index = df.index.str.replace(r"^\('\s*", "", regex=True).str.replace(r"\s*',\)$", "", regex=True)
    return df[['COVAS']]


def top_k_ids(df, k):
    k = min(k, len(df))
    return list(df.sort_values('COVAS', ascending=False).head(k).index)


def overlap(a, b):
    a = list(a)
    b = list(b)
    a_set = set(a)
    if len(a_set) == 0:
        return 0.0
    return len(a_set & set(b)) / len(a_set)


# Expected FFNN files (produced by your NN pipeline)
ffnn_files = {
    'malignant': results_dir / 'COVA_score_malignant.csv',
    'benign': results_dir / 'COVA_score_benign.csv',
}

# Random Forest files (produced by this script)
rf_files = {
    'malignant': deep_output_dir / 'rf_COVA_score_malignant.csv',
    'benign': deep_output_dir / 'rf_COVA_score_benign.csv',
}

k = 10
rows = []

for class_name in ['malignant', 'benign']:
    if not ffnn_files[class_name].exists():
        raise FileNotFoundError(
            f"Missing FFNN score file: {ffnn_files[class_name]}. "
            f"Run your FFNN pipeline and ensure it exports COVA_score_{class_name}.csv into the results/ folder."
        )
    if not rf_files[class_name].exists():
        raise FileNotFoundError(f"Missing RF score file: {rf_files[class_name]}")

    df_ffnn = _load_scores(ffnn_files[class_name])
    df_rf = _load_scores(rf_files[class_name])

    common_ids = df_ffnn.index.intersection(df_rf.index)

    # Restrict to intersection for rank-based comparisons
    f_common = df_ffnn.loc[common_ids]
    r_common = df_rf.loc[common_ids]

    # Top-k overlap on intersection only (to avoid comparing non-shared samples)
    top_ffnn = top_k_ids(f_common, k)
    top_rf = top_k_ids(r_common, k)

    ov = overlap(top_ffnn, top_rf)

    if len(common_ids) > 1:
        rho, p = spearmanr(f_common['COVAS'].values, r_common['COVAS'].values)
    else:
        rho, p = (np.nan, np.nan)

    rows.append({
        'class': class_name,
        'common_samples': int(len(common_ids)),
        'top_k': int(k),
        'top_k_overlap_rf_vs_ffnn': float(ov),
        'spearman_rho_rf_vs_ffnn': float(rho) if rho is not None else np.nan,
        'spearman_p_rf_vs_ffnn': float(p) if p is not None else np.nan,
    })

comparison_df = pd.DataFrame(rows)
comparison_df.to_csv(deep_output_dir / 'comparison_RF_vs_FFNN_BC.csv', index=False)

print('Saved RF vs FFNN comparison to:', deep_output_dir / 'comparison_RF_vs_FFNN_BC.csv')
print(comparison_df)


# %%
