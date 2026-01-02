# pipeline_full_MotM.py
# Make covaslib discoverable even in VS Code's interactive window
# %%
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers import LeakyReLU
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import spearmanr

from covaslib.classification import get_correct_classification
from covaslib.shap_analysis import get_shap_values_for_correct_classification, get_feature_distribution
from covaslib.covas import get_COVA_matrix, get_COVA_score
from covaslib.plotting import custom_decision_plot

# -----------------------------------------------------------------------
# Helper utilities for tau sensitivity analysis
# -----------------------------------------------------------------------
def get_top_k_ids(cova_scores, class_name, k):
    """Return the IDs of the top-k samples by COVAS Score for a given class."""
    ids = list(cova_scores[class_name]['IDs'])
    scores = np.array(cova_scores[class_name]['COVAS Score']).reshape(-1)
    k = min(k, len(scores))
    top_idx = np.argsort(scores)[::-1][:k]
    return [str(ids[i]) for i in top_idx]


def compute_overlap(a, b):
    """Compute |A ∩ B| / |A| for two iterables A and B."""
    a_set = set(a)
    if len(a_set) == 0:
        return 0.0
    return len(a_set.intersection(set(b))) / float(len(a_set))


def get_COVA_matrix_raw(class_labels, shap_values_right_class, feature_names):
    """
    Calculate a RAW COVA matrix without normalization.
    Each cell is abs(raw SHAP value). Rows are samples, columns are features.
    """
    if not isinstance(class_labels, list) or not all(isinstance(c, str) for c in class_labels):
        raise TypeError("class_labels must be a list of strings")
    if not isinstance(feature_names, list) or not all(isinstance(f, str) for f in feature_names):
        raise TypeError("feature_names must be a list of strings")

    cova_matrix_dict = {}
    for class_name in class_labels:
        shap_vals = shap_values_right_class[class_name]['values']
        raw_df = pd.DataFrame(shap_vals, columns=feature_names)
        cova_df = raw_df.copy()
        for feature in feature_names:
            cova_df[feature] = np.abs(raw_df[feature])
        cova_matrix_dict[class_name] = cova_df
    return cova_matrix_dict

def get_global_feature_distribution(class_labels, shap_values_right_class, feature_names):
    """
    Compute global mean/std per feature across all correctly classified samples
    (ignoring class labels).
    """
    global_dist = {}

    all_shap = []
    for class_name in class_labels:
        all_shap.append(shap_values_right_class[class_name]['values'])
    all_shap = np.vstack(all_shap)

    for j, feature in enumerate(feature_names):
        vals = all_shap[:, j]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if std == 0.0:
            std = 1.0
        global_dist[feature] = {'mean': mean, 'std': std}

    return {c: global_dist for c in class_labels}

def get_feature_zscore_outliers(X, ids, top_k):
    """Identify feature-space outliers using mean absolute z-scores (top-k IDs)."""
    X_z = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    scores = np.mean(np.abs(X_z), axis=1)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [str(i) for i in ids.iloc[top_idx]['ID']]


def mean_abs_feature_zscore(X, idx):
    """Compute mean absolute feature-wise z-score for selected sample indices."""
    X_sel = X[idx]
    X_z = (X_sel - np.mean(X, axis=0)) / np.std(X, axis=0)
    return float(np.mean(np.abs(X_z)))

# Set seeds
random.seed(100)
np.random.seed(100)
tf.random.set_seed(100)

####################################################################
#---------------Here goes your individual Dataset code-------------#
####################################################################
# Load data
file_path = "/Users/sebastian/Library/Mobile Documents/com~apple~CloudDocs/Uni/Paper/Programmierung/data/FIFA 2018 Statistics.csv"
data = pd.read_csv(file_path)
data['Own goals'] = data['Own goals'].fillna(0)
data['Man of the Match'] = data['Man of the Match'].apply(lambda x: 1 if x == 'Yes' else 0)
ids = pd.DataFrame()
ids['ID'] = data['Team'] + ' ~ ' + data['Date']
# Select relevant features and drop unnecessary columns
features = data.drop(columns=['Date', 'Team', 'Opponent', 'Man of the Match', 'Round', 'PSO', 'Goals in PSO', 'Own goal Time'])
feature_names = features.columns.tolist()
# Handle missing values by filling with the mean (for simplicity)
features = features.fillna(features.mean())
X = features
y = data['Man of the Match'].values # Target variable as numpy array
class_labels = ['Not MotM', 'MotM']

# Example decision plot for the first class
output_dir = Path(__file__).resolve().parents[1] / 'results'
####################################################################

# Split and scale
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(X, y, ids ,test_size=0.3, random_state=100)
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

# Evaluate model
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# %%
# Run full COVAS pipeline
correct_classification = get_correct_classification(model, X_test_scaled, y_test, ids_test, class_labels)
shap_vals = get_shap_values_for_correct_classification(model, X_train_scaled, X_test_scaled, correct_classification, class_labels)
feat_dist = get_feature_distribution(shap_vals, feature_names)
COVA_matrices = get_COVA_matrix('continuous', class_labels, shap_vals, feature_names, feat_dist)
COVA_scores = get_COVA_score(class_labels, COVA_matrices, shap_vals)



# ---------------------------------------------------------------
# Threshold (tau) sensitivity analysis (Reviewer 1)
# ---------------------------------------------------------------
# We vary the z-score threshold tau and analyze how the top-ranked samples and score
# distributions change. This is a robustness analysis, not hyperparameter tuning.

taus = [1.0, 1.5, 2.0, 2.5, 3.0]
top_k = 10

sensitivity_rows = []
overlap_rows = []

for class_name in class_labels:
    prev_top_ids = None

    for tau in taus:
        # Build thresholded COVA matrices and corresponding scores
        cova_matrices_tau = get_COVA_matrix(
            'threshold',
            class_labels,
            shap_vals,
            feature_names,
            feat_dist,
            threshold_std=tau,
        )
        cova_scores_tau = get_COVA_score(class_labels, cova_matrices_tau, shap_vals)

        # Extract scores for this class
        ids = list(cova_scores_tau[class_name]['IDs'])
        scores = np.array(cova_scores_tau[class_name]['COVAS Score']).reshape(-1)

        # Summary statistics
        sensitivity_rows.append({
            'class': class_name,
            'tau': tau,
            'n_samples': len(scores),
            'score_mean': float(np.mean(scores)) if len(scores) > 0 else np.nan,
            'score_std': float(np.std(scores)) if len(scores) > 0 else np.nan,
        })

        # Top-k overlap to previous tau (stability of detected atypical samples)
        top_ids = get_top_k_ids(cova_scores_tau, class_name, top_k)
        if prev_top_ids is not None:
            overlap_rows.append({
                'class': class_name,
                'tau_prev': prev_tau,
                'tau_curr': tau,
                'top_k': min(top_k, len(scores)),
                'top_k_overlap': compute_overlap(prev_top_ids, top_ids),
            })

        prev_top_ids = top_ids
        prev_tau = tau

# Save sensitivity results
sensitivity_df = pd.DataFrame(sensitivity_rows)
overlap_df = pd.DataFrame(overlap_rows)

sensitivity_df.to_csv(output_dir / 'tau_sensitivity_summary_MotM.csv', index=False)
overlap_df.to_csv(output_dir / 'tau_sensitivity_topk_overlap_MotM.csv', index=False)

print('Saved tau sensitivity analysis results')


# ---------------------------------------------------------------
# Ablation A : z-score normalization removed
# ---------------------------------------------------------------

# Build RAW (non-normalized) COVA matrices and scores
cova_matrices_raw = get_COVA_matrix_raw(class_labels, shap_vals, feature_names)
cova_scores_raw = get_COVA_score(class_labels, cova_matrices_raw, shap_vals)

ablation_overlap_rows = []
ablation_spearman_rows = []

for class_name in class_labels:
    # Top-k overlap between standard (z) and raw variants
    top_ids_z = get_top_k_ids(COVA_scores, class_name, top_k)
    top_ids_raw = get_top_k_ids(cova_scores_raw, class_name, top_k)

    ablation_overlap_rows.append({
        'class': class_name,
        'top_k': top_k,
        'top_k_overlap_z_vs_raw': compute_overlap(top_ids_z, top_ids_raw),
    })

    # Spearman rank correlation of full score rankings
    scores_z = np.array(COVA_scores[class_name]['COVAS Score']).reshape(-1)
    scores_r = np.array(cova_scores_raw[class_name]['COVAS Score']).reshape(-1)

    if len(scores_z) > 1 and len(scores_r) > 1:
        rho, p_val = spearmanr(scores_z, scores_r)
    else:
        rho, p_val = (np.nan, np.nan)

    ablation_spearman_rows.append({
        'class': class_name,
        'spearman_rho_z_vs_raw': float(rho) if rho is not None else np.nan,
        'spearman_p_z_vs_raw': float(p_val) if p_val is not None else np.nan,
    })

ablation_overlap_df = pd.DataFrame(ablation_overlap_rows)
ablation_spearman_df = pd.DataFrame(ablation_spearman_rows)

ablation_overlap_df.to_csv(output_dir / 'ablation_z_vs_raw_topk_overlap_MotM.csv', index=False)
ablation_spearman_df.to_csv(output_dir / 'ablation_z_vs_raw_spearman_MotM.csv', index=False)

print('Saved Ablation A (z vs raw) results')

# ---------------------------------------------------------------
# Ablation B: class-conditional vs global SHAP distributions
# ---------------------------------------------------------------

feat_dist_global = get_global_feature_distribution(
    class_labels,
    shap_vals,
    feature_names
)

COVA_matrices_global = get_COVA_matrix(
    'continuous',
    class_labels,
    shap_vals,
    feature_names,
    feat_dist_global
)

COVA_scores_global = get_COVA_score(
    class_labels,
    COVA_matrices_global,
    shap_vals
)

ablationB_overlap = []
ablationB_spearman = []

for class_name in class_labels:
    top_std = get_top_k_ids(COVA_scores, class_name, top_k)
    top_glb = get_top_k_ids(COVA_scores_global, class_name, top_k)

    ablationB_overlap.append({
        'class': class_name,
        'top_k_overlap_std_vs_global': compute_overlap(top_std, top_glb)
    })

    s_std = np.array(COVA_scores[class_name]['COVAS Score'])
    s_glb = np.array(COVA_scores_global[class_name]['COVAS Score'])

    rho, p = spearmanr(s_std, s_glb)
    ablationB_spearman.append({
        'class': class_name,
        'spearman_rho_std_vs_global': rho
    })

pd.DataFrame(ablationB_overlap).to_csv(
    output_dir / 'ablation_std_vs_global_topk_overlap_MotM.csv',
    index=False
)
pd.DataFrame(ablationB_spearman).to_csv(
    output_dir / 'ablation_std_vs_global_spearman_MotM.csv',
    index=False
)

print('Saved Ablation B (std vs global) results')


# ---------------------------------------------------------------
# Comparison with feature-space outlier methods (Reviewer 1)
# ---------------------------------------------------------------

comparison_rows = []

# Prepare feature matrix and IDs for correctly classified samples
all_ids = []
all_X = []

for class_name in class_labels:
    idx = correct_classification[class_name]['index'].values
    all_X.append(X_test_scaled[idx])
    all_ids.append(ids_test.iloc[idx])

X_all = np.vstack(all_X)
ids_all = pd.concat(all_ids).reset_index(drop=True)

# --- COVAS top-k (reference)
covas_scores_all = []
covas_ids_all = []

for class_name in class_labels:
    covas_scores_all.extend(COVA_scores[class_name]['COVAS Score'])
    covas_ids_all.extend([str(i) for i in COVA_scores[class_name]['IDs']])

covas_scores_all = np.array(covas_scores_all)
covas_ids_all = np.array(covas_ids_all)

covas_top_idx = np.argsort(covas_scores_all)[::-1][:top_k]
covas_top_ids = list(covas_ids_all[covas_top_idx])

# --- LOF
lof = LocalOutlierFactor(n_neighbors=20)
lof.fit(X_all)
lof_factor = -lof.negative_outlier_factor_
lof_top_idx = np.argsort(lof_factor)[::-1][:top_k]
lof_top_ids = [str(i) for i in ids_all.iloc[lof_top_idx]['ID']]

comparison_rows.append({
    'method': 'LOF',
    'top_k': top_k,
    'overlap_with_COVAS': compute_overlap(covas_top_ids, lof_top_ids),
})

# --- Feature-wise z-score
zscore_top_ids = get_feature_zscore_outliers(X_all, ids_all, top_k)

comparison_rows.append({
    'method': 'Feature z-score',
    'top_k': top_k,
    'overlap_with_COVAS': compute_overlap(covas_top_ids, zscore_top_ids),
})

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(output_dir / 'comparison_LOF_zscore_vs_COVAS_MotM.csv', index=False)

print('Saved comparison with LOF and z-score outliers')


# ---------------------------------------------------------------
# Auxiliary analysis: feature-space extremeness of detected outliers
# ---------------------------------------------------------------

extremeness_rows = []

# Indices of selected samples
covas_idx = covas_top_idx
lof_idx = lof_top_idx
zscore_idx = np.array([ids_all.index[ids_all['ID'] == i][0] for i in zscore_top_ids])

extremeness_rows.append({
    'method': 'COVAS',
    'mean_abs_feature_zscore': mean_abs_feature_zscore(X_all, covas_idx),
})

extremeness_rows.append({
    'method': 'LOF',
    'mean_abs_feature_zscore': mean_abs_feature_zscore(X_all, lof_idx),
})

extremeness_rows.append({
    'method': 'Feature z-score',
    'mean_abs_feature_zscore': mean_abs_feature_zscore(X_all, zscore_idx),
})

extremeness_df = pd.DataFrame(extremeness_rows)
extremeness_df.to_csv(output_dir / 'feature_space_extremeness_MotM.csv', index=False)

print('Saved feature-space extremeness analysis')


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
        save_path=output_dir / f"decision_plot_{class_name}.png",
        dpi=600,
        show=False,   # oder True, wenn du sie sehen willst
    )


# Export individual COVA components per class
for class_name in class_labels:
    # Score
    score_df = pd.DataFrame(COVA_scores[class_name]['COVAS Score'])
    score_df.index.name = 'ID'
    score_df.to_csv(output_dir / f'COVA_score_{class_name}.csv')

    # Matrix
    matrix_df = pd.DataFrame(COVA_scores[class_name]['COVAS Matrix'], index=COVA_scores[class_name]['IDs'], columns=feature_names)
    matrix_df.index.name = 'ID'
    matrix_df.to_csv(output_dir / f'COVA_matrix_{class_name}.csv')

    # IDs
    ids_df = pd.DataFrame(COVA_scores[class_name]['IDs'])
    ids_df.to_csv(output_dir / f'COVA_IDs_{class_name}.csv', index=False)
    
    print(f"Exported COVA components for class '{class_name}'")
# %%
