# run_full_covas_pipeline.py
# Make covaslib discoverable even in VS Code's interactive window
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

from covaslib.classification import get_correct_classification
from covaslib.shap_analysis import get_shap_values_for_correct_classification, get_feature_distribution
from covaslib.covas import get_COVA_matrix, get_COVA_score
from covaslib.plotting import custom_decision_plot

# Set seeds
random.seed(100)
np.random.seed(100)
tf.random.set_seed(100)


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

# Run full COVAS pipeline
correct_classification = get_correct_classification(model, X_test_scaled, y_test, ids_test, class_labels)
shap_vals = get_shap_values_for_correct_classification(model, X_train_scaled, X_test_scaled, correct_classification, class_labels)
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



# %% Uncertaty Covas correlation analaysis

from scipy.stats import spearmanr

def binary_entropy(p, eps=1e-12):
    """Binary entropy for probabilities p in [0,1]."""
    p = np.clip(p, eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def get_positive_class_proba(model, X):
    """Return p(y=1|x) for both Keras models and sklearn classifiers."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # Keras NN typically returns probabilities directly
    p = model.predict(X)
    return np.asarray(p).reshape(-1)

# --- 1) Compute probabilities on the full test set
p_pos = get_positive_class_proba(model, X_test_scaled)

# --- 2) Build per-class correlation summary (only correctly classified samples)
rows = []

for class_id, class_name in enumerate(class_labels):
    idx = correct_classification[class_name]['index'].values
    ids_for_class = correct_classification[class_name]['ID'].tolist()

    # Probabilities and uncertainty on this subset
    p_subset = p_pos[idx]
    unc_entropy = binary_entropy(p_subset)
    unc_margin = 1.0 - np.abs(p_subset - 0.5) * 2.0   # scaled to [0,1], higher = more uncertain

    # COVAS scores (align by IDs) -- robust to DataFrame/list/ndarray outputs
    covas_scores_raw = COVA_scores[class_name]['COVAS Score']
    covas_ids_raw = COVA_scores[class_name]['IDs']

    # Extract 1D scores
    if isinstance(covas_scores_raw, pd.DataFrame):
        covas_scores_1d = covas_scores_raw.iloc[:, 0].values
    elif isinstance(covas_scores_raw, pd.Series):
        covas_scores_1d = covas_scores_raw.values
    else:
        covas_scores_1d = np.asarray(covas_scores_raw).reshape(-1)

    # Extract 1D IDs
    if isinstance(covas_ids_raw, pd.DataFrame):
        covas_ids_1d = covas_ids_raw.iloc[:, 0].astype(str).tolist()
    elif isinstance(covas_ids_raw, pd.Series):
        covas_ids_1d = covas_ids_raw.astype(str).tolist()
    else:
        covas_ids_1d = [str(x) for x in covas_ids_raw]

    covas_series = pd.Series(covas_scores_1d, index=pd.Index(covas_ids_1d, dtype=str), name='COVAS')

    df_u = pd.DataFrame({
        'ID': [str(x) for x in ids_for_class],
        'p_pos': p_subset,
        'uncertainty_entropy': unc_entropy,
        'uncertainty_margin': unc_margin,
    }).set_index('ID')

    df_u = df_u.join(covas_series.rename('COVAS'), how='inner')

    # Spearman correlations
    if len(df_u) > 1:
        rho_ent, p_ent = spearmanr(df_u['COVAS'].values, df_u['uncertainty_entropy'].values)
        rho_mar, p_mar = spearmanr(df_u['COVAS'].values, df_u['uncertainty_margin'].values)
    else:
        rho_ent, p_ent = (np.nan, np.nan)
        rho_mar, p_mar = (np.nan, np.nan)

    rows.append({
        'class': class_name,
        'n_samples': int(len(df_u)),
        'spearman_rho_COVAS_vs_entropy': float(rho_ent) if rho_ent is not None else np.nan,
        'spearman_p_COVAS_vs_entropy': float(p_ent) if p_ent is not None else np.nan,
        'spearman_rho_COVAS_vs_margin': float(rho_mar) if rho_mar is not None else np.nan,
        'spearman_p_COVAS_vs_margin': float(p_mar) if p_mar is not None else np.nan,
    })

    # Optional: export per-instance table for transparency
    df_u.to_csv(output_dir / f'uncertainty_vs_COVAS_{class_name}.csv')

summary_unc_df = pd.DataFrame(rows)
summary_unc_df.to_csv(output_dir / 'uncertainty_correlation_summary.csv', index=False)

print("Saved uncertainty correlation summary to:", output_dir / 'uncertainty_correlation_summary.csv')
print(summary_unc_df)
