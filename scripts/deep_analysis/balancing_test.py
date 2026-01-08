# balancing_test.py
# Make covaslib discoverable even in VS Code's interactive window
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import random
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.datasets import load_breast_cancer

from scipy.stats import spearmanr

from covaslib.classification import get_correct_classification
from covaslib.shap_analysis import get_shap_values_for_correct_classification, get_feature_distribution
from covaslib.covas import get_COVA_matrix, get_COVA_score


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
random.seed(100)
np.random.seed(100)
tf.random.set_seed(100)


# ---------------------------------------------------------------------
# Load dataset (Breast Cancer)
# ---------------------------------------------------------------------
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names.tolist()
class_labels = data.target_names.tolist()  # ['malignant', 'benign']

ids = pd.DataFrame({
    'ID': ['patient ' + str(i) for i in range(len(X))]
})

# Output directories
results_dir = Path(__file__).resolve().parents[1] / 'results'
deep_output_dir = results_dir / 'deep_analysis_results'
deep_output_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Split and scale
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids, test_size=0.3, random_state=100, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------------------
# Compute class weights (to mitigate class imbalance during training)
# ---------------------------------------------------------------------
classes = np.unique(y_train)
class_weights_arr = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}
print('Class weights:', class_weights)


# ---------------------------------------------------------------------
# Train model
# ---------------------------------------------------------------------
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=10,
    batch_size=16,
    class_weight=class_weights,
)

loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")


# ---------------------------------------------------------------------
# Run COVAS pipeline (balanced model) and export results
# ---------------------------------------------------------------------
correct_classification = get_correct_classification(model, X_test_scaled, y_test, ids_test, class_labels)
shap_vals = get_shap_values_for_correct_classification(
    model,
    X_train_scaled,
    X_test_scaled,
    correct_classification,
    class_labels
)

feat_dist = get_feature_distribution(shap_vals, feature_names)
COVA_matrices = get_COVA_matrix('continuous', class_labels, shap_vals, feature_names, feat_dist)
COVA_scores = get_COVA_score(class_labels, COVA_matrices, shap_vals)

# Export individual COVA components per class
for class_name in class_labels:
    # Score
    score_df = pd.DataFrame(COVA_scores[class_name]['COVAS Score'])
    # Ensure a stable, predictable column name across all exports
    if score_df.shape[1] == 1:
        score_df.columns = ['COVAS']
    score_df.index.name = 'ID'
    score_df.to_csv(deep_output_dir / f'COVA_score_{class_name}_balanced.csv')

    # Matrix
    matrix_df = pd.DataFrame(
        COVA_scores[class_name]['COVAS Matrix'],
        index=COVA_scores[class_name]['IDs'],
        columns=feature_names
    )
    matrix_df.index.name = 'ID'
    matrix_df.to_csv(deep_output_dir / f'COVA_matrix_{class_name}_balanced.csv')

    # IDs
    ids_df = pd.DataFrame(COVA_scores[class_name]['IDs'], columns=['ID'])
    ids_df.to_csv(deep_output_dir / f'COVA_IDs_{class_name}_balanced.csv', index=False)

    print(f"Exported COVA components for class '{class_name}' (balanced model).")


# ---------------------------------------------------------------------
# Compare unbalanced vs balanced COVAS scores (paper-ready summary)
# ---------------------------------------------------------------------

def _clean_index_as_id(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the CSV index to plain string IDs (e.g., 'patient 152')."""
    df = df.copy()
    df.index = df.index.astype(str)
    # Clean ID strings saved as tuples like ("patient 152",)
    df.index = df.index.str.replace(r"^\('\s*", "", regex=True).str.replace(r"\s*',\)$", "", regex=True)
    df.index = df.index.str.replace(r"^\(\"\s*", "", regex=True).str.replace(r"\s*\",\)$", "", regex=True)
    return df


def _get_score_series(df: pd.DataFrame) -> pd.Series:
    """Return the COVAS score series, accepting common column naming variants."""
    for col in ["COVAS", "COVAS Score", "COVA score", "COVA Score"]:
        if col in df.columns:
            return df[col]
    # If the CSV has a single unnamed column, use it
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    raise KeyError(
        "Could not find a COVAS score column. Expected one of: 'COVAS', 'COVAS Score'."
    )


def compute_topk_overlap(a_ids, b_ids, k: int) -> float:
    """Compute |A ∩ B| / k for two ranked ID lists (top-k overlap)."""
    if k <= 0:
        return float('nan')
    a_set = set(a_ids[:k])
    b_set = set(b_ids[:k])
    return len(a_set & b_set) / k


def compare_balanced_vs_unbalanced(class_name: str, k: int = 10) -> dict:
    """Return paper-ready summary metrics for a given class."""
    # Baseline (unbalanced) files live in the main results directory
    unbalanced_path = results_dir / f"COVA_score_{class_name}.csv"
    # Balanced files are exported into deep_analysis_results
    balanced_path = deep_output_dir / f"COVA_score_{class_name}_balanced.csv"

    df_u = _clean_index_as_id(pd.read_csv(unbalanced_path, index_col=0))
    df_b = _clean_index_as_id(pd.read_csv(balanced_path, index_col=0))

    s_u = _get_score_series(df_u)
    s_b = _get_score_series(df_b)

    common_ids = s_u.index.intersection(s_b.index)
    n_common = int(len(common_ids))

    # Default NaNs for edge cases
    rho = float('nan')
    p = float('nan')
    overlap = float('nan')
    k_eff = int(min(k, n_common)) if n_common > 0 else 0

    if n_common > 0:
        s_u_common = s_u.loc[common_ids]
        s_b_common = s_b.loc[common_ids]

        # Rank agreement
        rho, p = spearmanr(s_u_common, s_b_common)

        # Top-k overlap (computed on common IDs only)
        u_ranked = s_u_common.sort_values(ascending=False).index.tolist()
        b_ranked = s_b_common.sort_values(ascending=False).index.tolist()
        overlap = compute_topk_overlap(u_ranked, b_ranked, k_eff)

        mean_u = float(s_u_common.mean())
        mean_b = float(s_b_common.mean())
        std_u = float(s_u_common.std())
        std_b = float(s_b_common.std())
    else:
        mean_u = float('nan')
        mean_b = float('nan')
        std_u = float('nan')
        std_b = float('nan')

    return {
        "class": class_name,
        "n_common": n_common,
        "k_eff": k_eff,
        "mean_unbalanced": mean_u,
        "mean_balanced": mean_b,
        "std_unbalanced": std_u,
        "std_balanced": std_b,
        "topk_overlap": overlap,
        "spearman_rho": float(rho),
        "spearman_p": float(p),
    }


def print_paper_summary(results: pd.DataFrame) -> None:
    """Pretty-print summary in console."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print("\n" + "=" * 70)
    print("Class imbalance impact (balanced vs unbalanced) — paper-ready summary")
    print(results)


# Run comparisons and export summary
rows = []
for _cls in ['benign', 'malignant']:
    rows.append(compare_balanced_vs_unbalanced(_cls, k=10))

results_df = pd.DataFrame(rows)

# Console output
print_paper_summary(results_df)

# Save CSV summary next to other results
results_df.to_csv(deep_output_dir / 'class_imbalance_summary.csv', index=False)

print("\nSaved:")
print("-", deep_output_dir / 'class_imbalance_summary.csv')
