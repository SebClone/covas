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

from covaslib.classification import get_correct_classification
from covaslib.shap_analysis import get_shap_values_for_correct_classification, get_feature_distribution
from covaslib.covas import get_COVA_matrix, get_COVA_score
from covaslib.plotting import custom_decision_plot

from scipy.stats import spearmanr

def topk_overlap(ids_a, ids_b, k=10):
    a = list(ids_a)[:k]
    b = list(ids_b)[:k]
    if len(a) == 0:
        return np.nan
    return len(set(a).intersection(set(b))) / len(a)

SEEDS = [0, 50, 100]
results_per_seed = {}

####################################################################
#---------------Here goes your individual Dataset code-------------#
####################################################################
# Load data
file_path = "/Users/sebastian/Library/Mobile Documents/com~apple~CloudDocs/Uni/Paper/Programmierung/data/FIFA 2018 Statistics.csv"
data = pd.read_csv(file_path)
data['Own goals'] = data['Own goals'].fillna(0)
data['Man of the Match'] = data['Man of the Match'].apply(lambda x: 1 if x == 'Yes' else 0)
ids = (data['Team'].astype(str) + ' ~ ' + data['Date'].astype(str)).astype(str).values
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
output_dir.mkdir(parents=True, exist_ok=True)
####################################################################

for seed in SEEDS:
    print(f"\nRunning seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Split and scale
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.3, random_state=seed, stratify=y
    )
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
    model.fit(X_train_scaled, y_train, epochs=10, batch_size=16, verbose=0)

    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Run COVAS pipeline
    correct_classification = get_correct_classification(
        model, X_test_scaled, y_test, ids_test, class_labels
    )
    shap_vals = get_shap_values_for_correct_classification(
        model, X_train_scaled, X_test_scaled, correct_classification, class_labels
    )
    feat_dist = get_feature_distribution(shap_vals, feature_names)
    COVA_matrices = get_COVA_matrix(
        'continuous', class_labels, shap_vals, feature_names, feat_dist
    )
    COVA_scores = get_COVA_score(class_labels, COVA_matrices, shap_vals)

    results_per_seed[seed] = COVA_scores

rows = []
ref_seed = SEEDS[0]

for class_name in class_labels:
    ref_scores_raw = results_per_seed[ref_seed][class_name]['COVAS Score']
    ref_ids_raw = results_per_seed[ref_seed][class_name]['IDs']

    # Robust 1D extraction
    ref_scores = np.asarray(ref_scores_raw).reshape(-1)
    if isinstance(ref_ids_raw, (pd.DataFrame, pd.Series)):
        ref_ids = ref_ids_raw.values.reshape(-1).astype(str).tolist()
    else:
        ref_ids = [str(x) for x in ref_ids_raw]

    for seed in SEEDS[1:]:
        cur_scores_raw = results_per_seed[seed][class_name]['COVAS Score']
        cur_ids_raw = results_per_seed[seed][class_name]['IDs']

        cur_scores = np.asarray(cur_scores_raw).reshape(-1)
        if isinstance(cur_ids_raw, (pd.DataFrame, pd.Series)):
            cur_ids = cur_ids_raw.values.reshape(-1).astype(str).tolist()
        else:
            cur_ids = [str(x) for x in cur_ids_raw]

        # Align by common IDs
        common = list(set(ref_ids).intersection(set(cur_ids)))
        if len(common) < 5:
            continue

        s_ref = pd.Series(ref_scores, index=ref_ids).loc[common]
        s_cur = pd.Series(cur_scores, index=cur_ids).loc[common]

        rho, p = spearmanr(s_ref, s_cur)
        overlap = topk_overlap(
            s_ref.sort_values(ascending=False).index,
            s_cur.sort_values(ascending=False).index,
            k=10
        )

        rows.append({
            'class': class_name,
            'seed_ref': ref_seed,
            'seed_cmp': seed,
            'n_common': len(common),
            'spearman_rho': rho,
            'spearman_p': p,
            'topk_overlap': overlap
        })

stability_df = pd.DataFrame(rows)
print("\nSeed stability rows:", len(stability_df))
print(stability_df)
stability_df.to_csv(output_dir / 'seed_stability_MotM.csv', index=False)
print("\nSeed stability results saved to seed_stability_MotM.csv")

# Export individual COVA components per class
# for class_name in class_labels:
#     # Score
#     score_df = pd.DataFrame(COVA_scores[class_name]['COVAS Score'])
#     score_df.index.name = 'ID'
#     score_df.to_csv(output_dir / f'COVA_score_{class_name}.csv')

#     # Matrix
#     matrix_df = pd.DataFrame(COVA_scores[class_name]['COVAS Matrix'], index=COVA_scores[class_name]['IDs'], columns=feature_names)
#     matrix_df.index.name = 'ID'
#     matrix_df.to_csv(output_dir / f'COVA_matrix_{class_name}.csv')

#     # IDs
#     ids_df = pd.DataFrame(COVA_scores[class_name]['IDs'])
#     ids_df.to_csv(output_dir / f'COVA_IDs_{class_name}.csv', index=False)
    
#     print(f"Exported COVA components for class '{class_name}'")
# %%
