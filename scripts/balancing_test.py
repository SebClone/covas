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
from sklearn.utils.class_weight import compute_class_weight
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
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids, test_size=0.3, random_state=100, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute class weights (to mitigate class imbalance during training)
classes = np.unique(y_train)
class_weights_arr = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = {int(c): float(w) for c, w in zip(classes, class_weights_arr)}
print('Class weights:', class_weights)


# Train model
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

# Evaluate model
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# Run full COVAS pipeline
correct_classification = get_correct_classification(model, X_test_scaled, y_test, ids_test, class_labels)
shap_vals = get_shap_values_for_correct_classification(model, X_train_scaled, X_test_scaled, correct_classification, class_labels)
feat_dist = get_feature_distribution(shap_vals, feature_names)
COVA_matrices = get_COVA_matrix('continuous', class_labels, shap_vals, feature_names, feat_dist)
COVA_scores = get_COVA_score(class_labels, COVA_matrices, shap_vals)

# Export individual COVA components per class
for class_name in class_labels:
    # Score
    score_df = pd.DataFrame(COVA_scores[class_name]['COVAS Score'])
    score_df.index.name = 'ID'
    score_df.to_csv(output_dir / f'COVA_score_{class_name}_balanced.csv')

    # Matrix
    matrix_df = pd.DataFrame(COVA_scores[class_name]['COVAS Matrix'], index=COVA_scores[class_name]['IDs'], columns=feature_names)
    matrix_df.index.name = 'ID'
    matrix_df.to_csv(output_dir / f'COVA_matrix_{class_name}.csv')

    # IDs
    ids_df = pd.DataFrame(COVA_scores[class_name]['IDs'])
    ids_df.to_csv(output_dir / f'COVA_IDs_{class_name}.csv', index=False)
    
    print(f"Exported COVA components for class '{class_name}'")


# %% Balancing test
df_u = pd.read_csv(output_dir / "COVA_score_benign.csv", index_col=0)
df_b = pd.read_csv(output_dir / "COVA_score_benign_balanced.csv", index_col=0)

df_u.index = df_u.index.astype(str)
df_b.index = df_b.index.astype(str)

# Clean ID strings saved as tuples like ("patient 152",)
df_u.index = df_u.index.str.replace(r"^\('", "", regex=True).str.replace(r"',\)$", "", regex=True)
df_b.index = df_b.index.str.replace(r"^\('", "", regex=True).str.replace(r"',\)$", "", regex=True)

common_ids = df_u.index.intersection(df_b.index)
print("Common samples:", len(common_ids))

# Compare distributions
summary = pd.DataFrame({
    "mean_unbalanced": [df_u["COVAS"].mean()],
    "mean_balanced": [df_b["COVAS"].mean()],
    "std_unbalanced": [df_u["COVAS"].std()],
    "std_balanced": [df_b["COVAS"].std()],
})
print(summary)

# Top-k Overlap
k = 10
u_common = df_u.loc[common_ids].sort_values("COVAS", ascending=False)
b_common = df_b.loc[common_ids].sort_values("COVAS", ascending=False)

top_u = set(u_common.head(k).index)
top_b = set(b_common.head(k).index)

overlap = len(top_u & top_b) / k
print("Top-k overlap:", overlap)

# Spearman correlation
from scipy.stats import spearmanr

rho, p = spearmanr(
    df_u.loc[common_ids]["COVAS"],
    df_b.loc[common_ids]["COVAS"],
)
print("Spearman rho:", rho)
print("Spearman p:", p)
# %%
