#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:42:08 2024

@author: sebastian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import shap

def get_right_classifications(model,X, y_test, class_keys):  
    
    # Get model predictions on test-set
    model_predictions_probabilities = model.predict(X)
    model_predictions = (model_predictions_probabilities > 0.5).astype(int)  # Finds maximum value in each row
    
    # Get real label of test data
    # true_labels = np.argmax(y_test, axis=1)
    true_labels = np.array(y_test).reshape((y_test.shape[0],1)) # Adapted for this dataset
    
    # Dynamically identify matching indices for each class
    right_class_indices = {}
    for class_id in np.unique(model_predictions):
        # Check wether the model prediction is the same as the real label
        match_indices = np.where((model_predictions == class_id) & (true_labels == class_id))[0]
        # right_class_indices[f'class {class_id}'] = match_indices # Key names as class 0 and class 1
        right_class_indices[class_keys[class_id]] = match_indices
        
    return right_class_indices


def creat_decision_plot(shap_values_dict, X_test, features, class_labels):
    # xp_dir: string
    #     Directory used to save the results of the experiment
    height_in_inches = 10 #placeholder
    width_in_pixels = 2926
    DPI = 300
    # Convert width from pixels to inches
    width_in_inches = width_in_pixels / DPI
    
    # Iterate over each class in shap_values_dict and create decision plot
    for class_key in class_labels:
        # Checks wether there are shap values or not
        if len(shap_values_dict[class_key]) == 0 :
            print(f'No SHAP values for {class_key}')
            continue  # Skip to the next class if no shap values
        ### Creation of the decision plots for each summary dircetion
        # Decision plot for sum over time
        fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches))
        shap.decision_plot(shap_values_dict[class_key]['base value'],shap_values_dict[class_key]['values'] ,X_test ,feature_names , link="logit") #, link="logit"
        ax.set_title(f'Decision plot for {class_key}', fontsize=20, pad=20, loc='center') # english: SHAP Decision Plot for {class_key} over features
        plt.tick_params(axis='x', labelsize=12)  # Change x-axis tick size
        plt.tick_params(axis='y', labelsize=12)
        plt.rcParams['axes.labelsize'] = 16
        # plt.savefig(xp_dir + "shap/" +f"descision_plot_class_{class_number}_featrue.png", bbox_inches="tight", dpi=DPI)
        plt.show()


def get_shap_values_for_right_classifications(model, X_train, X_test, class_labels, feature_names):
    explainer = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(X_test_scaled)
    shap_raw_values = shap_values.values
    shap_base_values = shap_values.base_values
    
    shap_values_right_class = {}
    for class_label in class_labels:
        # Get the indices of the right-classified cases for the current class
        indices = right_class_indices[class_label]
        class_ids = ids.iloc[indices]
        
        # Extract the corresponding SHAP values for those cases
        shap_values_right_class[class_label] = {'values' : shap_raw_values[indices],
                                                'base value': np.mean(shap_base_values),
                                                'ids' : class_ids['ID'].tolist()
                                                }
        
    creat_decision_plot(shap_values_right_class, X_test, feature_names, class_labels)

    return shap_values_right_class


def get_distribution_info_for_features(feature_names, considered_data):
    feature_distribution_info = {}
    for feature in range(len(feature_names)):
        considered_feature= {feature_names[feature] : considered_data[:,feature]}

        feature_mean = np.mean(considered_feature[feature_names[feature]])
        feature_std_dev = np.std(considered_feature[feature_names[feature]])
        feature_distribution_info[feature_names[feature]] = {'fdata'   : considered_data[:,feature],
                                                              'mean'   : feature_mean,
                                                              'std'    : feature_std_dev
                                                              }
    return feature_distribution_info

def create_COVAS_scoring(considered_data, feature_names, prob_ids_right_class):  
    if considered_data.shape[0] == 0:
        print('No correct classified cases for this class!')
        feature_distribution_info = get_distribution_info_for_features(feature_names, considered_data) # Gives back the mean and standartdivation for each feature
        COVAS_score = []
        COVAS_scoring_df = pd.DataFrame(COVAS_score,columns= ['COVAS'], index=prob_ids_right_class).sort_values(by='COVAS', ascending=False)
    else:
        feature_distribution_info = get_distribution_info_for_features(feature_names, considered_data) # Gives back the mean and standartdivation for each feature
        # COVAS_matrix = create_COVAS_matrix(feature_distribution_info, considered_data, feature_names, prob_ids_right_class)
        
        raw_matrix = pd.DataFrame(considered_data, columns=feature_names)
        COVAS_matrix = raw_matrix.copy()
        for feature in raw_matrix.columns:
            mean = feature_distribution_info[feature]['mean']
            std = feature_distribution_info[feature]['std']
            COVAS_matrix[feature] = np.abs((COVAS_matrix[feature] - mean) / std)   
        
        # Create COVAS score
        number_of_CO_cases = np.sum(np.array(COVAS_matrix), axis=1) # Takes all CO cases
        number_of_features = COVAS_matrix.shape[1]
        COVAS_score = number_of_CO_cases/number_of_features
        COVAS_scoring_df = pd.DataFrame(COVAS_score,columns= ['COVAS'], index=prob_ids_right_class).sort_values(by='COVAS', ascending=False)
        COVAS_matrix = pd.DataFrame(COVAS_matrix, columns= feature_names)
        COVAS_matrix.index = prob_ids_right_class
    return COVAS_scoring_df, feature_distribution_info, COVAS_matrix



def COVAS_scoring(shap_values_dict, feature_names, class_labels):
    com_COVAS_socring_dict = {}

    num_classes = len(class_labels)  # Determine the number of classes from shap_values
    
    for class_id in range(num_classes):
        class_key = class_labels[class_id]  # Construct the key for each class
        
        considered_data = shap_values_dict[class_key]['values']
        prob_ids_right_class = shap_values_dict[class_key]['ids']

        # Construct the awkwardness scoring for each class dynamically
        com_COVAS_socring_dict[class_key] = create_COVAS_scoring(considered_data , feature_names, prob_ids_right_class)

    
    com_COVAS_score_all_prob = pd.concat([com_COVAS_socring_dict[cls][0] for cls in com_COVAS_socring_dict]).sort_index()
    
    # for class_id, data in com_COVAS_socring_dict.items():
    #     if (com_COVAS_socring_dict[f'{class_id}'][0].empty) or (data==0):
    #         print(f'No classifications for {class_id}')
    #     else:
    #         # If not empty, create the plot
    #         # Assuming create_COVAS_score_plot() is your function for plotting
    #         # and it can handle the data directly from your dictionary
    #         plot_title = f'Klasse {class_id.split()[-1]}'  # Extracts the class number and creates a title # english: Class {class_id.split()[-1]} 
    #         create_COVAS_score_plot(data, plot_title, 'Kumulativ', xp_dir) # change back to english if needed (Cumulative)
            
    return com_COVAS_socring_dict, com_COVAS_score_all_prob



####### Individual Code
# Setting seeds for reproducability
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

#### Import data
file_path = "/Users/sebastian/Library/Mobile Documents/com~apple~CloudDocs/Uni/Paper/Programmierung/data/FIFA 2018 Statistics.csv"
data = pd.read_csv(file_path)
data['Own goals'] = data['Own goals'].fillna(0)
data['Man of the Match'] = data['Man of the Match'].apply(lambda x: 1 if x == 'Yes' else 0)
ids = pd.DataFrame()
ids['ID'] = data['Team'] + ' ~ ' + data['Date']
# Select relevant features and drop unnecessary columns
features = data.drop(columns=['Date', 'Team', 'Opponent', 'Man of the Match', 'Round', 'PSO', 'Goals in PSO', 'Own goal Time'])
feature_names = features.columns.tolist()
target = data['Man of the Match']

# Handle missing values by filling with the mean (for simplicity)
features = features.fillna(features.mean())

class_labels = ['Not MotM', 'MotM']


#### Data split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
train_ids, test_ids = train_test_split(ids, test_size=0.3, random_state=42)

# Scale the features for better performance of the neural network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#### Model
# Define a simple neural network model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=20, batch_size=16, verbose=0)
# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

right_class_indices = get_right_classifications(model, X_test_scaled, y_test, class_labels)
 
shap_dict = get_shap_values_for_right_classifications(model, X_train_scaled, X_test_scaled, class_labels, feature_names)
COVA_scoring_dict, COVAS_ALL = COVAS_scoring(shap_dict, feature_names, class_labels)

NOT_MotM_COVAS = COVA_scoring_dict['Not MotM'][0]
NOT_MotM_COVA_matrix = COVA_scoring_dict['Not MotM'][2]
   
MotM_COVAS = COVA_scoring_dict['MotM'][0]
MotM_COVA_matrix = COVA_scoring_dict['MotM'][2]

