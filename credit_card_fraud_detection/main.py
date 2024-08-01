from inspect_data import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import shap


training_data = processed_train_data
testing_data = processed_test_data

# Extract features and target from the training data
X_train = training_data.drop(columns=['is_fraud'])  # Drop the target column
y_train = training_data['is_fraud']  # Target column

# Extract features and target from the testing data
X_test = testing_data.drop(columns=['is_fraud'])  # Drop the target column
y_test = testing_data['is_fraud']  # Target column

# Perform K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=1)

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for train_index, val_index in kf.split(X_train):
    X_train_data, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_data, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Initialize and train the model
    model = RandomForestClassifier(random_state=1)
    model.fit(X_train_data, y_train_data)

    # Make predictions on the validation data
    y_val_pred = model.predict(X_val)

    # Calculate metrics
    accuracy_list.append(accuracy_score(y_val, y_val_pred))
    precision_list.append(precision_score(y_val, y_val_pred))
    recall_list.append(recall_score(y_val, y_val_pred))
    f1_list.append(f1_score(y_val, y_val_pred))

# Print average metrics
print(f"Average Accuracy: {np.mean(accuracy_list)}")
print(f"Average Precision: {np.mean(precision_list)}")
print(f"Average Recall: {np.mean(recall_list)}")
print(f"Average F1 Score: {np.mean(f1_list)}")

# Train the model on the full training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_test_pred = model.predict(X_test)

# Calculate metrics for the testing data
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"Test Accuracy: {accuracy}")
print(f"Test Precision: {precision}")
print(f"Test Recall: {recall}")
print(f"Test F1 Score: {f1}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Plot the confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance plot using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

# SHAP summary plot
shap.summary_plot(shap_values[1], X_train)
