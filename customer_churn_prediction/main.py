from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from feature_engineering import wrangle

# Define the path to the dataset
path = '../data/customer_churn/Churn_Modelling.csv'

# Preprocess the data
data = wrangle(path)
print(data.head())

# Split the data into features and target
X = data.drop(columns=['Exited'])
y = data['Exited']

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize and train the Gradient Boosting model
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions on validation and test sets
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Evaluate the model
val_accuracy = accuracy_score(y_val, y_val_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Validation Accuracy: {val_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')

print('Classification Report (Test Set):')
print(classification_report(y_test, y_test_pred))
print('Confusion Matrix (Test Set):')
print(confusion_matrix(y_test, y_test_pred))
