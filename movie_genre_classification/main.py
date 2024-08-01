from wrangle_split import WrangleSplitClassifier
from nlp import MovieGenreClassifier

train_file_path = '../data/genre_classification_dataset/train_data.txt'
test_file_path = '../data/genre_classification_dataset/test_data.txt'
train_data_classifier = WrangleSplitClassifier(train_file_path)  # default test size set to 0.2
test_data_classifier = WrangleSplitClassifier(test_file_path, test_size=None)
train_data = train_data_classifier.wrangle()
test_data = test_data_classifier.wrangle()
X_train, X_test, y_train, y_test = train_data_classifier.preprocess_and_split()

classifier = MovieGenreClassifier()
classifier.build_and_evaluate_models(X_train, X_test, y_train, y_test)

# Predict on new data
predictions = classifier.predict(test_data)
print(predictions)
