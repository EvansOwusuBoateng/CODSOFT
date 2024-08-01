from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


class MovieGenreClassifier:
    def __init__(self):
        self.best_model = None

    def build_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        # Define a pipeline combining TF-IDF and SVM
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', SVC(kernel='linear'))
        ])

        # Parameters for GridSearchCV
        parameters = {
            'tfidf__max_df': [0.8, 0.9, 1.0],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'clf__C': [0.1, 1, 10]
        }

        # Train and evaluate the model using GridSearchCV
        grid_search = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        y_pred = grid_search.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"SVM accuracy: {accuracy:.4f}")

        # Save the best model
        self.best_model = grid_search.best_estimator_

    def predict(self, texts):
        if self.best_model is None:
            raise Exception("Model has not been trained yet. Call the build_and_evaluate_models method first.")
        return self.best_model.predict(texts)
