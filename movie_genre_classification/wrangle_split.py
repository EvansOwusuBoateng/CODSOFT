import pandas as pd
from sklearn.model_selection import train_test_split


class WrangleSplitClassifier:
    def __init__(self, file_path, test_size=0.2):
        self.file_path = file_path
        self.df = None
        self.test_size = test_size  # Set the test size as an attribute

    def wrangle(self):

        # Read the .txt file
        with open(self.file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Parse the lines into a list of dictionaries
        data = []
        for line in lines:
            parts = line.strip().split(':::')
            if len(parts) == 4:  # Training data with genre
                data.append({
                    'ID': parts[0],
                    'Title': parts[1],
                    'Genre': parts[2],
                    'Description': parts[3]
                })
            elif len(parts) == 3:  # Test data without genre
                data.append({
                    'ID': parts[0],
                    'Title': parts[1],
                    'Description': parts[2]
                })

        # Convert list of dictionaries into DataFrame
        self.df = pd.DataFrame(data)

        # Set ID as the index
        self.df.set_index('ID', inplace=True)

        return self.df

    def preprocess_and_split(self):
        # Combine Title and Description into a single feature
        self.df['Text'] = self.df['Title'] + ' ' + self.df['Description']
        # Drop rows where Genre is None (test data)
        self.df = self.df.dropna(subset=['Genre'])
        X = self.df['Text']
        y = self.df['Genre']
        return train_test_split(X, y, test_size=self.test_size, random_state=42)
