import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

class DataPreprocessing:

    def __init__(self, file_path):
        self.df = self.load_data(file_path)

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        return df

    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great-circle distance between two points
        on the Earth's surface specified in decimal degrees.
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        r = 6371  # Radius of Earth in kilometers. Use 3956 for miles.
        return c * r

    def check_data_integrity(self):
        """
        Check for duplicates, missing values, and unique entries.
        """
        df = self.df

        # Check for duplicates
        duplicates = df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")

        # Check for missing values
        missing_values = df.isnull().sum()
        print(f"Missing values per column:\n{missing_values}")

        # Check the number of unique entries in a key column (e.g., transaction ID)
        unique_entries = df['trans_num'].nunique()
        total_entries = len(df)
        print(f"Total entries: {total_entries}")
        print(f"Unique entries: {unique_entries}")

        return duplicates, missing_values, unique_entries, total_entries

    def wrangle(self):
        df = self.df

        # Convert trans_date_trans_time to datetime
        if 'trans_date_trans_time' in df.columns:
            df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

            # Extract date features
            df['trans_year'] = df['trans_date_trans_time'].dt.year
            df['trans_month'] = df['trans_date_trans_time'].dt.month
            df['trans_day'] = df['trans_date_trans_time'].dt.day
            df['trans_hour'] = df['trans_date_trans_time'].dt.hour
            df['trans_dayofweek'] = df['trans_date_trans_time'].dt.dayofweek

            # Set trans_date_trans_time as index
            df.set_index('trans_date_trans_time', inplace=True)

        # Calculate age from dob if dob exists
        if 'dob' in df.columns:
            current_date = datetime.now()
            df['age'] = current_date.year - pd.to_datetime(df['dob']).dt.year

        # Feature extraction for distance calculation using Haversine formula
        if 'lat' in df.columns and 'long' in df.columns and 'merch_lat' in df.columns and 'merch_long' in df.columns:
            df['distance'] = self.haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
            df.drop(['merch_lat', 'merch_long'], axis=1, inplace=True)

        # Drop unnecessary columns if they exist
        columns_to_drop = ['Unnamed: 0', 'first', 'last', 'street', 'city', 'state', 'zip', 'lat', 'long', 'dob', 'unix_time', 'trans_num']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # Handle missing values
        for col in df.select_dtypes(include=['float', 'int']).columns:
            df[col] = df[col].fillna(df[col].median())

        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        self.df = df
        return df

    def drop_highly_correlated_features(self, threshold=0.8):
        """
        Drop features from the DataFrame that have a correlation greater than or equal to the specified threshold.
        """
        df = self.df

        # Select only numeric columns for correlation calculation
        numeric_df = df.select_dtypes(include=[np.number])

        # Calculate the correlation matrix
        correlation_matrix = numeric_df.corr().abs()  # Use absolute values for correlation

        # Create a set to hold features to drop
        features_to_drop = set()

        # Iterate over the correlation matrix
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if correlation_matrix.iloc[i, j] >= threshold:
                    # Get the name of the features
                    colname = correlation_matrix.columns[i]
                    features_to_drop.add(colname)

        # Drop the features from the DataFrame
        df_dropped = df.drop(columns=features_to_drop)

        print(f"Dropped features: {features_to_drop}")

        self.df = df_dropped

    def analyze_data(self):
        """
        Analyze the dataset for outliers, correlation, and data distribution.
        """
        df = self.df

        # Set the aesthetic style of the plots
        sns.set(style="whitegrid")

        # 1. Check for Outliers
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numerical_cols, 1):
            plt.subplot(3, len(numerical_cols) // 3 + 1, i)
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
        plt.tight_layout()
        plt.show()

        # 2. Correlation Matrix
        plt.figure(figsize=(12, 8))
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix')
        plt.show()

        # 3. Data Distribution
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numerical_cols, 1):
            plt.subplot(3, len(numerical_cols) // 3 + 1, i)
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.show()

    def feature_extraction(self):
        """
        Perform binary encoding on binary categorical columns and ordinal encoding on columns with multiple unique values.
        """
        df = self.df

        # Identify binary and non-binary categorical columns
        binary_cols = [col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() == 2]
        multi_value_cols = [col for col in df.select_dtypes(include=['object']).columns if df[col].nunique() > 2]

        # Binary encoding for binary categorical columns
        for col in binary_cols:
            unique_values = df[col].unique()
            mapping = {unique_values[0]: 0, unique_values[1]: 1}
            df[col] = df[col].map(mapping)

        # Ordinal encoding for columns with multiple unique values
        for col in multi_value_cols:
            df[col] = df[col].astype('category').cat.codes

        self.df = df
        return df
