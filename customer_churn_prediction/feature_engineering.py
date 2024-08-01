import pandas as pd
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)


def wrangle(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Drop RowNumber column
    df.drop(columns=['RowNumber'], inplace=True)

    # Set CustomerId as the index
    df.set_index('CustomerId', inplace=True)

    # Initialize LabelEncoder
    le = LabelEncoder()

    # List of columns with object data type
    object_cols = df.select_dtypes(include=['object']).columns

    # Apply label encoding to object columns
    for col in object_cols:
        df[col] = le.fit_transform(df[col])

    return df
