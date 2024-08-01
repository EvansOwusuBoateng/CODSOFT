import pandas as pd
pd.set_option('display.max_columns', None)

file_path = '../data/customer_churn/Churn_Modelling.csv'
data = pd.read_csv(file_path)
print(data.columns)
print(data.info())
print(data)
print(data.HasCrCard.nunique())
