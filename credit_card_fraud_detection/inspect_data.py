import pandas as pd
from data_preprocessing import DataPreprocessing

pd.set_option('display.max_columns', None)

# set training and test data to data pre-processing
train_data_path = '../data/credit_card_fraud_detection/fraudTrain.csv'
test_data_path = '../data/credit_card_fraud_detection/fraudTest.csv'
ccfd_train = DataPreprocessing(train_data_path)
ccfd_test = DataPreprocessing(test_data_path)

print('-------Train Data---------')
train_data = ccfd_train.wrangle()
print(train_data)

print('-------Test Data---------')
test_data = ccfd_test.wrangle()
print(test_data)

# inspect training data
ccfd_train.drop_highly_correlated_features()
ccfd_train.analyze_data()

processed_train_data = ccfd_train.feature_extraction()
processed_test_data = ccfd_test.feature_extraction()
print(ccfd_train.feature_extraction())
print(ccfd_test.feature_extraction())
