#Import Libraries
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import date
import re
from unicodedata import category
import numpy as np
import pandas as pd
from pyparsing import col
pd.set_option('display.max_columns', None)

#Import previous data cleaning function
from data_cleaning import CleanAndMerge

def preProcessing(categorical_features, numerical_features):
    categorical_preprocessor = OneHotEncoder(handle_unknown='ignore')
    numerical_preprocessor = StandardScaler()
    preprocessor = ColumnTransformer([
        ('one-hot-encoder', categorical_preprocessor, categorical_features),
        ('std-scaler', numerical_preprocessor, numerical_features)]
    )
    return preprocessor

if __name__ == '__main__':
    #Clean and Merge data
    _revenue = pd.read_csv('daily_revenue.csv')
    _listings = pd.read_csv('listings.csv')
    df = CleanAndMerge(_revenue, _listings)

    #Filter only relevant data
    df['Month'] = pd.DatetimeIndex(df['date']).month
    df.dropna(inplace=True)
    
    features = df[['Localizacao', 'Categoria', 'Qtde_Quartos', 'Month']]
    revenue_target = df[['revenue']]
    price_target = df[['last_offered_price']]

    categorical_features = ['Localizacao', 'Categoria', 'Month']
    numerical_features = ['Qtde_Quartos']


    preprocessor = preProcessing(categorical_features, numerical_features)
    revenue_model = make_pipeline(preprocessor, LinearRegression())
    price_model = make_pipeline(preprocessor, LinearRegression())
    
    # # Manual train-test split
    # feature_train = features[features['Qtde_Quartos'] != 2]
    # feature_test = features[features['Qtde_Quartos'] == 2]

    # revenue_target_train_index = feature_train.index
    # revenue_target_train = revenue_target[revenue_target.index.isin(revenue_target_train_index)]

    # revenue_target_test_index = feature_test.index
    # revenue_target_test = revenue_target[revenue_target.index.isin(revenue_target_test_index)]

    # # Automatic train-test split
    feature_train, feature_test, revenue_target_train, revenue_target_test = train_test_split(
        features, revenue_target, random_state=42
    )
    _ = revenue_model.fit(X = feature_train, y = revenue_target_train)
    revenue_score = revenue_model.score(feature_test, revenue_target_test)
    print(f'The score of the revenue model on the test data was {revenue_score:.4f}')
    
    cv_results = cross_validate(revenue_model, features, revenue_target, cv=5)
    scores = cv_results["test_score"]
    print("The mean cross-validation accuracy for the revenue model is: "
       f"{scores.mean():.4f} +/- {scores.std():.4f}")
    jur2Q = pd.DataFrame([['JUR', 'MASTER', 2, 3]], columns=['Localizacao', 'Categoria', 'Qtde_Quartos', 'Month'])
    predicted_revenue = revenue_model.predict(jur2Q)
    print(f'The predicted value for revenue is: {float(predicted_revenue):.2f}')
    jur1Q = df.query('Localizacao == "JUR" & Categoria == "MASTER" & Qtde_Quartos == 1 & Month == 3')['revenue'].mean()
    jur3Q = df.query('Localizacao == "JUR" & Categoria == "MASTER" & Qtde_Quartos == 3 & Month == 3')['revenue'].mean()
    print(f'The mean revenue for a JURMASTER1Q listing is: {jur1Q:.2f}, and for a JURMASTER3Q is {jur3Q:.2f}')

    # Automatic train-test split
    feature_train, feature_test, price_target_train, price_target_test = train_test_split(
        features, price_target, random_state=42
    )

    # # Manual train-test split
    # price_target_train_index = feature_train.index
    # price_target_train = price_target[price_target.index.isin(price_target_train_index)]

    # price_target_test_index = feature_test.index
    # price_target_test = price_target[price_target.index.isin(price_target_test_index)]

    _ = price_model.fit(X = feature_train, y = price_target_train)
    price_score = price_model.score(feature_test, price_target_test)
    print(f'The score of the price model on the test data was {price_score:.4f}')
    
    cv_results = cross_validate(price_model, features, price_target, cv=5)
    scores = cv_results["test_score"]
    print("The mean cross-validation accuracy for the price model is: "
       f"{scores.mean():.4f} +/- {scores.std():.4f}")
    jur2Q = pd.DataFrame([['JUR', 'MASTER', 2, 3]], columns=['Localizacao', 'Categoria', 'Qtde_Quartos', 'Month'])
    predicted_price = price_model.predict(jur2Q)
    print(f'The predicted value for price is: {float(predicted_price):.2f}')
    jur1Q = df.query('Localizacao == "JUR" & Categoria == "MASTER" & Qtde_Quartos == 1 & Month == 3')['last_offered_price'].mean()
    jur3Q = df.query('Localizacao == "JUR" & Categoria == "MASTER" & Qtde_Quartos == 3 & Month == 3')['last_offered_price'].mean()
    print(f'The mean price for a JURMASTER1Q listing is: {jur1Q:.2f}, and for a JURMASTER3Q is {jur3Q:.2f}')

    