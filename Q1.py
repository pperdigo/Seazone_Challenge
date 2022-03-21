#Import Libraries
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector, ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from datetime import date
import re
from unicodedata import category
import numpy as np
import pandas as pd
from pyparsing import col
pd.set_option('display.max_columns', None)

#Import previous data cleaning function
from data_cleaning import CleanAndMerge

def featureSelection(df):
    #Separate features and target
    categorical_selector = selector(dtype_include = object)
    numerical_selector = selector(dtype_exclude = object)

    categorical_features = categorical_selector(df)
    numerical_features = numerical_selector(df)

    return categorical_features, numerical_features

def preProcessing(categorical_features, numerical_features):
    categorical_preprocessor = OneHotEncoder(handle_unknown='ignore')
    numerical_preprocessor = StandardScaler()
    preprocessor = ColumnTransformer([
        ('one-hot-encoder', categorical_preprocessor, categorical_features),
        ('std-scaler', numerical_preprocessor, numerical_features)]
    )
    return preprocessor

def trainModel(preprocessor, modelType, features, target):
    model = make_pipeline(preprocessor, modelType)
    feature_train, feature_test, target_train, target_test = train_test_split(
        features, target
    )
    print(feature_train, feature_test, target_train, target_test)
    _ = model.fit(feature_train, target_train)
    score = model.score(feature_test, target_test)
    return score

# if __name__ == 'main':
#Clean and Merge data
_revenue = pd.read_csv('daily_revenue.csv')
_listings = pd.read_csv('listings.csv')
df = CleanAndMerge(_revenue, _listings)

#Filter only relevant data
df['Month'] = pd.DatetimeIndex(df['date']).month
dff = df[df['Month'] == 3]

features = dff[['Localizacao', 'Categoria', 'Capacidade', 'Qtde_Quartos', 'Taxa de Limpeza']]
target = dff['revenue']

categorical_features, numerical_features = featureSelection(dff)
preprocessor = preProcessing(categorical_features, numerical_features)
score = trainModel(preprocessor, LinearRegression, features, target)   
print(score)
