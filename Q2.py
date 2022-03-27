#Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
#Tools
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, TimeSeriesSplit
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
#Other libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

#Import previous data cleaning function
from data_cleaning import CleanAndMerge

if __name__ == '__main__':
    #Clean and Merge data
    _revenue = pd.read_csv('daily_revenue.csv')
    _listings = pd.read_csv('listings.csv')
    df = CleanAndMerge(_revenue, _listings)

    # # FEATURE ENGINEERING
    
    # Grouping DF by date
    df = df[['creation_date', 'revenue']]
    df = df.groupby(pd.Grouper(key='creation_date', freq='1M')).sum()
    
    # Creating targets
    df['revenue_next_1_month'] = df['revenue'].shift(-1)
    df['revenue_next_2_month'] = df['revenue'].shift(-2)
    df['revenue_next_3_month'] = df['revenue'].shift(-3)
    df['revenue_next_4_month'] = df['revenue'].shift(-4)
    df['revenue_next_5_month'] = df['revenue'].shift(-5)
    df['revenue_next_6_month'] = df['revenue'].shift(-6)
    df['revenue_next_7_month'] = df['revenue'].shift(-7)
    df['revenue_next_8_month'] = df['revenue'].shift(-8)
    df['revenue_next_9_month'] = df['revenue'].shift(-9)
    df.dropna(subset=['revenue_next_1_month', 'revenue_next_2_month', 'revenue_next_3_month',
                      'revenue_next_4_month', 'revenue_next_5_month', 'revenue_next_6_month',
                      'revenue_next_7_month', 'revenue_next_8_month', 'revenue_next_9_month', ], inplace=True)
    
    # Creating features
    
    #Lags
    
    df['revenue_last_1_month'] = df['revenue'].shift(1)
    df['revenue_last_2_month'] = df['revenue'].shift(2)

    #Diffs

    df['rev_diff_1_month'] = df['revenue'] - df['revenue_last_1_month']
    df['rev_diff_2_month'] = df['revenue'] - df['revenue_last_2_month']

    #Rolling average
    df['quarter_rolling_avg'] = df['revenue'].rolling(4).mean()
