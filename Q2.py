#Models
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
#Tools
from sklearn.preprocessing import  MaxAbsScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_validate, TimeSeriesSplit, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn import metrics
#Other libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)

#Import previous data cleaning function
from data_cleaning import CleanAndMerge

def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))

if __name__ == '__main__':
    #Clean and Merge data
    _revenue = pd.read_csv('daily_revenue.csv')
    _listings = pd.read_csv('listings.csv')
    df = CleanAndMerge(_revenue, _listings)

    # Grouping DF by date
    df = df[['creation_date', 'revenue']]
    df = df.groupby(pd.Grouper(key='creation_date', freq='1D')).sum()

    #FEATURES
    df['n-1_revenue'] = df['revenue'].shift(1)
    df['n-1_diff'] = df['revenue'].diff()

    df.dropna(inplace=True)

    features = ['n-1_revenue', 'n-1_diff']
    targets = ['revenue']

    X_train = df[features][df.index <= pd.Timestamp(2022,1,1)]
    X_test = df[features][df.index > pd.Timestamp(2022,1,1)]

    y_train = df[targets][df.index <= pd.Timestamp(2022,1,1)]
    y_test = df[targets][df.index > pd.Timestamp(2022,1,1)]
    
    models = []
    models.append(('DT', DecisionTreeRegressor()))
    # Evaluate each model in turn
    results = []
    names = []
    for name, model in models:
            # TimeSeries Cross validation
        tscv = TimeSeriesSplit(n_splits=10)
            
        cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
        results.append(cv_results)
        names.append(name)
        print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
            
    # Compare Algorithms
    # plt.boxplot(results, labels=names)
    
    model = DecisionTreeRegressor(random_state=1)
    param_search = {
        'max_depth' : [i for i in range(5,15)]
    }
    tscv = TimeSeriesSplit(n_splits=10)
    gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=param_search, scoring = 'r2')
    gsearch.fit(X_train, y_train)
    best_score = gsearch.best_score_
    best_model = gsearch.best_estimator_

    y_true = y_test.values
    y_pred = best_model.predict(X_test)
    # regression_results(y_true, y_pred)

    start_date = df.index.max() + timedelta(1)
    end_date = pd.Timestamp(2022,12,31)
    
    dates = pd.date_range(start_date, end_date)
    for date in dates:
        yesterday = date - timedelta(1)
        p = best_model.predict(df.loc[[yesterday], features])[0]
        new_line = {date: p}
        new_line = pd.Series(data=new_line, name='revenue')
        new_line = pd.DataFrame(new_line)
        df = pd.concat([df,new_line])
        df['n-1_revenue'] = df['revenue'].shift(1)
        df['n-1_diff'] = df['revenue'].diff()

    df['revenue'].plot.line()
    plt.show()


    df2019 = df[df.index <= pd.Timestamp(2019,12,31)]
    rev2019 = df2019['revenue'].sum()
    df2020 = df[df.index <= pd.Timestamp(2020,12,31)]
    rev2020 = df2020['revenue'].sum()
    df2021 = df[df.index <= pd.Timestamp(2021,12,31)]
    rev2021 = df2021['revenue'].sum()
    df2022 = df[df.index <= pd.Timestamp(2022,12,31)]
    rev2022 = df2022['revenue'].sum()
    
    print(f'2019 revenue: {rev2019}')
    print(f'2020 revenue: {rev2020}')
    print(f'2021 revenue: {rev2021}')
    print(f'Expected revenue for 2022: {rev2022}')