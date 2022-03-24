#Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from xgboost import XGBRegressor
#Tools
from skopt import gp_minimize
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
#Other libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
pd.set_option('display.max_columns', None)

#Import previous data cleaning function
from data_cleaning import CleanAndMerge

def inputMissingValues(df, columnsList):       
    df = df[columnsList]
    model = LinearRegression()
    imputer = IterativeImputer(estimator=model, verbose=0, max_iter=50, tol=1e-9, random_state=1)
    imputer.fit(df)
    imputed_df = imputer.transform(df)
    imputed_df = pd.DataFrame(imputed_df, columns=df.columns)
    
    return imputed_df

def preProcessing(categorical_features, numerical_features):
    categorical_preprocessor = OneHotEncoder(handle_unknown='ignore')
    numerical_preprocessor = StandardScaler()
    preprocessor = ColumnTransformer([
        ('one-hot-encoder', categorical_preprocessor, categorical_features),
        ('std-scaler', numerical_preprocessor, numerical_features)]
    )
    return preprocessor

def runModel(model, Xtrain, Xtest, Ytrain_rev, Ytest_rev, Ytrain_price, Ytest_price):
    
    if model[0] in ['Random Forest', 'XGBoost']:
        Ytrain_rev, Ytest_rev, Ytrain_price, Ytest_price = np.ravel(Ytrain_rev), np.ravel(Ytest_rev), np.ravel(Ytrain_price), np.ravel(Ytest_price)

    rev_model = model[1]
    price_model = model[2]

    print(f'The Cross Validation Scores (MSE) for the {model[0]} Models are:')
    cv_rev = cross_val_score(estimator=rev_model, X=Xtrain, y= Ytrain_rev, scoring=make_scorer(mean_squared_error), cv=10)
    cv_price = cross_val_score(estimator=price_model, X=Xtrain, y= Ytrain_price, scoring=make_scorer(mean_squared_error), cv=10)
    print(f'Revenue: {cv_rev.mean():.4f} +/- {(cv_rev.std()) * 2:.4f}')
    print(f'Price: {cv_price.mean():.4f} +/- {cv_price.std() * 2:.4f}')
    print(f'The final test scores (RMSE) for the {model[0]} Models are')
    rev_model.fit(X=Xtrain, y=Ytrain_rev)
    mse_rev = mean_squared_error(Ytest_rev,model[1].predict(Xtest))
    price_model.fit(X=Xtrain, y=Ytrain_price)
    mse_price = mean_squared_error(Ytest_price,model[1].predict(Xtest))
    print(f'Revenue: {(mse_rev):.4f}')
    print(f'Price: {(mse_price):.4f}')

    jur2Q = pd.DataFrame([['JUR', 'MASTER', 2, 3, 1, 0]], columns=['Localizacao', 'Categoria', 'Qtde_Quartos', 'Month', 'occupancy', 'blocked'])

    print(f'Prediction of revenue {model[0]} model')
    print(rev_model.predict(jur2Q))
    print(f'Prediction of price {model[0]} model')
    print(price_model.predict(jur2Q))

    plot_tree(rev_model)
    plot_tree(price_model)
    plt.show()

    print('\n')
    print('-------------------------')
    print('\n')
    return
    
if __name__ == '__main__':
    #Clean and Merge data
    _revenue = pd.read_csv('daily_revenue.csv')
    _listings = pd.read_csv('listings.csv')
    df = CleanAndMerge(_revenue, _listings)

    #Filter only relevant data
    df['Month'] = pd.DatetimeIndex(df['date']).month

    #Total bed count has higher correlation to number of rooms than individual bed counts.
    camas = ['Cama Casal', 'Cama Solteiro', 'Cama Queen', 'Cama King', 'Sofa Cama Solteiro']
    df.loc[:,'Qtde_Camas'] = df.loc[:,camas].sum(axis=1)

    #Fill "Qtde_Quartos" with MICE imputing
    columnsToBeFilled = ['Qtde_Camas', 'Travesseiros', 'Banheiros','Capacidade', 'Qtde_Quartos']
    df[columnsToBeFilled] = inputMissingValues(df, columnsToBeFilled)

    #Separating features and targets for revenue and price models
    features = df[['Localizacao', 'Categoria', 'Qtde_Quartos', 'Month', 'occupancy', 'blocked']]
    revenue_target = df[['revenue']]
    price_target = df[['last_offered_price']]

    #Defining the preprocessor
    categorical_features = ['Localizacao', 'Categoria']
    numerical_features = ['Qtde_Quartos', 'Month', 'occupancy', 'blocked']
    preprocessor = preProcessing(categorical_features, numerical_features)

    #Defining the models
    r_lr = make_pipeline(preprocessor, LinearRegression(n_jobs=-1))
    p_lr = make_pipeline(preprocessor, LinearRegression(n_jobs=-1))
    
    r_dtr = make_pipeline(preprocessor, DecisionTreeRegressor(random_state=0, max_depth=10))
    p_dtr = make_pipeline(preprocessor, DecisionTreeRegressor(random_state=0, max_depth=10))
    #rfr = make_pipeline(preprocessor, RandomForestRegressor(random_state=0, max_depth=10, n_estimators=500))
    ##I can't, for the life of me, get Random Forests to train properly. As soon as the program tries training it, all the active python processes go from 95% usage down to 7% and I get no outputs.
    ##Since the challenge is on a tight schedule, I will abandon Random Forests.
    
    r_gbr = make_pipeline(preprocessor, XGBRegressor(n_estimators=100,
                                                   learning_rate=1e-3,
                                                   max_depth=10,
                                                   random_state=0))
    p_gbr = make_pipeline(preprocessor, XGBRegressor(n_estimators=100,
                                                   learning_rate=1e-3,
                                                   max_depth=10,
                                                   random_state=0))

    models = [#('Linear Regression',r_lr, p_lr),
              ('Decision Tree',r_dtr, p_dtr),
              #('XGBoost', r_gbr, p_gbr)
             ]
    
    #Separating train and test sets for both models
    features_train, features_test, revenue_target_train, revenue_target_test = train_test_split(
        features, revenue_target, random_state=0, test_size=0.2
    )

    _, __, price_target_train, price_target_test = train_test_split(
        features, price_target, random_state=0, test_size=0.2
    )
 
    #Call the functions
    for model in models:
        runModel(model, features_train, features_test, revenue_target_train, revenue_target_test, price_target_train, price_target_test)
    
    jur1Q = df.query('Localizacao == "JUR" & Categoria == "MASTER" & Qtde_Quartos == 1 & Month == 3 & occupancy == 1 & blocked == 0')
    jur3Q = df.query('Localizacao == "JUR" & Categoria == "MASTER" & Qtde_Quartos == 3 & Month == 3 & occupancy == 1 & blocked == 0')
    
    print('Description of JUR1Q')
    print(jur1Q[['revenue', 'last_offered_price']].describe())
    
    print('\nDescription of JUR3Q')
    print(jur3Q[['revenue', 'last_offered_price']].describe())