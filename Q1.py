#Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
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

# runLinearRegression(lr, features_train, features_test, revenue_target_train, revenue_target_test, price_target_train, price_target_test)
def runLinearRegression(lr, Xtrain, Xtest, Ytrain_rev, Ytest_rev, Ytrain_price, Ytest_price):
    price_lr, rev_lr = lr, lr
    print('The Cross Validation Scores (RMSE) for the Linear Regression Models are:')
    cv_rev = cross_val_score(estimator=rev_lr, X=Xtrain, y= Ytrain_rev, scoring=make_scorer(mean_squared_error), n_jobs=-1, cv=10)
    cv_price = cross_val_score(estimator=price_lr, X=Xtrain, y= Ytrain_price, scoring=make_scorer(mean_squared_error), n_jobs=-1, cv=10)
    print(f'Revenue: {cv_rev.mean():.4f} +/- {cv_rev.std() * 2:.4f}')
    print(f'Price: {cv_price.mean():.4f} +/- {cv_price.std() * 2:.4f}')
    print('The final test scores (RMSE) for the Linear Regression Models are')
    rev_lr.fit(X=Xtrain, y=Ytrain_rev)
    mse_rev = mean_squared_error(Ytest_rev,rev_lr.predict(Xtest))
    price_lr.fit(X=Xtrain, y=Ytrain_price)
    mse_price = mean_squared_error(Ytest_price,price_lr.predict(Xtest))
    print(f'Revenue: {mse_rev:.4f}')
    print(f'Price: {mse_price:.4f}')
    print('\n')
    print('-------------------------')
    print('\n')

    return

def runDecisionTreeRegressor(dtr, Xtrain, Xtest, Ytrain_rev, Ytest_rev, Ytrain_price, Ytest_price):
    price_dtr, rev_dtr = dtr, dtr
    print('The Cross Validation Scores (RMSE) for the Decision Tree Models are:')
    cv_rev = cross_val_score(estimator=rev_dtr, X=Xtrain, y= Ytrain_rev, scoring=make_scorer(mean_squared_error), n_jobs=-1, cv=10)
    cv_price = cross_val_score(estimator=price_dtr, X=Xtrain, y= Ytrain_price, scoring=make_scorer(mean_squared_error), n_jobs=-1, cv=10)
    print(f'Revenue: {cv_rev.mean():.4f} +/- {cv_rev.std() * 2:.4f}')
    print(f'Price: {cv_price.mean():.4f} +/- {cv_price.std() * 2:.4f}')
    print('The final test scores (RMSE) for the Decision Tree Models are')
    rev_dtr.fit(X=Xtrain, y=Ytrain_rev)
    mse_rev = mean_squared_error(Ytest_rev,rev_dtr.predict(Xtest))
    price_dtr.fit(X=Xtrain, y=Ytrain_price)
    mse_price = mean_squared_error(Ytest_price,price_dtr.predict(Xtest))
    print(f'Revenue: {mse_rev:.4f}')
    print(f'Price: {mse_price:.4f}')
    print('\n')
    print('-------------------------')
    print('\n')
    return

def runRandomForestRegressor(rfr, Xtrain, Xtest, Ytrain_rev, Ytest_rev, Ytrain_price, Ytest_price):
    Ytrain_rev, Ytest_rev, Ytrain_price, Ytest_price = np.ravel(Ytrain_rev), np.ravel(Ytest_rev), np.ravel(Ytrain_price), np.ravel(Ytest_price)
    price_rfr, rev_rfr = rfr, rfr
    print('The Cross Validation Scores (RMSE) for the Random Forest Models are:')
    cv_rev = cross_val_score(estimator=rev_rfr, X=Xtrain, y= Ytrain_rev, scoring=make_scorer(mean_squared_error), n_jobs=1, cv=10)
    cv_price = cross_val_score(estimator=price_rfr, X=Xtrain, y= Ytrain_price, scoring=make_scorer(mean_squared_error), n_jobs=1, cv=10)
    print(f'Revenue: {cv_rev.mean():.4f} +/- {cv_rev.std() * 2:.4f}')
    print(f'Price: {cv_price.mean():.4f} +/- {cv_price.std() * 2:.4f}')
    print('The final test scores (RMSE) for the Random Forest Models are')
    rev_rfr.fit(X=Xtrain, y=Ytrain_rev)
    mse_rev = mean_squared_error(Ytest_rev,rev_rfr.predict(Xtest))
    price_rfr.fit(X=Xtrain, y=Ytrain_price)
    mse_price = mean_squared_error(Ytest_price,price_rfr.predict(Xtest))
    print(f'Revenue: {mse_rev:.4f}')
    print(f'Price: {mse_price:.4f}')
    print('\n')
    print('-------------------------')
    print('\n')
    return

def runModel(model, Xtrain, Xtest, Ytrain_rev, Ytest_rev, Ytrain_price, Ytest_price):
    
    if model[0] in ['Random Forest', 'XGBoost']:
        Ytrain_rev, Ytest_rev, Ytrain_price, Ytest_price = np.ravel(Ytrain_rev), np.ravel(Ytest_rev), np.ravel(Ytrain_price), np.ravel(Ytest_price)
    
    print(f'The Cross Validation Scores (MSE) for the {model[0]} Models are:')
    cv_rev = cross_val_score(estimator=model[1], X=Xtrain, y= Ytrain_rev, scoring=make_scorer(mean_squared_error), cv=10)
    cv_price = cross_val_score(estimator=model[1], X=Xtrain, y= Ytrain_price, scoring=make_scorer(mean_squared_error), cv=10)
    print(f'Revenue: {cv_rev.mean():.4f} +/- {(cv_rev.std()) * 2:.4f}')
    print(f'Price: {cv_price.mean():.4f} +/- {cv_price.std() * 2:.4f}')
    print(f'The final test scores (RMSE) for the {model[0]} Models are')
    model[1].fit(X=Xtrain, y=Ytrain_rev)
    mse_rev = mean_squared_error(Ytest_rev,model[1].predict(Xtest))
    model[1].fit(X=Xtrain, y=Ytrain_price)
    mse_price = mean_squared_error(Ytest_price,model[1].predict(Xtest))
    print(f'Revenue: {(mse_rev):.4f}')
    print(f'Price: {(mse_price):.4f}')
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
    features = df[['Localizacao', 'Categoria', 'Qtde_Quartos', 'Month']]
    revenue_target = df[['revenue']]
    price_target = df[['last_offered_price']]

    #Defining the preprocessor
    categorical_features = ['Localizacao', 'Categoria']
    numerical_features = ['Qtde_Quartos', 'Month']
    preprocessor = preProcessing(categorical_features, numerical_features)

    #Defining the models
    lr = make_pipeline(preprocessor, LinearRegression(n_jobs=-1))
    dtr = make_pipeline(preprocessor, DecisionTreeRegressor(random_state=0, max_depth=10))
    
    #rfr = make_pipeline(preprocessor, RandomForestRegressor(random_state=0, max_depth=10, n_estimators=500))
    ##I can't, for the life of me, get Random Forests to train properly. As soon as the program tries training it, all the active python processes go from 95% usage down to 7% and I get no outputs.
    ##Since the challenge is on a tight schedule, I will abandon Random Forests.
    
    gbr = make_pipeline(preprocessor, XGBRegressor(n_estimators=100,
                                                   learning_rate=1e-3,
                                                   max_depth=10,
                                                   random_state=0))

    models = [('Linear Regression',lr),
              ('Decision Tree',dtr),
              ('XGBoost', gbr)
            ]
    
    #Separating train and test sets for both models
    features_train, features_test, revenue_target_train, revenue_target_test = train_test_split(
        features, revenue_target, random_state=0, test_size=0.2
    )

    _, __, price_target_train, price_target_test = train_test_split(
        features, price_target, random_state=0, test_size=0.2
    )
 
    #Call the functions
    trained_models = {}
    for model in models:
        runModel(model, features_train, features_test, revenue_target_train, revenue_target_test, price_target_train, price_target_test)
    
    final_price_model, final_rev_model = dtr, dtr

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, price_target, test_size=0.25, random_state=42)
    
    final_price_model.fit(Xtrain, Ytrain)

    print('Final MSE of price model')
    mse = mean_squared_error(final_price_model.predict(Xtest), Ytest)
    print(mse)

    print('Prediction of price model')
    jur2Q = df.query('Localizacao == "JUR" & Categoria == "MASTER" & Qtde_Quartos == 2 & Month == 3')
    print(final_price_model.predict(jur2Q))

    jur1Q = df.query('Localizacao == "JUR" & Categoria == "MASTER" & Qtde_Quartos == 1 & Month == 3')
    jur3Q = df.query('Localizacao == "JUR" & Categoria == "MASTER" & Qtde_Quartos == 3 & Month == 3')
    
    print('Description of JUR1Q')
    print(jur1Q[['revenue', 'last_offered_price']].describe())
    
    print('\nDescription of JUR3Q')
    print(jur3Q[['revenue', 'last_offered_price']].describe())

    
    # model = DecisionTreeRegressor(min_samples_leaf=30, random_state=0)
    # revenue_model = make_pipeline(preprocessor, model)
    # price_model = make_pipeline(preprocessor, model)
    
    
    # # parameters = {'decisiontreeregressor__max_depth': range(1,50)}

    # # cv_revenue = GridSearchCV(revenue_model, param_grid = parameters, n_jobs=-1)
    # # _ = cv_revenue.fit(features_train, revenue_target_train)
    # # best_revenue_model = cv_revenue.best_estimator_
    # # print('Cross validation results for revenue (score and best parameters):')
    # # print (cv_revenue.best_score_, cv_revenue.best_params_) 
    
    # print('Independent test set score for revenue')
    # print(best_revenue_model.score(features_test, revenue_target_test))

    # jur2Q = pd.DataFrame([['JUR', 'MASTER', 2, 3]], columns=['Localizacao', 'Categoria', 'Qtde_Quartos', 'Month'])

    # print(f'The prediction for revenue is: {best_revenue_model.predict(jur2Q)}')

    # features_train, features_test, price_target_train, price_target_test = train_test_split(
    #     features, price_target, random_state=0, test_size=0.2
    # )

    # cv_price = GridSearchCV(price_model, param_grid = parameters, n_jobs=-1)
    # _ = cv_price.fit(features_train, price_target_train)
    # best_price_model = cv_price.best_estimator_
    # print('Cross validation results for price (score and best parameters):')
    # print (cv_price.best_score_, cv_price.best_params_) 
    
    # print('Independent test set score for price')
    # print(best_price_model.score(features_test, price_target_test))

    # print(f'The prediction for price is: {best_price_model.predict(jur2Q)}')

    # jur1Q = df.query('Localizacao == "JUR" & Categoria == "MASTER" & Qtde_Quartos == 1 & Month == 3')
    # jur3Q = df.query('Localizacao == "JUR" & Categoria == "MASTER" & Qtde_Quartos == 3 & Month == 3')
    
    # # jur1Q.hist(column='revenue')
    # # jur3Q.hist(column='revenue')
    # # plt.show()

    # print('revenue')
    # print('1Q')
    # print(jur1Q['revenue'].mean())
    # print(jur1Q['revenue'].std())

    # print('3Q')
    # print(jur3Q['revenue'].mean())
    # print(jur3Q['revenue'].std())

    # print('price')
    # print('1Q')
    # print(jur1Q['last_offered_price'].mean())
    # print(jur1Q['last_offered_price'].std())

    # print('3Q')
    # print(jur3Q['last_offered_price'].mean())
    # print(jur3Q['last_offered_price'].std())