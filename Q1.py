#Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
#Tools
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, KFold
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
    imputer = IterativeImputer(estimator=model, verbose=0, max_iter=100, tol=1e-9, random_state=1)
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

def baselineScore(df, target):
    iterable = set(zip(df['Categoria'], df['Localizacao']))

    r2 = []
    mse = []

    predict = []
    y_true = []
    
    for category, region in iterable:
        # Our "Features"
        one_room = df[(df['Categoria'] == category) & (df['Localizacao'] == region) & (df['Qtde_Quartos'] == 1) & (df['occupancy'] == 1) & (df['blocked'] == 0)][target].mean()
        three_rooms = df[(df['Categoria'] == category) & (df['Localizacao'] == region) & (df['Qtde_Quartos'] == 3 & (df['occupancy'] == 1) & (df['blocked'] == 0))][target].mean()
        
        # Our truth value
        two_rooms = df[(df['Categoria'] == category) & (df['Localizacao'] == region) & (df['Qtde_Quartos'] == 2) & (df['occupancy'] == 1) & (df['blocked'] == 0)][target].mean()
        
        if np.isnan(one_room) or np.isnan(two_rooms) or np.isnan(three_rooms):
            continue # do not compute metrics if we don't have all the features or the truth value
        
        #prediction and truth
        predict.append(np.mean(np.array([one_room, three_rooms])))
        y_true.append(two_rooms)

    # Metrics
    r2.append(r2_score(y_true, predict))
    mse.append(mean_squared_error(y_true, predict))

    print('Baseline results')
    print(f'R-Squared: {float(r2[0]):.4f}')
    print(f'Mean Squared Error: {float(mse[0]):.4f}')

if __name__ == '__main__':
    #Clean and Merge data
    _revenue = pd.read_csv('daily_revenue.csv')
    _listings = pd.read_csv('listings.csv')
    df = CleanAndMerge(_revenue, _listings)

    # # FEATURE ENGINEERING

    #Separate date in date hierarchy
    df['Month'] = df['date'].dt.month
    df['Year'] = df['date'].dt.year
    df['Day'] = df['date'].dt.day

    #Total bed count has higher correlation to number of rooms than individual bed counts.
    camas = ['Cama Casal', 'Cama Solteiro', 'Cama Queen', 'Cama King', 'Sofa Cama Solteiro']
    df.loc[:,'Qtde_Camas'] = df.loc[:,camas].sum(axis=1)

    #Fill "Qtde_Quartos" with MICE imputing
    columnsToBeFilled = ['Qtde_Camas', 'Travesseiros', 'Banheiros','Capacidade', 'Qtde_Quartos']
    df[columnsToBeFilled] = inputMissingValues(df, columnsToBeFilled)

    #Separating features and targets for the model
    features = df[['Localizacao', 'Categoria', 'Qtde_Quartos', 'Month', 'occupancy', 'blocked']]

    values_to_be_predicted = ['revenue', 'last_offered_price']
    for value_to_be_predicted in values_to_be_predicted:
        print('\n')
        print('-------------------------------------------------------------')
        print(f'{value_to_be_predicted} model')
        print('-------------------------------------------------------------')
        print('\n')

        target = df[[value_to_be_predicted]]

        # # PREPROCESSING
        categorical_features = ['Localizacao', 'Categoria']
        numerical_features = ['Qtde_Quartos', 'Month', 'occupancy', 'blocked']
        preprocessor = preProcessing(categorical_features, numerical_features)

        # # MODEL SELECTION

        # Set Baseline
        baselineScore(df, value_to_be_predicted)
        print('\n')
        print('-------------------------------------------------------------')
        print('\n')
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, train_size=0.2, random_state=1
        )

        models = []
        models.append(('LinReg', make_pipeline(preprocessor, LinearRegression())))
        models.append(('Ridge', make_pipeline(preprocessor, Ridge())))
        models.append(('Lasso', make_pipeline(preprocessor, Lasso())))
        models.append(('ElasticNet', make_pipeline(preprocessor, ElasticNet())))
        models.append(('DecisionTreeRegressor', make_pipeline(preprocessor, DecisionTreeRegressor(max_depth=5))))

        results = []
        names = []

        scoring = {'R-Squared': make_scorer(r2_score),
                'MSE': make_scorer(mean_squared_error)
                }
        print('Results for the models tested with no tuning')
        print('\n')
        for name, model in models:
            kfold = KFold(n_splits=10)
            cv_results_R2 = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=-1)['test_R-Squared']
            cv_results_mse = cross_validate(model, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=-1)['test_MSE']
            names.append(name)
            print(f'{name} R2: {cv_results_R2.mean()} +/- {cv_results_R2.std()}')
            print(f'{name} Mean Squared Error: {cv_results_mse.mean()} +/- {cv_results_mse.std()}')
        print('\n')
        print('-------------------------------------------------------------')
        print('\n')

        # # HYPERPARAMETERS TUNING
        
        # Parameter Grid
        params = {
            #'decisiontreeregressor__min_samples_leaf': range(1,100),
            'decisiontreeregressor__max_depth': range(1,25)
        }

        # Build the model
        model = make_pipeline(preprocessor, DecisionTreeRegressor(random_state=1))
        kfold = KFold(n_splits=10)
        grid = GridSearchCV(estimator=model, param_grid=params, scoring=make_scorer(r2_score), cv=kfold, n_jobs=-1)
        grid_result = grid.fit(X_train, y_train)

        # Show the results
        print("Best model : %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        best_model = grid.best_estimator_
        print('\n')
        print('-------------------------------------------------------------')
        print('\n')

        # # FINAL TEST

        predict = best_model.predict(X_test)
        mse = mean_squared_error(y_test, predict)
        r2 = r2_score(y_test, predict)

        print('Final validation of model')
        print(f'RMSE: {mse}')
        print(f'R-Squared: {r2}')
        print('\n')
        print('-------------------------------------------------------------')
        print('\n')

        # # FORECASTING

        jur2Q = pd.DataFrame([['JUR', 'MASTER', 2, 3, 1, 0]], columns=['Localizacao', 'Categoria', 'Qtde_Quartos', 'Month', 'occupancy', 'blocked'])
        answer = best_model.predict(jur2Q)
        print(f'The asked prediction is: {answer}')

        jur1Q = df.query('Localizacao == "JUR" & Categoria == "MASTER" & Qtde_Quartos == 1 & Month == 3 & occupancy == 1 & blocked == 0')
        jur3Q = df.query('Localizacao == "JUR" & Categoria == "MASTER" & Qtde_Quartos == 3 & Month == 3 & occupancy == 1 & blocked == 0')
        
        # print('Description of JUR1Q')
        # print(jur1Q[[value_to_be_predicted]].describe())
        
        # print('\nDescription of JUR3Q')
        # print(jur3Q[[value_to_be_predicted]].describe())

        baseline_predict = (jur1Q[value_to_be_predicted].mean() + jur3Q[value_to_be_predicted].mean())/2
        print(f'The baseline predicted: {baseline_predict}')