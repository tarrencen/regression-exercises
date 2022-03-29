import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import env
import acquire as acq
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_regression, SelectKBest, RFE

def get_splits(df):
    train, test = train_test_split(df, test_size= 0.2, random_state=302)
    train, validate = train_test_split(train, test_size= 0.3, random_state=302)
    return train, validate, test
    


def isolate_lm_target(train, validate, test, target):
    '''
    Takes in train/validate/test splits and a target variable and returns corresponding X and y splits with
    target variable isolated (y_train, y_validate, y_test), ready for modeling.
    '''
    X_train = train.drop(columns= [target])
    y_train = train[[target]]

    X_validate = validate.drop(columns= [target])
    y_validate = validate[[target]]

    X_test = test.drop(columns= [target])
    y_test= test[[target]]

    X_train_dummies = pd.get_dummies(X_train.select_dtypes(exclude=np.number), dummy_na=False, drop_first=True)
    X_train = pd.concat([X_train, X_train_dummies], axis=1, ignore_index=False)

    X_validate_dummies = pd.get_dummies(X_validate.select_dtypes(exclude=np.number), dummy_na=False, drop_first=True)
    X_validate = pd.concat([X_validate, X_validate_dummies], axis=1, ignore_index=False)

    X_test_dummies = pd.get_dummies(X_test.select_dtypes(exclude=np.number), dummy_na=False, drop_first=True)
    X_test = pd.concat([X_test, X_test_dummies], axis=1, ignore_index=False)
    return X_train, y_train, X_validate, y_validate, X_test, y_test


def select_kbest(X, y, k):
    '''
    Takes in a dataframe(X) a pandas Series(target variable y) and a user input integer(k, number of desired 
    features from X) and returns a list of features(columns) from X that are best suited for a linear model to 
    predict y.
    '''
    f_selector = SelectKBest(f_regression, k)
    f_selector.fit(X,y)
    f_mask = f_selector.get_support()
    f_feature = X.iloc[:,f_mask].columns.tolist()
    return f_feature


def rfe(X, y, k):
    '''
    Takes in a dataframe(X) a pandas Series (target variable y) and a user input integer (k, number of desired 
    features from X) and returns a list of features(columns) from X and a dataframe that ranks all features of X 
    (behind k selected) that are best suited for a linear model to predict y.
    '''
    lm = LinearRegression()
    rfe = RFE(lm, k)
    rfe.fit(X, y)

    rfe_mask = rfe.support_
    rfe_feature = X.iloc[:,rfe_mask].columns.tolist()
    var_ranks = rfe.ranking_
    var_names = X.columns.tolist()
    rfe_ranked = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    return rfe_feature, rfe_ranked





