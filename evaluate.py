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

def get_splits(df):
    train, test = train_test_split(df, test_size= 0.2, random_state=302)
    train, validate = train_test_split(train, test_size= 0.3, random_state=302)
    return train, test, validate
    


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

    X_train = X_train.select_dtypes(include='number')
    X_validate = X_validate.select_dtypes(include='number')
    X_test = X_test.select_dtypes(include='number')
    return X_train, y_train, X_validate, y_validate, X_test, y_test



