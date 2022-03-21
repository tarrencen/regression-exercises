import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer
from acquire import get_telco_data
from prep import prep_telco

def plot_variable_pairs(df):
    '''
    Takes in a DataFrame as input and returns plots of all pairwise relationships along 
    with the the regression line for each pair
    '''
    df = df.select_dtypes(include=np.float64)
    sns.pairplot(data=df, kind='reg', diag_kind='kde', plot_kws= {'line_kws': {'color': 'green'}}, dropna=True)
    
        
def months_to_years(telco_train):
    telco_train['tenure_years'] = telco_train.tenure / 12
    return telco_train

def plot_cat_and_cont_vars(df):
    cat_vars = df.select_dtypes(include=np.uint8)
    cont_vars= df.select_dtypes(include=(np.int64, np.float64))
    cats_plotted = sns.catplot(cat_vars, row=cat_vars.shape[0], col= cat_vars.shape[1], kind='boxen', legend_out=True)
    conts_plotted = plot_variable_pairs(cont_vars)
    return cats_plotted, conts_plotted


