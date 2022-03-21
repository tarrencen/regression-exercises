import pandas as pd
import numpy as np
import acquire as acq
import wrangle as wr
from env import get_db_url
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

iris_df = acq.get_iris_data()

def prep_iris(iris_df):
    '''
    Takes in a DataFrame of the iris dataset as acquired and returns a cleaned DF
    Args: iris_df, pandas DataFrame with expected columns and feature names
    Return: cleaned iris_df, pandas DF with cleaning operations performed
    '''
    
    iris_df = acq.get_iris_data()
    iris_df = iris_df.drop_duplicates()
    cols_to_drop = ['species_id', 'measurement_id']
    iris_df = iris_df.drop(columns= (cols_to_drop))
    iris_df = iris_df.rename(columns= {'species_id.1': 'species_id', 'species_name': 'species'})
    dummy_df = pd.get_dummies(iris_df[['species']], dummy_na=False, drop_first=[True])
    iris_df = pd.concat([iris_df, dummy_df], axis=1)
    return iris_df

def prep_titanic(titanic_df):
    '''
    Takes in a pandas DataFrame of the Titanic dataset as acquired and returns a cleaned version of the DF, and train, validate, and test splits
    Args: titanic_df, pandas DF with expected columns and feature names
    Return: titanic_df_clean and splits of it (train, validate, and test) 
    '''
    titanic_df = acq.get_titanic_data()
    titanic_df = titanic_df.drop_duplicates()
    titanic_df = titanic_df.drop(columns= ['passenger_id', 'age', 'embarked', 'class', 'deck'])
    titanic_df = titanic_df.fillna('Southampton')
    titanic_dummy_df = pd.get_dummies(titanic_df[['sex', 'embark_town']], dummy_na=False, drop_first= [True, True])
    titanic_df = pd.concat([titanic_df, titanic_dummy_df], axis=1)
    titanic_df_clean = titanic_df.drop(columns= ['sex', 'embark_town'])
    titanic_train, titanic_test = train_test_split(titanic_df_clean, train_size = 0.8, stratify= titanic_df_clean.survived,
        random_state= 302)
    titanic_train, titanic_validate = train_test_split(titanic_train, train_size = 0.7, stratify= titanic_train.survived, random_state= 302)
    return titanic_df_clean, titanic_train, titanic_validate, titanic_test


def prep_alt_titanic(titanic_df):
    '''
    Takes in a pandas DataFrame of the Titanic dataset as acquired and returns a cleaned version of the DF, and train, validate, and test splits
    Args: titanic_df, pandas DF with expected columns and feature names
    Return: titanic_df_clean and splits of it (train, validate, and test) 
    '''
    titanic_df = acq.get_titanic_data()
    titanic_df = titanic_df.drop_duplicates()
    titanic_df = titanic_df.drop(columns= ['passenger_id', 'embarked', 'class', 'deck'])
    titanic_df = titanic_df.fillna('Southampton')
    titanic_dummy_df = pd.get_dummies(titanic_df[['sex', 'embark_town']], dummy_na=False, drop_first= [True, True])
    titanic_df = pd.concat([titanic_df, titanic_dummy_df], axis=1)
    alt_titanic_df_clean = titanic_df.drop(columns= ['sex', 'embark_town'])
    alt_titanic_train, alt_titanic_test = train_test_split(alt_titanic_df_clean, train_size = 0.8, stratify= alt_titanic_df_clean.survived,
        random_state= 302)
    alt_titanic_train, alt_titanic_validate = train_test_split(alt_titanic_train, train_size = 0.7, stratify= alt_titanic_train.survived, random_state= 302)
    return alt_titanic_df_clean, alt_titanic_train, alt_titanic_validate, alt_titanic_test

def prep_telco(telco_df):
    '''
    Takes in a pandas DataFrame of Telco data and returns a clean and prepped DF (telco_df_clean) along with train, 
    validate, and test splits
    '''
    
    telco_df = telco_df.drop_duplicates()
    telco_df = telco_df.drop(columns= ['contract_type_id.1', 'payment_type_id.1', 'internet_service_type_id.1'])
    telco_df.total_charges = telco_df.total_charges.replace(' ', np.nan).astype(float)
    telco_df = telco_df.dropna()
    cat_cols = [col for col in telco_df.columns if telco_df[col].dtype == 'O']
    telco_cats = list(telco_df[cat_cols].columns)
    telco_cats = telco_cats[1:]
    telco_dummies = pd.get_dummies(telco_df[telco_cats], dummy_na=False, drop_first=True)
    telco_df_clean = pd.concat([telco_df, telco_dummies], axis=1)
    telco_train, telco_test = train_test_split(telco_df_clean, train_size = 0.8, stratify= telco_df_clean.churn_Yes, random_state= 302)
    telco_train, telco_validate =  train_test_split(telco_train, train_size= 0.7, stratify= telco_train.churn_Yes, random_state= 302)
    return telco_df_clean, telco_train, telco_validate, telco_test 