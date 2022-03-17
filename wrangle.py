import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from env import get_db_url
from pydataset import data
import os

def show_codeup_dbs():
    '''
    Returns a list of the databases residing the Codeup SQL server
    '''
    url = get_db_url('employees')
    codeup_dbs = pd.read_sql('SHOW DATABASES', url)
    print('List of Codeup DBs:\n')
    return codeup_dbs

def get_prop_vals():
    '''
    Returns a DataFrame composed of selected column data from the properties_2017 table in the zillow database on
    Codeup's SQL serve
    '''
    filename = 'prop_vals.csv'
    if os.path.exists(filename):
        print('Reading from CSV file...')
        return pd.read_csv(filename)
    query = ''' 
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips, propertylandusetypeid
    FROM properties_2017
    '''
    print('Getting a fresh copy from SQL database...')
    prop_vals = pd.read_sql(query, url)
    print('Copying to CSV...')
    prop_vals.to_csv(filename)
    return prop_vals

def wrangle_zillow():
    '''
    Returns a cleaned subset of prop_vals DataFrame
    '''
    prop_vals = get_prop_vals()
    prop_vals = prop_vals.rename(columns={
    'bedroomcnt': 'bedrooms', 
    'bathroomcnt': 'bathrooms', 
    'calculatedfinishedsquarefeet': 'calcfin_sqft', 
    'taxvaluedollarcnt': 'tax_val',
    'yearbuilt': 'yr_built',
    'taxamount': 'tax_amt',
    'fips': 'fips',
    'propertylandusetypeid': 'prop_use_id'
    })
    single_fams = prop_vals[prop_vals.prop_use_id == 261]
    #single_fams[['bedrooms', 'bathrooms', 'calcfin_sqft', 'tax_val', 'yr_built', 'fips']] = single_fams[['bedrooms', 'bathrooms', 'calcfin_sqft', 'tax_val', 'fips']].astype('int64')
    single_fams = single_fams.drop(columns= ['prop_use_id'])
    prop_vals_clean = single_fams.dropna()
    prop_vals_clean.bedrooms = prop_vals_clean.bedrooms.astype('int')
    prop_vals_clean.bathrooms = prop_vals_clean.bathrooms.astype('int')
    prop_vals_clean.calcfin_sqft = prop_vals_clean.calcfin_sqft.astype('int')
    prop_vals_clean.tax_val = prop_vals_clean.tax_val.astype('int')
    prop_vals_clean.yr_built = prop_vals_clean.yr_built.astype('int')
    prop_vals_clean.fips = prop_vals_clean.fips.astype('int')
    return prop_vals_clean
