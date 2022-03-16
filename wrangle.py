import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from env import get_db_url
from pydataset import data
import os

def show_codeup_dbs():
    url = get_db_url('employees')
    codeup_dbs = pd.read_sql('SHOW DATABASES', url)
    print('List of Codeup DBs:\n')
    return codeup_dbs

