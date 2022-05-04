#%% Import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from statistics import mean
from sklearn.model_selection import KFold   
import joblib 

# %% Load dữ liệu
raw_data = pd.read_csv('Dataset/dataset.csv')

# %% Introview dữ liệu
print('\n____________ Dataset info ____________')
print(raw_data.info())  

print('\n____________ Counts on a feature ____________')
print(raw_data['language'].value_counts()) 

print('\n____________ Statistics of numeric features ____________')
print(raw_data.describe())    


# %%
