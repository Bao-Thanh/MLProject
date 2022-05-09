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
from collections import Counter
import seaborn as sns
from matplotlib.pyplot import xlim
from sklearn import naive_bayes
from pandas.plotting import scatter_matrix  
import os
from ast import literal_eval

# Hàm trả về tần số xuất hiện  của các sample
# trong 1 feature non-numeric
# x là label , y là value
def tanso_featurenonnumeric(features):
    x = []
    y = []
    features = features
    freqs = Counter(features)
    temp = np.array(list(freqs.items()))
    for i in range(1,len(freqs.values())):
        x.append(temp[i][0])
        y.append(temp[i][1])
    return np.array(x), np.array(y)
