#%% 
# Import các thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from statistics import mean
from sklearn.model_selection import KFold   
import joblib 
from collections import Counter
from matplotlib.pyplot import xlim
from sklearn import naive_bayes
from pandas.plotting import scatter_matrix  
import os
from ast import literal_eval
# Import các hàm
import function as func

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  SGDRegressor
from sklearn.linear_model import LinearRegression
import sklearn.linear_model

# %% Load dữ liệu
raw_data = pd.read_csv('Model/movies_metadata.csv')

  
# %%  Convert JSON to array data feature 

raw_data['genres'] = raw_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
raw_data['spoken_languages'] = raw_data['spoken_languages'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
raw_data['production_companies'] = raw_data['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
raw_data['production_countries'] = raw_data['production_countries'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
raw_data['year'] = pd.to_datetime(raw_data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
#Can phai chay
raw_data =raw_data.fillna(0)

# %%
# Remove unused features
raw_data.drop(columns = ["belongs_to_collection"], inplace=True) 
raw_data.drop(columns = ["id"], inplace=True) 

# %% Use IMDB's weighted rating formula
vote_counts = raw_data[raw_data['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = raw_data[raw_data['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
C

# %%
m = vote_counts.quantile(0.95)
m

# %%
qualified  = raw_data[(raw_data['vote_count'] >= m) & (raw_data['vote_count'].notnull()) & (raw_data['vote_average'].notnull())]
qualified ['vote_count'] = qualified ['vote_count'].astype('int')
qualified ['vote_average'] = qualified ['vote_average'].astype('int')

# %%
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

# %%
raw_data['wr'] = qualified.apply(weighted_rating, axis=1)
raw_data = raw_data.fillna(0)
# chuyen du lieu trong genres, spoken_languages ra thanh dong
raw_data = raw_data.explode('genres')
raw_data = raw_data.explode('spoken_languages')
raw_data = raw_data.explode('production_countries')

# %%
raw_data_replace = raw_data
genres_name =["Action","Fantasy","Crime","Romance","Family","Thriller",
"Comedy","Science Fiction","Drama","Adventure","History","Horror","Mystery","War",
"Documentary","Animation","Music"]
# chuyen du lieu trong genres ra so
raw_data_replace = raw_data_replace.replace(to_replace ="Action",
                 value ="0")
raw_data_replace = raw_data_replace.replace(to_replace ="Fantasy",
                 value ="1")
raw_data_replace = raw_data_replace.replace(to_replace ="Crime",
                 value ="2")
raw_data_replace = raw_data_replace.replace(to_replace ="Romance",
                 value ="3")
raw_data_replace = raw_data_replace.replace(to_replace ="Family",
                 value ="4")
raw_data_replace = raw_data_replace.replace(to_replace ="Thriller",
                 value ="5")
raw_data_replace = raw_data_replace.replace(to_replace ="Comedy",
                 value ="6")
raw_data_replace = raw_data_replace.replace(to_replace ="Science Fiction",
                 value ="7")
raw_data_replace = raw_data_replace.replace(to_replace ="Drama",
                 value ="8")
raw_data_replace = raw_data_replace.replace(to_replace ="Adventure",
                 value ="9")
raw_data_replace = raw_data_replace.replace(to_replace ="History",
                 value ="10")
raw_data_replace = raw_data_replace.replace(to_replace ="Horror",
                 value ="12")
raw_data_replace = raw_data_replace.replace(to_replace ="Mystery",
                 value ="13")
raw_data_replace = raw_data_replace.replace(to_replace ="War",
                 value ="14")
raw_data_replace = raw_data_replace.replace(to_replace ="Documentary",
                 value ="15")
raw_data_replace = raw_data_replace.replace(to_replace ="Animation",
                 value ="17")
raw_data_replace = raw_data_replace.replace(to_replace ="Music",
                 value ="18")

train_set, test_set = train_test_split(raw_data_replace, test_size=0.2, random_state=42) 

# %%
train_set_labels = train_set["wr"].copy()
train_set = train_set.drop(columns = "wr") 
test_set_labels = test_set["wr"].copy()
test_set = test_set.drop(columns = "wr") 

# %%
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values  

# %%
num_feat_names = ['revenue', 'runtime', 'vote_average', 'vote_count', 'year'] 
cat_feat_names = ['adult', 'budget', 'genres', 'homepage', 'imdb_id',
       'original_language', 'original_title', 'overview', 'popularity',
       'poster_path', 'production_companies', 'production_countries',
       'release_date','spoken_languages', 'status',
       'tagline', 'title', 'video', 'year']

# %%
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place 
    ('cat_encoder', OneHotEncoder()) # convert categorical data into one-hot vectors
    ])   

# %%
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", copy=True)), # copy=False: imputation will be done in-place 
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True)) # Scale features to zero mean and unit variance
    ])  

# %%
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])  

# %%
processed_train_set_val = full_pipeline.fit_transform(train_set.astype(str))
print('\n____________ Processed feature values ____________')
print(processed_train_set_val[[0, 1, 2, 3, 4],:].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature + 1 added features + 35 cols of onehotvector for categorical features.' %(len(num_feat_names)))
#
# %%