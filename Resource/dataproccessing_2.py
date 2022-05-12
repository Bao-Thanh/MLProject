#%% 
# Import các thư viện
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
# Import các hàm
import function as func

# %% Load dữ liệu
raw_data = pd.read_csv('Model/movies_metadata.csv')

  


# %% Introview dữ liệu
print('\n____________ Dataset info ____________')
print(raw_data.info())  
# Categorical: adult,, original_language,spoken_languages, status, video, production_companies (list),spoken_languages (list), genres (list), production_countries (list)
# Numerical: (continuous): budget, popularity, revenue, runtime, vote_average, vote_count, release_date (timeseries)
# Blank or Null: belongs_to_collection, homepage, tagline
# Contain Error/Typo: original_title (string), overview (string), poster_path (string), tagline (string), title (string), homepage (string),  production_companies (list),spoken_languages (list), genres (list), production_countries (list)

# %%  Convert JSON to array data feature 

raw_data['genres'] = raw_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
raw_data['production_companies'] = raw_data['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
raw_data['production_countries'] = raw_data['production_countries'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
raw_data['spoken_languages'] = raw_data['spoken_languages'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
raw_data['year'] = pd.to_datetime(raw_data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

# %% Convert object to category
features = ['adult', 'original_language', 'status', 'video']
def convert_cat(df, features):
    for feature in features:
        df[feature] = df[feature].astype("category") 

convert_cat(raw_data, features)



# %%
print('\n____________ Counts on revenue feature ____________')
print(raw_data['revenue'].value_counts()) 

print('\n____________ Statistics of numeric features ____________')
print(raw_data.describe())  

print('\n____________ Statistics of category features ____________')
print(raw_data.describe(include=['category']))  


# %%  Draw Diagram

if 0:
    x, y = func.tanso_featurenonnumeric(raw_data['original_language'])
    col_map = plt.get_cmap('Paired')
    plt.bar(x, y, width=2, color=col_map.colors, edgecolor='k', 
        linewidth=2)
    plt.figure(figsize=(10000, 10000)) 
    plt.show()
    #plt.savefig('figures/language.png', format='png', dpi=300)

if 0:
    x, y = func.tanso_featurenonnumeric(raw_data['extenstion'])
    col_map = plt.get_cmap('Paired')
    plt.bar(x, y, width=2, color=col_map.colors, edgecolor='k', 
        linewidth=2)
    plt.figure(figsize=(10000, 10000)) 
    plt.show()

if 0:
    #raw_data.hist(bins=10, figsize=(10,5)) #bins: no. of intervals
    raw_data.hist(figsize=(10,5)) #bins: no. of intervals
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.tight_layout()
    plt.show()

if 0:
    features_to_plot = ["revenue", "runtime", "vote_average","vote_count"]
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.savefig('PART 1-2-3/figures/scatter_mat_all_feat.png', format='png', dpi=300)
    plt.show()

if 0:
    sns.countplot(data = raw_data, x='adult', hue='genres', palette='Blue')

# %%
# Remove unused features
raw_data.drop(columns = ["belongs_to_collection"], inplace=True) 

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

# %% Fill in missing values
for x in raw_data.columns:
    raw_data[x] = raw_data[x].fillna(0)


# %%
print('\n____________ Dataset info ____________')
print(raw_data.info())  


