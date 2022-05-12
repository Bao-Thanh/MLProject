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
df = pd.read_csv('Dataset/movies_metadata.csv')

# %% Introview dữ liệu
print('\n____________ Dataset info ____________')
print(df.info())  

# %% Convert JSON to array data feature 

df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['production_companies'] = df['production_companies'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['production_countries'] = df['production_countries'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
df['spoken_languages'] = df['spoken_languages'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# %% Convert timeseries feature
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

# %% Introview dữ liệu
print('\n____________ Dataset info ____________')
print(df.info())  

print('\n____________ Dataset value ____________')
print(df.head())  

print('\n____________ Counts on revenue feature ____________')
print(df['revenue'].value_counts()) 

print('\n____________ Counts on budget feature ____________')
print(df['budget'].value_counts()) 

print('\n____________ Statistics of numeric features ____________')
print(df.describe())  

print('\n____________ Statistics of revenue feature ____________')
print(df['revenue'].describe())

print('\n____________ Statistics of budget feature ____________')
print(df['budget'].describe())  


# %% Remove unused features
df.drop(columns = ["belongs_to_collection"], inplace=True) 
df.drop(columns = ["id"], inplace=True) 
df.drop(columns = ["imdb_id"], inplace=True) 
df.drop(columns = ["homepage"], inplace=True) 
df.drop(columns = ["tagline"], inplace=True) 
df.drop(columns = ["poster_path"], inplace=True) 
df.drop(columns = ["video"], inplace=True) 
df.drop(columns = ["original_title"], inplace=True) 
df.drop(columns = ["title"], inplace=True) 
df.drop(columns = ["adult"], inplace=True)
df.drop(columns = ["overview"], inplace=True) 
df.drop(columns = ["original_language"], inplace=True) 
df.drop(columns = ["release_date"], inplace=True) 

# %% Convert list type feature to other sample
df = df.explode('genres')
df = df.explode('spoken_languages')
df = df.explode('production_companies') 
df = df.explode('production_countries')

# %% Convert object feature to category feature
cat_feat_names = ["genres","production_companies", "production_countries", "status","spoken_languages", "year"]
def convert_cat(df, cat_feat_names):
    for feature in cat_feat_names:
        df[feature] = df[feature].astype("category")
convert_cat(df, cat_feat_names)

# %% Trực quan hóa dữ liệu
if 0:
    df.plot(kind="scatter", y="runtime", x="revenue", alpha=0.2)
    #plt.axis([0, 5, 0, 10000])
    plt.savefig('figs/scatter_revenue_vs_runtime_feat.png', format='png', dpi=300)
    plt.show()

if 0:
    g = sns.countplot(data=df, x='genres')
    g.set_xticklabels(g.get_xticklabels(),rotation=90)
    fig = g.get_figure()
    fig.savefig('figs/frequency_genres.png')

if 0:
    g = sns.countplot(data=df, x='spoken_languages')
    g.set_xticklabels(g.get_xticklabels(),rotation=90)
    fig = g.get_figure()
    fig.savefig('figs/frequency_spoken_languages.png')

# %%
corr_matrix = df.corr()
print(corr_matrix) # print correlation matrix
print('\n',corr_matrix["revenue"].sort_values(ascending=False))

# %% Tách dataset ra tập train và test 
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.3, random_state=42) 

# %%
print(train_set.info())
print(test_set.info())

# %%
train_set_labels = train_set["revenue"].copy()
train_set = train_set.drop(columns = "revenue") 
test_set_labels = test_set["revenue"].copy()
test_set = test_set.drop(columns = "revenue") 

# %%
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values

# %%
num_feat_names = ['budget','popularity', 'runtime', 'vote_average', 'vote_count'] 
cat_feat_names = ['genres','production_companies', 'production_countries','spoken_languages', 'status', "year"]

# %%
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place 
    ('cat_encoder', OneHotEncoder()) 
    ]) 

num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)), 
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True))
    ])  

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline) ])

# %%
processed_train_set_val = full_pipeline.fit_transform(train_set)
print('\n____________ Processed feature values ____________')
print(processed_train_set_val[[0, 1, 2, 3, 4],:].toarray())
print(processed_train_set_val.shape)
print('We have %d numeric feature + 9863 cols of onehotvector for categorical features.' %(len(num_feat_names)))
joblib.dump(full_pipeline, r'models/full_pipeline.pkl')

# %%
onehot_cols = []
for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
    onehot_cols = onehot_cols + val_list.tolist()
columns_header = train_set.columns.tolist() + onehot_cols
for name in cat_feat_names:
    columns_header.remove(name)
processed_train_set = pd.DataFrame(processed_train_set_val.toarray(), columns = columns_header)
print('\n____________ Processed dataframe ____________')
print(processed_train_set.info())
print(processed_train_set.head())


# %% Try to train

