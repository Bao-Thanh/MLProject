#%% 
# Import các thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from statistics import mean
from sklearn.model_selection import KFold   
import joblib 
from collections import Counter
import math 
#import seaborn as sns
from matplotlib.pyplot import xlim
from sklearn import naive_bayes
from pandas.plotting import scatter_matrix  
import os
from ast import literal_eval
# Import các hàm
import function as func

# %% Load dữ liệu
#df = pd.read_csv('Dataset/movies_metadata.csv')
df = pd.read_csv('model/movies_metadata.csv')

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

#if 0:
#    g = sns.countplot(data=df, x='genres')
#    g.set_xticklabels(g.get_xticklabels(),rotation=90)
#    fig = g.get_figure()
#    fig.savefig('figs/frequency_genres.png')

#if 0:
#    g = sns.countplot(data=df, x='spoken_languages')
#    g.set_xticklabels(g.get_xticklabels(),rotation=90)
#    fig = g.get_figure()
#    fig.savefig('figs/frequency_spoken_languages.png')

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

raw_data_replace = df
genres_name =["Action","Fantasy","Crime","Romance","Family","Thriller",
"Comedy","Science Fiction","Drama","Adventure","History","Horror","Mystery","War",
"Documentary","Animation","Music"]
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
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(raw_data_replace, test_size=0.3, random_state=42) 
# %%
# check max column value
column = test_set["vote_count"]
max_value = column.max()
print(max_value)
# reduce to fit float 64 of column
train_set['vote_count'] = train_set['vote_count']/1000
test_set['vote_count'] = test_set['vote_count']/1000
# %% early stopping
poly_scaler = Pipeline([
    ("MinMax Scaling", MinMaxScaler()),
#    ("poly_features", PolynomialFeatures(degree=70, include_bias=False)),    
    ("std_scaler", StandardScaler())  ])
X_train_poly_scaled = poly_scaler.fit_transform(train_set[['vote_average','vote_count','popularity']].values)
X_val_poly_scaled = poly_scaler.transform(test_set[['vote_average','vote_count','popularity']].values)
#X_train_poly_scaled[np.isnan(X_train_poly_scaled)] = 0
#X_val_poly_scaled[np.isnan(X_val_poly_scaled)] = 0
# 9.4. Do early stopping 
sgd_reg = SGDRegressor(max_iter=1, tol=-np.inf, # tol<0: allow loss to increase
                       warm_start=True,   # warm_start=True: init fit() with result from previous run
                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42) 
n_iter_wait = 200
minimum_val_error = np.inf  
from copy import deepcopy
train_errors, val_errors = [], []

for epoch in range(1000):
    # Train and compute val. error:
    sgd_reg.fit(X_train_poly_scaled, train_set['genres'].values)  # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(test_set['genres'].values, y_val_predict)
    # Save the best model:
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = deepcopy(sgd_reg)   
    # Stop after n_iter_wait loops with no better val. error:
    if best_epoch < epoch - n_iter_wait:
        break

    # Save for plotting purpose:
    val_errors.append(val_error)
    y_train_predict = sgd_reg.predict(X_train_poly_scaled)
    train_errors.append(mean_squared_error(train_set[['genres']].values, y_train_predict)) 
train_errors = np.sqrt(train_errors) # convert to RMSE
val_errors = np.sqrt(val_errors)
# %%Print best epoch and model
# best_epoch
# best_model.intercept_, best_model.coef_  
importance = best_model.coef_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# %%
# Plot learning curves
if 1:
    best_val_error = val_errors[best_epoch]
    plt.plot(val_errors, "b-", linewidth=2, label="Validation set")
    plt.plot(train_errors, "r-", linewidth=2, label="Training set")
    plt.annotate('Best model',xytext=(best_epoch, best_val_error+0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 xy=(best_epoch, best_val_error), ha="center", fontsize=14,  )      
    plt.xlim(0, epoch)
    plt.grid()
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Root Mean Squared Error", fontsize=14)
    plt.title("Learning curves w.r.t. the training time")
    plt.show()

# %%
#test line predict of best model
column_predict_test=test_set[['vote_average','vote_count','popularity']]
poly_scaled_test_predict = poly_scaler.fit_transform(column_predict_test)
predicted_beast_model_thing = best_model.predict(poly_scaled_test_predict)
#get predict of row 1
sample_id = 30
print(test_set['genres'].iloc[sample_id])
print(math.ceil(predicted_beast_model_thing[sample_id]))
# %%
# Softmax regression
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
x = train_set[['popularity','runtime','vote_count','genres','budget']].values
y = train_set['vote_average'].values.astype(str)
x_test = test_set[['popularity','runtime','vote_count','genres','budget']].values
y_test = test_set['vote_average'].values.astype(str)
softmax_reg = LogisticRegression(multi_class="multinomial", # multinomial: use Softmax regression
                                 solver="lbfgs", random_state=42) # C=10
softmax_reg.fit(x, y)

# try predict
sample_id = 90
softmax_reg.predict_proba([x_test[sample_id]]) 
print(softmax_reg.predict([x_test[sample_id]]) )
y_test[sample_id]
mean_squared_error(y_test,softmax_reg.predict(x_test) )
# %%
