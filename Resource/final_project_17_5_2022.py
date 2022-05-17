'''
This source code is uploaded on Github at the following link: https://github.com/Bao-Thanh/MLProject 

'''
#%% 
# Import các thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score   
import joblib 
from collections import Counter
#import seaborn as sns
from matplotlib.pyplot import xlim
from sklearn import naive_bayes
from pandas.plotting import scatter_matrix  
import os
from ast import literal_eval
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
# Import các hàm
import function as func


'''Prepare data'''

# %% Load dữ liệu
# Link data: https://www.kaggle.com/code/rounakbanik/movie-recommender-systems/data?select=movies_metadata.csv
df = pd.read_csv('Dataset/movies_metadata_regression.csv')

# %% Introview dữ liệu
print('\n____________ Dataset info ____________')
print(df.info())  

# %% Convert JSON to array feature

df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# %% Convert JSON to one hot dataframe and temp
genre_list=df['genres'].tolist()
temp = []
for genre in genre_list:
    temp = temp + genre
Genres = pd.DataFrame(temp)
unique_genres = Genres[0].unique()
columns = unique_genres   

# %%
index = range(len(df))
df_Genre_list= pd.DataFrame(index = index, columns = columns)
df_Genre_list=df_Genre_list.fillna(0)
for row in range(len(df_Genre_list)):
    for col in genre_list[row]:
        df_Genre_list.loc[row,col]=1

# %%
df = pd.concat([df,df_Genre_list],axis=1)

# %% Convert timeseries feature
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

# %% Interview dữ liệu
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
df.drop(columns = ["genres"], inplace=True) 
df.drop(columns = ["id"], inplace=True) 
df.drop(columns = ["homepage"], inplace=True) 
df.drop(columns = ["tagline"], inplace=True) 
df.drop(columns = ["original_title"], inplace=True) 
df.drop(columns = ["title"], inplace=True) 
df.drop(columns = ["overview"], inplace=True) 
df.drop(columns = ["release_date"], inplace=True) 
df.drop(columns = ["production_companies"], inplace=True) 
df.drop(columns = ["production_countries"], inplace=True) 
df.drop(columns = ["spoken_languages"], inplace=True) 
df.drop(columns = ["keywords"], inplace=True) 


# %% Data visualization: trực quan hóa dữ liệu
if 0:
    df.plot(kind="scatter", y="runtime", x="revenue", alpha=0.2)
    plt.savefig('figs/scatter_revenue_vs_runtime_feat.png', format='png', dpi=300)
    plt.show()

#if 0:
#    g = sns.countplot(data=df, x='genres')
#    g.set_xticklabels(g.get_xticklabels(),rotation=90)
#    fig = g.get_figure()
#    fig.savefig('figs/frequency_genres.png')

if 0:
    scatter_matrix(df)
    plt.savefig('figs/scatter_matrix.png', format='png', dpi=300)
    plt.show()

# %% Tách dataset ra tập train và test 
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df, test_size=0.3, random_state=42) 

# %%
print(train_set.info())
print(test_set.info())

# %%
train_set_labels = train_set["popularity"].copy()
train_set = train_set.drop(columns = "popularity") 
test_set_labels = test_set["popularity"].copy()
test_set = test_set.drop(columns = "popularity") 

'''Data processing'''
# %%
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values

num_feat_names = ['budget','revenue', 'runtime', 'vote_average', 'vote_count']
cat_feat_names = ["year", "status", "original_language"]
onehot_feat_names = columns

cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place 
    ('cat_encoder', OneHotEncoder(handle_unknown = "ignore")) 
    ]) 

num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)), 
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True))
    ])  

onehot_pipeline = Pipeline([
    ('selector', ColumnSelector(onehot_feat_names)),
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)), 
    ]) 

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
    ("onehot_pipeline", onehot_pipeline)])

# %%
processed_train_set_val = full_pipeline.fit_transform(train_set.astype(str))
print('\n____________ Processed feature values ____________')
print(processed_train_set_val[[0, 1, 2, 3, 4],:])
print(processed_train_set_val.shape)
print('We have {0} numeric feature + {1} categorical features + {2} one hot features.'.format(len(num_feat_names), len(cat_feat_names), len(onehot_feat_names)))
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

# # %% 
# processed_train_set.to_csv('Dataset/result.csv')


# # %%
# df_new = pd.read_csv('Dataset/result.csv')

# %%
def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse      
import joblib
def store_model(model, model_name = ""):
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'models/' + model_name + '_model.pkl')
def load_model(model_name):
    model = joblib.load('models/' + model_name + '_model.pkl')
    #print(model)
    return model

'''Train'''

# %% 
print('\n____________ Fine-tune models ____________')
def print_search_result(grid_search, model_name = ""): 
    print("\n====== Fine-tune " + model_name +" ======")
    print('Best hyperparameter combination: ',grid_search.best_params_)
    print('Best rmse: ', np.sqrt(-grid_search.best_score_))  
    #print('Best estimator: ', grid_search.best_estimator_) # NOTE: require refit=True in  SearchCV
    print('Performance of hyperparameter combinations:')
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        print('rmse =', np.sqrt(-mean_score).round(decimals=1), params) 

# %% Train model by Lasso (fine tunning)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

cv = KFold(n_splits=5,shuffle=True,random_state=42) 

model = RandomForestRegressor()
param_dist = {"max_depth": [10,35, None],
              "max_features": ["auto", "sqrt", "log2"],
              "min_samples_split": [2, 5, 10],
              "bootstrap": [True, False],
              "n_estimators": [100, 150]}

grid_search = GridSearchCV(model, param_dist, cv=cv, scoring='neg_mean_squared_error', return_train_score=True, 
        refit=True)
grid_search.fit(processed_train_set_val, train_set_labels)
joblib.dump(grid_search,'saved_objects/RandomForestRegressor_grid_search.pkl')
print_search_result(grid_search, model_name = "RandomForestRegressor") 

 
# %%
#search = joblib.load('saved_objects/RandomForestRegressor_grid_search.pkl')
search = grid_search
best_model = search.best_estimator_

print('\n____________ ANALYZE AND TEST YOUR SOLUTION ____________')
print('SOLUTION: ' , best_model)
store_model(best_model, model_name="SOLUTION")

# %%
if type(best_model).__name__ == "RandomForestRegressor":
    # Print features and importance score  (ONLY on rand forest)
    feature_importances = best_model.feature_importances_
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    feature_names = train_set.columns.tolist() + onehot_cols
    for name in cat_feat_names:
        feature_names.remove(name)
    print('\nFeatures and importance score: ')
    print(*sorted(zip( feature_names, feature_importances.round(decimals=4)), key = lambda row: row[1], reverse=True),sep='\n')

# %%
#full_pipeline = joblib.load(r'models/full_pipeline.pkl')
processed_test_set = full_pipeline.transform(test_set)  

# %%
r2score, rmse = r2score_and_rmse(best_model, processed_test_set, test_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

# %%
# print("\nTest data: \n", test_set.iloc[0:9])
print("\nPredictions: ", best_model.predict(processed_test_set[0:9]).round(decimals=1))
print("Labels:      ", list(test_set_labels[0:9]),'\n')

# %%
