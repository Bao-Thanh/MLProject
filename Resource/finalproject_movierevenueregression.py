'''
This source code is uploaded on Github at the following link: https://github.com/Bao-Thanh/MLProject 

'''
#%% 
# Import các thư viện
from cmath import nan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score   
import joblib 
from sklearn.preprocessing import PolynomialFeatures
from collections import Counter
import seaborn as sns
from matplotlib.pyplot import xlim
from sklearn import naive_bayes
from pandas.plotting import scatter_matrix  
import os
from ast import literal_eval
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
# Import các hàm
import function as func


'''Prepare data'''

# %% Load dữ liệu
# Link data: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv
df = pd.read_csv('Dataset/movies_metadata_regression.csv')

# %% Introview dữ liệu
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

# %% Convert JSON to array feature

df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# %% Convert JSON to one hot dataframe and temp
genre_list=df['genres'].tolist()
temp = []
for genre in genre_list:
    temp = temp + genre
Genres = pd.DataFrame(df['genres'].tolist())
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
df.drop(columns = ["original_language"], inplace=True) 
df.drop(columns = ["status"], inplace=True)

# %% Data visualization: trực quan hóa dữ liệu
if 0:
    g = sns.histplot(df['revenue'], bins=25)
    fig = g.get_figure()
    fig.savefig('figs/histplot_revenue.png')

if 0:
    g = sns.kdeplot(df['revenue'])
    fig = g.get_figure()
    fig.savefig('figs/kdeplot_revenue.png')

if 0:
    from pandas.plotting import scatter_matrix   
    features_to_plot = ['budget','popularity', 'runtime', 'vote_average', 'vote_count']
    scatter_matrix(df[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.savefig('figs/scatter_mat_all_feat.png', format='png', dpi=300)
    plt.show()

if 0:
    g = sns.histplot(df['status'], bins=25)
    fig = g.get_figure()
    fig.savefig('figs/histplot_status.png')

if 0:
    g = sns.histplot(df['original_language'], bins=25)
    fig = g.get_figure()
    fig.savefig('figs/histplot_original_language.png')

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

'''Data processing'''
# %%
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values

num_feat_names = ["year",'budget','popularity', 'runtime', 'vote_average', 'vote_count']
# cat_feat_names = []
onehot_feat_names = columns

# cat_pipeline = Pipeline([
#     ('selector', ColumnSelector(cat_feat_names)),
#     ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value = "NO INFO", copy=True)), # complete missing values. copy=False: imputation will be done in-place 
#     ('cat_encoder', OneHotEncoder(handle_unknown = "ignore")) 
#     ]) 

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
    # ("cat_pipeline", cat_pipeline),
    ("onehot_pipeline", onehot_pipeline)])

# %%
processed_train_set_val = full_pipeline.fit_transform(train_set.astype(str))
print('\n____________ Processed feature values ____________')
print(processed_train_set_val[[0, 1, 2, 3, 4],:])
print(processed_train_set_val.shape)
print('We have {0} numeric feature + {1} one hot features.'.format(len(num_feat_names), len(onehot_feat_names)))
joblib.dump(full_pipeline, r'models/full_pipeline.pkl')

# %%
# onehot_cols = []
# for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
#     onehot_cols = onehot_cols + val_list.tolist()
columns_header = train_set.columns.tolist()
# for name in cat_feat_names:
#     columns_header.remove(name)
processed_train_set = pd.DataFrame(processed_train_set_val, columns = columns_header)
print('\n____________ Processed dataframe ____________')
print(processed_train_set.info())
print(processed_train_set.head())

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

# %%
full_pipeline = joblib.load(r'models/full_pipeline.pkl')
processed_test_set = full_pipeline.transform(test_set) 

# %%
scores = []
names = ['Linear Regression', 'Polynomial','Ridge Regresion','Lasso Regresion','ElasticNet',
'Random Forest Regressor','AdaBoost Regressor', 'Bagging Regressor','SVR',
'KNeighbors Regressor','XGBoost Regressor','Gradient Boosting Regressor',
'ExtraTreesRegressor','SGDRegressor','DecisionTreeRegressor']


'''Linear Regression'''
# %% 

model = LinearRegression()
model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)


'''Polynomial'''
# %% 
model = Pipeline([ ('poly_feat_adder', PolynomialFeatures(degree=5)),('lin_reg', LinearRegression())])
model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

'''Ridge Regresion'''
# %%
model = Ridge(solver = 'sag', alpha = 5, random_state = 42)
model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

'''Lasso Regresion'''
# %%
model =  Lasso(alpha = 0.0005 ,random_state = 42)
model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

'''ElasticNet'''
# %%
model = ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=42)
model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

''''ElasticNet'''
# %% 
model = ElasticNet(alpha=0.01, l1_ratio=0.7, random_state=42)
model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

'''Random Forest Regressor'''
# %% 
model = RandomForestRegressor(n_estimators = 3000, max_features = "sqrt", min_samples_split = 5, bootstrap = True, max_depth= 35, random_state=42)
model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

'''AdaBoost Regressor'''
# %%
model =  AdaBoostRegressor(n_estimators = 300, learning_rate = 0.06, loss ='exponential', random_state = 42)
model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

'''Bagging Regressor'''
# %%
model =  BaggingRegressor(n_estimators = 1000, max_features = 0.6, random_state = 42)
model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

'''SVR'''
# %%
model = SVR(kernel = 'linear',degree = 30, gamma = 'scale')
model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

'''KNeighbors Regressor'''
# %%
model = KNeighborsRegressor(n_neighbors=10, weights='distance')
model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

'''XGBoost Regressor'''
# %%
model = XGBRegressor(max_depth= 15, learning_rate = 0.1,
              booster= 'gblinear', objective='reg:squarederror', random_state=42)

model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

'''Gradient Boosting Regressorr'''
# %%
model = GradientBoostingRegressor(loss='huber',max_depth=4,max_features='log2',n_estimators=200,
                                                       random_state=42)

model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

'''ExtraTreesRegressor'''
# %%
model = ExtraTreesRegressor(max_leaf_nodes=16, n_estimators=200, max_samples=.7, 
                                 bootstrap=True, random_state=42)


model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

'''SGDRegressor'''
# %%
model = SGDRegressor()

model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)

'''Decision Tree'''
# %%
model = DecisionTreeRegressor(min_samples_split=5, min_samples_leaf= 12,max_depth=10,max_features="sqrt" ,random_state=42)

model.fit(processed_train_set_val, train_set_labels)
score = model.score(processed_test_set, test_set_labels)
scores.append(score)


# %% Convert 2 list to dataframe
data = {'Model': names, 'Score': scores}
df = pd.DataFrame(data)
df.sort_values('Score', inplace=True, ascending=False) # sort 'score' feature to find top 3 model
df

# %% Implement Ensemble learning with top 3 model

models = list()
models.append(('forest',  RandomForestRegressor(n_estimators = 3000, max_features = "sqrt", min_samples_split = 5, bootstrap = True, max_depth= 35, random_state=42)))
models.append(('knn',  KNeighborsRegressor(n_neighbors=10, weights='distance')))
models.append(('bag',  BaggingRegressor(n_estimators = 1000, max_features = 0.6, random_state = 42)))


# Use voting method 
voting = VotingRegressor(estimators=models)
voting.fit(processed_train_set_val, train_set_labels)
print('\n____________ VotingRegressor ____________')
r2score, rmse = r2score_and_rmse(voting, processed_train_set_val, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(voting)    

# Use stacking method 
stacking = StackingRegressor(estimators=models, final_estimator=ExtraTreesRegressor(max_leaf_nodes=16, n_estimators=200, max_samples=.7, 
                                 bootstrap=True, random_state=42))
stacking.fit(processed_train_set_val, train_set_labels)
store_model(stacking)
print('\n____________ StackingRegressor ____________')
r2score, rmse = r2score_and_rmse(stacking, processed_train_set_val, train_set_labels)
print('\nR2 score (on training data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))
store_model(stacking)  

# Compute R2 score and root mean squared error
  
# Predict labels for some training instances
#print("Input data: \n", train_set.iloc[0:9])
# print("\nPredictions: ", model.predict(processed_train_set_val[0:9]).round(decimals=1))
# print("Labels:      ", list(train_set_labels[0:9])) 

# %%
voting_best_model = joblib.load('models/VotingRegressor_model.pkl')
stacking_best_model = joblib.load('models/StackingRegressor_model.pkl')

store_model(voting_best_model, model_name="VOTING")
store_model(stacking_best_model, model_name="STACKING")

# %% Find best method for test set
print('\n____________ VotingRegressor ____________')
r2score, rmse = r2score_and_rmse(voting_best_model, processed_test_set, test_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

print('\n____________ StackingRegressor ____________')
r2score, rmse = r2score_and_rmse(stacking_best_model, processed_test_set, test_set_labels)
print('\nPerformance on test data:')
print('R2 score (on test data, best=1):', r2score)
print("Root Mean Square Error: ", rmse.round(decimals=1))

# %%
# print("\nTest data: \n", test_set.iloc[0:9])
print("\nPredictions: ", stacking_best_model.predict(processed_test_set[0:9]).round(decimals=1))
print("Labels:      ", list(test_set_labels[0:9]),'\n')

