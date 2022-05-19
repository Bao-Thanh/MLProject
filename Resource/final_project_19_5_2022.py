'''
This source code is uploaded on Github at the following link: https://github.com/Bao-Thanh/MLProject 

'''
#%% 
# Import các thư viện
import io
import threading
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
# Import các hàm
import function as func
import joblib
from tkinter import *
import tkinter as tk

'''Prepare data'''
fit_in_grid_search_fine_tunning_line_220_true_false = 0
class ColumnSelector(BaseEstimator, TransformerMixin):
        def __init__(self, feature_names):
            self.feature_names = feature_names
        def fit(self, dataframe, labels=None):
            return self
        def transform(self, dataframe):
            return dataframe[self.feature_names].values


def r2score_and_rmse(model, train_data, labels): 
    r2score = model.score(train_data, labels)
    from sklearn.metrics import mean_squared_error
    prediction = model.predict(train_data)
    mse = mean_squared_error(labels, prediction)
    rmse = np.sqrt(mse)
    return r2score, rmse 
def print_search_result(grid_search, model_name = ""):
    line_2 = "Best hyperparameter combination: " +str(grid_search.best_params_)
    root_print_search_result = Toplevel(root)
    root_print_search_result.title("data info")
    scrollbar = Scrollbar(root_print_search_result)
    scrollbar.pack( side = RIGHT, fill = Y )
    scrollbarX = Scrollbar(root_print_search_result,orient="horizontal")
    scrollbarX.pack( side = BOTTOM, fill = X )
    mylist = Text(root_print_search_result, yscrollcommand = scrollbar.set,
    xscrollcommand= scrollbarX.set)
    mylist.insert(END, "\n====== Fine-tune " + str(model_name) +" ======"+
    line_2 +
    'Best rmse: '+ str(-grid_search.best_score_)+"\n"+
    'Performance of hyperparameter combinations:'+
          "")
    cv_results = grid_search.cv_results_
    for (mean_score, params) in zip(cv_results["mean_test_score"], cv_results["params"]):
        mylist.insert(END,'rmse ='+ str(np.sqrt(-mean_score).round(decimals=1))+str(params)+"\n") 

    mylist.pack( side = LEFT, fill = BOTH )
    scrollbar.config( command = mylist.yview )
    scrollbarX.config(command = mylist.xview)
        #print('Best estimator: ', grid_search.best_estimator_) # NOTE: require refit=True in  SearchCV

def store_model(model, model_name = ""):
    if model_name == "": 
        model_name = type(model).__name__
    joblib.dump(model,'models/' + model_name + '_model.pkl')
def load_model(model_name):
    model = joblib.load('models/' + model_name + '_model.pkl')
        #print(model)
    return model
def print_dataset_info(data_set):
    root_print_dataset_info = Toplevel(root)
    root_print_dataset_info.title("data info")
    scrollbar = Scrollbar(root_print_dataset_info)
    scrollbar.pack( side = RIGHT, fill = Y )
    scrollbarX = Scrollbar(root_print_dataset_info,orient="horizontal")
    scrollbarX.pack( side = BOTTOM, fill = X )
    mylist = Text(root_print_dataset_info, yscrollcommand = scrollbar.set,
    xscrollcommand= scrollbarX.set)
    mylist.insert(END, "'____________ Dataset info ____________'\n")
    buffer = io.StringIO()
    data_set.info(buf=buffer)
    buffer_value = buffer.getvalue()
    mylist.insert(END,
          str(buffer_value)+"\n")
    mylist.insert(END,
          "'____________ Dataset value ____________'\n"+
          str(data_set.head())+
          "\n'____________ Counts on revenue feature ____________'\n"+
          str(data_set['revenue'].value_counts())+"\n"+
          "'____________ Counts on budget feature ____________'\n"+
          str(data_set['budget'].value_counts())+"\n"+
          "'____________ Statistics of numeric features ____________'\n"+
          str(data_set.describe())+"\n"+
          "'____________ Statistics of revenue feature ____________'\n"+
          str(data_set['revenue'].describe())+"\n"+
          "'____________ Statistics of budget feature ____________'\n"+
          str(data_set['budget'].describe())+
          "")
    mylist.pack( side = LEFT, fill = BOTH )
    scrollbar.config( command = mylist.yview )
    scrollbarX.config(command = mylist.xview)
def set_cat_num_onehot_full_pipeline(cat_feat_names, num_feat_names, onehot_feat_names):
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
    return cat_pipeline, num_pipeline, onehot_pipeline, full_pipeline

def set_processed_train_set(full_pipeline, processed_train_set_val,train_set,cat_feat_names):
    onehot_cols = []
    for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
        onehot_cols = onehot_cols + val_list.tolist()
    columns_header = train_set.columns.tolist() + onehot_cols
    for name in cat_feat_names:
        columns_header.remove(name)
    processed_train_set = pd.DataFrame(processed_train_set_val.toarray(), columns = columns_header)
    
    root_set_processed_train_set = Toplevel(root)
    root_set_processed_train_set.title("data info")
    scrollbar = Scrollbar(root_set_processed_train_set)
    scrollbar.pack( side = RIGHT, fill = Y )
    scrollbarX = Scrollbar(root_set_processed_train_set,orient="horizontal")
    scrollbarX.pack( side = BOTTOM, fill = X )
    mylist = Text(root_set_processed_train_set, yscrollcommand = scrollbar.set,
    xscrollcommand= scrollbarX.set)
    mylist.insert(END, "'____________ Processed dataframe ____________'\n")
    buffer = io.StringIO()
    processed_train_set.info(buf=buffer)
    buffer_value = buffer.getvalue()
    mylist.insert(END,
          str(buffer_value)+"\n")
    mylist.insert(END,
    str(processed_train_set.head())+
    +"")
    mylist.pack( side = LEFT, fill = BOTH )
    scrollbar.config( command = mylist.yview )
    scrollbarX.config(command = mylist.xview)
        #print('Best estimator: ', grid_search.best_estimator_) # NOTE: require refit=True in  SearchCV
    return processed_train_set   
def fit_transform_and_dump_set(train_set, full_pipeline,num_feat_names,cat_feat_names,onehot_feat_names):
    processed_train_set_val = full_pipeline.fit_transform(train_set.astype(str))
    
    root_fit_transform_and_dump_set = Toplevel(root)
    root_fit_transform_and_dump_set.title("data info")
    scrollbar = Scrollbar(root_fit_transform_and_dump_set)
    scrollbar.pack( side = RIGHT, fill = Y )
    scrollbarX = Scrollbar(root_fit_transform_and_dump_set,orient="horizontal")
    scrollbarX.pack( side = BOTTOM, fill = X )
    mylist = Text(root_fit_transform_and_dump_set, yscrollcommand = scrollbar.set,
    xscrollcommand= scrollbarX.set)
    mylist.insert(END, "'____________ Processed feature values ____________'\n"+
    str(processed_train_set_val[[0, 1, 2, 3],:])+"\n"+
    str(processed_train_set_val.shape)+"\n"+
    'We have {0} numeric feature + {1} categorical features + {2} one hot features.'.format(len(num_feat_names), len(cat_feat_names), len(onehot_feat_names))+
    "")
    mylist.pack( side = LEFT, fill = BOTH )
    scrollbar.config( command = mylist.yview )
    scrollbarX.config(command = mylist.xview)
# Tai sao lai dump o day khi ko dong toi full_pipeline
    joblib.dump(full_pipeline, r'models/full_pipeline.pkl')
    return processed_train_set_val
def train_model_by_lasso(processed_train_set_val, train_set_labels):
    cv = KFold(n_splits=5,shuffle=True,random_state=42) 

    model = RandomForestRegressor()
#    param_dist = {  "n_estimators"      : [1500, 3000],
#                "max_features"      : ["sqrt", "log2"],
#                "min_samples_split" : [4,5,6],
#                "bootstrap"         : [True, False],
#                'max_depth'         : [15,35,70]}

    param_dist = {"max_depth": [10,35, None],
            "max_features": ["auto", "sqrt", "log2"],
            "min_samples_split": [2, 5, 10],
            "bootstrap": [True, False],
            "n_estimators": [100, 150]}
    grid_search = GridSearchCV(model, param_dist, cv=cv, scoring='neg_mean_squared_error', return_train_score=True, 
        refit=True)
    if fit_in_grid_search_fine_tunning_line_220_true_false:
        grid_search.fit(processed_train_set_val, train_set_labels)
        joblib.dump(grid_search,'saved_objects/RandomForestRegressor_grid_search.pkl')
    print_search_result(grid_search, model_name = "RandomForestRegressor")
    return grid_search
def load_and_get_best_model(search):
    search = joblib.load('saved_objects/RandomForestRegressor_grid_search.pkl')
    best_model = search.best_estimator_
       
    root_load_and_get_best_model = Toplevel(root)
    root_load_and_get_best_model.title("data info")
    scrollbar = Scrollbar(root_load_and_get_best_model)
    scrollbar.pack( side = RIGHT, fill = Y )
    scrollbarX = Scrollbar(root_load_and_get_best_model,orient="horizontal")
    scrollbarX.pack( side = BOTTOM, fill = X )
    mylist = Text(root_load_and_get_best_model, yscrollcommand = scrollbar.set,
    xscrollcommand= scrollbarX.set)
    mylist.insert(END, "'____________ ANALYZE AND TEST YOUR SOLUTION ____________'\n"+
    "SOLUTION: "+ str(best_model)+
    "")
    mylist.pack( side = LEFT, fill = BOTH )
    scrollbar.config( command = mylist.yview )
    scrollbarX.config(command = mylist.xview)
    
    store_model(best_model, model_name="SOLUTION")
    return best_model, search
def print_features_and_importance_score(best_model,full_pipeline,cat_feat_names,train_set):
    if type(best_model).__name__ == "RandomForestRegressor":
        # Print features and importance score  (ONLY on rand forest)
        feature_importances = best_model.feature_importances_
        onehot_cols = []
        for val_list in full_pipeline.transformer_list[1][1].named_steps['cat_encoder'].categories_: 
            onehot_cols = onehot_cols + val_list.tolist()
        feature_names = train_set.columns.tolist() + onehot_cols
        
        root_print_features_and_importance_score = Toplevel(root)
        root_print_features_and_importance_score.title("data info")
        scrollbar = Scrollbar(root_print_features_and_importance_score)
        scrollbar.pack( side = RIGHT, fill = Y )
        scrollbarX = Scrollbar(root_print_features_and_importance_score,orient="horizontal")
        scrollbarX.pack( side = BOTTOM, fill = X )
        mylist = Text(root_print_features_and_importance_score, yscrollcommand = scrollbar.set,
        xscrollcommand= scrollbarX.set)
        for name in cat_feat_names:
            feature_names.remove(name)
        mylist.insert(END, "Features and importance score: "+"\n")
        x = (sorted(zip( feature_names, feature_importances.round(decimals=4)), key = lambda row: row[1], reverse=True))
        for in_x in x:
            mylist.insert(END, str(in_x)+"\n")
        mylist.pack( side = LEFT, fill = BOTH )
        scrollbar.config( command = mylist.yview )
        scrollbarX.config(command = mylist.xview)

def fit_train_set(search,processed_train_set_val,train_set_labels, train_set):
    model = search.best_estimator_
    model.fit(processed_train_set_val, train_set_labels)
# Compute R2 score and root mean squared error
    
    root_fit_train_set = Toplevel(root)
    root_fit_train_set.title("data info")
    scrollbar = Scrollbar(root_fit_train_set)
    scrollbar.pack( side = RIGHT, fill = Y )
    scrollbarX = Scrollbar(root_fit_train_set,orient="horizontal")
    scrollbarX.pack( side = BOTTOM, fill = X )
    mylist = Text(root_fit_train_set, yscrollcommand = scrollbar.set,
    xscrollcommand= scrollbarX.set)
    r2score, rmse = r2score_and_rmse(model, processed_train_set_val, train_set_labels)
    mylist.insert(END, "'____________ RandomForestRegressor ____________'\n"+
    "R2 score (on training data, best=1): "+str(r2score)+
    "Root Mean Square Error: "+str(rmse.round(decimals=1))+
    "Input data: \n"+str(train_set.iloc[0:9])+
    "\nPredictions: "+str(model.predict(processed_train_set_val[0:9]).round(decimals=1))+"\n"+
    "Labels:      "+str( list(train_set_labels[0:9]))+
    "")
    mylist.pack( side = LEFT, fill = BOTH )
    scrollbar.config( command = mylist.yview )
    scrollbarX.config(command = mylist.xview)
    store_model(model)      
    # Predict labels for some training instances
    return model
def xu_ly_cot_genres_years(data_set):
# Convert JSON to array feature
    data_set['genres'] = data_set['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

# Convert JSON to one hot dataframe and temp
    genre_list=data_set['genres'].tolist()
    temp = []
    for genre in genre_list:
        temp = temp + genre
    Genres = pd.DataFrame(data_set['genres'].tolist())
    unique_genres = Genres[0].unique()
    columns = unique_genres   


    index = range(len(data_set))
    df_Genre_list= pd.DataFrame(index = index, columns = columns)
    df_Genre_list=df_Genre_list.fillna(0)
    for row in range(len(df_Genre_list)):
        for col in genre_list[row]:
            df_Genre_list.loc[row,col]=1
    data_set = pd.concat([data_set,df_Genre_list],axis=1)
#    Convert timeseries feature
    data_set['year'] = pd.to_datetime(data_set['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    return data_set, columns
def train_and_dump_model(num_feat_names, cat_feat_names, onehot_feat_names, test_set,train_set,train_set_labels):
    cat_pipeline, num_pipeline, onehot_pipeline, full_pipeline = set_cat_num_onehot_full_pipeline(cat_feat_names, num_feat_names, onehot_feat_names)

    processed_train_set_val = fit_transform_and_dump_set(train_set, full_pipeline,num_feat_names,cat_feat_names,onehot_feat_names)

    '''Train'''
    print('\n____________ Fine-tune models ____________')

# Train model by Lasso (fine tunning)

    grid_search = train_model_by_lasso(processed_train_set_val, train_set_labels)

    best_model, search = load_and_get_best_model(grid_search)
# 
    print_features_and_importance_score(best_model,full_pipeline,cat_feat_names,train_set)

#  Try to fit train set
    model = fit_train_set(search,processed_train_set_val,train_set_labels, train_set)

#full_pipeline = joblib.load(r'models/full_pipeline.pkl')
    processed_test_set = full_pipeline.transform(test_set)  
    return best_model, processed_test_set
class Main(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
    def initUI(self):
        self.parent.title("Cuoi Ky Hoc May")
  
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)
  
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Add to model", command=self.add_to_model)
        fileMenu.add_command(label="Train model", command=self.thearding_train_model)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=fileMenu)
        global e1
        e1 = tk.Entry(root)
        tk.Label(root, text="Budget").grid(row=0)
        e1.grid(row=0, column=1)
        global e2
        e2 = tk.Entry(root)
        tk.Label(root, text="original_language").grid(row=1)
        e2.grid(row=1, column=1)
        global e3
        e3 = tk.Entry(root)
        tk.Label(root, text="popularity").grid(row=2)
        e3.grid(row=2, column=1)
        global e4
        e4 = tk.Entry(root)
        tk.Label(root, text="runtime").grid(row=3)
        e4.grid(row=3, column=1)
        global e5
        e5 = tk.Entry(root)
        tk.Label(root, text="status").grid(row=4)
        e5.grid(row=4, column=1)
        global e6
        e6 = tk.Entry(root)
        tk.Label(root, text="vote_average").grid(row=5)
        e6.grid(row=5, column=1)
        global e7
        e7 = tk.Entry(root)
        tk.Label(root, text="vote_count").grid(row=6)
        e7.grid(row=6, column=1)
        global i1
        i1=IntVar()
        c1 = Checkbutton(root, text = "Action", variable=i1)
        c1.grid(row=9)
        global i2
        i2=IntVar()
        c2 = Checkbutton(root, text = "Adventure", variable=i2)
        c2.grid(row=10)
        global i3
        i3=IntVar()
        c3 = Checkbutton(root, text = "Fantasy", variable=i3)
        c3.grid(row=11)
        global i4
        i4=IntVar()
        c4 = Checkbutton(root, text = "Animation", variable=i4)
        c4.grid(row=12)
        global i5
        i5=IntVar()
        c5 = Checkbutton(root, text = "Science Fiction", variable=i5)
        c5.grid(row=13)
        global i6
        i6=IntVar()
        c6 = Checkbutton(root, text = "Crime", variable=i6)
        c6.grid(row=14)
        global i7
        i7=IntVar()
        c7 = Checkbutton(root, text = "Drama", variable=i7)
        c7.grid(row=15)
        global i8
        i8=IntVar()
        c8 = Checkbutton(root, text = "Thriller", variable=i8)
        c8.grid(row=16)
        global i9
        i9=IntVar()
        c9 = Checkbutton(root, text = "Family", variable=i9)
        c9.grid(row=17)
        global i10
        i10=IntVar()
        c10 = Checkbutton(root, text = "Western", variable=i10)
        c10.grid(row=18)
        global e8
        e8 = tk.Entry(root)
        tk.Label(root, text="year").grid(row=19)
        e8.grid(row=19, column=1)
        tk.Button(root,text='Add',command=self.add_it).grid(row=20, 
                                   column=0)
        self.txt = Text(self)
    def add_it(self):
        data = {'budget' : [str(e1.get())], 
        'original_language' : [str(e2.get())],
        'popularity':[str(e3.get())],
        'runtime':[str(e4.get())],
        'status':[str(e5.get)],
        'vote_average':[str(e6.get())],
        'vote_count':[str(e7.get())],
        'Action':[str(i1.get())],
        'Adventure':[str(i2.get())],
        'Fantasy':[str(i3.get())],
        'Animation':[str(i4.get())],
        'Science Fiction':[(i5.get())],
        'Crime':[(i6.get())],
        'Drama':[(i7.get())],
        'Thirller':[(i8.get())],
        'Family':[(i9.get())],
        'Western':[(i10.get())],
        'year':[str(e8.get())]}
        data_panda = pd.DataFrame(data)
        root_fit_train_set2 = Toplevel(root)
        root_fit_train_set2.title("data info")
        scrollbar = Scrollbar(root_fit_train_set2)
        scrollbar.pack( side = RIGHT, fill = Y )
        scrollbarX = Scrollbar(root_fit_train_set2,orient="horizontal")
        scrollbarX.pack( side = BOTTOM, fill = X )
        mylist = Text(root_fit_train_set2, yscrollcommand = scrollbar.set,
        xscrollcommand= scrollbarX.set)
        mylist.insert(END, "input value"+str(e1.get())+" "+str(e2.get())+" "+str(e3.get())+" "+str(e4.get())+" "+str(e5.get())+" "+str(e6.get())+" "+
        str(e7.get())+" "+" "+str(i1.get())+" "+str(i2.get())+" "+
        str(i3.get())+" "+str(i4.get())+" "+str(i5.get())+" "+str(i6.get())+" "+
        str(i7.get())+" "+str(i8.get())+" "+str(i9.get())+" "+str(i10.get())+" "+str(e8.get())+"\n")
        full_pipeline = joblib.load(r'models/full_pipeline.pkl')
        search = joblib.load('saved_objects/RandomForestRegressor_grid_search.pkl')
        best_model = search.best_estimator_
        processed_test_set = full_pipeline.transform(data_panda)
        mylist.insert(END,
        "\nPredictions: "+str(best_model.predict(processed_test_set).round(decimals=1))+
        "")
        mylist.pack( side = LEFT, fill = BOTH )
        scrollbar.config( command = mylist.yview )
        scrollbarX.config(command = mylist.xview)
        
    def add_to_model(self):  
        global df
        global columns
    # Load dữ liệu
    # Link data: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata?select=tmdb_5000_movies.csv
        df = pd.read_csv('Dataset/movies_metadata_regression.csv')
        print_dataset_info(df)
        df, columns = xu_ly_cot_genres_years(df)
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

        #if 0:
        #    g = sns.histplot(df['revenue'], bins=25)
        #    fig = g.get_figure()
        #    fig.savefig('figs/histplot_revenue.png')

        #if 0:
        #    g = sns.kdeplot(df['revenue'])
        #    fig = g.get_figure()
        #    fig.savefig('figs/kdeplot_revenue.png')

        if 0:
            from pandas.plotting import scatter_matrix   
            features_to_plot = ['budget','popularity', 'runtime', 'vote_average', 'vote_count']
            scatter_matrix(df[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
            plt.savefig('figs/scatter_mat_all_feat.png', format='png', dpi=300)
            plt.show()
    def thearding_train_model(self):
        threading.Thread(target=self.train_model_it).start()
    def train_model_it(self):

        #  Tách dataset ra tập train và test 
        train_set, test_set = train_test_split(df, test_size=0.3, random_state=42) 
        root_fit_train_set = Toplevel(root)
        root_fit_train_set.title("data info")
        scrollbar = Scrollbar(root_fit_train_set)
        scrollbar.pack( side = RIGHT, fill = Y )
        scrollbarX = Scrollbar(root_fit_train_set,orient="horizontal")
        scrollbarX.pack( side = BOTTOM, fill = X )
        mylist = Text(root_fit_train_set, yscrollcommand = scrollbar.set,
        xscrollcommand= scrollbarX.set)
        buffer = io.StringIO()
        train_set.info(buf=buffer)
        buffer_value = buffer.getvalue()
        mylist.insert(END,"train_set info\n"+
            str(buffer_value)+"\n")
        buffer = io.StringIO()
        test_set.info(buf=buffer)
        buffer_value = buffer.getvalue()
        mylist.insert(END,"test_set info\n"+
            str(buffer_value)+"\n")
        mylist.pack( side = LEFT, fill = BOTH )
        scrollbar.config( command = mylist.yview )
        scrollbarX.config(command = mylist.xview)

        
        train_set_labels = train_set["revenue"].copy()
        train_set = train_set.drop(columns = "revenue") 
        test_set_labels = test_set["revenue"].copy()
        test_set = test_set.drop(columns = "revenue") 

        '''Data processing'''
        

        num_feat_names = ['budget','popularity', 'runtime', 'vote_average', 'vote_count']
        cat_feat_names = ["year", "status", "original_language"]
        onehot_feat_names = columns

        

        best_model, processed_test_set = train_and_dump_model(num_feat_names, cat_feat_names, onehot_feat_names, test_set,train_set,train_set_labels)

        r2score, rmse = r2score_and_rmse(best_model, processed_test_set, test_set_labels)
        
        root_fit_train_set2 = Toplevel(root)
        root_fit_train_set2.title("data info")
        scrollbar = Scrollbar(root_fit_train_set2)
        scrollbar.pack( side = RIGHT, fill = Y )
        scrollbarX = Scrollbar(root_fit_train_set2,orient="horizontal")
        scrollbarX.pack( side = BOTTOM, fill = X )
        mylist = Text(root_fit_train_set2, yscrollcommand = scrollbar.set,
        xscrollcommand= scrollbarX.set)
        mylist.insert(END, "Performance on test data:\n"
        +"R2 score (on test data, best=1): ")
        #mylist.insert(END,(r2score)+"\n")
        mylist.insert(END,
        "Root Mean Square Error: "+str(rmse.round(decimals=1))+
        "\nPredictions: "+str(best_model.predict(processed_test_set[0:9]).round(decimals=1))+
        "\nLabels:      "+
        "")
        for inside_list_thing in list(test_set_labels[0:9]):
            mylist.insert(END,str(inside_list_thing)+"\n")
        mylist.pack( side = LEFT, fill = BOTH )
        scrollbar.config( command = mylist.yview )
        scrollbarX.config(command = mylist.xview)

root = tk.Tk()
Main(root)
root.geometry("480x480+100+100")
root.mainloop()
# %%
