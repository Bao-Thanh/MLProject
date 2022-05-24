'''
This source code is uploaded on Github at the following link: https://github.com/Bao-Thanh/MLProject 

'''
#%% 
# Import các thư viện
from cmath import nan
from glob import glob
from tkinter.filedialog import Open
from matplotlib.figure import Figure
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
#import seaborn as sns
from matplotlib.pyplot import xlim
from sklearn import naive_bayes
from pandas.plotting import scatter_matrix  
import os
from ast import literal_eval
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
#from xgboost import XGBRegressor 
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

# UI
import joblib
from tkinter import *
import tkinter as tk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import math

#%%
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values

#print("\nPredictions: ", stacking_best_model.predict(processed_test_set[0:9]).round(decimals=1))
#print("Labels:      ", list(test_set_labels[0:9]),'\n')
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
        fileMenu.add_command(label="Predict from file", command=self.predict_it)
        fileMenu.add_separator()
        fileMenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=fileMenu)
        global e1
        e1 = tk.Entry(root)
        tk.Label(root, text="Budget").grid(row=0)
        e1.grid(row=0, column=1)
        global e3
        e3 = tk.Entry(root)
        tk.Label(root, text="popularity").grid(row=0, column=2)
        e3.grid(row=0, column=3)
        global e4
        e4 = tk.Entry(root)
        tk.Label(root, text="runtime").grid(row=0, column=4)
        e4.grid(row=0, column=5)
        global e6
        e6 = tk.Entry(root)
        tk.Label(root, text="vote_average").grid(row=1, column=0)
        e6.grid(row=1, column=1)
        global e7
        e7 = tk.Entry(root)
        tk.Label(root, text="vote_count").grid(row=1, column=2)
        e7.grid(row=1, column=3)
        global e8
        e8 = tk.Entry(root)
        tk.Label(root, text="year").grid(row=1, column=4)
        e8.grid(row=1, column=5)
        global i1
        i1=IntVar()
        c1 = Checkbutton(root, text = "Action", variable=i1)
        c1.grid(row=2, column = 1)
        global i2
        i2=IntVar()
        c2 = Checkbutton(root, text = "Adventure", variable=i2)
        c2.grid(row=2, column = 2)
        global i3
        i3=IntVar()
        c3 = Checkbutton(root, text = "Fantasy", variable=i3)
        c3.grid(row=2, column = 3)
        global i4
        i4=IntVar()
        c4 = Checkbutton(root, text = "Animation", variable=i4)
        c4.grid(row=2, column = 4)
        global i5
        i5=IntVar()
        c5 = Checkbutton(root, text = "Science Fiction", variable=i5)
        c5.grid(row=2, column = 5)
        global i6
        i6=IntVar()
        c6 = Checkbutton(root, text = "Drama", variable=i6)
        c6.grid(row=3, column = 1)
        global i7
        i7=IntVar()
        c7 = Checkbutton(root, text = "Thriller", variable=i7)
        c7.grid(row=3, column = 2)
        global i8
        i8=IntVar()
        c8 = Checkbutton(root, text = "Family", variable=i8)
        c8.grid(row=3, column = 3)
        global i9
        i9=IntVar()
        c9 = Checkbutton(root, text = "Comedy", variable=i9)
        c9.grid(row=3, column = 4)
        global i10
        i10=IntVar()
        c10 = Checkbutton(root, text = "History", variable=i10)
        c10.grid(row=3, column = 5)
        global i11
        i11=IntVar()
        c11 = Checkbutton(root, text = "War", variable=i11)
        c11.grid(row=4, column = 1)
        global i12
        i12=IntVar()
        c12 = Checkbutton(root, text = "Western", variable=i12)
        c12.grid(row=4, column = 2)
        global i13
        i13=IntVar()
        c13 = Checkbutton(root, text = "Romance", variable=i13)
        c13.grid(row=4, column = 3)
        global i14
        i14=IntVar()
        c14 = Checkbutton(root, text = "Crime", variable=i14)
        c14.grid(row=4, column = 3)
        global i15
        i15=IntVar()
        c15 = Checkbutton(root, text = "Mystery", variable=i15)
        c15.grid(row=4, column = 4)
        global i16
        i16=IntVar()
        c16 = Checkbutton(root, text = "Horror", variable=i16)
        c16.grid(row=4, column = 5)
        global i17
        i17=IntVar()
        c17 = Checkbutton(root, text = "Documentary", variable=i17)
        c17.grid(row=5, column = 1)
        global i18
        i18=IntVar()
        c18 = Checkbutton(root, text = "Music", variable=i18)
        c18.grid(row=5, column = 2)
        global i19
        i19=IntVar()
        c19 = Checkbutton(root, text = "TV Movie", variable=i19)
        c19.grid(row=5, column = 3)
        global i20
        i20=IntVar()
        c20 = Checkbutton(root, text = "None", variable=i20)
        c20.grid(row=5, column = 4)
        global i21
        i21=IntVar()
        c21 = Checkbutton(root, text = "Foreign", variable=i21)
        c21.grid(row=5, column = 5)
        tk.Button(root,text='Add',command=self.predict_input).grid(row=20, 
                                   column=0)
        self.txt = Text(self)
    def predict_input(self):
        data = {'budget' : [str(e1.get())], 
        'popularity':[str(e3.get())],
        'runtime':[str(e4.get())],
        'vote_average':[str(e6.get())],
        'vote_count':[str(e7.get())],
        'Action':[str(i1.get())],
        'Adventure':[str(i2.get())],
        'Fantasy':[str(i3.get())],
        'Animation':[str(i4.get())],
        'Science Fiction':[(i5.get())],
        'Drama':[str(i6.get())],
        'Thirller':[(i7.get())],
        'Family':[(i8.get())],
        'Comedy':[(i9.get())],
        'History':[(i10.get())],
        'War':[(i11.get())],
        'Western':[(i12.get())],
        'Romance':[(i13.get())],
        'Crime':[(i14.get())],
        'Mystery':[(i15.get())],
        'Horror':[(i16.get())],
        'Documentary':[(i17.get())],
        'Music':[(i18.get())],
        'TV Movie':[(i19.get())],
        'None':[(i20.get())],
        'Foreign':[(i21.get())],
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
        mylist.insert(END, "input value:\n")
        for i in data :
            mylist.insert(END, str(i)+": "+str(data[i])+"\n")
        full_pipeline = joblib.load(r'models/full_pipeline.pkl')
        search = joblib.load('saved_objects/StackingRegressor_model.pkl')
        best_model = search.best_estimator_
        processed_test_set = full_pipeline.transform(data_panda)
        mylist.insert(END,
        "\nPredictions: "+str(best_model.predict(processed_test_set).round(decimals=1))+
        "")
        mylist.pack( side = LEFT, fill = BOTH )
        scrollbar.config( command = mylist.yview )
        scrollbarX.config(command = mylist.xview)
    def predict_it(self):
        global fVtypes
        fVtypes = [('Video', '*.csv')]
        dlg = Open(self, filetypes = fVtypes)
        fl = dlg.show()
        if fl != '':
            df = pd.read_csv(fl)
            full_pipeline = joblib.load(r'models/full_pipeline.pkl')
            search = joblib.load('saved_objects/StackingRegressor_model.pkl')
            best_model = search.best_estimator_
            processed_test_set = full_pipeline.transform(df.astype(str))
            predictions = best_model.predict(processed_test_set)
            x = np.array(range(0, len(predictions)))
            y = np.array(predictions)
            root_fit_train_set2 = Toplevel(root)
            root_fit_train_set2.title("data info")
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(122)
            ax.plot(x, y, color = "red", marker = "o",linestyle='')
            ax.set_xlabel("row")
            ax.set_title("predicted from file")
            ax.set_ylabel("predicted value")
            new_list = range(math.floor(min(x)), math.ceil(max(x))+1)            
            ax.set_xticks(new_list)
            canvas = FigureCanvasTkAgg(fig,
                               master = root_fit_train_set2)  
            canvas.draw()
            canvas.get_tk_widget().pack()
            toolbar = NavigationToolbar2Tk(canvas,
                                        root_fit_train_set2)
            toolbar.update()
            canvas.get_tk_widget().pack()
root = tk.Tk()
Main(root)
root.mainloop()

# %%
