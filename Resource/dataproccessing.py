#%% 
# Import các thư viện
import library as lb
# Import các hàm
import function as func

# %% Load dữ liệu
raw_data = lb.pd.read_csv('Model/movies_metadata.csv')

  
# %%  Convert JSON to array data feature 

raw_data['genres'] = raw_data['genres'].fillna('[]').apply(lb.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
raw_data['belongs_to_collection'] = raw_data['belongs_to_collection'].fillna('[]').apply(lb.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
raw_data['production_companies'] = raw_data['production_companies'].fillna('[]').apply(lb.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
raw_data['spoken_languages'] = raw_data['spoken_languages'].fillna('[]').apply(lb.literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# %% Introview dữ liệu
print('\n____________ Dataset info ____________')
print(raw_data.info())  

print('\n____________ Counts on revenue feature ____________')
print(raw_data['revenue'].value_counts()) 

print('\n____________ Statistics of numeric features ____________')
print(raw_data.describe())  


# %%  Draw Diagram

if 0:
    x, y = func.tanso_featurenonnumeric(raw_data['original_language'])
    col_map = lb.plt.get_cmap('Paired')
    lb.plt.bar(x, y, width=2, color=col_map.colors, edgecolor='k', 
        linewidth=2)
    lb.plt.figure(figsize=(10000, 10000)) 
    lb.plt.show()
    #plt.savefig('figures/language.png', format='png', dpi=300)

if 0:
    x, y = func.tanso_featurenonnumeric(raw_data['extenstion'])
    col_map = lb.plt.get_cmap('Paired')
    lb.plt.bar(x, y, width=2, color=col_map.colors, edgecolor='k', 
        linewidth=2)
    lb.plt.figure(figsize=(10000, 10000)) 
    lb.plt.show()

if 0:
    #raw_data.hist(bins=10, figsize=(10,5)) #bins: no. of intervals
    raw_data.hist(figsize=(10,5)) #bins: no. of intervals
    lb.plt.rcParams['xtick.labelsize'] = 10
    lb.plt.rcParams['ytick.labelsize'] = 10
    lb.plt.tight_layout()
    lb.plt.show()

if 0:
    lb.features_to_plot = ["revenue", "runtime", "vote_average","vote_count"]
    lb.scatter_matrix(raw_data[lb.features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    lb.plt.savefig('PART 1-2-3/figures/scatter_mat_all_feat.png', format='png', dpi=300)
    lb.plt.show()


# %%
