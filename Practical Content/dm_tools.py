import numpy as np
import pandas as pd
import pydot
from io import StringIO
from sklearn.tree import export_graphviz

def data_prep():
    df = pd.read_csv('datasets/veteran.csv')
    #Interval to nominal demcluster
    df['DemCluster'] = df['DemCluster'].astype(str)
    #Binary 0/1 dem home owner
    dem_home_owner_map = {'U':0, 'H': 1}
    df['DemHomeOwner'] = df['DemHomeOwner'].map(dem_home_owner_map)
    
    #Erronous vals in demmedincome
    mask = df['DemMedIncome'] < 1
    df.loc[mask, 'DemMedIncome'] = np.nan

    
    #impute missing vals in demage with mean
    df['DemAge'].fillna(df['DemAge'].mean(), inplace=True)
    #impute med income using mean
    df['DemMedIncome'].fillna(df['DemMedIncome'].mean(), inplace=True)
    #impute gift avg using mean
    df['GiftAvgCard36'].fillna(df['GiftAvgCard36'].mean(), inplace=True)
    
    df.drop(['ID', 'TargetD'], axis=1, inplace=True)
    df = pd.get_dummies(df)
    return df

def analyse_feature_importance(dm_model, feature_names, n_to_display=20):
    # grab feature importances from the model
    importances = dm_model.feature_importances_

    # sort them out in descending order
    indices = np.argsort(importances)
    indices = np.flip(indices, axis=0)
    # limit to 20 features, you can leave this out to print out everything
    indices = indices[:n_to_display]
    for i in indices:
        print(feature_names[i], ':', importances[i])
    
def visualize_decision_tree(dm_model, feature_names, save_name):
    dotfile = StringIO()
    export_graphviz(dm_model, out_file=dotfile, feature_names=feature_names)
    graph = pydot.graph_from_dot_data(dotfile.getvalue())
    graph[0].write_png(save_name) # saved in the following file
