import math
import operator
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score

class k_means:
    
    model = 0
    centroids = 0
    labels = 0
    silhouette_score = 0

    def __init__(self, X, k):
        self.X = X
        self.k = k

    # model
    def k_means_model(self):
        self.model = 0 # KMeans(n_clusters=self.k,random_state=0).fit(self.X)
        self.labels = 1# self.model.labels_
        self.centroids = 2# self.model.cluster_centers_
        self.sil_coef = 3# silhouette_score(self.X, labels, metric='euclidean')

    # representation
    def __repr__(self):
        return "kmeans({})".format(self.k)


km1 = k_means(0,2)
km1.kmeans_model()
print(km1.labels)

def k_means(self,X_columns, y_column, k, n_features=2):  
    try:
        X = self.df_input[X_columns]
        y = self.df_input[[y_column]]
        features = pd.concat([X,y],axis=1)
        scaler = MinMaxScaler()
        for col in features.columns:
            features[col] = scaler.fit_transform(features[[col]].astype(float))
        if(n_features!=0):
            sel_feat = self.km_feature_select(X_columns, y_column, 5, n_features =n_features)
            features = features[sel_feat]
        kmeans_model = KMeans(n_clusters=k,random_state=0).fit(features)
        labels = kmeans_model.labels_
        sil_coef = np.around(silhouette_score(features, labels, metric='euclidean'), decimals=4)
        centroids = np.around(kmeans_model.cluster_centers_, decimals=4)
        labels_df = pd.DataFrame(data=labels)
        labels_df.columns = ['clusters']
        labeled_features= pd.concat([features,labels_df], axis=1)
        return centroids,sil_coef,labeled_features
    except Exception as e:
        print(e)