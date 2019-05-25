"""
This program is a module for generating visualization and numerical results using k-means clustering.
"""
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .preprocessor import Preprocessing
from matplotlib import style
from matplotlib import cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PowerTransformer
pd.options.mode.chained_assignment = None
style.use('seaborn-bright')

class Kmeans(Preprocessing):
    """
    This represents the class for generating data visualizations and analysis using K-means clustering.
    """
    
    version = "1.0"

    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        """
        pass

    def km_feature_select(self,X_columns, y_column, k, n_features = 2):         

        """
        Select the best n number of features using silhouette analysis and forward stepwise regression

        Parameters
        ----------
        X_columns : numpy array
            Column names of the predictor features 
        y_column : str
            Column name of the target feature
        k : int
            number of clusters 
        n_features : int (default=2)
            number of features to be selected
        Returns
        -------
        numpy array
            The selected features

        """

        try:
            X = self.df_input[X_columns]
            y = self.df_input[[y_column]]
            features = pd.concat([X,y],axis=1)
            features_selected = [y_column]
            while(n_features>1):
                temp_selected = ""
                temp_coef = 0
                for col in X_columns:
                    temp_feat_sel = np.append(features_selected,col)
                    kmeans_model = KMeans(n_clusters=k,random_state=0).fit(features[temp_feat_sel])
                    labels = kmeans_model.labels_
                    sil_coef = silhouette_score(features[temp_feat_sel], labels, metric='euclidean')
                    if((col not in features_selected) and (sil_coef>temp_coef)):
                        temp_coef = sil_coef
                        temp_selected = col
                features_selected = np.append(features_selected,temp_selected)
                n_features -= 1
            return features_selected

        except Exception as e:
                    print(e)

    def k_means(self,X_columns, y_column, k, n_features=2):
        
        """
        Perform k-means clustering

        Parameters
        ----------
        X_columns : numpy array
            Column names of the predictor features 
        y_column : str
            Column name of the target feature
        k : int
            number of clusters 
        n_features : int (default=2)
            number of features to be selected
        
        Returns
        -------
        centroids,sil_coef,labeled_features
        numpy array
            Centroids of the clusters generated
        float
            Predicted values of the target feature
        pandas Dataframe
            The dataset with the cluster labels 
        """     

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

    def k_means_cc(self, X_columns, y_column, k, n_features=0):
        """
        Perform k-means clustering and visualize the results

        Parameters
        ----------
        X_columns : numpy array
            Column names of the predictor features 
        y_column : str
            Column name of the target feature
        k : int
            number of clusters 
        n_features : int (default=2)
            number of features to be selected
        
        Returns
        -------
        figure
            Scatter plots and the silhouette coefficient
        figure
            Centroid Chart of the cluster centroids
        """
        try:
            centroids, sil_coef,labeled_features = self.k_means(X_columns, y_column, k, n_features=n_features)
            

            fig = plt.figure(1)
            fig.subplots_adjust(hspace=0.5)
            for x in range(0,centroids.shape[1]):
                ax = fig.add_subplot(round(centroids.shape[1]/2), 2, x+1)
                cmap = 'Set1'
                for c in range(centroids.shape[0]):
                    temp_df = labeled_features[labeled_features['clusters'] == c]
                    ax.scatter(temp_df[labeled_features.columns[x]], temp_df[labeled_features.columns[0]], label=c, cmap=cmap[c])
                ax.set_xlabel(labeled_features.columns[x])
                ax.axis('tight')
                if(x == 0):
                    plt.legend(title='Clusters', loc='upper right',bbox_to_anchor=(-0.1, 1.1))
            fig.text(0.04, 0.5, labeled_features.columns[0], va='center', rotation='vertical')
            plt.annotate("Silhouette Coefficient: "+ str(sil_coef),xy=(10, 10), xycoords='figure pixels')
            plt.suptitle("Scatter Plot of "+labeled_features.columns[0]+" against other Features")
                       
            fig2 = plt.figure(2)
            ax2 = fig2.subplots()
            for k in range(centroids.shape[0]):
                plt.plot(range(centroids.shape[1]),centroids[k], label=str(k)+": "+str(centroids[k]))
            plt.xticks(range(centroids.shape[1]),labeled_features.columns[:-1])
            plt.xlabel("Features")
            plt.ylabel("Location")
            plt.setp( ax2.xaxis.get_majorticklabels(), rotation=-15, ha="left" )
            plt.legend(loc="upper right",title='Clusters (Centroids)')
            plt.title("Centroid Chart ("+y_column+")")
            plt.tight_layout()

            plt.show()

        except Exception as e:
                print(e)
