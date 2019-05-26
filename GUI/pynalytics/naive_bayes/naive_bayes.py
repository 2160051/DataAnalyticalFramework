"""
This module is a framework for generating visualization and numerical results using naive Bayes classification.
"""
import math
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .preprocessor import Preprocessing
from matplotlib import style
from matplotlib import cm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PowerTransformer
pd.options.mode.chained_assignment = None
style.use('seaborn-bright')

class NaiveBayes():
    """
    This represents the class for generating data visualizations and analysis using naive Bayes classification.
    """
    
    version = "1.0"

    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        """
        self.X = None
        self.y = None
        self.model = None
    
    def naive_bayes(self,X, y, cv_kfold=10, bin_strat='uniform', feature_selection=True):

        """
        Perform naive Bayes (Gaussian) classification

        Parameters
        ----------
        X_columns : numpy array
            Column names of the predictor features 
        y_column : str
            Column name of the target feature
        cv_kfold : int
            The number of folds/splits for cross validation
        class_bins : int (default=0)
            The number of bins to class the target feature 
        bin_strat : {'uniform', 'quantile', 'kmeans'}, (default='uniform')
            Strategy of defining the widths of the bins

            uniform:
                All bins have identical widths
            quantile:
                All bins have the same number of points
            kmeans:
                Values of each bins have the same k-means cluster centroid
            feature_selection : binary (default=True)
                Determines if nb_feature_select is to be applied
        
        Returns
        -------
        numpy array
            True values of the target feature
        numpy array
            Predicted values of the target feature
        float
            Accuracy of the model (based on `balanced_accuracy_score<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html>`_).
        """
        
        try:

            valid_strategy = ('uniform', 'quantile', 'kmeans')
            if bin_strat not in valid_strategy:
                raise ValueError("Valid options for 'bin_strat' are {}. "
                             "Got strategy={!r} instead."
                             .format(valid_strategy, bin_strat))
            valid_feature_selection = {1,0}

            if feature_selection not in valid_feature_selection:
                raise ValueError("Valid options for 'bin_strat' are {}. "
                             "Got strategy={!r} instead."
                             .format(valid_feature_selection, feature_selection))

            X = self.df_input[X_columns]
            y = self.df_input[[y_column]]

            scaler = MinMaxScaler()
            for col in X.columns:
                X[col] = scaler.fit_transform(X[[col]].astype(float))

            if(class_bins!=0):
                est = KBinsDiscretizer(n_bins=class_bins, encode='ordinal', strategy='kmeans')
                if(bin_strat=='percentile'):
                    est = KBinsDiscretizer(n_bins=class_bins, encode='ordinal', strategy='percentile')
                elif(bin_strat=='uniform'):
                    est = KBinsDiscretizer(n_bins=class_bins, encode='ordinal', strategy='uniform')
                y[[y.columns[0]]] = est.fit_transform(y[[y.columns[0]]])

            if(feature_selection):
                X = X[self.nb_feature_select(LogisticRegression(solver='lbfgs', multi_class='auto'), X, y, cv_kfold=10)]
                print("Features Selected: ")
                for x in X.columns:
                    print(x, end = ", ")
                
            X = X.values.tolist()

            nb = GaussianNB()
            y_true_values,y_pred_values = [], []

            y = y[y.columns[0]].values.tolist()
            
            if(cv_kfold!=0):
                kf = KFold(n_splits=cv_kfold)

                
                kf.get_n_splits(X)
                accuracy = []

                for train_index, test_index in kf.split(X,y):

                    X_test = [X[ii] for ii in test_index]
                    X_train = [X[ii] for ii in train_index]
                    y_test = [y[ii] for ii in test_index]
                    y_train = [y[ii] for ii in train_index]

                    nb.fit(X_train,y_train)
                    y_pred =nb.predict(X_test)
                    accuracy = np.append(accuracy, np.around(balanced_accuracy_score(y_test, y_pred),decimals=4))
                    y_pred_values = np.append(y_pred_values, y_pred)
                    y_true_values = np.append(y_true_values, y_test)
                total_accuracy = np.around(np.sum(accuracy)/cv_kfold, decimals=4)
            else:
                nb.fit(X,y)
                y_pred =nb.predict(X)
                y_true_values =  y
                y_pred_values = y_pred
                total_accuracy = np.around(balanced_accuracy_score(y_true_values, y_pred_values),decimals=4)

            return y_true_values, y_pred_values, total_accuracy
            


        except Exception as e:
            print(e)

    def naive_bayes_cm(self,X_columns, y_column,cv_kfold=10, class_bins=0, bin_strat='uniform', feature_selection=True):

        """
        Perform naive Bayes (Gaussian) classification and visualize the results

        Parameters
        ----------
        X_columns : numpy array
            Column names of the predictor features 
        y_column : str
            Column name of the target feature
        cv_kfold : int
            The number of folds/splits for cross validation
        class_bins : int (default=0)
            The number of bins to class the target feature 
        bin_strat : {'uniform', 'quantile', 'kmeans'}, (default='uniform')
            Strategy of defining the widths of the bins

            uniform:
                All bins have identical widths
            quantile:
                All bins have the same number of points
            kmeans:
                Values of each bins have the same k-means cluster centroid
            feature_selection : binary (default=True)
                Determines if nb_feature_select is to be applied
        
        Returns
        -------
        figure
            Visualization (confusion matrix) of the naive Bayes classifier 

        """

        try:

            valid_strategy = ('uniform', 'quantile', 'kmeans')
            if bin_strat not in valid_strategy:
                raise ValueError("Valid options for 'bin_strat' are {}. "
                             "Got bin_strat={!r} instead."
                             .format(valid_strategy, bin_strat))
            valid_feature_selection = {True,False}
            
            if feature_selection not in valid_feature_selection:
                raise ValueError("Valid options for 'bin_strat' are {}. "
                             "Got feature_selection={!r} instead."
                             .format(valid_feature_selection, feature_selection))            

            y_true, y_pred, accuracy = self.naive_bayes(X_columns, y_column, cv_kfold=cv_kfold, class_bins=class_bins, bin_strat=bin_strat, feature_selection=feature_selection)
            cm = confusion_matrix(y_true, y_pred)
            cm_norm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100
            
            ticks = []

            if(class_bins!=0):
                y = self.df_input[[y_column]]
                est = KBinsDiscretizer(n_bins=class_bins, encode='ordinal', strategy='kmeans')
                if(bin_strat=='percentile'):
                    est = KBinsDiscretizer(n_bins=class_bins, encode='ordinal', strategy='percentile')
                elif(bin_strat=='uniform'):
                    est = KBinsDiscretizer(n_bins=class_bins, encode='ordinal', strategy='uniform')
                new_y = est.fit_transform(y[[y.columns[0]]])
                new_df = pd.DataFrame(new_y)
                edges = est.bin_edges_[0]
                new_df = pd.concat([new_df,y],axis=1)
                first = True
                for bins in new_df[0].unique():
                    if (first):
                        ticks.append(str(int(round(edges[int(bins)])))+" - "+str(int(round(edges[int(bins+1)]))))
                        first = False
                    else:
                        ticks.append(str(int(round(edges[int(bins)]))+1)+" - "+str(int(round(edges[int(bins+1)]))))


            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),ylabel='True Label',xlabel='Predicted Label')

            thresh = cm.max() / 2
            for x in range(cm_norm.shape[0]):
                for y in range(cm_norm.shape[1]):
                    if(x==y):
                        ax.text(y,x,f"{cm[x,y]}({cm_norm[x,y]:.2f}%)",ha="center", va="center", fontsize=12, color="white" if cm[x, y] > thresh else "black")
                    else:
                        ax.text(y,x,f"{cm[x,y]}({cm_norm[x,y]:.2f}%)",ha="center", va="center", color="white" if cm[x, y] > thresh else "black")
            ax.annotate("Accuracy: "+ str(accuracy),xy=(0.25, 0.9), xycoords='figure fraction')
            if(class_bins!=0):
                plt.xticks(np.arange(cm.shape[1]),ticks)
                plt.yticks(np.arange(cm.shape[0]),ticks)
            plt.title("Naive Bayes Confusion Matrix ("+y_column+")", y=1.05)
            plt.subplots_adjust(left=0)
            plt.show()
        except Exception as e:
                print(e)

    def nb_feature_select(self,estimator, X, y,cv_kfold=5):

        try:
            selector = RFECV(estimator, step=1,cv=cv_kfold, min_features_to_select=round((len(X.columns)/2)))
            selector = selector.fit(X,y)
            support = selector.support_
            selected = []
            for a, s in zip(X.columns, support):
                if(s):
                    selected.append(a)
            return selected
        except Exception as e:
            print(e)
