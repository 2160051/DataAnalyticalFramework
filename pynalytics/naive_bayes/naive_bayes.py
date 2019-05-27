"""
This module is a framework for generating visualization and numerical results using naive Bayes classification.
"""
import math
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import cm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
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
    
    def naive_bayes(self,X, y, cv_kfold=10, feature_selection=True):
        
        try:

            nb = GaussianNB()
            y_true_values,y_pred_values = [], []
            
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
                    accuracy = np.append(accuracy, balanced_accuracy_score(y_test, y_pred))
                    y_pred_values = np.append(y_pred_values, y_pred)
                    y_true_values = np.append(y_true_values, y_test)
                total_accuracy = np.around(np.sum(accuracy)/cv_kfold, decimals=4)
            else:
                nb.fit(X,y)
                y_pred =nb.predict(X)
                y_true_values =  y
                y_pred_values = y_pred
                total_accuracy = balanced_accuracy_score(y_true_values, y_pred_values)

            return y_true_values, y_pred_values, total_accuracy
            


        except Exception as e:
            print(e)

    def naive_bayes_cm(self,X_columns, y_column,cv_kfold=10, class_bins=0, bin_strat='uniform', feature_selection=True):


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
