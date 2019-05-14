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

class naive_bayes:
    
    model = 0

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_test = 0
        self.y_test = 0

    @classmethod
    def train_test_data(self, X_test, y_test, X_train, y_train):
        self.X = X_test
        self
        

    # model
    def k_means_model(self):
        self.model = 0 # KMeans(n_clusters=self.k,random_state=0).fit(self.X)
        self.labels = 1# self.model.labels_
        self.centroids = 2# self.model.cluster_centers_
        self.sil_coef = 3# silhouette_score(self.X, labels, metric='euclidean')

    # representation
    def __repr__(self):
        return "kmeans({})".format(self.k)




def nb_feature_select(self,estimator, X, y,cv_kfold=5):
    """
    Select the best features and cross-validated selection of best number of features using recursive feature elimination
    Parameters
    ----------
    estimator : object
        A supervised learning estimator that can provide feature importance either through a ``coef_``
        attribute or through a ``feature_importances_`` attribute.
    X : pandas DataFrame
        The features to be selected
    y : pandas DataFrame
        The target feature as a basis of feature importance
    cv_kfold : int (default=5)
        The number of folds/splits for cross validation
    # Returns
    -------
    numpy array
        The selected features
    """
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
def naive_bayes(self,X_columns, y_column, cv_kfold=10, class_bins=0, bin_strat='uniform', feature_selection=True):
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
                print(x, end=", ")
            
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