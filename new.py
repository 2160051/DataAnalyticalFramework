import pandas as pd
import numpy as np
import operator
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PowerTransformer
from framework import DataAnalyticalFramework
test_frame = DataAnalyticalFramework('Data Analytics.csv')
disease_arr = test_frame.get_column(3, 54)
attribute_arr = test_frame.get_column(54, 80)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=Warning)


# x = test_frame.df_input[['Swamps','Streams']]
# y = test_frame.df_input[['Person-to-Person Contact']]

# poly = PolynomialFeatures(degree = 2)
# x_poly = poly.fit_transform(x)   

# model = LinearRegression()
# model.fit(x_poly, y)
# intercept_arr = model.intercept_

# print(intercept_arr)

# x,y,z = test_frame.naive_bayes(attribute_arr, 'Person-to-Person Contact', cv_kfold=3,bin_strat='uniform' ,class_bins=3,feature_selection=0)
# print(type(z))
# nb_feature_select(self,estimator, X, y,cv=5)
# true_val = []

print('Person-to-Person Contact')
test_frame.naive_bayes_cm(attribute_arr, 'Person-to-Person Contact', cv_kfold=5, class_bins=3,bin_strat='uniform',feature_selection=0)
print('Airborne Transmission')
test_frame.naive_bayes_cm(attribute_arr, 'Airborne Transmission', cv_kfold=10, class_bins=3,bin_strat='kmeans',feature_selection=0)
print('Droplet Spread')
test_frame.naive_bayes_cm(attribute_arr, 'Droplet Spread', cv_kfold=5, class_bins=3,bin_strat='uniform',feature_selection=1)
print('Vector-Borne Transmission')
test_frame.naive_bayes_cm(attribute_arr, 'Vector-Borne Transmission', cv_kfold=10, class_bins=3,bin_strat='kmeans',feature_selection=0)
print('Vehicle-Borne Transmission')
test_frame.naive_bayes_cm(attribute_arr, 'Vehicle-Borne Transmission', cv_kfold=5, class_bins=3,bin_strat='uniform',feature_selection=0)

# print('Person-to-Person Contact')
# x,y,a=test_frame.naive_bayes(attribute_arr, 'Person-to-Person Contact', cv_kfold=10, class_bins=3,bin_strat='kmeans',feature_selection=1)
# print(a)
# x,y,a=test_frame.naive_bayes(attribute_arr, 'Person-to-Person Contact', cv_kfold=10, class_bins=3,bin_strat='uniform',feature_selection=1)
# print(a)
# print('Airborne Transmission')
# x,y,a=test_frame.naive_bayes(attribute_arr, 'Airborne Transmission', cv_kfold=10, class_bins=3,bin_strat='kmeans',feature_selection=1)
# print(a)
# x,y,a=test_frame.naive_bayes(attribute_arr, 'Airborne Transmission', cv_kfold=10, class_bins=3,bin_strat='uniform',feature_selection=1)
# print(a)
print('Droplet Spread')
# x,y,a=test_frame.naive_bayes(attribute_arr, 'Droplet Spread', cv_kfold=10, class_bins=3,bin_strat='kmeans',feature_selection=1)
# print(a)
# x,y,a=test_frame.naive_bayes(attribute_arr, 'Droplet Spread', cv_kfold=10, class_bins=3,bin_strat='uniform',feature_selection=1)
# print(a)
# print('Vector-Borne Transmission')
# x,y,a=test_frame.naive_bayes(attribute_arr, 'Vector-Borne Transmission', cv_kfold=5, class_bins=3,bin_strat='kmeans',feature_selection=1)
# print(a)
# x,y,a=test_frame.naive_bayes(attribute_arr, 'Vector-Borne Transmission', cv_kfold=5, class_bins=3,bin_strat='uniform',feature_selection=1)
# print(a)
# print('Vehicle-Borne Transmission')
# x,y,a=test_frame.naive_bayes(attribute_arr, 'Vehicle-Borne Transmission', cv_kfold=5, class_bins=3,bin_strat='kmeans',feature_selection=1)
# print(a)
# x,y,a=test_frame.naive_bayes(attribute_arr, 'Vehicle-Borne Transmission', cv_kfold=5, class_bins=3,bin_strat='uniform',feature_selection=1)
# print(a)