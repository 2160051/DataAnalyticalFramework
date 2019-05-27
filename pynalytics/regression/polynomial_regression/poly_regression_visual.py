"""
This program is a module for generating visualizations using Polynomial Regression.
"""
import math
import operator
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from .poly_regression_num import PolyRegressionRes
from matplotlib import style
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
pd.options.mode.chained_assignment = None
style.use('seaborn-bright')

class PolyRegressionVis(PolyRegressionRes):
    """
    This represents the class for generating data visualizations using Polynomial Regression.
    """
    
    version = "1.0"

    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        """
        pass

    def polynomial_reg(self, dependent, independent):
        """

        Generates the visualization for polynomial regression

        Parameters
        ----------
        dependent : 2D pandas DataFrame
            the dependent(y) variable specified
        independent : 2D pandas DataFrame
            the independent(x) variable specified

        Returns
        -------
        figure
            visualization of the polynomial regression
        """

        try:
            x_column = independent.columns.values
            y_column = dependent.columns.values

            x = independent
            y = dependent

            poly = PolynomialFeatures(degree = 2)
            X_fit = poly.fit_transform(x) 
            lin = LinearRegression()
            lin.fit(X_fit, y) 
            y_poly_pred = lin.predict(poly.fit_transform(x))

            plt.scatter(x, y, color = 'red')
            plt.plot(x, y_poly_pred, color='blue', label=self.poly_eq(dependent, independent))
            plt.legend(fontsize=9, loc="upper right")
            plt.title("Polynomial Regression of " + x_column[0] + " and " + y_column[0])
            plt.show()
                
        except Exception as e:
            print(e)


