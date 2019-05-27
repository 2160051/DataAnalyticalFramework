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
from .poly_regression__num import PolyRegressionRes
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

    def polynomial_reg(self, independent, dependent):
        """

        Generates the visualization for polynomial regression

        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified

        Returns
        -------
        figure
            visualization of the polynomial regression
        """

        try:
            if isinstance(independent, str) and isinstance(dependent, str):
                x = self.df_input[independent]
                y = self.df_input[[dependent]]
            elif isinstance(independent, pd.DataFrame) and isinstance(dependent, pd.DataFrame):
                x = independent
                y = dependent

            x = x[:, np.newaxis]
            y = y[: np.newaxis]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)  

            model = LinearRegression()
            model.fit(x_poly, y)
            y_poly_pred = model.predict(x_poly)

            plt.scatter(x, y, color = 'red')
            sort_axis = operator.itemgetter(0)
            sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
            x, y_poly_pred = zip(*sorted_zip)
            plt.plot(x, y_poly_pred, color='blue', label=self.poly_eq(independent, dependent))
            plt.legend(fontsize=9, loc="upper right")

            plt.title("Polynomial Regression of " + independent + " and " + dependent)
            plt.show()
                
        except Exception as e:
            print(e)

    def polynomial_reg_summary(self, independent, dependent):
        """

        Generates the calculated value of the coefficient of determination(R²) of the polynomial regression 

        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified
        second_indep : str, optional
            the second independent(x) variable specified used for the multiple linear regression

        Parameters
        ----------
        str
            calculated value of the coefficient of determination(R²) of the polynomial regression 
        """

        try:
            poly_rsquared = self.get_poly_rsquared(independent, dependent)
            poly_pearsonr = self.get_poly_pearsonr(independent, dependent)
            print("Pearson correlation coefficient(R) of the polynomial regression of " + independent + " and " + dependent + ": " + str(poly_pearsonr))
            print("R\xb2 of the polynomial regression of " + independent + " and " + dependent + ": " + str(poly_rsquared))

        except Exception as e:
            print(e)
