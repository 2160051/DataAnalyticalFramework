"""
This program is a module for generating numerical results using Polynomial Regression.
"""
import math
import operator
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from ...preprocessor import Preprocessing
from matplotlib import style
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
pd.options.mode.chained_assignment = None
style.use('seaborn-bright')

class PolyRegressionRes(Preprocessing):
    """
    This represents the class for generating numerical results using Polynomial Regression.
    """
    
    version = "1.0"

    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        """
        pass

    def get_poly_intercept(self, independent, dependent):
        """

        Returns the calculated intercept of the polynomial regression
        
        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified

        Returns
        -------
        float
            intercept of the polynomial regression
        """

        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)   

            model = LinearRegression()
            model.fit(x_poly, y)
            intercept_arr = model.intercept_
            return round(intercept_arr[0], 4)
        except Exception as e:
            print(e)
    
    def get_poly_coeff(self, independent, dependent):
        """

        Returns a list containing the correlation coefficients of the polynomial regression
        
        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified

        Returns
        -------
        list
            list of correlation coefficients of the polynomial regression
        """

        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)   

            model = LinearRegression()
            model.fit(x_poly, y)
            return model.coef_
        except Exception as e:
            print(e)

    def get_poly_rsquared(self, independent, dependent):
        """

        Returns the calculated coefficient of determination(R²) of the polynomial regression
        
        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified

        Returns
        -------
        float
            calculated coefficient of determination(R²) of the polynomial regression
        """

        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)   

            model = LinearRegression()
            model.fit(x_poly, y)
            y_poly_pred = model.predict(x_poly)
            r2 = r2_score(y,y_poly_pred)
            return round(r2, 4)
        except Exception as e:
            print(e)

    def get_poly_pearsonr(self, independent, dependent):
        """

        Returns the calculated Pearson correlation coefficient of the polynomial regression
        
        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified

        Returns
        -------
        float
            calculated Pearson correlation coefficient of the polynomial regression
        """

        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)   

            model = LinearRegression()
            model.fit(x_poly, y)
            y_poly_pred = model.predict(x_poly)
            r2 = r2_score(y,y_poly_pred)
            pearsonr = math.sqrt(r2)
            return round(pearsonr, 4)
        except Exception as e:
            print(e) 
    
    def poly_eq(self, independent, dependent):
        """

        Returns the equation of the polynomial regression

        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified
        
        Returns
        -------
        str
            line equation of the polynomial regression
        """

        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)  

            model = LinearRegression()
            model.fit(x_poly, y)
            coef_arr = model.coef_
            intercept_arr = model.intercept_
            
            poly_equation = "y = " + str(round(coef_arr[0][2], 4)) + "x\xB2"
            
            if(coef_arr[0][1] < 0):
                poly_equation += " + (" + str(round(coef_arr[0][1], 4)) + "x" + ")"
            else:
                poly_equation += " + " + str(round(coef_arr[0][1], 4)) + "x"
            
            if(intercept_arr[0] < 0):
                poly_equation += " + (" + str(round(intercept_arr[0], 4)) + ")"
            else:
                poly_equation += " + " + str(round(intercept_arr[0], 4))
           
            return  poly_equation
        except Exception as e:
            print(e)
