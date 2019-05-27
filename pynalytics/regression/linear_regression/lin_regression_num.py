"""
This program is a module for generating numerical results using linear regression.
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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
pd.options.mode.chained_assignment = None
style.use('seaborn-bright')

class LinRegressionRes(Preprocessing):
    """
    This represents the class for generating numerical results using Linear Regression.
    """
    
    version = "1.0"

    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        """
        pass

    def get_slope(self, dependent, independent):
        """

        Returns the slope of the regression

        Returns the calculated slope(m) of the simple linear regression given that there is no second independent variable specified, else it will return a list containing the calculated slope(m) of the multiple linear regression
        
        Parameters
        ----------
        dependent : str
            the dependent(y) variable specified
        independent : str
            independent(x) variable specified used for linear regression
        
        Returns
        -------
        float
            calculated slope(m) of the simple linear regression
        """

        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit() 
            coef_df = model.params
            return round(coef_df[independent], 4)
        except Exception as e:
            print(e)

    def get_intercept(self, dependent, independent):
        """

        Returns the calculated intercept of the simple linear regression
        
        Parameters
        ----------
        dependent : str
            the dependent(y) variable specified
        independent : str
            the independent(x) variable specified

        Returns
        -------
        float
            intercept of the simple linear regression
        """

        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            lm = LinearRegression()
            lm.fit(x, y)
            b = lm.intercept_
            return round(b[0], 4)
        except Exception as e:
            print(e)

    def get_rsquare(self, dependent, *independent):
        """

        Returns the calculated coefficient of determination(R²) of the regression

        Returns the calculated coefficient of determination(R²) of the simple linear regression given that there is no second independent variable specified, else it will return the calculated coefficient of determination(R²) of the mulltiple linear regression 
        
        Parameters
        ----------
        dependent : str
            the dependent(y) variable specified
        independent : tuple
            independent(x) variable specified used for linear regression
        
        Returns
        -------
        float
            coefficient of determination(R²) of the regression
        """

        try:
            independent = list(independent)
            x = self.df_input[independent]
            y = self.df_input[[dependent]]

            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            return round(model.rsquared, 4)
        except Exception as e:
            print(e)

    def get_adj_rsquare(self, dependent, *independent):
        """
        
        Returns the calculated adjusted coefficient of determination(R²) of the regression

        returns the calculated adjusted coefficient of determination(R²) of the simple linear regression given that there is no second independent variable specified, else it will return the calculated adjusted coefficient of determination(R²) of the mulltiple linear regression 
        
        Parameters
        ----------
        dependent: str
            the dependent(y) variable specified
        independent : tuple
            the independent(x) variable specified

        Returns
        -------
        float
            calculated adjusted coefficient of determination(R²) of the regression
        """
        try:
            independent = list(independent)
            x = self.df_input[independent]
            y = self.df_input[[dependent]]

            x = sm.add_constant(x)           
            model = sm.OLS(y, x).fit()
            return round(model.rsquared_adj, 4)
        except Exception as e:
            print(e)
        
    def get_pearsonr(self, dependent, *independent):
        """

        Returns the calculated Pearson correlation coefficient of the regression

        Returns the calculated Pearson correlation coefficient of the simple linear regression given that there is no second independent variable specified, else it will return the calculated coefficient of determination(R²) of the mulltiple linear regression 
        
        Parameters
        ----------
        dependent : str
            the dependent(y) variable specified
        independent : tuple
            the independent(x) variable specified
        
        Returns
        -------
        float
            Pearson correlation coefficient of the regression
        """

        try:
            independent = list(independent)
            x = self.df_input[independent]
            y = self.df_input[[dependent]]

            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            r2 = model.rsquared
            pearsonr = math.sqrt(r2)
            return round(pearsonr, 4)
        except Exception as e:
            print(e)

    def get_pvalue(self, dependent, *independent):
        """

        Returns the calculated P-value/s of the regression

        Returns the dataframe containing calculated P-value/s of the simple linear regression given that there is no second independent variable specified, else it will return the calculated P-value/s of the mulltiple linear regression 
        
        Parameters
        ----------
        dependent : str
            the dependent(y) variable specified
        independent : tuple
            the independent(x) variable specified
        
        Returns
        -------
        pandas Dataframe
            dataframe containing the P-value/s of the regression
        """

        try:
            independent = list(independent)
            x = self.df_input[independent]
            y = self.df_input[[dependent]]

            x = sm.add_constant(x)           
            model = sm.OLS(y, x).fit()
            pvalue = model.pvalues
            return pvalue
        except Exception as e:
            print(e)

    def line_eq(self, dependent, independent):
        """

        Returns the line equation of the simple linear regression 
        
        Parameters
        ----------
        dependent : str
            the dependent(y) variable specified
        independent : str
            the independent(x) variable specified
        
        Returns
        -------
        str
            line equation of the simple linear regression
        """

        try:
            m = self.get_slope(dependent, independent)
            b = self.get_intercept(dependent, independent)
            lin_equation = "y = " + str(m) + "x "
            if(b < 0):
                lin_equation += "+ (" + str(m) + ")"
            else:
                lin_equation += "+ " + str(b)
            
            return lin_equation
        except Exception as e:
            print(e)
