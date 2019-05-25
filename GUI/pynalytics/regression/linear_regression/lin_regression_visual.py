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
from .lin_regression_num import LinRegressionRes
from matplotlib import style
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
pd.options.mode.chained_assignment = None
style.use('seaborn-bright')

class LinRegressionVis(LinRegressionRes):
    """
    This represents the class for generating numerical results using Linear Regression.
    """
    
    version = "1.0"

    def __init__(self):  
        """

        Initializes the use of the class and its functions 
        """
        pass

    def scatter_plot(self, dependent, *independent):
        """

        Generates the visualization of the scatter plot

        Generates the 2D visualization of scatter plot given that no second independent variable is specified, else it will generate a 3D visualization

        Parameters
        ----------
        dependent : str
            the dependent(y) variable specified
        independent : tuple
            the independent(x) variable specified
        
        Returns
        -------
        figure
            visualization of the scatter plot
        """

        try:
            independent = list(independent)
            if(len(independent) == 1):
                x = self.df_input[independent]
                y = self.df_input[dependent]

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(x, y, color = 'red')
                ax.set_xlabel(independent)
                ax.set_ylabel(dependent)
                ax.axis('tight')
                plt.title("Scatter Plot of " + dependent + " and " + independent)
                plt.show()
            elif(len(independent) > 1):
                x = self.df_input[independent[0]]
                y = self.df_input[[dependent]]
                z = self.df_input[independent[1]]

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, color = 'red')
                ax.set_xlabel(independent[0])
                ax.set_ylabel("Number of cases of " + dependent)
                ax.set_zlabel(independent[1])
                ax.axis('tight')
                plt.title("Scatter Plot of " + dependent + ", " + independent[0] + " and " + independent[1])
                plt.show()
        except Exception as e:
            print(e)

    def linear_regression(self, dependent, independent):
        """

        Generates the visualization for simple linear regression

        Parameters
        ----------
        dependent : str
            the dependent(y) variable specified
        independent : str
            the independent(x) variable specified
        
        Returns
        -------
        figure
            visualization of the simple linear regression
        """

        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            lm = LinearRegression()
            model = lm.fit(x, y)
            x_new = np.linspace(self.df_input[independent].min() - 5, self.df_input[independent].max() + 5, 50)
            y_new = model.predict(x_new[:, np.newaxis])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x_new, y_new, color = 'blue', label=self.line_eq(dependent, independent))
            ax.legend(fontsize=9, loc="upper right")
            ax.scatter(x, y, color = 'red')
            ax.set_xlabel(independent)
            ax.set_ylabel(dependent)
            ax.axis('tight')
            plt.title("Linear Regression of " + independent + " and " + dependent)
            plt.show()
                
        except Exception as e:
            print(e)

    def linear_reg_summary(self, dependent, *independent):
        """

        Generates the calculated statistical values of the regression

        Generates the calculated statistical values for the linear regression such as the standard error, coefficient of determination(R²) and p-value, in table form

        Parameters
        ----------
        dependent : str
            the dependent(y) variable specified
        independent : tuple
            the independent(x) variable specified
        
        Returns
        -------
        statsmodels.summary
            table summary containing the calculated statistical values of the regression
        """

        try:
            independent = list(independent)
            x = self.df_input[independent]
            y = self.df_input[[dependent]]

            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            print(model.summary())
        except Exception as e:
            print(e)

    def regression_table(self, dependent, *independent):

        """

        Generates the summary of the calculated values for P-value, correlation coefficient, coefficient of determination(R²) and adjusted coefficient of determination of regression, in table form

        Parameters
        ----------
        dependent : list
            list containing the string value of the dependent(y) variables
        independent : tuple
            the independent(x) variable specified

        Returns
        -------
        pandas Dataframe
            summary of the calculated values for P-value, correlation coefficient, coefficient of determination(R²) and adjusted coefficient of determination of regression
        """

        try:
            independent = list(independent)
            coeff_det = []
            adj_coeff_det = []
            pearsonr = []
            pvalue = []

            for step in independent:
                pvalue_df = self.get_pvalue(dependent, step)
                pvalue.append(round(pvalue_df.loc[step], 4))
                coeff_det.append(self.get_rsquare(dependent, step))
                adj_coeff_det.append(self.get_adj_rsquare(dependent, step))
                pearsonr.append(self.get_pearsonr(dependent, step))

            table_content =  {"Attribute (x)": independent, "P-Value": pvalue, "Coefficient of Determination (R^2)": coeff_det, "Adjusted Coefficient of Determination (R^2)": adj_coeff_det, "Pearson Correlation Coefficient (R)": pearsonr}
            table_df = pd.DataFrame(table_content)
            return table_df
        except Exception as e:
            print(e)
