# from pynalytics.preprocessor import Preprocessing
# from pynalytics.regression.lin_regression_visual import LinRegressionVis
from pynalytics.preprocessor import Preprocessing
from pynalytics.regression.linear_regression.lin_regression_num import LinRegressionRes
from pynalytics.regression.linear_regression.lin_regression_visual import LinRegressionVis
from pynalytics.regression.polynomial_regression.poly_regression_num import PolyRegressionRes
from pynalytics.regression.polynomial_regression.poly_regression_visual import PolyRegressionVis
import os
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

csv_file = os.path.dirname(__file__) + "/regrex1.csv"
# csv_file = os.path.dirname(__file__) + "/Position_Salaries.csv"
# csv_file = os.path.dirname(__file__) + "/Data Analytics.csv"
lin_res = LinRegressionRes()
lin_vis = LinRegressionVis()
poly_res = PolyRegressionRes()
poly_vis = PolyRegressionVis()

df = pd.read_csv(csv_file)
#FOR LINRES
# print(lin_res.get_slope(df[["y"]], df[["x"]]))
# print(lin_res.get_intercept(df[["y"]], df[["x"]]))
# print(lin_res.get_rsquare(df["y"] ,df[["x", "z"]])) # how to input for MULTIPLE INDEPENDENTS
# print(lin_res.get_adj_rsquare(df["y"] ,df[["x", "z"]]))
# print(lin_res.get_pearsonr(df["y"] ,df[["x", "z"]]))
# print(lin_res.get_pvalue(df["y"] ,df[["x"]]))
# print(lin_res.line_eq(df[["y"]], df[["x"]]))

#FOR LINVIS
# lin_vis.scatter_plot(df[["y"]], df[["x"]]) #2D
# lin_vis.scatter_plot(df[["y"]], df[["x", "z"]]) #3D
# lin_vis.linear_regression(df[["y"]], df[["x"]])
# lin_vis.linear_reg_summary(df[["y"]], df[["x", "z"]])
# print(lin_vis.lin_regression_table(df["y"] ,df[["x", "z"]]))

#FOR POLYRES
# print(poly_res.get_poly_intercept(df[["y"]], df[["x"]]))
# print(poly_res.get_poly_coeff(df[["y"]], df[["x"]]))
# print(poly_res.get_poly_rsquared(df[["y"]], df[["x"]]))
# print(poly_res.get_poly_pearsonr(df[["y"]], df[["x"]]))
# print(poly_res.poly_eq(df[["y"]], df[["x"]]))

#FOR POLYVIS
# poly_vis.polynomial_reg(df[["y"]], df[["x"]])
# poly_vis.polynomial_reg_summary(df[["y"]], df[["x"]])
# print(poly_vis.poly_reg_table(df[["Gonorrhea"]], df[["TAVE_D", "AVE_TMIN_D", "AVE_TMAX_D", "GDP per capita"]]))

x = df[["z"]]
y = df[["y"]]
fig = poly_vis.polynomial_reg(y, x)
# fig = lin_vis.linear_regression(y, x)
# print(poly_vis.fig_to_html(fig))
# poly_vis.fig_show(fig)



