import eel
import pandas as pd
import numpy as np
import json
from pynalytics.regression.linear_regression.lin_regression_num import LinRegressionRes
from pynalytics.regression.linear_regression.lin_regression_visual import LinRegressionVis
from pynalytics.regression.polynomial_regression.poly_regression_num import PolyRegressionRes
from pynalytics.regression.polynomial_regression.poly_regression_visual import PolyRegressionVis
from pynalytics.k_means import Centroid_Chart, Scatter_Matrix, Kmeans

eel.init('web')

df = pd.read_csv('Sample_Data.csv')

print(df.head())

@eel.expose
def csvUpload(csvfile):

    # Convert to dictionary
    dicts = {}
    for x in csvfile:
        dicts[x[0]] = x[1:]

    df = pd.DataFrame.from_dict(dicts,orient='index')
    df.columns = df.iloc[0]
    df = df.iloc[1:]


# eel.csvUpload()(table)

rsquare = None
adj_rsquare = None
pearsonr = None
reg_summary = None
reg_table = None
slope = None
intercept = None
line_eq = None
poly_intercept= None
poly_coeff = None
poly_rsquare = None
poly_pearsonr = None
poly_eq = None
poly_summary = None
poly_table = None
sil_coef = None
centroids = None
labeled_df = None

@eel.expose
def kmeans():
    df = pd.read_csv('Data Analytics.csv')
    df = df[['Glaciers','Forests','Locales']]
    km = Kmeans(df,3)

    return km.sil_coef(),km.centroids(),km.labeled_dataset()

sil_coef,centroids,labeled_df = kmeans()

@eel.expose
def lin_num():
    lin_res = LinRegressionRes()
    df = pd.read_csv('Data Analytics.csv')
    x = df[["TAVE_D"]]
    y = df[["Gonorrhea"]]

    return lin_res.get_rsquare(y, x), lin_res.get_adj_rsquare(y,  x), lin_res.get_pearsonr(y, x), lin_res.linear_reg_summary(y, x), lin_res.lin_regression_table(y, x)

rsquare, adj_rsquare, pearsonr, reg_summary, reg_table = lin_num()

@eel.expose
def simp_lin_num():
    lin_res = LinRegressionRes()
    df = pd.read_csv('Data Analytics.csv')
    x = df[["TAVE_D"]]
    y = df[["Gonorrhea"]]

    return lin_res.get_slope(y, x), lin_res.get_intercept(y, x), lin_res.line_eq(y, x)

slope, intercept, line_eq = simp_lin_num()

@eel.expose
def lin_scatter2D():
    lin_vis = LinRegressionVis()
    df = pd.read_csv('Data Analytics.csv')
    x = df[["TAVE_D"]]
    y = df[["Gonorrhea"]]
    fig = lin_vis.scatter_plot(y, x)

    lin_vis.fig_show(fig)
    # lin_vis.fig_to_html(fig)

@eel.expose
def lin_scatter3D():
    lin_vis = LinRegressionVis()
    df = pd.read_csv('Data Analytics.csv')
    x = df[["TAVE_D", "AVE_TMIN_D"]]
    y = df[["Gonorrhea"]]
    fig = lin_vis.scatter_plot(y, x)

    lin_vis.fig_show(fig)
    # lin_vis.fig_to_html(fig)

@eel.expose
def lin_regression():
    lin_vis = LinRegressionVis()
    df = pd.read_csv('Data Analytics.csv')
    x = df[["TAVE_D"]]
    y = df[["Gonorrhea"]]
    fig = lin_vis.linear_regression(y, x)

    lin_vis.fig_show(fig)
    # lin_vis.fig_to_html(fig)

@eel.expose
def poly_num():
    poly_res = PolyRegressionRes()
    df = pd.read_csv('Data Analytics.csv')
    x = df[["TAVE_D"]]
    y = df[["Gonorrhea"]]

    return poly_res.get_poly_intercept(y, x), poly_res.get_poly_coeff(y, x), poly_res.get_poly_rsquared(y, x), poly_res.get_poly_pearsonr(y, x), poly_res.poly_eq(y, x), poly_res.polynomial_reg_summary(y, x), poly_res.poly_reg_table(y, x)

poly_intercept, poly_coeff, poly_rsquare, poly_pearsonr, poly_eq, poly_summary, poly_table = poly_num()

@eel.expose
def poly_regression():
    poly_vis = PolyRegressionVis()
    df = pd.read_csv('Data Analytics.csv')
    x = df[["TAVE_D"]]
    y = df[["Gonorrhea"]]
    fig = poly_vis.polynomial_reg(y, x)

    poly_vis.fig_show(fig)
    # poly_vis.fig_to_html(fig)

@eel.expose
def kmeans_visuals():
    cc = Centroid_Chart()
    fig = cc.centroid_chart(centroids,x_labels=labeled_df.columns[:-1].values)

    cc.fig_show(fig)
    # cc.fig_to_html(fig)

# kmeans_visuals()

eel.start('main.html', size=(1920, 1080))
