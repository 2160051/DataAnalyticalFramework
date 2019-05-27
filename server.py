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

df = pd.read_csv('/Users/britanny/Documents/School Files/Thesis/Framework/data.csv')

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

#Create table
@eel.expose
def table():
    tabledata = df.to_html()
    return(''+ tabledata +'')

#Send columns
@eel.expose
def columns():
    columnsList = list(df.columns.values)
    return(columnsList)

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

#@eel.expose
#def kmeans():
#    df = pd.read_csv('Data Analytics.csv')
#    df = df[['Glaciers','Forests','Locales']]
#    km = Kmeans(df,3)
#
#    return km.sil_coef(),km.centroids(),km.labeled_dataset()

#sil_coef,centroids,labeled_df = kmeans()
#
#@eel.expose
#def lin_num():
#    lin_res = LinRegressionRes()
#    df = pd.read_csv('Data Analytics.csv')
#    x = df[["TAVE_D"]]
#    y = df[["Gonorrhea"]]
#
#    return lin_res.get_rsquare(y, x), lin_res.get_adj_rsquare(y,  x), lin_res.get_pearsonr(y, x), lin_res.linear_reg_summary(y, x), lin_res.lin_regression_table(y, x)

#rsquare, adj_rsquare, pearsonr, reg_summary, reg_table = lin_num()

#@eel.expose
#def simp_lin_num():
#    lin_res = LinRegressionRes()
#    df = pd.read_csv('Data Analytics.csv')
#    x = df[["TAVE_D"]]
#    y = df[["Gonorrhea"]]

#    return lin_res.get_slope(y, x), lin_res.get_intercept(y, x), lin_res.line_eq(y, x)

#slope, intercept, line_eq = simp_lin_num()

#@eel.expose
#def lin_scatter2D():
#    lin_vis = LinRegressionVis()
#    df = pd.read_csv('Data Analytics.csv')
#    x = df[["TAVE_D"]]
#    y = df[["Gonorrhea"]]
#    fig = lin_vis.scatter_plot(y, x)

#    lin_vis.fig_show(fig)
    # lin_vis.fig_to_html(fig)

#@eel.expose
#def lin_scatter3D():
#    lin_vis = LinRegressionVis()
#    df = pd.read_csv('Data Analytics.csv')
#    x = df[["TAVE_D", "AVE_TMIN_D"]]
#    y = df[["Gonorrhea"]]
#    fig = lin_vis.scatter_plot(y, x)

#    lin_vis.fig_show(fig)
    # lin_vis.fig_to_html(fig)

#@eel.expose
#def lin_regression():
#    lin_vis = LinRegressionVis()
#    df = pd.read_csv('Data Analytics.csv')
#    x = df[["TAVE_D"]]
#    y = df[["Gonorrhea"]]
#    fig = lin_vis.linear_regression(y, x)

#    lin_vis.fig_show(fig)
    # lin_vis.fig_to_html(fig)

#@eel.expose
#def poly_num():
#    poly_res = PolyRegressionRes()
#    df = pd.read_csv('Data Analytics.csv')
#    x = df[["TAVE_D"]]
#    y = df[["Gonorrhea"]]

#    return poly_res.get_poly_intercept(y, x), poly_res.get_poly_coeff(y, x), poly_res.get_poly_rsquared(y, x), poly_res.get_poly_pearsonr(y, x), poly_res.poly_eq(y, x), poly_res.polynomial_reg_summary(y, x), poly_res.poly_reg_table(y, x)

#poly_intercept, poly_coeff, poly_rsquare, poly_pearsonr, poly_eq, poly_summary, poly_table = poly_num()

#@eel.expose
#def poly_regression():
#    poly_vis = PolyRegressionVis()
#    df = pd.read_csv('Data Analytics.csv')
#    x = df[["TAVE_D"]]
#    y = df[["Gonorrhea"]]
#    fig = poly_vis.polynomial_reg(y, x)

#   poly_vis.fig_show(fig)
    # poly_vis.fig_to_html(fig)

#@eel.expose
#def kmeans_visuals():
#    cc = Centroid_Chart()
#    fig = cc.centroid_chart(centroids,x_labels=labeled_df.columns[:-1].values)

#    cc.fig_show(fig)
    # cc.fig_to_html(fig)

# kmeans_visuals()

#GUI functions
@eel.expose
def kmeans_sil_coef(kdf,c):
    kc = int(c)
    kmdf = df[kdf]
    km = Kmeans(kmdf,kc)
    return str(km.sil_coef())

@eel.expose
def kmeans_centroids(kdf,c):
    kc = int(c)
    kmdf = df[kdf]
    km = Kmeans(kmdf,kc)
    return str(km.centroids())

@eel.expose
def kmeans_centroid_chart(kdf, c):
    kc = int(c)
    kmdf = df[kdf]
    km = Kmeans(kmdf,kc)

    cc = Centroid_Chart()
    fig = cc.centroid_chart(km.centroids(),x_labels=kmdf.columns.values)
    return(''+ cc.fig_to_html(fig) +'')

@eel.expose
def lin_num_rsquare(dv, idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    return str(lin_res.get_rsquare(y, x))

@eel.expose
def lin_adj_rsquare(dv, idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    return str(lin_res.get_adj_rsquare(y, x))

@eel.expose
def lin_pearson(dv, idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    return str(lin_res.get_pearsonr(y, x))

#bug
@eel.expose
def lin_pvalue(dv, idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    return str(lin_res.get_pvalue(y, x))

@eel.expose
def lin_regression(dv, idv):
    lin_vis = LinRegressionVis()
    x = df[[idv]]
    y = df[[dv]]
    fig = lin_vis.linear_regression(y, x)

    #return(''+ lin_vis.fig_to_html(fig)+ '')

@eel.expose
def lin_rtable(dv ,idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    #print(lin_res.linear_reg_summary(y, x))
    return(''+ lin_res.lin_regression_table(y, x).to_html() +'')

#Simple Linear Regression
@eel.expose
def simp_lin_num_slope(dv, idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    return str(lin_res.get_slope(y, x)) #lin_res.get_intercept(y, x), lin_res.line_eq(y, x)

@eel.expose
def simp_lin_num_intercept(dv, idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    return str(lin_res.get_intercept(y, x))

@eel.expose
def simp_lin_num_rslope(dv, idv):
    lin_res = LinRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    return str(lin_res.line_eq(y, x))
#

@eel.expose
def poly_int(dv, idv):
    poly_res = PolyRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    return str(poly_res.get_poly_intercept(y, x))

@eel.expose
def poly_coefficient(dv, idv):
    poly_res = PolyRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    return str(poly_res.get_poly_coeff(y, x))

@eel.expose
def poly_rsquared(dv, idv):
    poly_res = PolyRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    return str(poly_res.get_poly_rsquared(y, x))

@eel.expose
def poly_pearson_r(dv, idv):
    poly_res = PolyRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    return str(poly_res.get_poly_pearsonr(y, x))

@eel.expose
def poly_equation(dv, idv):
    poly_res = PolyRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    return str(poly_res.poly_eq(y, x))

@eel.expose
def poly_regression(dv, idv):
    poly_vis = PolyRegressionVis()
    x = df[[idv]]
    y = df[[dv]]
    fig = poly_vis.polynomial_reg(y, x)

    #return(''+poly_vis.fig_to_html(fig)+'')

#error
@eel.expose
def poly_rtable(dv, idv):
    poly_res = PolyRegressionRes()
    x = df[[idv]]
    y = df[[dv]]

    #print(poly_res.polynomial_reg_summary(y, x)) 
    return(''+ poly_res.poly_reg_table(y, x).tohtml() +'')

eel.start('main.html', size=(1920, 1080))
