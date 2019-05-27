import eel
import pandas as pd
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
def csvUpload(data):
    # df = pd.read_csv(data)
    print(df)

eel.csvUpload()(table)

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
def regression():
    df = pd.read_csv('Data Analytics.csv')
    df = df[['Glaciers','Forests','Locales']]
    km = Kmeans(df,3)

    return km.sil_coef(),km.centroids(),km.labeled_dataset()


sil_coef,centroids,labeled_df = kmeans()


@eel.expose
def kmeans_visuals():
    cc = Centroid_Chart()
    fig = cc.centroid_chart(centroids,x_labels=labeled_df.columns[:-1].values)

    cc.fig_show(fig)
    # cc.fig_to_html(fig)

# kmeans_visuals()

eel.start('main.html', size=(1920, 1080))
