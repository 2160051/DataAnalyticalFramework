import eel
import pandas as pd
import numpy as np
import json
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
def kmeans_visuals():
    cc = Centroid_Chart()
    fig = cc.centroid_chart(centroids,x_labels=labeled_df.columns[:-1].values)

    cc.fig_show(fig)
    # cc.fig_to_html(fig)

# kmeans_visuals()

eel.start('main.html', size=(1920, 1080))
