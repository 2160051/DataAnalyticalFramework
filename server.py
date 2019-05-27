import eel
import pandas as pd
import json
# import pynalitics

eel.init('web')

df = ''

@eel.expose
def csvSend(csvFile):
    print(csvFile)

@eel.expose
def table():
    df = pd.read_csv('/Users/britanny/Documents/School Files/Thesis/Framework/biostats.csv')
    tabledata = df.to_html()
    return(''+ tabledata +'') 

# @eel.expose
# def kmeans_visualization():

eel.start('main.html', size=(1920, 1080))