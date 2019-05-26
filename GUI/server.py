import eel
import pandas as pd
from matplotlib.pyplot import figure
import mpld3
eel.init('web')

@eel.expose
def test():
    x = '2131231'
    return x

@eel.expose
def table():
    df = pd.read_csv('/Users/britanny/Documents/School Files/Thesis/Framework/Data Analytics.csv')
    columns = df.columns.values
    csv = pd.read_csv('/Users/britanny/Documents/School Files/Thesis/Framework/Data Analytics.csv', names=columns)
    tabledata = csv.to_html()
    return('"""'+ tabledata +'"""')


@eel.expose
def plot():

    fig = figure()
    ax = fig.gca()
    ax.plot([1,2,3,4])

    return mpld3.fig_to_html(fig)

plot()

eel.start('main.html', size=(1920, 1080))
