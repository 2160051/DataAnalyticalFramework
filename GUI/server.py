import eel
import pandas as pd

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

eel.start('main.html', size=(1920, 1080))
