import eel

eel.init('web')

@eel.expose
def test():
    x = '2131231'
    return x

eel.start('main.html', size=(1920, 1080))