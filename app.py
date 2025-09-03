from flask import Flask
from flask import render_template

app = Flask (__name__)

@app.route("/")
def home():
    name = None
    name = "Flask"
    return f"Hello, {name}!"

@app.route('/casos_de_uso_ML')
def casos_de_uso_ML():
    return render_template('casos_de_uso_ML.html')

@app.route('/navbar')
def navbar():
    return render_template('navbar.html')

@app.route('/index')
def index():
    Myname = "Machine_Learning"
    return render_template('index.html', name=Myname)

if __name__== '__main__':
    app.run(debug=True)