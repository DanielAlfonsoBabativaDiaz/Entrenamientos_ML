from flask import Flask
from flask import render_template
from flask import request
import LinearRegression
import joblib


app = Flask (__name__)
model = joblib.load("linear_regression_model.pkl")

@app.route('/casos_de_uso_ML')
def casos_de_uso_ML():
    return render_template('casos_de_uso_ML.html')

@app.route('/navbar')
def navbar():
    return render_template('navbar.html')

@app.route('/')
def home():
    Myname = "Machine_Learning"
    return render_template('index.html', name=Myname)

#@app.route('/LinearRegression', methods=['GET', 'POST'])
#def linearRegression():
    #calculatedResult = None
    #if request.method == 'POST':
        #calculatedResult = linearRegression.CalcuateGrade("5")
    #return "Final Grade Predicted: " + str(calculatedResult)

@app.route("/prediccion", methods=["GET", "POST"])
def calculateGrade():
    calculateResult = None
    if request.method == "POST":
        Rainfall = float(request.form["rainfall"])
        Temperature = float(request.form["temperature"])
        predicted_coffe_price = model.predict([[Rainfall, Temperature]])
        calculateResult = predicted_coffe_price[0]
    return render_template("prediccion.html", result = calculateResult)

if __name__== '__main__':
    app.run(debug=True)