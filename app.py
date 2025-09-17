from flask import Flask
from flask import render_template
from flask import request
from modeloCurso import evaluar_modelo
import LinearRegression
import joblib
import numpy as np
import pickle


app = Flask (__name__)
model = joblib.load("linear_regression_model.pkl")

@app.route('/casos_de_uso_ML')
def casos_de_uso_ML():
    return render_template('casos_de_uso_ML.html')

@app.route('/conceptos_basicos')
def conceptos_basicos():
    return render_template('conceptos_basicos.html')

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

with open('./PKL/modelo_aprobacion.pkl', 'rb') as f:
    model = pickle.load(f)

with open('./PKL/scaler_aprobacion.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route("/Cursos", methods=["GET", "POST"])
def prediccionCurso():
    resultado = None

    if request.method == "POST":
        horas = float(request.form["horas"])
        foros = int(request.form["foros"])
        nivel = request.form["nivel"]

        niveles = {"Secundaria": 0, "Tecnico": 1, "Universitario": 2}
        nivel_codificado = niveles.get(nivel, 0)

        entrada = np.array([[horas, foros, nivel_codificado]])
        entrada_escalada = scaler.transform(entrada)
        prediccion = model.predict(entrada_escalada)

        resultado = "Sí" if prediccion[0] == 1 else "No"

    # Métricas del modelo (siempre visibles)
    accuracy, report_html, conf_matrix_img = evaluar_modelo(model, scaler)

    return render_template("Cursos.html",
                           result=resultado,
                           accuracy=accuracy,
                           report_html=report_html,
                           conf_matrix_img=conf_matrix_img)

if __name__== '__main__':
    app.run(debug=True)