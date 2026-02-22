from flask import Flask,render_template,request
import pickle
import numpy as np
from joblib import load

app = Flask(__name__)

# load model + scaler
model = pickle.load(open("floods.save","rb"))
sc = load("transform.save")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict')
def index():
    return render_template("index.html")

@app.route('/data_predict',methods=['POST'])
def predict():

    temp = float(request.form['temp'])
    hum  = float(request.form['Hum'])
    db   = float(request.form['db'])
    ap   = float(request.form['ap'])
    aal  = float(request.form['aal'])
    mar  = float(request.form['mar'])
    jun  = float(request.form['jun'])
    octa = float(request.form['oct'])

    data = np.array([[temp,hum,db,ap,aal,mar,jun,octa,0,0]])

    prediction = model.predict(sc.transform(data))

    if prediction[0]==0:
        return render_template("nochance.html")
    else:
        return render_template("chance.html")

if __name__ == "__main__":
    app.run(debug=True)