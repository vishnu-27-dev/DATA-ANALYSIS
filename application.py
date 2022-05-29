
from flask import Flask,render_template,request
import pandas as pd
import pickle
app=Flask(__name__)

model=pickle.load(open("LinearRegressionModel.pkl","rb"))
car=pd.read_csv("Cleaned car.csv")

@app.route('/')
def index():
    Make = sorted(car["Make"].unique())
    Fuel_Type=sorted(car["Fuel_Type"].unique())
    return render_template('index.html',Make=Make,Fuel_Type=Fuel_Type)

@app.route('/predict',methods=['POST'])
def predict():
    Make = request.form.get('Make')
    Fuel_Type = request.form.get("Fuel_Type")
    print(Make,Fuel_Type)
    prediction = model.predict(pd.DataFrame([[Make,Fuel_Type]],columns=['Make','Fuel_Type']))
    print(prediction)
    return ""

if __name__=="__main__":
    app.run(debug=True)