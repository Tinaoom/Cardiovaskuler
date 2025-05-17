from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Modeli yükle
with open('cardio_risk_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        height = int(request.form['height'])
        weight = float(request.form['weight'])
        ap_hi = int(request.form['ap_hi'])
        ap_lo = int(request.form['ap_lo'])
        cholesterol = int(request.form['cholesterol'])
        gluc = int(request.form['gluc'])
        smoke = int(request.form['smoke'])
        alco = int(request.form['alco'])
        active = int(request.form['active'])

        age_years = round(age / 365)
        bmi = weight / ((height / 100) ** 2)
        is_hypertension = 1 if (ap_hi >= 140 or ap_lo >= 90) else 0

        input_features = pd.DataFrame({
            'age_years': [age_years],
            'gender': [gender],
            'height': [height],
            'weight': [weight],
            'ap_hi': [ap_hi],
            'ap_lo': [ap_lo],
            'cholesterol': [cholesterol],
            'gluc': [gluc],
            'smoke': [smoke],
            'alco': [alco],
            'active': [active],
            'bmi': [bmi],
            'is_hypertension': [is_hypertension]
        })

        prediction = model.predict(input_features.values)

        if prediction[0] == 1:
            result = 'Yüksek'
        else:
            result = 'Düşük'

        return render_template('index.html', result=result) # result.html yerine index.html render ediliyor

if __name__ == '__main__':
    app.run(debug=True)