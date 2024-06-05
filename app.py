import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
classifier = pickle.load(open('models/model_v1.0.0.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def root():
    if request.method == 'POST':
        print(request.form)
        highbp = 1 if 'highbp' in request.form.keys() else 0
        highchol = 1 if 'highchol' in request.form.keys() else 0
        cholcheck = 1 if 'cholcheck' in request.form.keys() else 0
        gender = 1 if 'gender' in request.form.keys() else 0
        bmi = request.form.get('bmi')
        age = request.form['age']
        df = pd.DataFrame({
            'HighBP': [highbp],
            'HighChol': [highchol],
            'CholCheck': [cholcheck],
            'BMI': [bmi],
            'Sex': [gender],
            'Age': [age]
        })
        # print(f'SEBELUM>>> {df[["BMI", "Age"]]}')
        df[["BMI", "Age"]] = scaler.transform(df[["BMI", "Age"]])
        # print(f'SESUDAH>>> {df[["BMI", "Age"]]}')
        if classifier.predict(df)[0] == 1:
            return 'Terdeteksi Diabetes'
        return 'Tidak Terdeteksi Diabetes'
    return render_template('forms.html')

if __name__ == '__main__':
    app.run(debug=True)