from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from model import Model

df = pd.read_csv('C:/Users/user/Downloads/archive/SaYoPillow.csv')
df.columns = ['laju dengkuran', 'laju pernapasan',
           'suhu tubuh', 'gerakan anggota badan',
           'oksigen darah', 'gerakan mata',
           'jam tidur', 'detak jantung','stress level']

app = Flask(__name__)
model = Model(df)
@app.route('/home')
def home():
    return render_template('home.html')

def predict(listt):
    pred = np.array(listt).reshape(1, -1)
    pred = model.scaler.transform(pred)
    result = model.log.predict(pred)
    return result[0]

@app.route('/', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        laju_dengkuran, laju_pernapasan, suhu_tubuh, gerakan_anggota_badan, oksigen_darah, gerakan_mata,jam_tidur, detak_jantung = request.form['laju dengkuran'], request.form['laju pernapasan'], request.form['suhu tubuh'], request.form['gerakan anggota badan'], request.form['oksigen darah'], request.form['gerakan mata'], request.form['jam tidur'], request.form['detak jantung']
        result = [[laju_dengkuran, laju_pernapasan, suhu_tubuh,
        gerakan_anggota_badan, oksigen_darah, gerakan_mata, jam_tidur, detak_jantung]]
        result = predict(result)
        return render_template('halaman1.html', result=result)
    else:
        lanjut = ''
        return render_template('home.html', result=lanjut )
if __name__ == '__main__':
    app.run(debug=True)
    
