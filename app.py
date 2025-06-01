from flask import Flask, render_template, request
from flask import send_file
import numpy as np
import pickle
import csv
import os

app = Flask(__name__)

# Modelni yuklash
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Fayl nomi
RESULTS_FILE = 'predictions.csv'

# Faylga sarlavha yozish (agar yo‘q bo‘lsa)
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['rooms', 'area', 'age', 'predicted_price'])

@app.route('/results')
def results():
    import pandas as pd
    if os.path.exists(RESULTS_FILE):
        df = pd.read_csv(RESULTS_FILE)
        records = df.to_dict(orient='records')
    else:
        records = []
    return render_template('results.html', data=records)

@app.route('/download')
def download_csv():
    if os.path.exists(RESULTS_FILE):
        return send_file(RESULTS_FILE, as_attachment=True)
    return "CSV fayl topilmadi", 404

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    if request.method == 'POST':
        rooms = int(request.form['rooms'])
        area = int(request.form['area'])
        age = int(request.form['age'])

        input_data = np.array([[rooms, area, age]])
        predicted_price = model.predict(input_data)[0]

        # Natijani faylga yozish
        with open(RESULTS_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([rooms, area, age, round(predicted_price, 2)])

    return render_template('index.html', price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
