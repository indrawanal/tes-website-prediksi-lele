from flask import Flask, render_template
import tensorflow as tf
import numpy as np
import joblib
import plotly.graph_objects as go
import pandas as pd

app = Flask(__name__)

# Memuat model dan scaler
model = tf.keras.models.load_model('model_lele.h5')
scaler_X = joblib.load('scaler_lele_X.pkl')
scaler_y = joblib.load('scaler_lele_y.pkl')

# Data provinsi dan kota untuk input
cities_by_province = {
    'JAWA BARAT': ['Bandung', 'Cirebon', 'Bogor'],
    'JAWA TENGAH': ['Semarang', 'Magelang'],
    'JAWA TIMUR': ['Surabaya', 'Malang']
}

# Fungsi untuk melakukan prediksi berurutan
def predict_future(model, last_known_data, n_years, scaler_X, scaler_y):
    predictions = []
    current_input = last_known_data.copy()

    for _ in range(n_years * 12):  # Prediksi bulanan selama n_years
        pred = model.predict(current_input.reshape(1, 1, current_input.shape[1]))
        predictions.append(pred[0][0])
        current_input = np.roll(current_input, -1, axis=1)
        current_input[0, -1] = pred

    predictions_rescaled = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    predictions_rescaled = np.where(predictions_rescaled < 0, 0, predictions_rescaled)
    return predictions_rescaled

@app.route('/')
def index():
    # Prediksi harga untuk beberapa kota
    predictions_for_cities = {}
    cities = ['Bandung', 'Cirebon', 'Bogor', 'Semarang', 'Surabaya']
    n_years = 2  # Misalnya prediksi untuk 2 tahun ke depan

    for city in cities:
        # Tentukan provinsi berdasarkan kota
        if city in cities_by_province['JAWA BARAT']:
            province = 'JAWA BARAT'
        elif city in cities_by_province['JAWA TENGAH']:
            province = 'JAWA TENGAH'
        else:
            province = 'JAWA TIMUR'
        
        city_one_hot = np.zeros(len(cities_by_province[province]))
        city_index = cities_by_province[province].index(city)
        city_one_hot[city_index] = 1

        # Ambil data terakhir dari dataset untuk input model
        last_known_data = np.array([[0.5] * scaler_X.scale_.shape[0]])
        last_known_data[0, -len(city_one_hot):] = city_one_hot

        # Normalisasi data
        last_known_data = scaler_X.transform(last_known_data)

        # Prediksi harga lele untuk 2 tahun ke depan
        future_predictions = predict_future(model, last_known_data, n_years, scaler_X, scaler_y)

        # Membuat grafik prediksi untuk setiap kota
        monthly_labels = pd.date_range(start='2025-01-01', periods=n_years * 12, freq='M')

        # Menyimpan grafik dan prediksi untuk setiap kota
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly_labels,
            y=future_predictions,
            mode='lines+markers',
            name=f'Prediksi Harga {city}'
        ))
        fig.update_layout(
            title=f'Prediksi Harga Lele di {city}, {province}',
            xaxis_title='Tanggal',
            yaxis_title='Harga (Rp)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True)
        )

        predictions_for_cities[city] = fig.to_html(full_html=False)

    return render_template('index.html', predictions_for_cities=predictions_for_cities)

if __name__ == '__main__':
    app.run(debug=True)
