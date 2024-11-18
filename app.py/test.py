from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import joblib

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

    # Lakukan prediksi untuk n_years ke depan (bulanan)
    for _ in range(n_years * 12):
        pred = model.predict(current_input.reshape(1, 1, current_input.shape[1]))
        predictions.append(pred[0][0])
        current_input = np.roll(current_input, -1, axis=1)
        current_input[0, -1] = pred

    # Rescale hasil prediksi
    predictions_rescaled = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    predictions_rescaled = np.where(predictions_rescaled < 0, 0, predictions_rescaled)
    return predictions_rescaled

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari user
        province = request.form['province']
        city = request.form['city']
        n_years = int(request.form['years'])

        # Pastikan kota ada dalam data kota yang tersedia untuk provinsi
        if city not in cities_by_province[province]:
            return f"Error: Kota {city} tidak ditemukan dalam provinsi {province}."

        # Buat one-hot encoding untuk kota yang dipilih
        city_one_hot = np.zeros(len(cities_by_province[province]))
        city_index = cities_by_province[province].index(city)
        city_one_hot[city_index] = 1

        # Ambil data terakhir dari dataset untuk input model (misal data terakhir dari X_test)
        last_known_data = np.array([[0.5] * scaler_X.scale_.shape[0]])  # Ganti dengan data riil
        last_known_data[0, -len(city_one_hot):] = city_one_hot  # Masukkan one-hot encoding kota

        # Normalisasi data
        last_known_data = scaler_X.transform(last_known_data)

        # Prediksi harga lele untuk 1 tahun ke depan
        future_predictions = predict_future(model, last_known_data, n_years, scaler_X, scaler_y)

        # Hitung rata-rata harga per tahun
        yearly_averages = [np.mean(future_predictions[i * 12:(i + 1) * 12]) for i in range(n_years)]

        predictions_list = {
            "monthly_predictions": future_predictions.tolist(),
            "yearly_averages": yearly_averages,
        }

        return render_template('result.html', predictions=predictions_list, years=n_years)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
