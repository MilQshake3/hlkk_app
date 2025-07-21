from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_caching import Cache
import requests
import logging
import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_swagger_ui import get_swaggerui_blueprint

# H.L.K.K Application Version 1.0
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')
app.config['CACHE_TYPE'] = 'filesystem'
app.config['CACHE_DIR'] = 'cache'
app.config['CACHE_DEFAULT_TIMEOUT'] = 300
app.config['CACHE_THRESHOLD'] = 100

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Set up caching
cache = Cache(app)

# Set up Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Simulated user database
users = {'admin': generate_password_hash('securepassword123')}

# Swagger UI for API documentation (Version 1.0)
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "HI Lava Kuyi Koka-koka API v1.0"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cache.memoize()
def predict():
    try:
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'Invalid data'}), 400
        prediction = predict_crash_point(data)
        return jsonify(prediction)
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500

@app.route('/fetch_data', methods=['GET'])
def fetch_data():
    try:
        api_key = os.getenv('ODDS_API_KEY', 'YOUR_365_ODDS_API_KEY')
        url = 'https://api.365oddsapi.com/betway/odds'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return jsonify(data)
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err}")
        return jsonify({'error': 'HTTP error occurred'}), 500
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Error occurred: {req_err}")
        return jsonify({'error': 'Error occurred'}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            user = User(username)
            login_user(user)
            return redirect(url_for('dashboard'))
        return 'Invalid credentials', 401
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.id == 'admin':
        return render_template('admin_dashboard.html')
    return render_template('dashboard.html')

@app.route('/prediction_history', methods=['GET'])
@login_required
def prediction_history():
    history = [
        {'crash_point': 2.5, 'timestamp': '2023-10-01T12:00:00Z'},
        {'crash_point': 3.1, 'timestamp': '2023-10-02T12:00:00Z'}
    ]
    return jsonify(history)

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': '1.0'})

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

def train_model():
    try:
        data = pd.read_csv('data/historical_data.csv')
        features = data[['feature1', 'feature2', 'feature3']].values
        labels = data['crash_point'].values
        features = features.reshape((features.shape[0], 1, features.shape[1]))
        
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(1, features.shape[2])))
        model.add(LSTM(50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(features, labels, epochs=10, batch_size=32, validation_split=0.2)
        model.save('models/aviator_prediction_model.h5')
        logging.info("Model trained and saved successfully")
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def predict_crash_point(data):
    try:
        model = load_model('models/aviator_prediction_model.h5')
        input_data = np.array(data['features']).reshape(1, 1, 3)
        prediction = model.predict(input_data)
        return {
            'crash_point': float(prediction[0][0]),
            'confidence': 0.95,
            'version': '1.0'
        }
    except Exception as e:
        logging.error(f"Error predicting crash point: {e}")
        raise

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    train_model()
    app.run(debug=True)
