from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load and preprocess the data
df = pd.read_csv("gold_price_data.csv")
df.columns = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]

def preprocess_data(df):
    df = df.dropna().copy()

    # Clean unnecessary characters
    df['Date'] = df['Date'].str.replace('"', '').str.strip()

    # Convert Date column to datetime
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    except Exception as e:
        raise ValueError(f"Could not convert Date column: {e}")

    # Check for invalid dates
    if df['Date'].isnull().any():
        invalid_dates = df[df['Date'].isnull()]
        raise ValueError(f"Invalid dates found: {invalid_dates}")

    # Convert numeric columns
    numeric_columns = ['Price', 'Open', 'High', 'Low']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col].replace({',': ''}, regex=True), errors='coerce')
    
    df['Vol.'] = pd.to_numeric(df['Vol.'].replace({',': '', 'K': ''}, regex=True), errors='coerce') * 1000
    df['Change %'] = pd.to_numeric(df['Change %'].replace({'%': ''}, regex=True), errors='coerce') / 100

    df = df.dropna() # làm sạch lại dữ liệu
    df['DayOfYear'] = df['Date'].dt.dayofyear # thêm cột ngày trong năm
    return df.sort_values('Date')

# Preprocess data
df = preprocess_data(df)

# Prepare features and target
X = df[['DayOfYear', 'Open', 'High', 'Low', 'Vol.', 'Change %']].values
y = df['Price'].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train models
# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Lasso Regression
lasso_model = Lasso(alpha=0.1, max_iter=1000)
lasso_model.fit(X_train, y_train)

# Neural Network
neural_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, activation='relu', 
                          solver='adam', random_state=42)
neural_model.fit(X_train, y_train)

# Bagging
base_estimator = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
bagging_model = BaggingRegressor(estimator=base_estimator, n_estimators=10, random_state=42)
bagging_model.fit(X_train, y_train)

# Prediction functions
def predict_with_model(model, inputs):
    scaled_inputs = scaler.transform([inputs])
    return model.predict(scaled_inputs)[0]

# Combined prediction
def predict_combined(inputs):
    predictions = [
        predict_with_model(model, inputs)
        for model in [linear_model, lasso_model, neural_model, bagging_model]
    ]
    return np.mean(predictions)

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    return {
        'MSE': mean_squared_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred)
    }

# Input model
class PredictionInput(BaseModel):
    date: str
    open: float
    high: float
    low: float
    vol: float
    change_percentage: float

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Prepare input data
        day_of_year = datetime.strptime(input_data.date, "%Y-%m-%d").timetuple().tm_yday
        inputs = [
            day_of_year,
            input_data.open,
            input_data.high,
            input_data.low,
            input_data.vol,
            input_data.change_percentage
        ]

        # Get predictions
        predictions = {
            "Linear Regression": predict_with_model(linear_model, inputs),
            "Lasso": predict_with_model(lasso_model, inputs),
            "Neural Network": predict_with_model(neural_model, inputs),
            "Bagging": predict_with_model(bagging_model, inputs),
            "Combined": predict_combined(inputs)
        }

        # Calculate metrics for test set
        metrics = {
            "Linear Regression": calculate_metrics(y_test, linear_model.predict(X_test)),
            "Lasso": calculate_metrics(y_test, lasso_model.predict(X_test)),
            "Neural Network": calculate_metrics(y_test, neural_model.predict(X_test)),
            "Bagging": calculate_metrics(y_test, bagging_model.predict(X_test))
        }

        return {
            "predictions": predictions,
            "metrics": metrics
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# [Previous Python code remains the same until the HTML response]

@app.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dự Đoán Giá Vàng | Gold Price Prediction</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {
                --primary-color: #FFD700;
                --secondary-color: #C5AA6A;
                --dark-gold: #B8860B;
                --light-gold: #FFF8DC;
                --gradient-gold: linear-gradient(135deg, #FFD700, #FDB931);
            }
            
            body { 
                background-color: #f8f9fa;
                min-height: 100vh;
                background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                padding: 20px;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .container {
                max-width: 1200px;
                margin-top: 2rem;
            }
            
            .card {
                border: none;
                border-radius: 15px;
                box-shadow: 0 10px 20px rgba(0,0,0,0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                background: white;
                overflow: hidden;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 15px 30px rgba(0,0,0,0.15);
            }
            
            .card-header {
                background: var(--gradient-gold);
                color: #2C3E50;
                border-bottom: none;
                padding: 1.5rem;
                text-align: center;
                font-weight: bold;
            }
            
            .card-title {
                color: #2C3E50;
                font-size: 1.8rem;
                font-weight: 700;
                margin-bottom: 1.5rem;
            }
            
            .form-label {
                color: #2C3E50;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            
            .form-control {
                border-radius: 10px;
                border: 2px solid #e9ecef;
                padding: 0.75rem;
                transition: all 0.3s ease;
            }
            
            .form-control:focus {
                border-color: var(--primary-color);
                box-shadow: 0 0 0 0.2rem rgba(255, 215, 0, 0.25);
            }
            
            .btn-predict {
                background: var(--gradient-gold);
                border: none;
                color: #2C3E50;
                font-weight: bold;
                padding: 12px 25px;
                border-radius: 10px;
                transition: all 0.3s ease;
                width: 100%;
                margin-top: 1rem;
            }
            
            .btn-predict:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(255, 215, 0, 0.4);
            }
            
            .results {
                padding: 1.5rem;
            }
            
            .result-item {
                background: var(--light-gold);
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                border-left: 4px solid var(--dark-gold);
            }
            
            .result-value {
                font-size: 1.2rem;
                color: var(--dark-gold);
                font-weight: bold;
            }
            
            .metrics-card {
                background: #FFFFFF;
                border-radius: 10px;
                padding: 1rem;
                margin-bottom: 1rem;
                border: 1px solid #e9ecef;
            }
            
            .loading-spinner {
                display: none;
                text-align: center;
                padding: 2rem;
            }
            
            .gold-icon {
                color: var(--dark-gold);
                margin-right: 0.5rem;
            }

            .footer {
                margin-top: 3rem;
                padding: 1.5rem 0;
                background: var(--gradient-gold);
                border-radius: 15px;
                box-shadow: 0 -5px 15px rgba(0,0,0,0.1);
                text-align: center;
            }
            
            .footer-content {
                color: #2C3E50;
                font-weight: 500;
            }
            
            .footer-divider {
                width: 50%;
                margin: 1rem auto;
                border-top: 2px solid rgba(44, 62, 80, 0.2);
            }
            
            .social-links {
                margin-top: 1rem;
            }
            
            .social-links a {
                color: #2C3E50;
                margin: 0 10px;
                font-size: 1.2rem;
                transition: transform 0.3s ease;
                display: inline-block;
            }
            
            .social-links a:hover {
                transform: translateY(-3px);
            }
            
            /* Animation classes */
            .fade-in {
                animation: fadeIn 0.5s ease-in;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            /* Responsive adjustments */
            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                
                .card {
                    margin-bottom: 1rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Header -->
            <div class="text-center mb-5">
                <h1 class="display-4 fw-bold" style="color: var(--dark-gold);">
                    <i class="fas fa-coins gold-icon"></i>
                    Dự Đoán Giá Vàng
                </h1>
                <p class="lead text-muted">Hệ thống dự đoán giá vàng sử dụng Machine Learning</p>
            </div>
            
            <div class="row g-4">
                <!-- Form Card -->
                <div class="col-md-6">
                    <div class="card h-100">
                        <div class="card-header">
                            <h3 class="mb-0">
                                <i class="fas fa-chart-line gold-icon"></i>
                                Nhập Thông Tin
                            </h3>
                        </div>
                        <div class="card-body">
                            <form id="predictionForm">
                                <div class="mb-3">
                                    <label for="date" class="form-label">
                                        <i class="far fa-calendar-alt gold-icon"></i>
                                        Ngày
                                    </label>
                                    <input type="date" class="form-control" id="date" required>
                                </div>
                                <div class="mb-3">
                                    <label for="open" class="form-label">
                                        <i class="fas fa-door-open gold-icon"></i>
                                        Giá Mở Cửa
                                    </label>
                                    <input type="number" class="form-control" id="open" step="0.01" required>
                                </div>
                                <div class="mb-3">
                                    <label for="high" class="form-label">
                                        <i class="fas fa-arrow-up gold-icon"></i>
                                        Giá Cao Nhất
                                    </label>
                                    <input type="number" class="form-control" id="high" step="0.01" required>
                                </div>
                                <div class="mb-3">
                                    <label for="low" class="form-label">
                                        <i class="fas fa-arrow-down gold-icon"></i>
                                        Giá Thấp Nhất
                                    </label>
                                    <input type="number" class="form-control" id="low" step="0.01" required>
                                </div>
                                <div class="mb-3">
                                    <label for="vol" class="form-label">
                                        <i class="fas fa-chart-bar gold-icon"></i>
                                        Khối Lượng Giao Dịch
                                    </label>
                                    <input type="number" class="form-control" id="vol" required>
                                </div>
                                <div class="mb-3">
                                    <label for="change_percentage" class="form-label">
                                        <i class="fas fa-percentage gold-icon"></i>
                                        Tỷ Lệ Thay Đổi (%)
                                    </label>
                                    <input type="number" class="form-control" id="change_percentage" step="0.01" required>
                                </div>
                                <button type="button" class="btn btn-predict" onclick="predict()">
                                    <i class="fas fa-search-dollar"></i>
                                    Dự Đoán
                                </button>
                            </form>
                        </div>
                    </div>
                </div>
                
                <!-- Results Card -->
                <div class="col-md-6">
                    <div class="card h-100">
                        <div class="card-header">
                            <h3 class="mb-0">
                                <i class="fas fa-chart-pie gold-icon"></i>
                                Kết Quả Dự Đoán
                            </h3>
                        </div>
                        <div class="card-body">
                            <div id="loading-spinner" class="loading-spinner">
                                <div class="spinner-border text-warning" role="status">
                                    <span class="visually-hidden">Đang tính toán...</span>
                                </div>
                                <p class="mt-2">Đang xử lý dữ liệu...</p>
                            </div>
                            <div id="result"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <!-- Footer Section -->
            <footer class="footer">
                <div class="container">
                    <div class="footer-content">
                        <div class="row">
                            <div class="col-12">
                                <i class="fas fa-coins gold-icon"></i>
                                <h4 class="mb-2">Hệ Thống Dự Đoán Giá Vàng</h4>
                                <div class="footer-divider"></div>
                                <p class="mb-2">© 2025 Gold Price Prediction System. All rights reserved.</p>
                                <p class="mb-3">Developed by <strong>64HTTT1-10</strong></p>
                                <div class="social-links">
                                    <a href="#" title="Facebook"><i class="fab fa-facebook"></i></a>
                                    <a href="#" title="Twitter"><i class="fab fa-twitter"></i></a>
                                    <a href="#" title="LinkedIn"><i class="fab fa-linkedin"></i></a>
                                    <a href="#" title="GitHub"><i class="fab fa-github"></i></a>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </footer>
        </div>

        <script>
            async function predict() {
                const loadingSpinner = document.getElementById('loading-spinner');
                const resultDiv = document.getElementById('result');
                
                // Show loading spinner
                loadingSpinner.style.display = 'block';
                resultDiv.innerHTML = '';
                
                const inputData = {
                    date: document.getElementById('date').value,
                    open: parseFloat(document.getElementById('open').value),
                    high: parseFloat(document.getElementById('high').value),
                    low: parseFloat(document.getElementById('low').value),
                    vol: parseFloat(document.getElementById('vol').value),
                    change_percentage: parseFloat(document.getElementById('change_percentage').value)
                };

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(inputData)
                    });

                    const data = await response.json();
                    
                    // Hide loading spinner
                    loadingSpinner.style.display = 'none';
                    
                    if (response.ok) {
                        let html = '<div class="results fade-in">';
                        
                        // Predictions section
                        html += '<h4 class="mb-4"><i class="fas fa-chart-line gold-icon"></i>Dự Đoán:</h4>';
                        for (const [model, value] of Object.entries(data.predictions)) {
                            html += `
                                <div class="result-item">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <strong>${model}:</strong>
                                        <span class="result-value">${value.toFixed(2)}</span>
                                    </div>
                                </div>`;
                        }

                        // Metrics section
                        html += '<h4 class="mt-4 mb-3"><i class="fas fa-calculator gold-icon"></i>Độ Chính Xác:</h4>';
                        for (const [model, metrics] of Object.entries(data.metrics)) {
                            html += `
                                <div class="metrics-card">
                                    <h5 class="text-primary mb-3">${model}</h5>
                                    <div class="row">
                                        <div class="col-6">
                                            <small class="text-muted">MSE:</small>
                                            <div class="fw-bold">${metrics.MSE.toFixed(6)}</div>
                                        </div>
                                        <div class="col-6">
                                            <small class="text-muted">R²:</small>
                                            <div class="fw-bold">${metrics['R²'].toFixed(6)}</div>
                                        </div>
                                    </div>
                                </div>`;
                        }
                        
                        html += '</div>';
                        resultDiv.innerHTML = html;
                    } else {
                        resultDiv.innerHTML = `
                            <div class="alert alert-danger fade-in">
                                <i class="fas fa-exclamation-triangle me-2"></i>
                                Lỗi: ${data.detail}
                            </div>`;
                    }
                } catch (error) {
                    loadingSpinner.style.display = 'none';
                    resultDiv.innerHTML = `
                        <div class="alert alert-danger fade-in">
                            <i class="fas fa-exclamation-triangle me-2"></i>
                            Lỗi: ${error.message}
                        </div>`;
                }
            }
        </script>
    </body>
    </html>
    """