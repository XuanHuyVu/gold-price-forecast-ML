from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = FastAPI()

# Load and preprocess the data
df = pd.read_csv("gold_price_data.csv")
df.columns = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]

def preprocess_data(df):
    df = df.dropna().copy()

    # Loại bỏ các ký tự không cần thiết (dấu ngoặc kép, khoảng trắng thừa)
    df['Date'] = df['Date'].str.replace('"', '', regex=False)
    df['Date'] = df['Date'].str.strip()

    # Chuyển đổi cột 'Date' thành datetime, với định dạng rõ ràng
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')  # Định dạng ngày tháng là mm/dd/yyyy
    except Exception as e:
        raise ValueError(f"Không thể chuyển đổi cột 'Date': {e}")

    # Kiểm tra nếu có giá trị NaT (Not a Time) trong cột 'Date' sau khi chuyển đổi
    if df['Date'].isnull().any():
        invalid_dates = df[df['Date'].isnull()]
        raise ValueError(f"Các giá trị không hợp lệ trong cột 'Date': {invalid_dates}")

    # Chuyển các cột số liệu thành kiểu số (numeric)
    columns_to_convert = ['Price', 'Open', 'High', 'Low']
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col].replace({',': ''}, regex=True), errors='coerce')
    df['Vol.'] = pd.to_numeric(df['Vol.'].replace({',': '', 'K': ''}, regex=True), errors='coerce') * 1000
    df['Change %'] = pd.to_numeric(df['Change %'].replace({'%': ''}, regex=True), errors='coerce') / 100

    df = df.dropna().copy()  # Xóa các hàng chứa giá trị NaN
    df['DayOfYear'] = df['Date'].dt.dayofyear  # Thêm cột 'DayOfYear' từ cột 'Date'
    df = df.sort_values('Date').copy()  # Sắp xếp theo ngày
    return df


df = preprocess_data(df)

# Features and target variables
X = df[['DayOfYear', 'Open', 'High', 'Low', 'Vol.', 'Change %']].values
y = df['Price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mô hình hồi quy tuyến tính
# Tính toán tham số m, b cho mô hình hồi quy tuyến tính
def linear_regression(X, y):
    x = X[:, 1] 
    N = len(y)
    m = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x ** 2) - (np.sum(x) ** 2))
    b = np.mean(y) - (m * np.mean(x))
    return m, b

# Lấy tham số m, b từ mô hình hồi quy tuyến tính
m, b = linear_regression(X_train, y_train)

# Hàm dự đoán giá vàng dựa trên mô hình hồi quy tuyến tính
def predict_gold_price_linear(inputs: np.ndarray) -> float:
    return m * inputs[1] + b

# Mô hình Lasso
lasso_model = Lasso(alpha=0.1, max_iter=10000)
lasso_model.fit(X_train, y_train)

def predict_gold_price_lasso(inputs: np.ndarray) -> float:
    return lasso_model.predict([inputs])[0]

# Mô hình Neural Network
neural_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=10000, activation='relu', solver='adam', random_state=42)
neural_model.fit(X_train, y_train)

def predict_gold_price_neural(inputs: np.ndarray) -> float:
    return neural_model.predict([inputs])[0]

# Mô hình Bagging
base_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=10000, activation='relu', solver='adam', random_state=42)
bagging_model = BaggingRegressor(estimator = base_model, n_estimators=10, random_state=42)
bagging_model.fit(X_train, y_train)

def predict_gold_price_bagging(inputs: np.ndarray) -> float: 
    return bagging_model.predict([inputs])[0] 

# Mô hình kết hợp
def predict_gold_price_combined(inputs: np.ndarray) -> float:
    pred_linear = predict_gold_price_linear(inputs)
    pred_lasso = predict_gold_price_lasso(inputs) 
    pred_neural = predict_gold_price_neural(inputs) 
    return np.mean([pred_linear, pred_lasso, pred_neural])

# Metrics calculations
mse_linear = mean_squared_error(y_test, [predict_gold_price_linear(x) for x in X_test]) # Using 'Open' for linear regression
r2_linear = r2_score(y_test, [predict_gold_price_linear(x) for x in X_test]) # Using 'Open' for linear regression

mse_lasso = mean_squared_error(y_test, lasso_model.predict(X_test)) # Using all features for Lasso
r2_lasso = r2_score(y_test, lasso_model.predict(X_test)) # Using all features for Lasso

mse_nn = mean_squared_error(y_test, neural_model.predict(X_test)) # Using all features for Neural Network
r2_nn = r2_score(y_test, neural_model.predict(X_test)) # Using all features for Neural Network

mse_bagging = mean_squared_error(y_test, bagging_model.predict(X_test))  # Using all features for Bagging
r2_bagging = r2_score(y_test, bagging_model.predict(X_test)) # Using all features for Bagging

# Input model for FastAPI
class PredictionInput(BaseModel):
    date: str # Format: YYYY-MM-DD
    open: float # Opening price
    high: float # Highest price
    low: float # Lowest price
    vol: float # Trading volume
    change_percentage: float # Change percentage

def preprocess_date(date: str) -> int:
    return datetime.strptime(date, "%Y-%m-%d").timetuple().tm_yday

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        day_of_year = preprocess_date(input_data.date)
        inputs = [day_of_year, input_data.open, input_data.high, input_data.low, input_data.vol, input_data.change_percentage]
        
        # Get predictions from all models
        pred_linear = predict_gold_price_linear(inputs)
        pred_lasso = predict_gold_price_lasso(inputs)
        pred_neural = predict_gold_price_neural(inputs)
        pred_bagging = predict_gold_price_bagging(inputs)
        pred_combined = predict_gold_price_combined(inputs)
        
        return {
            "Kết quả dự đoán": {
                "Giá vàng dự đoán theo Hồi quy tuyến tính": pred_linear,
                "Giá vàng dự đoán theo Lasso": pred_lasso,
                "Giá vàng dự đoán theo Neural Network (ReLU)": pred_neural,
                "Giá vàng dự đoán theo Bagging": pred_bagging,
                "Giá vàng dự đoán theo cách kết hợp": pred_combined
            },
            "MSE và R²": {
               "MSE Hồi quy tuyến tính": mse_linear,
                "R² Hồi quy tuyến tính": r2_linear,
                "MSE Hồi quy Lasso": mse_lasso,
                "R² Hồi quy Lasso": r2_lasso,
                "MSE Neural Network (ReLU)": mse_nn,
                "R² Neural Network (ReLU)": r2_nn,
                "MSE Bagging": mse_bagging,
                "R² Bagging": r2_bagging
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dự đoán giá vàng</title>
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <style>
            body {
                background-color: #f8f9fa;
            }
            .container {
                display: flex;
                justify-content: space-between;
            }
            .form-container, .result-container {
                width: 48%;
                padding: 20px;
                border-radius: 8px;
                background-color: white;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .form-container h3, .result-container h3 {
                text-align: center;
            }
            .gold-icon {
                width: 50px;
                height: 50px;
                margin-right: 10px;
            }
        </style>
    </head>
    <body>
        <div class="container mt-5">
            <div class="form-container">
                <h3>Nhập thông tin để dự đoán giá vàng</h3>
                <form id="predictionForm">
                    <div class="form-group">
                        <label for="date">Ngày (YYYY-MM-DD):</label>
                        <input type="date" id="date" class="form-control" required>
                    </div>
                    <div class="form-group">
                        <label for="open">Giá mở cửa:</label>
                        <input type="number" id="open" class="form-control" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="high">Giá cao nhất:</label>
                        <input type="number" id="high" class="form-control" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="low">Giá thấp nhất:</label>
                        <input type="number" id="low" class="form-control" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="vol">Khối lượng giao dịch:</label>
                        <input type="number" id="vol" class="form-control" step="any" required>
                    </div>
                    <div class="form-group">
                        <label for="change_percentage">Tỷ lệ thay đổi (%):</label>
                        <input type="number" id="change_percentage" class="form-control" step="any" required>
                    </div>
                    <button type="button" class="btn btn-primary" onclick="predict()">Dự đoán</button>
                </form>
            </div>
            <div class="result-container">
                <h3>Kết quả dự đoán</h3>
                <div id="result"></div>
            </div>
        </div>

        <script>
            async function predict() {
                const date = document.getElementById("date").value;
                const open = parseFloat(document.getElementById("open").value);
                const high = parseFloat(document.getElementById("high").value);
                const low = parseFloat(document.getElementById("low").value);
                const vol = parseFloat(document.getElementById("vol").value);
                const change_percentage = parseFloat(document.getElementById("change_percentage").value);

                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ date: date, open: open, high: high, low: low, vol: vol, change_percentage: change_percentage })
                });
                const data = await response.json();
                const resultDiv = document.getElementById("result");
                if (response.ok) {
                    resultDiv.innerHTML = "<h4>Kết quả dự đoán:</h4>";
                    for (const [key, value] of Object.entries(data["Kết quả dự đoán"])) {
                        resultDiv.innerHTML += `<p><strong>${key}:</strong> ${value.toFixed(2)}</p>`;
                    }
                    resultDiv.innerHTML += "<h4>MSE và R²:</h4>";
                    for (const [key, value] of Object.entries(data["MSE và R²"])) {
                        resultDiv.innerHTML += `<p><strong>${key}:</strong> ${value.toFixed(6)}</p>`;
                    }
                } else {
                    resultDiv.innerHTML = `<p class="text-danger">Lỗi: ${data.detail}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """