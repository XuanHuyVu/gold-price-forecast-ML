from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
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
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'])
    columns_to_convert = ['Price', 'Open', 'High', 'Low']
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col].replace({',': ''}, regex=True), errors='coerce')
    df['Vol.'] = pd.to_numeric(df['Vol.'].replace({',': '', 'K': ''}, regex=True), errors='coerce') * 1000
    df['Change %'] = pd.to_numeric(df['Change %'].replace({'%': ''}, regex=True), errors='coerce') / 100
    df = df.dropna().copy()
    df = df.sort_values('Date').copy()
    return df

df = preprocess_data(df)

# Use multiple features as input
X = df[['Open', 'High', 'Low', 'Vol.', 'Change %']].values
y = df['Price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression
def linear_regression(X, y):
    x = X[:, 0]  # Using only 'Open' for this example
    N = len(y)
    m = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x ** 2) - (np.sum(x) ** 2))
    b = np.mean(y) - (m * np.mean(x))
    return m, b

m, b = linear_regression(X_train, y_train)

def predict_gold_price_linear(inputs: np.ndarray) -> float:
    return m * inputs[0] + b  # Only using 'Open' for linear regression

# Lasso regression (sklearn)
lasso_model = Lasso(alpha=0.1, max_iter=1000)
lasso_model.fit(X_train, y_train)

def predict_gold_price_lasso(inputs: np.ndarray) -> float:
    return lasso_model.predict([inputs])[0]

# Neural Network
neural_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, activation='relu', solver='adam', random_state=42)
neural_model.fit(X_train, y_train)

def predict_gold_price_neural(inputs: np.ndarray) -> float:
    return neural_model.predict([inputs])[0]

# Bagging regression
bagging_model = BaggingRegressor(estimator=MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42), n_estimators=10, random_state=42)
bagging_model.fit(X_train, y_train)

def predict_gold_price_bagging(inputs: np.ndarray) -> float:
    return bagging_model.predict([inputs])[0]

# Combined prediction
def predict_gold_price_combined(inputs: np.ndarray) -> float:
    pred_linear = predict_gold_price_linear(inputs)
    pred_lasso = predict_gold_price_lasso(inputs)
    pred_neural = predict_gold_price_neural(inputs)
    return np.mean([pred_linear, pred_lasso, pred_neural])

# Metrics calculations
mse_linear = mean_squared_error(y_test, [predict_gold_price_linear(x) for x in X_test])
r2_linear = r2_score(y_test, [predict_gold_price_linear(x) for x in X_test])

mse_lasso = mean_squared_error(y_test, lasso_model.predict(X_test))
r2_lasso = r2_score(y_test, lasso_model.predict(X_test))

mse_nn = mean_squared_error(y_test, neural_model.predict(X_test))
r2_nn = r2_score(y_test, neural_model.predict(X_test))

mse_bagging = mean_squared_error(y_test, bagging_model.predict(X_test))
r2_bagging = r2_score(y_test, bagging_model.predict(X_test))

# Input model
class PredictionInput(BaseModel):
    open: float
    high: float
    low: float
    vol: float
    change_percentage: float

@app.post("/predict")
async def predict(input_data: PredictionInput):
    inputs = [input_data.open, input_data.high, input_data.low, input_data.vol, input_data.change_percentage]
    try:
        return {
            "Kết quả dự đoán": {
                "Giá vàng dự đoán theo Hồi quy tuyến tính": predict_gold_price_linear(inputs),
                "Giá vàng dự đoán theo Hồi quy Lasso": predict_gold_price_lasso(inputs),
                "Giá vàng dự đoán theo Neural Network (ReLU)": predict_gold_price_neural(inputs),
                "Giá vàng dự đoán theo Bagging": predict_gold_price_bagging(inputs),
                "Giá vàng dự đoán theo cách kết hợp": predict_gold_price_combined(inputs)
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
    </head>
    <body>
        <div class="container mt-5">
            <h2 class="text-center">Dự đoán giá vàng</h2>
            <form id="predictionForm" class="mt-4">
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
            <div id="result" class="mt-4"></div>
        </div>
        <script>
            async function predict() {
                const open = parseFloat(document.getElementById("open").value);
                const high = parseFloat(document.getElementById("high").value);
                const low = parseFloat(document.getElementById("low").value);
                const vol = parseFloat(document.getElementById("vol").value);
                const change_percentage = parseFloat(document.getElementById("change_percentage").value);
                const response = await fetch("/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ open: open, high: high, low: low, vol: vol, change_percentage: change_percentage })
                });
                const data = await response.json();
                const resultDiv = document.getElementById("result");
                if (response.ok) {
                    resultDiv.innerHTML = "<h4>Kết quả dự đoán:</h4>";
                    for (const [key, value] of Object.entries(data["Kết quả dự đoán"])) {
                        resultDiv.innerHTML += `<p>${key}: ${value.toFixed(6)}</p>`;
                    }
                    resultDiv.innerHTML += "<h4>MSE và R²:</h4>";
                    for (const [key, value] of Object.entries(data["MSE và R²"])) {
                        resultDiv.innerHTML += `<p>${key}: ${value.toFixed(6)}</p>`;
                    }
                } else {
                    resultDiv.innerHTML = `<p class="text-danger">Lỗi: ${data.detail}</p>`;
                }
            }
        </script>
    </body>
    </html>
    """
