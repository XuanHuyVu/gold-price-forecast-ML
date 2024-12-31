from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fastapi import HTTPException
from sklearn.linear_model import Lasso
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

app = FastAPI()

# Read and preprocess the data
df = pd.read_csv("gold_price_data.csv")
df.columns = ["Date", "Price", "Open", "High", "Low", "Vol", "Change %"]

# Convert string values to numeric
def convert_price(x):
    return float(str(x).replace(',', ''))

def convert_volume(x):
    return float(str(x).replace('K', '')) * 1000

def convert_percentage(x):
    return float(str(x).replace('%', ''))

# Apply conversions
df['Price'] = df['Price'].apply(convert_price)
df['Open'] = df['Open'].apply(convert_price)
df['High'] = df['High'].apply(convert_price)
df['Low'] = df['Low'].apply(convert_price)
df['Vol'] = df['Vol'].apply(convert_volume)
df['Change %'] = df['Change %'].apply(convert_percentage)

# Create visualization
def create_pairplot():
    sns.pairplot(df[['Price', 'Open', 'High', 'Low', 'Vol', 'Change %']])
    plt.suptitle('Biểu đồ Pair Plot của các biến Price, Open, High, Low, Vol và Change %', y=1.0)
    plt.show()

# Tiền xử lý dữ liệu
def preprocess_data(df):
    df_processed = df.copy()
    df_processed.dropna(inplace=True)
    df_processed.drop(columns=['Date'], inplace=True)
    return df_processed

df_processed = preprocess_data(df)

# Prepare features and target
X = df_processed[['Open', 'High', 'Low', 'Vol', 'Change %']].values
y = df_processed['Price'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear regression implementation
def linear_regression(X, y):
    N = len(y)
    X_with_bias = np.column_stack([np.ones(N), X])
    coefficients = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
    return coefficients

# Get linear regression coefficients
linear_coeffs = linear_regression(X, y)

# Prediction functions
def predict_gold_price_linear(features):
    return np.array([1] + features) @ linear_coeffs

def predict_gold_price_combined(features):
    # Chuẩn hóa features đầu vào
    features_scaled = scaler.transform([features])
    
    pred_linear = predict_gold_price_linear(features)
    pred_lasso = lasso_model.predict(features_scaled)[0]
    pred_neural = neural_model.predict(features_scaled)[0]
    pred_bagging = bagging_model.predict(features_scaled)[0]
    return np.mean([pred_linear, pred_lasso, pred_neural, pred_bagging])

# Initialize and train models
lasso_model = Lasso(
    alpha=0.01,  # Giảm alpha để giảm regularization
    max_iter=10000,  # Tăng số lần lặp tối đa
    tol=0.0001,  # Điều chỉnh tolerance
    warm_start=True,  # Sử dụng warm start
    random_state=42
)
lasso_model.fit(X_scaled, y)

neural_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, activation='relu', 
                          solver='adam', random_state=42)
neural_model.fit(X, y)

bagging_model = BaggingRegressor(
    estimator=MLPRegressor(hidden_layer_sizes=(64,32), max_iter=1000, random_state=42),
    n_estimators=10,
    random_state=42
)
bagging_model.fit(X, y)

def predict_gold_price_combined(features):
    pred_linear = predict_gold_price_linear(features)
    pred_lasso = lasso_model.predict([features])[0]
    pred_neural = neural_model.predict([features])[0]
    pred_bagging = bagging_model.predict([features])[0]
    return np.mean([pred_linear, pred_lasso, pred_neural, pred_bagging])

class PredictionInput(BaseModel):
    open: float
    high: float
    low: float
    vol: float
    change: float

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        features = [
            input_data.open,
            input_data.high,
            input_data.low,
            input_data.vol,
            input_data.change
        ]
        
        # Chuẩn hóa features cho dự đoán
        features_scaled = scaler.transform([features])

        predicted_linear = predict_gold_price_linear(features)
        predicted_lasso = lasso_model.predict(features_scaled)[0]
        predicted_neural = neural_model.predict(features_scaled)[0]
        predicted_bagging = bagging_model.predict(features_scaled)[0]
        predicted_combined = predict_gold_price_combined(features)

        # Tính toán metrics với dữ liệu đã chuẩn hóa
        y_pred_linear = np.array([predict_gold_price_linear(x) for x in X])
        y_pred_lasso = lasso_model.predict(X_scaled)
        y_pred_neural = neural_model.predict(X_scaled)
        y_pred_bagging = bagging_model.predict(X_scaled)
        y_pred_combined = np.array([predict_gold_price_combined(x) for x in X])

        metrics = {
            "MSE và R²": {
                "MSE Hồi quy tuyến tính": mean_squared_error(y, y_pred_linear),
                "R² Hồi quy tuyến tính": r2_score(y, y_pred_linear),
                "MSE Hồi quy Lasso": mean_squared_error(y, y_pred_lasso),
                "R² Hồi quy Lasso": r2_score(y, y_pred_lasso),
                "MSE Neural Network": mean_squared_error(y, y_pred_neural),
                "R² Neural Network": r2_score(y, y_pred_neural),
                "MSE Bagging": mean_squared_error(y, y_pred_bagging),
                "R² Bagging": r2_score(y, y_pred_bagging),
                "MSE Kết hợp": mean_squared_error(y, y_pred_combined),
                "R² Kết hợp": r2_score(y, y_pred_combined)
            }
        }

        return {
            "Kết quả dự đoán": {
                "Giá vàng dự đoán theo Hồi quy tuyến tính": predicted_linear,
                "Giá vàng dự đoán theo Hồi quy Lasso": predicted_lasso,
                "Giá vàng dự đoán theo Neural Network": predicted_neural,
                "Giá vàng dự đoán theo Bagging": predicted_bagging,
                "Giá vàng dự đoán theo cách kết hợp": predicted_combined
            },
            **metrics
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
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                background-color: #f8f8f8;
            }
            .container {
                display: flex;
                justify-content: space-between;
                width: 90%;
                padding: 20px;
            }
            .form-container {
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                width: 48%;
            }
            .result-container {
                background-color: #f0f0f0;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                width: 48%;
                max-height: 80vh;
                overflow-y: auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="form-container">
                <h2 class="text-center">Dự đoán giá vàng</h2>
                <form id="predictionForm">
                    <div class="form-group">
                        <label for="open">Giá mở cửa:</label>
                        <input type="number" id="open" class="form-control" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label for="high">Giá cao nhất:</label>
                        <input type="number" id="high" class="form-control" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label for="low">Giá thấp nhất:</label>
                        <input type="number" id="low" class="form-control" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label for="vol">Khối lượng giao dịch:</label>
                        <input type="number" id="vol" class="form-control" step="0.01" required>
                    </div>
                    <div class="form-group">
                        <label for="change">Tỷ lệ thay đổi (%):</label>
                        <input type="number" id="change" class="form-control" step="0.01" required>
                    </div>
                    <button type="button" class="btn btn-primary" onclick="predict()">Dự đoán</button>
                </form>
            </div>
            <div class="result-container" id="result">
                <h4>Kết quả dự đoán:</h4>
            </div>
        </div>

        <script>
            async function predict() {
                const formData = {
                    open: parseFloat(document.getElementById("open").value),
                    high: parseFloat(document.getElementById("high").value),
                    low: parseFloat(document.getElementById("low").value),
                    vol: parseFloat(document.getElementById("vol").value),
                    change: parseFloat(document.getElementById("change").value)
                };

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json"
                        },
                        body: JSON.stringify(formData)
                    });
                    
                    const data = await response.json();
                    const resultDiv = document.getElementById("result");
                    resultDiv.innerHTML = "<h4>Kết quả dự đoán:</h4>";
                    
                    for (const [key, value] of Object.entries(data["Kết quả dự đoán"])) {
                        resultDiv.innerHTML += `<p>${key}: ${value.toFixed(2)}</p>`;
                    }
                    
                    resultDiv.innerHTML += "<h4>MSE và R²:</h4>";
                    for (const [key, value] of Object.entries(data["MSE và R²"])) {
                        resultDiv.innerHTML += `<p>${key}: ${value.toFixed(6)}</p>`;
                    }
                } catch (error) {
                    console.error('Error:', error);
                    document.getElementById("result").innerHTML = 
                        "<h4>Lỗi:</h4><p>Có lỗi xảy ra trong quá trình dự đoán.</p>";
                }
            }
        </script>
    </body>
    </html>
    """