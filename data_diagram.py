from re import X
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
import numpy as np

app = FastAPI()

# Đọc dữ liệu từ file CSV
df = pd.read_csv("gold_price_data.csv")
df.columns = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]

# Tiền xử lý dữ liệu
def preprocess_data(df):
    # Xóa các dòng chứa giá trị thiếu
    df = df.dropna(inplace=True)

    # Chuyển cột 'Date' sang kiểu dữ liệu datetime và sắp xếp dữ liệu theo thời gian
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

# Tiền xử lý dữ liệu
df = preprocess_data(df)

# Kiểm tra và in ra số lượng giá trị thiếu trong mỗi cột
print(df.isnull().sum())

# Kiểm tra kiểu dữ liệu của các cột
print(df.dtypes)

# Chuyển các giá trị trong cột Price, Open, High, Low từ chuỗi với dấu phẩy thành số
df['Price'] = df['Price'].replace({',': ''}, regex=True).astype(float)
df['Open'] = df['Open'].replace({',': ''}, regex=True).astype(float)
df['High'] = df['High'].replace({',': ''}, regex=True).astype(float)
df['Low'] = df['Low'].replace({',': ''}, regex=True).astype(float)

# Chuyển cột Vol. từ dạng chuỗi (ví dụ: '98.72K') thành số
df['Vol.'] = df['Vol.'].replace({',': '', 'K': ''}, regex=True).astype(float) * 1000

# Chuyển cột "Change %" từ dạng chuỗi (ví dụ: '0.06%') thành số
df['Change %'] = df['Change %'].replace({'%': ''}, regex=True).astype(float) / 100

# Kiểm tra kết quả
print(df.head())

# Kiểm tra lại dữ liệu sau khi chuyển đổi
print(df.dtypes)

# Tạo biểu đồ phân tán (pair plot)
sns.pairplot(df[['Price', 'Open', 'Vol.', 'High', 'Low']])

plt.suptitle('Biểu đồ Pair Plot của các biến price, open, high, low, vol', y=1.0)

plt.show()