import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.stats import linregress

# CSVファイルのパスを指定
file_path = '/home/karita/shibata_karita/2025/influenza_england_1978_school/influenza_england_1978_school.csv'

# CSVファイルを読み込む
df = pd.read_csv(file_path)

# 1月22日から1月24日のデータを選択
subset = df.iloc[:3]
t = np.arange(len(subset))  # 経過時間 t (0, 1, 2)
I_t = subset["in_bed"].values  # 感染者数

# 指数関数モデル
def exponential_model(t, I_0, lambda_0):
    return I_0 * np.exp(lambda_0 * t)

# 対数変換
log_I_t = np.log(I_t)

# 線形回帰を使用
slope, intercept, _, _, _ = linregress(t, log_I_t)
lambda_0 = slope
I_0 = np.exp(intercept)

print(f"推定された初期感染者数 I_0: {I_0}")
print(f"推定された初期成長率 λ_0: {lambda_0}")