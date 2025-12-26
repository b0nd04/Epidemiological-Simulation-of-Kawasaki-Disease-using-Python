import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os
print(os.getcwd())

# CSVファイルのパスを指定
file_path = "C:\\Users\\81802\\OneDrive - 独立行政法人 国立高等専門学校機構 (1)\\岐阜高専\\本科\\5E\\5E卒業研究\\backup\\2025_02_14(network SIR model一旦終わり)\\influenza_england_1978_school\\influenza_england_1978_school.csv"

# CSVファイルを読み込む
df_influenza = pd.read_csv(file_path)

# 'in_bed' 列のみを抽出
df_in_bed = df_influenza[['in_bed']]

# SIRモデルの微分方程式
def sir_model(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def calculation_parameter(lambda_0, r_0, N, I_0):
    gamma = lambda_0 / (r_0 - 1)
    beta = (gamma * r_0) / (N - I_0)
    return gamma, beta

# 初期条件
N = 763  # 総人口
I_0 = 2.671176  # 感染者の数
lambda_0 = 1.260234  # 初期成長率
r_0 = 3.758621    # 基本再生産数

R_0 = 0   # 回復者の数
S_0 = N - I_0 - R_0  # 感受性人口

t = np.linspace(0, 13, 14)

# 感染率と回復率の計算
gamma, beta = calculation_parameter(lambda_0, r_0, N, I_0)

print(f"感染率: {beta}")
print(f"回復率: {gamma}")

# 初期値設定
y_0 = S_0, I_0, R_0

# 微分方程式を解く
ret = odeint(sir_model, y_0, t, args=(beta, gamma, N))
S, I, R = ret.T

# df_in_bedのデータが対応する時間軸（例: 0日目から13日目の場合）
x_in_bed = np.arange(len(df_in_bed))  # 0, 1, ..., 13

# 結果をプロット
plt.figure()
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')

# df_in_bedのデータをプロット
plt.scatter(x_in_bed, df_in_bed['in_bed'], color='purple', label='In Bed', marker='o')

plt.title('SIR Model', fontsize=16)
plt.xlabel('Days', fontsize=16)
plt.ylabel('Population', fontsize=16)
plt.xlim(0, 14)  # x軸の範囲設定。元々0~16だったけど、プレゼン用に14に変更している
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.legend()
# plt.grid()

# グラフを画像として保存
plt.savefig('sir_model.png', bbox_inches='tight')  # 画像ファイル名を指定
plt.show()  # プロットを表示