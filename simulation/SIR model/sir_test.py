import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# SIRモデルのパラメータ
beta = 0.3  # 感染率（一般的な値）
gamma = 0.1  # 回復率（一般的な値）

# 初期条件
S0 = 990  # 感受性のある人の数
I0 = 4    # 初期感染者の数
R0 = 0    # 初期回復者の数
N = S0 + I0 + R0  # 総人口

# 時間の設定
t = np.linspace(0, 160, 160)  # 0から160日までの時間

# SIRモデルの微分方程式
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N  # 感受性のある人の変化率
    dIdt = beta * S * I / N - gamma * I  # 感染者の変化率
    dRdt = gamma * I  # 回復者の変化率
    return dSdt, dIdt, dRdt

# 初期値の設定
y0 = S0, I0, R0

# 微分方程式を解く
ret = odeint(deriv, y0, t, args=(beta, gamma, N))
S, I, R = ret.T

# 結果をプロット
plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible (S)')  # 感受性のある人
plt.plot(t, I, 'r', label='Infected (I)')  # 感染者
plt.plot(t, R, 'g', label='Recovered (R)')  # 回復者
plt.title('SIR Model')  # タイトル
plt.xlabel('Time (days)')  # X軸ラベル
plt.ylabel('Number of individuals')  # Y軸ラベル
plt.legend()
plt.grid()

# 画像を保存
plt.savefig('sir_model.png', bbox_inches='tight')  # 画像ファイル名を指定
plt.show()