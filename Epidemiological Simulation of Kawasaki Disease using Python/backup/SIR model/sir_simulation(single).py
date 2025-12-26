import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# SIRモデルの微分方程式
def sir_model(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def calculation_parameter(lambda_0, r_0, N, I_0):
    gamma = lambda_0 / (r_0 - 1)
    beta = (gamma*r_0)/(N-I_0)
    return gamma , beta

# 初期条件
N = 352073  # 総人口
I_0 = 0.018109501945391855  # 感染者の数
lambda_0 = 4.296961157869274  # 初期成長率
r_0 = 1.0413793103448277    # 基本再生産数

R_0 = 0   # 回復者の数
S_0 = N - I_0 - R_0  # 感受性人口



t = np.linspace(0, 13, 14)


#感染率と回復率の計算
gamma , beta = calculation_parameter(lambda_0 , r_0 , N , I_0)

#初期値設定
y_0 = S_0, I_0, R_0

print(f"総人口N = {N}")
print(f"感染率 = {beta}")
print(f"回復率 = {gamma}")

# 微分方程式を解く
ret = odeint(sir_model, y_0, t, args=(beta, gamma, N))
S, I, R = ret.T

np.set_printoptions(suppress=True)  # 指数表現を無効にする
print(I)
print(R)

# 結果をプロット
plt.figure()
# plt.plot(t, S, 'b-', label='Susceptible')  # 感受性人口の折れ線グラフ
plt.plot(t, I, 'r-', label='Infected', marker = 'o')      # 感染者の折れ線グラフ
# plt.plot(t, R, 'g-', label='Recovered')     # 回復者の折れ線グラフ
plt.title('SIR Model')
plt.xlabel('Days')
plt.ylabel('Population')
# plt.xlim(0, 16)
plt.legend()
plt.grid()

# グラフを画像として保存
plt.savefig('sir_model.png',bbox_inches='tight')  # 画像ファイル名を指定
plt.close()