import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import os

# ===============================
# 1. パラメータCSVの読み込み
# ===============================
# ※　ここではカレントディレクトリにパラメータCSVファイルがある前提です。
param_csv = "/home/karita/shibata_karita/2025/network SIR model/results/19781201_19800131_monthly_iteration(1000)_network_parameter_survey.csv"  # 適宜パスを変更してください
params_df = pd.read_csv(param_csv)

# 各県のパラメータ（λ₀, I₀, r₀, K）を辞書形式に格納
params = {}
for idx, row in params_df.iterrows():
    pref = row["都道府県"]  # 例："茨城県", "栃木県", …
    params[pref] = {
        "lambda0": row["λ₀"],
        "I0": row["I₀"],
        "r0": row["r₀"],
        "K": row["K"]
    }
print("パラメータCSVを読み込みました。")
print(params)

# ===============================
# 2. 人口データの読み込み（0～4歳）
# ===============================
pop_file = '/home/karita/shibata_karita/2025/population/1980_Prefectural_Data_0-4_全域.csv'
pop_df = pd.read_csv(pop_file)
pop_df['都道府県コード'] = pop_df['都道府県コード'].astype(float)

# 関東7県の都道府県コード（8～14）
kanto_pref_codes = {
    "茨城県": 8,
    "栃木県": 9,
    "群馬県": 10,
    "埼玉県": 11,
    "千葉県": 12,
    "東京都": 13,
    "神奈川県": 14
}

# 各県の人口（0～4歳）を辞書に
populations = {}
for pref, code in kanto_pref_codes.items():
    # ※　列名は実際のCSVに合わせる。ここでは例として '０−４歳【人】' を使用
    pop_val = pop_df[pop_df['都道府県コード'] == code]['０−４歳【人】'].values[0]
    populations[pref] = pop_val
print("人口データを読み込みました：")
print(populations)

# ===============================
# 3. ネットワーク（通勤・流出入）データの読み込みと重みの計算
# ===============================
network_file = '/home/karita/shibata_karita/2025/network SIR model/kanto_commute_data(total).csv'
net_df = pd.read_csv(network_file, thousands=',')
net_df = net_df.fillna(0)

# 各都道府県の総移動人数を算出
all_prefectures = set(net_df['都道府県1']).union(set(net_df['都道府県2']))
total_flow = {
    pref: net_df.loc[(net_df['都道府県1'] == pref) | (net_df['都道府県2'] == pref), '移動人数'].sum()
    for pref in all_prefectures
}

# エッジの閾値（例：3%）を設定し、関東7県のみ対象とする
threshold = 0.03
neighbors_info = {pref: {} for pref in kanto_pref_codes.keys()}
for _, row in net_df.iterrows():
    pref1 = row['都道府県1']
    pref2 = row['都道府県2']
    movement = row['移動人数']
    if (pref1 in neighbors_info) and (pref2 in neighbors_info):
        if total_flow[pref1] > 0:
            w = movement / total_flow[pref1]
        else:
            w = 0
        if w > threshold:
            neighbors_info[pref1][pref2] = w
        # 逆方向
        if total_flow[pref2] > 0:
            w_rev = movement / total_flow[pref2]
        else:
            w_rev = 0
        if w_rev > threshold:
            neighbors_info[pref2][pref1] = w_rev
print("ネットワークの重み w_ij を計算しました。")
#print(neighbors_info)

# ===============================
# 4. ネットワークSIRモデルのシミュレーション関数
# ===============================
def calculation_parameter(lambda0, r0, N, I0):
    """
    与えられた λ₀, r₀, N, I₀ から γ, β を計算する。
    γ = λ₀/(r₀-1), β = (γ * r₀)/(N - I0)
    """
    gamma = lambda0 / (r0 - 1)
    beta = (gamma * r0) / (N - I0)
    return gamma, beta

def network_sir_ode(y, t, params, populations, nodes, neighbors_info):
    """
    各ノード i について
      dS_i/dt = -β_i S_i I_i - S_i * Σ_{j} (w_ij * K_i * I_j)
      dI_i/dt = β_i S_i I_i - γ_i I_i + S_i * Σ_{j} (w_ij * K_i * I_j)
      dR_i/dt = γ_i I_i
    としてシミュレーションする。
    """
    n = len(nodes)
    dydt = np.zeros(3*n)
    y_reshaped = y.reshape((n, 3))
    
    for i, node in enumerate(nodes):
        S = y_reshaped[i, 0]
        I = y_reshaped[i, 1]
        N_val = populations[node]
        lambda0 = params[node]['lambda0']
        r0 = params[node]['r0']
        I0 = params[node]['I0']
        gamma, beta = calculation_parameter(lambda0, r0, N_val, I0)
        # 局所感染
        local_infection = beta * S * I
        # 隣接ノードからの感染（重み付き）
        network_infection = 0.0
        for neighbor, w in neighbors_info[node].items():
            if neighbor in nodes:
                j = nodes.index(neighbor)
                I_neighbor = y_reshaped[j, 1]
                network_infection += w * I_neighbor
        # スケール調整係数 K を掛ける
        coupling_term = params[node]['K'] * S * network_infection
        
        dS = - local_infection - coupling_term
        dI = local_infection - gamma * I + coupling_term
        dR = gamma * I
        
        dydt[3*i]   = dS
        dydt[3*i+1] = dI
        dydt[3*i+2] = dR
        
    return dydt

def simulate_network_sir(params, populations, t, nodes, neighbors_info):
    """
    各ノードの初期条件 S0 = N - I0, I0, R0 = 0 として、
    odeint によりネットワークSIRモデルをシミュレーションする。
    戻り値は全時刻の状態ベクトルと、各ノードの感染者数 (I) の配列（時系列）。
    """
    n = len(nodes)
    y0 = np.zeros(3*n)
    for i, node in enumerate(nodes):
        I0 = params[node]['I0']
        N_val = populations[node]
        S0 = N_val - I0
        R0 = 0
        y0[3*i]   = S0
        y0[3*i+1] = I0
        y0[3*i+2] = R0
    sol = odeint(network_sir_ode, y0, t, args=(params, populations, nodes, neighbors_info))
    
    n_time = sol.shape[0]
    I_sim = np.zeros((n_time, n))
    for i in range(n):
        I_sim[:, i] = sol[:, 3*i+1]
    return sol, I_sim

# ===============================
# 5. シミュレーションの実行
# ===============================
# シミュレーション用の時間軸（例：0～100日、101ステップ）
t = np.linspace(0, 100, 101)

# ノードの順序は都道府県コードの昇順（例：["茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県"]）
nodes = sorted(kanto_pref_codes.keys(), key=lambda x: kanto_pref_codes[x])
print("シミュレーション対象ノード：", nodes)

# シミュレーション実行
sol, I_sim = simulate_network_sir(params, populations, t, nodes, neighbors_info)
print("シミュレーション完了。")

# ===============================
# 6. 結果のプロットと保存
# ===============================
plt.figure(figsize=(10, 6))
for i, node in enumerate(nodes):
    # 都道府県コードをラベルとして使用
    plt.plot(t, I_sim[:, i], label=str(kanto_pref_codes[node]), linewidth=2)

# plt.xlim(0, 14)
plt.xlabel("Months")
plt.ylabel("Infected Population")
plt.title("Network SIR Model Simulation")  # タイトルも設定
# plt.legend(title='都道府県コード')  # 凡例のタイトルを設定
plt.grid(True)
plt.tight_layout()

output_fig = "19781201_19800131_monthly_iteration(1000)network_SIR_simulation.png"
plt.savefig(output_fig, format="PNG")
print(f"シミュレーション結果グラフを {output_fig} に保存しました。")
plt.show()
