import pandas as pd
import numpy as np
from scipy.integrate import odeint
import networkx as nx
import matplotlib.pyplot as plt
import copy

# =============================================================================
# 1. 人口データの読み込み（対象：0～4歳）
# =============================================================================
pop_file = '/home/karita/shibata_karita/2025/population/1980_Prefectural_Data_0-4_全域.csv'
pop_df = pd.read_csv(pop_file)
pop_df['都道府県コード'] = pop_df['都道府県コード'].astype(float)

# 関東のみ対象とする：各都道府県コード（8～14）
kanto_pref_codes = {
    "茨城県": 8,
    "栃木県": 9,
    "群馬県": 10,
    "埼玉県": 11,
    "千葉県": 12,
    "東京都": 13,
    "神奈川県": 14
}

# 関東各県の人口（0～4歳）を辞書に
populations = {}
for pref, code in kanto_pref_codes.items():
    pop_val = pop_df[pop_df['都道府県コード'] == code]['０−４歳【人】'].values[0]
    populations[pref] = pop_val
print("関東各県の人口（0～4歳）：", populations)


# =============================================================================
# 2. 発症データの読み込み（CSVの最初の列（date列）は除去）
# =============================================================================
infection_file = '/home/karita/shibata_karita/2025/infection_data/19781201_19800131_monthly_counts.csv'
inf_df = pd.read_csv(infection_file)
inf_data = inf_df.iloc[:, 1:]  # 最初のdate列を除く
inf_data.columns = inf_data.columns.astype(float)

# 関東（コード8～14）の列のみ抽出
kanto_codes = list(kanto_pref_codes.values())
inf_data_kanto = inf_data[[code for code in inf_data.columns if code in kanto_codes]]

# 各都道府県（prefecture名）ごとに発症データの時系列を辞書に
I_actual_dict = {}
for pref, code in kanto_pref_codes.items():
    I_actual_dict[pref] = inf_data_kanto[code].values
print("発症データ（関東）を取得しました。")


# =============================================================================
# 3. 初期パラメータの読み込み
# =============================================================================
# ここでは各県の初期λ₀, I₀をCSVから取得。r₀は初期値2.0、Kは初期値1.0とする。
param_file = '/home/karita/shibata_karita/2025/SIR model/lambda_0/19781201_19800131_monthly_sample(5)initial_lambda_0.csv'
param_df = pd.read_csv(param_file, index_col=0)

initial_params = {}
for pref, code in kanto_pref_codes.items():
    lambda0_val = param_df.loc[code, 'lambda_0']
    I0_val = param_df.loc[code, 'I_0']
    initial_params[pref] = {'lambda0': lambda0_val, 'I0': I0_val, 'r0': 2.0, 'K': 1.0}
print("初期パラメータを設定しました。")


# =============================================================================
# 4. ネットワーク（通勤・流出入）データの読み込みと各エッジの重みw_ijの算出
# =============================================================================
network_file = '/home/karita/shibata_karita/2025/network SIR model/kanto_commute_data(total).csv'
net_df = pd.read_csv(network_file, thousands=',')
net_df = net_df.fillna(0)

# 各都道府県ごとの総移動人数を算出
all_prefectures = set(net_df['都道府県1']).union(set(net_df['都道府県2']))
total_flow = {pref: net_df.loc[(net_df['都道府県1'] == pref) | (net_df['都道府県2'] == pref), '移動人数'].sum()
              for pref in all_prefectures}

# エッジの閾値（例：3%）を設定し、関東のみ対象とする
threshold = 0.03
neighbors_info = {pref: {} for pref in kanto_pref_codes.keys()}
for _, row in net_df.iterrows():
    pref1 = row['都道府県1']
    pref2 = row['都道府県2']
    movement = row['移動人数']
    # 両方とも関東の都道府県であれば
    if pref1 in neighbors_info and pref2 in neighbors_info:
        if total_flow[pref1] > 0:
            w = movement / total_flow[pref1]
        else:
            w = 0
        if w > threshold:
            neighbors_info[pref1][pref2] = w
        # 逆方向も同様に
        if total_flow[pref2] > 0:
            w_rev = movement / total_flow[pref2]
        else:
            w_rev = 0
        if w_rev > threshold:
            neighbors_info[pref2][pref1] = w_rev
print("ネットワークの重みw_ijを計算しました。")
#print(neighbors_info)  # 必要に応じて確認


# =============================================================================
# 5. ネットワークSIRモデルの定義
# =============================================================================
def calculation_parameter(lambda_0, r0, N, I0):
    """
    与えられたλ₀, r₀, N, I₀からγ, βを計算する。
    """
    gamma = lambda_0 / (r0 - 1)
    beta = (gamma * r0) / (N - I0)
    return gamma, beta

def network_sir_ode(y, t, params, populations, nodes, neighbors_info):
    """
    y: 状態ベクトル（全ノード分、各ノードは [S, I, R] の順）
    params: 各ノードごとのパラメータ辞書（λ₀, r₀, I₀, K）
    populations: 各ノードの総人口（0～4歳）
    nodes: ノード名のリスト（例：["茨城県", "栃木県", ...]）
    neighbors_info: 各ノードごとに {隣接ノード: w_ij} の辞書
    """
    n = len(nodes)
    dydt = np.zeros(3*n)
    y_reshaped = y.reshape((n, 3))
    for i, node in enumerate(nodes):
        S = y_reshaped[i, 0]
        I = y_reshaped[i, 1]
        # R = y_reshaped[i, 2]  # 回復者数（微分方程式内では感染・回復項で用いるのみ）
        N = populations[node]
        lambda0 = params[node]['lambda0']
        r0 = params[node]['r0']
        I0 = params[node]['I0']
        gamma, beta = calculation_parameter(lambda0, r0, N, I0)
        # 局所感染（ノード内）
        local_infection = beta * S * I
        # ネットワーク感染：隣接ノードからの感染
        network_infection = 0.0
        for neighbor, w_ij in neighbors_info[node].items():
            if neighbor in nodes:
                j = nodes.index(neighbor)
                I_neighbor = y_reshaped[j, 1]
                network_infection += w_ij * I_neighbor
        # スケール調整係数Kを掛ける
        coupling_term = params[node]['K'] * S * network_infection
        # 微分方程式
        dS = - local_infection - coupling_term
        dI = local_infection - gamma * I + coupling_term
        dR = gamma * I
        dydt[3*i]   = dS
        dydt[3*i+1] = dI
        dydt[3*i+2] = dR
    return dydt

def simulate_network_sir(params, populations, t_data, nodes, neighbors_info):
    """
    全ノード分の初期条件（S, I, R）を作成し、odeintによりネットワークSIRモデルをシミュレーションする。
    戻り値は各時間における各ノードのI（感染者数）の配列（shape: (time_steps, n_nodes)）。
    """
    n = len(nodes)
    y0 = np.zeros(3*n)
    for i, node in enumerate(nodes):
        I0 = params[node]['I0']
        N = populations[node]
        S0 = N - I0
        R0 = 0
        y0[3*i]   = S0
        y0[3*i+1] = I0
        y0[3*i+2] = R0
    t = np.arange(len(t_data))
    sol = odeint(network_sir_ode, y0, t, args=(params, populations, nodes, neighbors_info))
    n_time = sol.shape[0]
    I_simulated = np.zeros((n_time, n))
    for i in range(n):
        I_simulated[:, i] = sol[:, 3*i+1]
    return I_simulated

def calculate_sigma_bar(I_actual, I_simulated):
    """
    実測値とシミュレーション値の二乗平均平方根（σ_bar）を返す。
    """
    residual = I_actual - I_simulated
    sigma_bar = np.sqrt(np.mean(residual**2))
    return sigma_bar


# =============================================================================
# 6. ネットワークSIRモデルのパラメータサーベイ（座標降下法）
# =============================================================================
def parameter_survey_network(t_data, I_actual_dict, populations, nodes, neighbors_info, initial_params, max_iterations=1000, tolerance=1e-6):
    """
    各ノードごとに、パラメータ（λ₀, r₀, I₀, K）を座標降下的に最適化する。
    I_actual_dict: {ノード名: 実測発症データ（時系列）}
    戻り値は最適化されたパラメータの辞書と、各ノードごとの最終σ_bar。
    """
    params = copy.deepcopy(initial_params)
    sigma_bar_prev = {node: float('inf') for node in nodes}
    
    for iteration in range(max_iterations):
        print(f"Iteration {iteration+1}")
        # 各ノードについて順次最適化
        for node in nodes:
            # --- パラメータ r₀ の最適化 ---
            r0_current = params[node]['r0']
            r0_candidates = np.linspace(r0_current * 0.9, r0_current * 1.1, 30)
            best_r0 = r0_current
            best_sigma = float('inf')
            for candidate in r0_candidates:
                temp_params = copy.deepcopy(params)
                temp_params[node]['r0'] = candidate
                sim_result = simulate_network_sir(temp_params, populations, t_data, nodes, neighbors_info)
                node_index = nodes.index(node)
                sigma = calculate_sigma_bar(I_actual_dict[node], sim_result[:, node_index])
                if sigma < best_sigma:
                    best_sigma = sigma
                    best_r0 = candidate
            params[node]['r0'] = best_r0

            # --- パラメータ λ₀ の最適化 ---
            lambda_current = params[node]['lambda0']
            lambda_candidates = np.linspace(lambda_current * 0.9, lambda_current * 1.1, 30)
            best_lambda = lambda_current
            best_sigma = float('inf')
            for candidate in lambda_candidates:
                temp_params = copy.deepcopy(params)
                temp_params[node]['lambda0'] = candidate
                sim_result = simulate_network_sir(temp_params, populations, t_data, nodes, neighbors_info)
                node_index = nodes.index(node)
                sigma = calculate_sigma_bar(I_actual_dict[node], sim_result[:, node_index])
                if sigma < best_sigma:
                    best_sigma = sigma
                    best_lambda = candidate
            params[node]['lambda0'] = best_lambda

            # --- 初期感染者数 I₀ の最適化 ---
            I0_current = params[node]['I0']
            I0_candidates = np.linspace(I0_current * 0.9, I0_current * 1.1, 30)
            best_I0 = I0_current
            best_sigma = float('inf')
            for candidate in I0_candidates:
                temp_params = copy.deepcopy(params)
                temp_params[node]['I0'] = candidate
                sim_result = simulate_network_sir(temp_params, populations, t_data, nodes, neighbors_info)
                node_index = nodes.index(node)
                sigma = calculate_sigma_bar(I_actual_dict[node], sim_result[:, node_index])
                if sigma < best_sigma:
                    best_sigma = sigma
                    best_I0 = candidate
            params[node]['I0'] = best_I0

            # --- スケール調整係数 K の最適化 ---
            K_current = params[node]['K']
            K_candidates = np.linspace(K_current * 0.9, K_current * 1.1, 30)
            best_K = K_current
            best_sigma = float('inf')
            for candidate in K_candidates:
                temp_params = copy.deepcopy(params)
                temp_params[node]['K'] = candidate
                sim_result = simulate_network_sir(temp_params, populations, t_data, nodes, neighbors_info)
                node_index = nodes.index(node)
                sigma = calculate_sigma_bar(I_actual_dict[node], sim_result[:, node_index])
                if sigma < best_sigma:
                    best_sigma = sigma
                    best_K = candidate
            params[node]['K'] = best_K
        
        # 収束判定：全ノードのσ_barの変化が各 tolerance 以下なら終了
        sim_result = simulate_network_sir(params, populations, t_data, nodes, neighbors_info)
        converged = True
        for node in nodes:
            node_index = nodes.index(node)
            sigma = calculate_sigma_bar(I_actual_dict[node], sim_result[:, node_index])
            if abs(sigma - sigma_bar_prev[node]) > tolerance:
                converged = False
            sigma_bar_prev[node] = sigma
        if converged:
            print(f"Convergence reached at iteration {iteration+1}")
            break
            
    return params, sigma_bar_prev


# =============================================================================
# 7. シミュレーション用の時間データ（発症データの長さに合わせる）
# =============================================================================
t_data = np.arange(len(inf_data_kanto))

# =============================================================================
# 8. ノードの順序設定（都道府県コードの昇順に並べる）
# 例：["茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県"]
nodes = sorted(kanto_pref_codes.keys(), key=lambda x: kanto_pref_codes[x])
print("シミュレーション対象ノード：", nodes)


# =============================================================================
# 9. パラメータサーベイの実行
# =============================================================================
max_iterations = 2000
optimized_params, final_sigma = parameter_survey_network(t_data, I_actual_dict, populations, nodes, neighbors_info, initial_params, max_iterations=max_iterations)
print("パラメータサーベイ完了。")


# =============================================================================
# 10. 結果の表示・CSVファイルとして保存
# =============================================================================
results_list = []
for node in nodes:
    code = kanto_pref_codes[node]
    results_list.append([code, node,
                         optimized_params[node]['lambda0'],
                         optimized_params[node]['I0'],
                         optimized_params[node]['r0'],
                         optimized_params[node]['K'],
                         final_sigma[node]])
    print(f"都道府県 {node} (コード {code}): λ₀={optimized_params[node]['lambda0']}, I₀={optimized_params[node]['I0']}, r₀={optimized_params[node]['r0']}, K={optimized_params[node]['K']}, σ_bar={final_sigma[node]}")

results_df = pd.DataFrame(results_list, columns=['都道府県コード', '都道府県', 'λ₀', 'I₀', 'r₀', 'K', 'σ_bar'])
output_file_path = f'19781201_19800131_monthly_iteration({max_iterations})_network_parameter_survey.csv'
results_df.to_csv(output_file_path, index=False)
print("最適化結果をCSVに保存しました。")


# =============================================================================
# ※（任意）ネットワークグラフの描画（もともとのネットワーク構築コードに準拠）
# =============================================================================
# 関東のノードのみの無向グラフを作成
G = nx.Graph()
for pref in kanto_pref_codes.keys():
    G.add_node(pref, code=kanto_pref_codes[pref])
for pref, neighbors in neighbors_info.items():
    for neighbor, w in neighbors.items():
        # 両方ともグラフに存在している場合のみエッジを追加
        if pref in G.nodes and neighbor in G.nodes:
            # すでにエッジがあればスキップ（無向グラフ）
            if G.has_edge(pref, neighbor):
                continue
            # エッジ属性として移動人数（または重みwの値）を付与
            G.add_edge(pref, neighbor, weight=w)

# ノードの配置：円配置
pos = nx.circular_layout(G)
node_labels = {node: f"{data['code']}" for node, data in G.nodes(data=True)}

nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500)
nx.draw_networkx_edges(G, pos, width=2)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_family="sans-serif")

# エッジラベル（移動人数を表示）
edge_labels = {(u, v): f"{int(d['weight']*100)}%" for u, v, d in G.edges(data=True)}

def offset_edge_labels(pos, edge_labels, offset=0.05):
    """エッジラベルの表示位置を少しずらす"""
    new_pos = {}
    for (n1, n2), label in edge_labels.items():
        x1, y1 = pos[n1]
        x2, y2 = pos[n2]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2  # 中点
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        ux, uy = (-dy / length, dx / length)
        new_pos[(n1, n2)] = (mx + offset * ux, my + offset * uy)
    return new_pos

label_pos = offset_edge_labels(pos, edge_labels, offset=0.05)
for (n1, n2), (lx, ly) in label_pos.items():
    plt.text(lx, ly, edge_labels[(n1, n2)], fontsize=10, ha='center', va='center',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
plt.axis('off')
plt.tight_layout()
plt.savefig("network_graph.png", format="PNG")
plt.show()
