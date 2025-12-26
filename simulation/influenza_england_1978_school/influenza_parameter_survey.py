import pandas as pd
import numpy as np
from scipy.integrate import odeint


# CSVファイルのパスを指定
file_path = '/home/karita/shibata_karita/2025/influenza_england_1978_school/influenza_england_1978_school.csv'

# CSVファイルを読み込む
df_influenza = pd.read_csv(file_path)

# 'in_bed' 列のみを抽出
df_in_bed = df_influenza[['in_bed']]

#初期値の設定
N=763
lambda_0 = 1.0797421246768601
I_0 = 2.902699980015461


# ----------パラメータサーベイ
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

# シミュレーションによる感染者数の推定
def simulate_sir(lambda_0, I_0, R_0, N, t_data):
    gamma , beta = calculation_parameter(lambda_0 , R_0 , N , I_0)

    # 初期値
    S_0 = N - I_0                 # 初期感受性者数
    R_0_initial = 0               # 初期回復者数
    y0 = [S_0, I_0, R_0_initial]

    # 時間範囲
    t = np.arange(len(t_data))

    # SIRモデルをシミュレーション
    result = odeint(sir_model, y0, t, args=(beta, gamma, N))
    S, I, R = result.T  # 結果の分解
    return I

# 残差の二乗平均 (σバー) を計算
def calculate_sigma_bar(I_actual, I_simulated):
    residual = I_actual - I_simulated
    sigma_bar = np.sqrt(np.mean(residual**2))
    return sigma_bar

# パラメータサーベイの実行 (動的停止条件付き)
def parameter_survey(t_data, I_actual, N, lambda_0_initial, I_0_initial, R_0_initial, max_iterations, tolerance=1e-6):
    lambda_0 = lambda_0_initial
    I_0 = I_0_initial
    R_0 = R_0_initial

    sigma_bar_prev = float('inf')  # 初期値を無限大に設定

    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")

        # Step 1: R_0 を変動させて最適化
        R_0_values = np.linspace(0.01, 3.0, 50)  # R_0 の探索範囲
        sigma_bar_values = []
        for R_0_candidate in R_0_values:
            I_simulated = simulate_sir(lambda_0, I_0, R_0_candidate, N, t_data)
            sigma_bar = calculate_sigma_bar(I_actual, I_simulated)
            sigma_bar_values.append(sigma_bar)
        R_0 = R_0_values[np.argmin(sigma_bar_values)]  # 最小値の R_0 を採用
        # print(f"  Optimized R_0: {R_0}")

        # Step 2: λ_0 を変動させて最適化
        lambda_0_values = np.linspace(lambda_0 * 0.9, lambda_0 * 1.1, 50)  # λ_0 の探索範囲
        sigma_bar_values = []
        for lambda_0_candidate in lambda_0_values:
            I_simulated = simulate_sir(lambda_0_candidate, I_0, R_0, N, t_data)
            sigma_bar = calculate_sigma_bar(I_actual, I_simulated)
            sigma_bar_values.append(sigma_bar)
        lambda_0 = lambda_0_values[np.argmin(sigma_bar_values)]  # 最小値の λ_0 を採用
        # print(f"  Optimized λ_0: {lambda_0}")

        # Step 3: I_0 を変動させて最適化
        I_0_values = np.linspace(I_0 * 0.9, I_0 * 1.1, 50)  # I_0 の探索範囲
        sigma_bar_values = []
        for I_0_candidate in I_0_values:
            I_simulated = simulate_sir(lambda_0, I_0_candidate, R_0, N, t_data)
            sigma_bar = calculate_sigma_bar(I_actual, I_simulated)
            sigma_bar_values.append(sigma_bar)
        I_0 = I_0_values[np.argmin(sigma_bar_values)]  # 最小値の I_0 を採用
        # print(f"  Optimized I_0: {I_0}")

        # 新しいσバーを計算
        I_simulated = simulate_sir(lambda_0, I_0, R_0, N, t_data)
        sigma_bar_current = calculate_sigma_bar(I_actual, I_simulated)
        print(sigma_bar_current)

        
        # 収束条件の確認
        if abs(sigma_bar_prev - sigma_bar_current) < tolerance:
            print(f"Convergence reached at iteration {iteration + 1}")
            break

        sigma_bar_prev = sigma_bar_current  # 前回の値を更新

    return lambda_0, I_0, R_0



# データを準備
t_data = np.arange(len(df_in_bed))  # 時間データ

# 結果を保存するためのリストを初期化
results_summary = []

# パラメータサーベイの繰り返し回数を指定
max_iterations = 1000


#最終的な残差の二条平均を初期化
sigma_bar_last = 0.0

for prefecture in df_in_bed.columns:
    I_actual = df_in_bed[prefecture].values  # 実データ

    if len(I_actual) < 2:
        print(f"都道府県 {prefecture} のデータが不足しています。")
        continue

    lambda_0_initial = lambda_0
    I_0_initial = I_0
    R_0_initial = 2  # 初期値

    print(f"{prefecture}のパラメータサーベイ")
    # パラメータサーベイを実行
    lambda_0, I0, R_0 = parameter_survey(t_data, I_actual, N, lambda_0_initial, I_0_initial, R_0_initial, max_iterations)

    #最終的な残差の二乗平均σバーを計算
    sigma_bar_last = calculate_sigma_bar(I_actual, simulate_sir(lambda_0, I0, R_0, N, t_data))

    # 結果を表示
    print(f"都道府県 {prefecture}: 最適化された λ_0: {lambda_0}, I_0: {I0}, R_0: {R_0}, I_0_actual: {I_actual[0]}, sigma_bar: {sigma_bar_last}")
    # 結果をリストに保存
    results_summary.append([prefecture, lambda_0, I0, R_0, I_actual[0], sigma_bar_last])  # sigma_bar_lastをリストに追加  # データフレーム形式で追加

# 結果をデータフレームに変換
results_df = pd.DataFrame(results_summary, columns=['都道府県コード', 'λ_0', 'I_0', 'R_0', 'I_0_actual', 'sigma_bar'])
print(results_df)


# # CSVファイルとして保存
# output_file_path = '/home/karita/shibata_karita/2025/influenza_england_1978_school/influenza_results_parameter_survey.csv'  # 保存したいファイルのパスを指定
# results_df.to_csv(output_file_path, index=False)  # index=Falseでインデックスを保存しない 