import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


# CSVファイルのパスを指定
file_path = '/home/karita/shibata_karita/2025/influenza_england_1978_school/influenza_england_1978_school.csv'

# CSVファイルを読み込む
df_influenza = pd.read_csv(file_path)

# 'in_bed' 列のみを抽出
df_in_bed = df_influenza[['in_bed']]

# 初期成長率λ_0を計算するのに使用する期間の指定
t_lambda_0 = 3

# 4行だけ抽出して別の変数に格納
extracted_rows = df_in_bed.head(t_lambda_0)  # 最初のn行を抽出


# λ_0を計算するためのモデル定義
def linear_model(t, lambda_0, log_I0):
    return lambda_0 * t + log_I0

def calculate_sigma_bar(I_actual, I_simulated):
    residual = I_actual - I_simulated
    sigma_bar = np.sqrt(np.mean(residual**2))
    return sigma_bar



# 都道府県ごとにλ_0を計算
results = {}
for prefecture in df_in_bed.columns:
    I_data = extracted_rows[prefecture].values  # 各都道府県の感染者数
    I_data = I_data.astype(int)
    t_data = np.arange(len(I_data))  # 時間データ (0, 1, 2, 3)

    # ゼロまたは負の値を除外
    valid_indices = np.where(I_data > 0)  # 正の値のインデックスを取得
    I_data = I_data[valid_indices]
    t_data = t_data[valid_indices]

    if len(I_data) < 2:  # フィッティングに必要なデータがない場合
        print(f" {prefecture} のフィッティングに十分なデータがありません。")
        continue

    # 自然対数を取る
    log_I_data = np.log(I_data)

    # 曲線フィッティング
    try:
        params, _ = curve_fit(linear_model, t_data, log_I_data)
        lambda_0 = params[0]  # 初期成長率
        log_I0 = params[1]  # 初期感染者数の対数
        results[prefecture] = {
            'lambda_0': lambda_0,
            'I_0': np.exp(log_I0)  # 初期感染者数
        }
    except Exception as e:
        print(f"{prefecture} のフィッティングでエラーが発生: {e}")

# 結果の表示
for prefecture, result in results.items():
    print(f"{prefecture}: 初期成長率 λ_0: {result['lambda_0']}, 初期感染者数 I_0: {result['I_0']}, 残差の二乗平均: {calculate_sigma_bar(extracted_rows['in_bed'] , result['I_0'])}")



# # 結果をCSVファイルに保存
# output_file_name = '19820215_19830331_weekly_initial_lambda_0.csv'
# results_df = pd.DataFrame.from_dict(results, orient='index')
# results_df.to_csv(output_file_name, encoding='utf-8')