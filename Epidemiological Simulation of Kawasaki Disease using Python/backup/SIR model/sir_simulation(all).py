import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines

# ----------0~4歳の各都道府県の人口を取得
file_path_p = '/home/karita/shibata_karita/2025/population/1985_Prefectural_Data_0-4.csv' 
result_p = pd.read_csv(file_path_p)

# '都道府県'列をfloat型に変換
result_p['都道府県コード'] = result_p['都道府県コード'].astype(float)

# # 結果を表示
# print(result_p)



# ----------各都道府県の発症データ取得
file_path_i = '/home/karita/shibata_karita/2025/infection_data/19851001_19860630_monthly_counts.csv' 
monthly_counts = pd.read_csv(file_path_i)

#発症データを保存する時に、はじめのdateのカラムを消す
extracted_row_filtered = monthly_counts.iloc[:, 1:]

# 都道府県コードをfloat型に変換
extracted_row_filtered.columns = extracted_row_filtered.columns.astype(float)

# # 結果を表示
# print(extracted_row_filtered)



# ----------パラメータサーベイによって求めた初期値データ取得
file_path_ps = '/home/karita/shibata_karita/2025/SIR model/parameter_survey/19851001_19860630_monthly_iteration(1000)_parameter_survey.csv' 
results_parameter_survey = pd.read_csv(file_path_ps)

# # 結果を表示
# print(results_parameter_survey)



#期間を指定
# 時間データを日付形式に変換
start_date = "1985-10-01"  # 開始日を指定
end_date = "1986-06-30"    # 終了日を指定

# 日付を加工
start_str = start_date.replace("-", "")  # '-'を取り除く
end_str = end_date.replace("-", "")        # '-'を取り除く

# 必要な部分を結合
period_str = f"{start_str}_{end_str}"

#Trueなら月ごと、Falseなら週ごとにデータを処理する
month_or_week = True

time_period_type = 'M' if month_or_week else 'W-MON'
month_or_week_str = 'monthly' if month_or_week else 'weekly'
Months_or_Weeks = 'Months' if month_or_week else 'Weeks'
Monthly_or_Weekly = 'Monthly' if month_or_week else 'Weekly'
period_str = period_str + '_' + month_or_week_str

date_range = pd.date_range(start=start_date, end=end_date, freq=time_period_type)
t_data = np.arange(len(date_range))

# 各月の初日を取得（x軸の目盛り用）
monthly_labels = pd.date_range(start=start_date, end=end_date, freq='MS')

# 各月の開始日に対応する `t_data` のインデックスを取得
monthly_ticks = [t_data[list(date_range).index(min(date_range, key=lambda d: abs(d - month)))] for month in monthly_labels]


# 既存のカラーマップから取得（42色）
colors = np.vstack([
    matplotlib.colormaps.get_cmap('tab20').colors, 
    matplotlib.colormaps.get_cmap('tab10').colors, 
    matplotlib.colormaps.get_cmap('Set3').colors
])
# 追加で'Paired'や'Accent'のカラーマップから補完
extra_colors = np.vstack([
    matplotlib.colormaps.get_cmap('Paired').colors,  # 12色
    matplotlib.colormaps.get_cmap('Accent').colors   # 8色
])
# すべての色を結合
colors = np.vstack([colors, extra_colors])
# 47色分確保
colors = colors[:47]




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



# 都道府県コードを設定
prefecture_codes = {
    '北海道': 1,'青森県': 2,'岩手県': 3,'宮城県': 4,'秋田県': 5,'山形県': 6,'福島県': 7,'茨城県': 8,'栃木県': 9,'群馬県': 10,'埼玉県': 11,'千葉県': 12,'東京都': 13,'神奈川県': 14,'新潟県': 15,
    '富山県': 16,'石川県': 17,'福井県': 18,'山梨県': 19,'長野県': 20,'岐阜県': 21,'静岡県': 22,'愛知県': 23,'三重県': 24,'滋賀県': 25,'京都府': 26,'大阪府': 27,'兵庫県': 28,'奈良県': 29,'和歌山県': 30,
    '鳥取県': 31,'島根県': 32,'岡山県': 33,'広島県': 34,'山口県': 35,'徳島県': 36,'香川県': 37,'愛媛県': 38,'高知県': 39,'福岡県': 40,'佐賀県': 41,'長崎県': 42,'熊本県': 43,'大分県': 44,
    '宮崎県': 45,'鹿児島県': 46,'沖縄県': 47,
}

# 辞書を逆にする
code_to_prefecture = {v: k for k, v in prefecture_codes.items()}

# 各都道府県の感染者数を保存するリスト
I_total = []

# 感染率と回復率データを格納するリスト
output_data = []

for prefecture in extracted_row_filtered.columns:
    I_actual = extracted_row_filtered[prefecture].values  # 実データ


    print(f"{code_to_prefecture[prefecture]}のsirモデルによるシミュレーション")

    N = result_p.loc[result_p['都道府県コード'] == prefecture, '０－４歳【人】'].values[0]
    lambda_0 = results_parameter_survey.loc[results_parameter_survey['都道府県コード'] == prefecture, 'λ_0'].values[0]
    I_0 = results_parameter_survey.loc[results_parameter_survey['都道府県コード'] == prefecture, 'I_0'].values[0]
    r_0 = results_parameter_survey.loc[results_parameter_survey['都道府県コード'] == prefecture, 'R_0'].values[0]
    R_0 = 0   # 回復者の数
    S_0 = N - I_0 - R_0  # 感受性人口
    
    #感染率と回復率の計算
    gamma , beta = calculation_parameter(lambda_0 , r_0 , N , I_0)

    # 保存するデータをリストに追加
    sigma_bar = results_parameter_survey.loc[results_parameter_survey['都道府県コード'] == prefecture, 'sigma_bar'].values[0]
    output_data.append({
        '都道府県コード': prefecture,
        'beta': beta,
        'gamma': gamma,
        'sigma_bar': sigma_bar
    })

    #初期値設定
    y_0 = S_0, I_0, R_0

    print(f"総人口N = {N}")
    print(f"感染率 = {beta}")
    print(f"回復率 = {gamma}")


    # 微分方程式を解く
    ret = odeint(sir_model, y_0, t_data, args=(beta, gamma, N))
    S, I, R = ret.T

    # 各都道府県の感染者数をリストに追加
    I_total.append(I)

    # 結果をプロット
    plt.figure(figsize=(10, 6))
    # plt.plot(t_data, S, 'b-', label='Susceptible')  # 感受性人口の折れ線グラフ
    plt.plot(t_data, I, 'r-', label='Infected', marker = 'o')      # 感染者の折れ線グラフ
    # plt.plot(t_data, R, 'g-', label='Recovered')     # 回復者の折れ線グラフ
    
    # print(f'I_actual: {len(I_actual)}')
    # print(f't_data: {len(t_data)}')

    # 発症データを追加
    plt.scatter(t_data, I_actual, color='blue', label='Reported Cases', zorder=5)

    graph_name = f'SIR Model : {prefecture}'
    plt.title(graph_name , fontsize=14)
    plt.xlabel(Months_or_Weeks, fontsize=12)
    plt.ylabel(f'{Monthly_or_Weekly} Infection Counts', fontsize=12)
    plt.legend()
    plt.grid()
    #X軸の目盛りを月ごとに設定（週ごとのデータを維持）
    plt.xticks(monthly_ticks, [date.strftime('%b %Y') for date in monthly_labels], rotation=45)
    plt.xlim(min(t_data), max(t_data))

    # グラフを画像として保存
    file_name = f'{code_to_prefecture[prefecture]}_{period_str}'
    plt.tight_layout()
    plt.savefig(file_name)  # 画像ファイル名を指定
    plt.close()
    

# すべての都道府県の感染者数を一つの図にまとめてプロット
plt.figure(figsize=(10, 6))
for I, prefecture in zip(I_total, extracted_row_filtered.columns):
    prefecture_int = int(prefecture) -1 #整数に変換
    plt.plot(t_data, I, label=prefecture , marker = 'o', color = colors[prefecture_int])  # 各都道府県の感染者数をプロット

plt.title('SIR Model - Total Infected by Prefecture', fontsize=14)
plt.xlabel(Months_or_Weeks, fontsize=12)
plt.ylabel(f'{Monthly_or_Weekly} Infection Counts', fontsize=12)
# plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
plt.grid()
#X軸の目盛りを月ごとに設定
plt.xticks(monthly_ticks, [date.strftime('%b %Y') for date in monthly_labels], rotation=45)
plt.xlim(min(t_data), max(t_data))
plt.tight_layout()
plt.savefig(f'all_{period_str}')  # まとめた図を保存
plt.close()




# --- まとめた図の保存後に追加するコード ---
# 凡例のみを描画するための新たな図と軸を作成
fig_legend, ax_legend = plt.subplots(figsize=(10, 6))
ax_legend.axis('off')  # 軸は非表示にする

# ダミーのラインオブジェクトを作成してハンドルとラベルのリストを用意
handles = []
labels = []
# extracted_row_filtered.columns は元々各都道府県のコード（float型）になっているので、
# 必要に応じて文字列に変換するなどの調整が可能
for prefecture in extracted_row_filtered.columns:
    prefecture_int = int(prefecture) - 1  # colorsリストのインデックスは0始まり
    # markerやlinestyleを実際のプロットに合わせる
    handles.append(mlines.Line2D([], [], color=colors[prefecture_int], marker='o', linestyle='-'))
    labels.append("Prefecture " + str(prefecture))

# 複数列で凡例を配置。
legend = fig_legend.legend(handles, labels, title="Prefecture",loc='center', ncol=3, fontsize=10, title_fontsize=12)

plt.tight_layout()
legend_file_name = f'legend (all_{period_str}).png'
plt.savefig(legend_file_name, bbox_inches='tight', dpi=300)
plt.close()




# 感染率と回復率をcsvファイルで保存
output_df = pd.DataFrame(output_data)
file_name2 = f"parameter_{period_str}.csv"
output_df.to_csv(file_name2, index=False, encoding='utf-8-sig')