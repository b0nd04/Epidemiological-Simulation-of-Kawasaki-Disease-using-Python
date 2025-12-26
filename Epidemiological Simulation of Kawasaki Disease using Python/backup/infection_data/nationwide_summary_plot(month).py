import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.transforms as mtransforms
import pandas as pd

# --- 既存のデータ読み込み・加工部分 ---
# CSVファイルのパス
file_path = '/home/karita/shibata_karita/2025/infection_data/nationwide_survey.csv'
# ヘッダーを設定してCSVファイルを読み込む
column_names = [
    'no_survey', 'no_patient', 'no_facility_pref', 'no_facility',
    'no_patient_facility', 'no_pref', 'no_city', 'gender', 'BD',
    'date_firstvisit', 'days_until_diagnosis', 'diagnosis', 'death', 'age_in_days'
]
df = pd.read_csv(file_path, header=None, names=column_names)

# 'date_firstvisit'を日付型に変換
df['date_firstvisit'] = pd.to_datetime(df['date_firstvisit'], errors='coerce')

# 'days_until_diagnosis'を数値型に変換（nanを含む場合）
df['days_until_diagnosis'] = pd.to_numeric(df['days_until_diagnosis'], errors='coerce')

# 発症日を計算し、新しいカラムとして追加
df['diagnosis_date'] = df['date_firstvisit'] - pd.to_timedelta(df['days_until_diagnosis'], unit='d')

# no_prefが1から47の範囲にあるデータのみを抽出
df_filtered = df[df['no_pref'].between(1, 47)]

# --- ここから月ごとの集計とプロット ---
# diagnosis_dateがNaTでないデータのみ抽出
df_filtered = df_filtered[df_filtered['diagnosis_date'].notnull()]

# 2011年から2022年までのデータにフィルタリング
start_date = pd.to_datetime('2011-01-01')
end_date   = pd.to_datetime('2022-12-31')
df_filtered = df_filtered[(df_filtered['diagnosis_date'] >= start_date) & 
                          (df_filtered['diagnosis_date'] <= end_date)]

# diagnosis_dateを月単位でグループ化し、各月の症例数をカウント
cases_per_month = df_filtered.groupby(pd.Grouper(key='diagnosis_date', freq='M')).size()

# プロットの作成
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cases_per_month.index, cases_per_month.values, marker='o', linestyle='-')

# 軸ラベル、タイトルの設定
ax.set_xlabel('Years')
ax.set_ylabel('Number of cases')
# ax.set_title('Monthly Trend of Onset Cases (2011-2022)')

# x軸の目盛り設定
# 主要目盛り：各年ごとに表示（大きめのラベル）
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
# 補助目盛り：2か月ごとに表示（小さめのラベル）
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%-m'))

# 目盛りラベルのサイズ調整
ax.tick_params(axis='x', which='major', labelsize=10 , pad = 15)
ax.tick_params(axis='x', which='minor', labelsize=6)

# ラベルが重なりやすい場合は回転させる（必要に応じて）
plt.setp(ax.get_xticklabels(), ha='right')

# 主要目盛ラベルを5ポイント分左にずらす
for label in ax.get_xticklabels(which='major'):
    offset = mtransforms.ScaledTranslation(-8/72., 0, fig.dpi_scale_trans)
    label.set_transform(label.get_transform() + offset)

plt.grid(True)

# グラフを画像として保存
plt.savefig('/home/karita/shibata_karita/2025/infection_data/monthly_trend.png', format='png', dpi=300)

# グラフの表示
plt.show()