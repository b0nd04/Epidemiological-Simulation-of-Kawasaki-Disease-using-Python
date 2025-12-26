import matplotlib.pyplot as plt
import pandas as pd


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

# diagnosis_dateがNaTでないデータのみ抽出
df_filtered = df_filtered[df_filtered['diagnosis_date'].notnull()]

# diagnosis_dateから年（西暦）を抽出して、各年の症例数をカウント
cases_per_year = df_filtered.groupby(df_filtered['diagnosis_date'].dt.year).size()

print(cases_per_year)

# プロットの作成
plt.figure(figsize=(10, 6))
plt.plot(cases_per_year.index, cases_per_year.values, marker='o', linestyle='-')
plt.xlabel('Years')
plt.ylabel('Number of cases')
plt.xticks(cases_per_year.index , rotation= -90)  # x軸の目盛りを各西暦に設定
# plt.title('Annual Trend of Onset Cases')
plt.grid(True)

# グラフを画像として保存
plt.savefig('/home/karita/shibata_karita/2025/infection_data/annual_trend.png', format='png', dpi=300)

# プロットを表示
plt.show()
