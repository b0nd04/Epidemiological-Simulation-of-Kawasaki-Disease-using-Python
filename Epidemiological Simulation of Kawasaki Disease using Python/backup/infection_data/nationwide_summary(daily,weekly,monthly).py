import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルのパス
file_path = '/home/karita/shibata_karita/2025/infection_data/nationwide_survey.csv'

# ヘッダーを設定してCSVファイルを読み込む
column_names = ['no_survey', 'no_patient', 'no_facility_pref', 'no_facility', 'no_patient_facility', 'no_pref', 'no_city', 'gender', 'BD', 'date_firstvisit', 'days_until_diagnosis', 'diagnosis', 'death', 'age_in_days']
df = pd.read_csv(file_path, header=None, names=column_names)

# 'date_firstvisit'を日付型に変換
df['date_firstvisit'] = pd.to_datetime(df['date_firstvisit'], errors='coerce')

# 'days_until_diagnosis'を数値型に変換（nanを含む場合）
df['days_until_diagnosis'] = pd.to_numeric(df['days_until_diagnosis'], errors='coerce')

# 発症日を計算し、新しいカラムとして追加
df['diagnosis_date'] = df['date_firstvisit'] - pd.to_timedelta(df['days_until_diagnosis'], unit='d')

# no_prefが1から47の範囲にあるデータのみを抽出
df_filtered = df[df['no_pref'].between(1, 47)]

# フィルタリングする年と月を設定
start_date = pd.to_datetime('2011-01-01')
end_date = pd.to_datetime('2022-12-31')

# 年と月に基づいてフィルタリング
df_filtered = df_filtered[(df_filtered['diagnosis_date'] >= start_date) & (df_filtered['diagnosis_date'] <= end_date)]

# 集計方法を選択するためのフラグ
daily_aggregation = False
weekly_aggregation = False
monthly_aggregation = True

# 集計を行う
if daily_aggregation:
    # 日ごとの発症者数を集計
    daily_counts = df_filtered.groupby(['diagnosis_date', 'no_pref']).size().unstack(fill_value=0)
    aggregation_period = 'daily'
    
elif weekly_aggregation:
    # 週ごとの発症者数を集計
    df_filtered['diagnosis_week'] = df_filtered['diagnosis_date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_counts = df_filtered.groupby(['diagnosis_week', 'no_pref']).size().unstack(fill_value=0)
    aggregation_period = 'weekly'
    
elif monthly_aggregation:
    # 月ごとの発症者数を集計
    df_filtered['diagnosis_year_month'] = df_filtered['diagnosis_date'].dt.to_period('M')
    monthly_counts = df_filtered.groupby(['diagnosis_year_month', 'no_pref']).size().unstack(fill_value=0)
    aggregation_period = 'monthly'
    
# 結果を表示
if daily_aggregation:
    print(daily_counts)
elif weekly_aggregation:
    print(weekly_counts)
elif monthly_aggregation:
    print(monthly_counts)

# 集計結果をファイルに保存
file_name = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{aggregation_period}_counts.csv"
if weekly_aggregation:
    weekly_counts.to_csv(file_name)
elif daily_aggregation:
    daily_counts.to_csv(file_name)
elif monthly_aggregation:
    monthly_counts.to_csv(file_name)
