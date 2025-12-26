import pandas as pd

# CSVファイルの読み込み
file_name = '/home/karita/shibata_karita/2025/infection_data/survey_25th_complete.csv' 
data = pd.read_csv(file_name)

# 平成の年を西暦に変換する関数
def convert_heisei_to_gregorian(year):
    return year + 1988

# 年を西暦に変換
data['西暦年'] = data['29'].apply(convert_heisei_to_gregorian)

# 月と日を整数に変換
data['月'] = data['1.5'].astype(int)
data['日'] = data['4'].astype(int)

# 無効な日付をフィルタリング
valid_dates = data[(data['月'] >= 1) & (data['月'] <= 12) & 
                   (data['日'] >= 1) & (data['日'] <= 31)]

# 発症日を日付型に変換
valid_dates['発症日'] = pd.to_datetime(valid_dates[['西暦年', '月', '日']].astype(str).agg('-'.join, axis=1), errors='coerce')

# 無効な日付がある行を削除
valid_dates = valid_dates.dropna(subset=['発症日'])

# 都道府県コードを使って、発症日でグループ化
grouped_data = valid_dates.groupby(['1.1', '発症日']).size().reset_index(name='発症者数')

# カラム名の変更
grouped_data = grouped_data.rename(columns={'1.1': '都道府県コード'})


# # 日ごとに集計
# daily_data = grouped_data.groupby(['都道府県コード', '発症日']).agg({'発症者数': 'sum'}).reset_index()
# # 週ごとに集計
# grouped_data['週'] = grouped_data['発症日'].dt.to_period('W').dt.start_time
# weekly_data = grouped_data.groupby(['都道府県コード', '週']).agg({'発症者数': 'sum'}).reset_index()
# 月ごとに集計
grouped_data['月'] = grouped_data['発症日'].dt.to_period('M').dt.start_time
monthly_data = grouped_data.groupby(['都道府県コード', '月']).agg({'発症者数': 'sum'}).reset_index()


# 抽出したい都道府県コードのリスト
target_codes = [8, 9, 10, 11, 12, 13, 14, 19]

# データをフィルタリング
filtered_data = monthly_data[monthly_data['都道府県コード'].isin(target_codes)]

# ピボットテーブルを作成
pivoted_data = filtered_data.pivot(index='月', columns='都道府県コード', values='発症者数').fillna(0)

# 2018年のデータのみをフィルタリング
pivoted_data_2018 = pivoted_data[pivoted_data.index.year == 2018]

# 結果を表示
print(pivoted_data_2018)
print(len(pivoted_data_2018))