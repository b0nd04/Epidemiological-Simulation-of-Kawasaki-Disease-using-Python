import pandas as pd
import numpy as np

file_path = '/home/karita/shibata_karita/2025/network SIR model/令和2年 従業・通学都道府県，男女，就業・通学別通勤者・通学者数 － 全国，都道府県（常住地）.csv'

commute_data = pd.read_csv(file_path, skiprows =14 )

filtered_commute_data = commute_data[commute_data['男女'] == '総数']

filtered_commute_data = filtered_commute_data[commute_data['就業・通学']== '総数']

filtered_commute_data = filtered_commute_data[filtered_commute_data['全国，都道府県（常住地）'] != '全国']

filtered_commute_data.rename(columns={'全国，都道府県（常住地）': '都道府県(常住地)'}, inplace=True)

filtered_commute_data = filtered_commute_data[['都道府県(常住地)' , '茨城県', '栃木県', '群馬県', '埼玉県', '千葉県', '東京都' , '神奈川県']]

filtered_commute_data = filtered_commute_data[filtered_commute_data['都道府県(常住地)'].isin(['東京都', '千葉県', '埼玉県', '神奈川県', '栃木県', '群馬県', '茨城県'])]

# 一致するセルのデータをNaNに置き換える
for column in ['茨城県', '栃木県', '群馬県', '埼玉県', '千葉県', '東京都', '神奈川県']:
    filtered_commute_data[column] = np.where(filtered_commute_data['都道府県(常住地)'] == column, np.nan, filtered_commute_data[column])

# カラムをfloat型に変換（NaNはそのまま）
for column in filtered_commute_data.columns:
    if column != '都道府県(常住地)':
        filtered_commute_data[column] = pd.to_numeric(filtered_commute_data[column].astype(str).str.replace(',', ''), errors='coerce')


# ------------------------------
# ここから都道府県間の移動人数を集計するコード
# ------------------------------

# 「都道府県(常住地)」をインデックスに設定
df = filtered_commute_data.set_index('都道府県(常住地)')

# 都道府県名のリストを取得（行名＝都道府県）
prefectures = df.index.tolist()

# 結果を格納するリストを初期化
result_data = []

# 各都道府県ペア（重複なし）について、両方向の移動人数を合計
for i in range(len(prefectures)):
    for j in range(i + 1, len(prefectures)):
        p1 = prefectures[i]
        p2 = prefectures[j]
        # p1からp2への移動人数（該当するセルが存在しない場合は 0）
        move_p1_to_p2 = df.loc[p1, p2] if p2 in df.columns else 0
        # p2からp1への移動人数
        move_p2_to_p1 = df.loc[p2, p1] if p1 in df.columns else 0
        total_movement = move_p1_to_p2 + move_p2_to_p1
        
        result_data.append({
            '都道府県1': p1,
            '都道府県2': p2,
            '移動人数': total_movement
        })

# 結果のDataFrameを作成
aggregated_df = pd.DataFrame(result_data)

# # CSVファイルとして出力
# output_file = 'aggregated_prefecture_movements.csv'
# aggregated_df.to_csv(output_file, index=False)