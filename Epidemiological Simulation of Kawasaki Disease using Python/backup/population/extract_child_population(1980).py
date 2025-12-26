import pandas as pd

# CSVファイルの読み込み
file_path_p = '/home/karita/shibata_karita/2025/population/昭和55年(1980年) 年齢別人口(都道府県別).csv' 
data_p = pd.read_csv(file_path_p, skiprows=11, encoding='shift_jis')

# 「男女」カラムが「男女総数」の行のみ抽出
filtered_data_p = data_p[data_p['男女Ａ030001'] == '男女総数']

#このcsvファイルには、全域と集中地域の二つのデータがあるため、片方を抽出
zenn = '全域'
filtered_data_p = filtered_data_p[filtered_data_p['全域・集中の別030002'] == zenn]

# 必要なカラムのみ抽出
result_p = filtered_data_p[['市区町村２．９030005', '０−４歳【人】']]

# Rename
result_p = result_p.rename(columns={'市区町村２．９030005': '都道府県'})

print(result_p.columns)



# # 「都道府県名」が「全国」、「全国市部」、「全国郡部」の行を削除
# result_p = result_p[~result_p['都道府県'].isin(['全国', '全国市部', '全国郡部'])]

# 都道府県コードを設定
prefecture_codes = {
    '北海道': 1,'青森県': 2,'岩手県': 3,'宮城県': 4,'秋田県': 5,'山形県': 6,'福島県': 7,'茨城県': 8,'栃木県': 9,'群馬県': 10,'埼玉県': 11,'千葉県': 12,'東京都': 13,'神奈川県': 14,'新潟県': 15,
    '富山県': 16,'石川県': 17,'福井県': 18,'山梨県': 19,'長野県': 20,'岐阜県': 21,'静岡県': 22,'愛知県': 23,'三重県': 24,'滋賀県': 25,'京都府': 26,'大阪府': 27,'兵庫県': 28,'奈良県': 29,'和歌山県': 30,
    '鳥取県': 31,'島根県': 32,'岡山県': 33,'広島県': 34,'山口県': 35,'徳島県': 36,'香川県': 37,'愛媛県': 38,'高知県': 39,'福岡県': 40,'佐賀県': 41,'長崎県': 42,'熊本県': 43,'大分県': 44,
    '宮崎県': 45,'鹿児島県': 46,'沖縄県': 47,
}

# 都道府県名のリスト
prefecture_list = list(prefecture_codes.keys())

# 都道府県名のみ抽出
result_p = result_p[result_p['都道府県'].isin(prefecture_list)]

# 都道府県コードを適用
result_p['都道府県コード'] = result_p['都道府県'].map(prefecture_codes)

# カンマを削除して整数型に変換
result_p['０−４歳【人】'] = result_p['０−４歳【人】'].str.replace(',', '').astype(int)

# 結果を表示
print(result_p)

print(len(result_p))


# 結果を新しいCSVファイルに保存する場合
result_p.to_csv(f'1980_Prefectural_Data_0-4_{zenn}.csv', index=False)

# 確認
# file_path2 = '/home/karita/shibata_karita/2025/population/filtered_data.csv'  
# data2 = pd.read_csv(file_path2)
# print(data2)