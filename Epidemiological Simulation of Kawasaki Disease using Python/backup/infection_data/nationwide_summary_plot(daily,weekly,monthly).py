import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib
import matplotlib.lines as mlines  # 凡例用のラインオブジェクト作成に必要

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

# フィルタリングする期間を設定
start_date = pd.to_datetime('2011-01-01')
end_date   = pd.to_datetime('2022-12-31')
df_filtered = df_filtered[(df_filtered['diagnosis_date'] >= start_date) & (df_filtered['diagnosis_date'] <= end_date)]

# 集計方法のフラグ（いずれか1つをTrueに設定）
daily_aggregation   = False
weekly_aggregation  = False
monthly_aggregation = True

# 既存のカラーマップから47色を取得
colors = np.vstack([
    matplotlib.colormaps.get_cmap('tab20').colors, 
    matplotlib.colormaps.get_cmap('tab10').colors, 
    matplotlib.colormaps.get_cmap('Set3').colors
])
extra_colors = np.vstack([
    matplotlib.colormaps.get_cmap('Paired').colors,  # 12色
    matplotlib.colormaps.get_cmap('Accent').colors   # 8色
])
colors = np.vstack([colors, extra_colors])
colors = colors[:47]

# ※ 各分岐ごとに aggregated_counts と period_str を設定する
if daily_aggregation:
    # 日ごとの発症者数を集計
    daily_counts = df_filtered.groupby(['diagnosis_date', 'no_pref']).size().unstack(fill_value=0)
    aggregated_counts = daily_counts  # 凡例作成用に共通変数に保存
    aggregation_period = 'daily'
    
    plt.figure(figsize=(10, 6))
    for pref in daily_counts.columns:
        prefecture_int = int(pref) - 1
        plt.plot(daily_counts.index, daily_counts[pref], marker='o', linewidth=2, label=f'Prefecture {pref}', color=colors[prefecture_int])
    plt.title('Daily Infection Counts', fontsize=14)
    plt.xlabel('Dates', fontsize=12)
    plt.ylabel('Number of Infections', fontsize=12)
    plt.grid()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    plt.xlim(daily_counts.index.min(), daily_counts.index.max())
    plt.tight_layout()
    
    period_str = f"{aggregation_period}_counts_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    plt.savefig(period_str + '.png')
    plt.close()

elif weekly_aggregation:
    # 週ごとの発症者数を集計
    df_filtered['diagnosis_week'] = df_filtered['diagnosis_date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly_counts = df_filtered.groupby(['diagnosis_week', 'no_pref']).size().unstack(fill_value=0)
    aggregated_counts = weekly_counts  # 凡例作成用に共通変数に保存
    aggregation_period = 'weekly'
    
    plt.figure(figsize=(10, 6))
    for pref in weekly_counts.columns:
        prefecture_int = int(pref) - 1
        plt.plot(weekly_counts.index, weekly_counts[pref], marker='o', linewidth=2, label=f'Prefecture {pref}', color=colors[prefecture_int])
    plt.title('Weekly Infection Counts', fontsize=14)
    plt.xlabel('Weeks', fontsize=12)
    plt.ylabel('Number of Infections', fontsize=12)
    plt.grid()
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    plt.xlim(weekly_counts.index.min(), weekly_counts.index.max())
    plt.tight_layout()
    
    period_str = f"{aggregation_period}_counts_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    plt.savefig(period_str + '.png')
    plt.close()

elif monthly_aggregation:
    # 月ごとの発症者数を集計
    df_filtered['diagnosis_year_month'] = df_filtered['diagnosis_date'].dt.to_period('M')
    monthly_counts = df_filtered.groupby(['diagnosis_year_month', 'no_pref']).size().unstack(fill_value=0)
    aggregated_counts = monthly_counts  # 凡例作成用に共通変数に保存
    aggregation_period = 'monthly'
    
    # x軸用の月ごとのタイムスタンプとラベルを作成
    t_data = monthly_counts.index.to_timestamp()
    monthly_ticks = t_data  # x軸の目盛り位置
    monthly_labels = [d.strftime('%b %Y') for d in t_data]
    
    plt.figure(figsize=(10, 6))
    # 各列は都道府県番号（1～47）と仮定
    for pref in monthly_counts.columns:
        prefecture_int = int(pref) - 1  
        plt.plot(t_data, monthly_counts[pref], marker='o', linewidth=2, label=f'{pref}', color=colors[prefecture_int])
    
    plt.title(f"Monthly Infection Counts ({start_date.strftime('%Y-%m-%d')}~{end_date.strftime('%Y-%m-%d')})", fontsize=14)
    plt.xlabel('Months', fontsize=12)
    plt.ylabel('Number of Infections', fontsize=12)
    plt.grid()
    plt.xticks(monthly_ticks, monthly_labels, rotation=45)
    plt.xlim(t_data.min(), t_data.max())
    plt.tight_layout()
    
    period_str = f"{aggregation_period}_counts_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
    plt.savefig(period_str + '.png')
    plt.close()

# --------------------------
# 以下、凡例のみを別ファイルとして保存する共通コード
# --------------------------
fig_legend, ax_legend = plt.subplots(figsize=(10, 6))
ax_legend.axis('off')  # 軸は非表示

handles = []
labels = []
# aggregated_counts.columns は各都道府県の番号（1～47）と仮定
for pref in aggregated_counts.columns:
    prefecture_int = int(pref) - 1
    # プロット時と同じ marker, linestyle, 色 を設定
    line = mlines.Line2D([], [], color=colors[prefecture_int], marker='o', linestyle='-', linewidth=2)
    handles.append(line)
    labels.append("Prefecture " + str(pref))

# 複数列で凡例を配置（ここでは例として3列）
legend = fig_legend.legend(handles, labels, title="Prefecture", loc='center', ncol=3, fontsize=10, title_fontsize=12)

plt.tight_layout()
legend_file_name = f'legend ({period_str}).png'
plt.savefig(legend_file_name, bbox_inches='tight', dpi=300)
plt.close()
