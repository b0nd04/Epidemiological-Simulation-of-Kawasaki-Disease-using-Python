import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# --- 1. CSVファイルの読み込み ---
df = pd.read_csv('/home/karita/shibata_karita/2025/network SIR model/kanto_commute_data(total).csv', thousands=',')
df = df.fillna(0)

# --- 2. 各都道府県の総移動人数の算出 ---
prefectures = set(df['都道府県1']).union(set(df['都道府県2']))
total_flow = {pref: df.loc[(df['都道府県1'] == pref) | (df['都道府県2'] == pref), '移動人数'].sum()
              for pref in prefectures}

# --- 3. 都道府県コードの定義 ---
pref_codes = {
    "茨城県": 8,
    "栃木県": 9,
    "群馬県": 10,
    "埼玉県": 11,
    "千葉県": 12,
    "東京都": 13,
    "神奈川県": 14
}

# 無向グラフの作成
G = nx.Graph()
for pref in prefectures:
    if pref in pref_codes:
        G.add_node(pref, code=pref_codes[pref])

# 閾値設定（3%）
threshold = 0.03
for _, row in df.iterrows():
    pref1, pref2, movement = row['都道府県1'], row['都道府県2'], row['移動人数']
    
    ratio1 = movement / total_flow[pref1] if total_flow[pref1] > 0 else 0
    ratio2 = movement / total_flow[pref2] if total_flow[pref2] > 0 else 0
    
    if (ratio1 > threshold) or (ratio2 > threshold):
        G.add_edge(pref1, pref2, weight=movement)

# --- 4. グラフの描画 ---
pos = nx.circular_layout(G)

# ノードラベル（都道府県コード）
node_labels = {node: f"{data['code']}" for node, data in G.nodes(data=True)}

nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500)
nx.draw_networkx_edges(G, pos, width=2)
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_family="sans-serif")

# --- 5. エッジラベルのオフセット調整 ---
edge_labels = {(u, v): f"{int(d['weight'])}" for u, v, d in G.edges(data=True)}

def offset_edge_labels(pos, edge_labels, offset=0.1):
    """エッジのラベルを適切な位置にオフセットする"""
    new_pos = {}
    for (n1, n2), label in edge_labels.items():
        x1, y1 = pos[n1]
        x2, y2 = pos[n2]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2  # エッジの中点
        dx, dy = x2 - x1, y2 - y1  # エッジの方向
        length = np.sqrt(dx**2 + dy**2)
        ux, uy = (-dy / length, dx / length)  # エッジに直交する単位ベクトル

        new_pos[(n1, n2)] = (mx + offset * ux, my + offset * uy)  # 少しずらした位置に配置
    return new_pos

label_pos = offset_edge_labels(pos, edge_labels, offset=0.05)

for (n1, n2), (lx, ly) in label_pos.items():
    plt.text(lx, ly, edge_labels[(n1, n2)],fontsize=10, ha='center', va='center',bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)) 

# plt.title("network_graph")
plt.axis('off')
plt.tight_layout()
plt.savefig("network_graph.png", format="PNG")
plt.show()




# 画像として保存
plt.savefig("network_graph.png", format="PNG")
plt.show()