#!/usr/bin/env python
# coding: utf-8

# <span style="font-size: 25px;">1、データの整理

# In[116]:


import numpy as np
import pandas as pd
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import folium
import matplotlib.pyplot as plt


# In[117]:


file_path = r'/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/13_Tokyo_20182_20232.csv'
data = pd.read_csv(file_path, encoding='cp932')
df=data


# In[118]:


#df[df['Math'] != -1]
# 「地域」列が「商業地」を含む行を削除
df = df[df['地域'] != '商業地']

# 「種類」列が「宅地(土地)」を含む行を削除
df = df[df['種類'] != '宅地(土地)']

# 「種類」列が「林地」を含む行を削除
df = df[df['種類'] != '林地']
df = df[df['種類'] != '農地']

# 「種類」列が「宅地(土地と建物)」を含む行を削除
df = df[df['種類'] != '宅地(土地と建物)']

# 結果を新しいCSVファイルに保存する（オリジナルのCSVを上書きしたくない場合）
df.to_csv('mansion data 201802-202302.csv', index=False)


# In[119]:


df_mansion = df


# In[120]:


print(df_mansion)


# In[121]:


new_column_names = {'取引価格（総額）': 'total_price'}
df_mansion.rename(columns=new_column_names, inplace=True)


# In[122]:


new_column_names = {'建ぺい率（％）': 'building_percentage'}
df_mansion.rename(columns=new_column_names, inplace=True)


# In[123]:


new_column_names = {'市区町村コード': 'city_code'}
df_mansion.rename(columns=new_column_names, inplace=True)


# In[124]:


new_column_names = {'最寄駅：距離（分）': 'nearest station distance_minutes'}
df_mansion.rename(columns=new_column_names, inplace=True)


# In[125]:


new_column_names = {'容積率（％）': 'volume_percentage'}
df_mansion.rename(columns=new_column_names, inplace=True)


# In[126]:


new_column_names = {'最寄駅：名称': 'nearest station'}
df_mansion.rename(columns=new_column_names, inplace=True)


# In[127]:


new_column_names = {'面積（㎡）': 'area_square'}
df_mansion.rename(columns=new_column_names, inplace=True)
print(df_mansion)


# In[128]:


#"建築年"列の和暦から西暦への変換

import re

def convert_to_seireki(wareki_str):
    if pd.isnull(wareki_str):  # NoneまたはNaNの場合はそのまま返す
        return wareki_str
    wareki_str = str(wareki_str)  # float型のNaNを文字列に変換
    era = wareki_str[:2]  # 元号部分を取得（例：昭和、平成、令和）
    year_match = re.search(r'(\d+)', wareki_str)  # 和暦の年数部分を正規表現で検索
    if year_match:
        year = int(year_match.group())
        if era == '昭和':
            seireki_year = 1925 + year
        elif era == '平成':
            seireki_year = 1988 + year
        elif era == '令和':
            seireki_year = 2018 + year
        else:
            seireki_year = None
    else:
        seireki_year = None
    return seireki_year

# "建築年"列の和暦から西暦への変換を適用
df_mansion['建築年（西暦）'] = df_mansion['建築年'].apply(convert_to_seireki)


# In[129]:


df_mansion['建築年（西暦）'].median()


# In[130]:


# 「建築年（西暦）」列の欠損値を2005で補完
df_mansion['建築年（西暦）'].fillna(2003, inplace=True)


# In[131]:


# 「建築年」列が数値型でない場合は変換
if not pd.api.types.is_numeric_dtype(df_mansion['建築年']):
    df_mansion['建築年'] = df_mansion['建築年'].apply(convert_to_seireki)

# 建築年数を計算して新しい列「建築年数」(architectural year)を追加
df_mansion['architectural year'] = 2023 - df_mansion['建築年（西暦）']

# 結果を表示
print(df_mansion)


# In[132]:


df_mansion.drop(['No','地域', '都道府県名','坪単価','取引価格（㎡単価）', '土地の形状', '間口', '延床面積（㎡）', '用途','前面道路：方位', '前面道路：種類', '改装', '取引の事情等', '前面道路：幅員（ｍ）'],         axis=1, inplace = True)


# In[133]:


df_mansion['total_price'] = pd.to_numeric(df_mansion['total_price'], errors='coerce')
df_mansion['area_square'] = pd.to_numeric(df_mansion['area_square'], errors='coerce')


# 新しい列を計算して追加
df_mansion['average_price_per_sqm'] = df_mansion['total_price'] / df_mansion['area_square']


# In[134]:


# 列「Unit price/ square meter」に欠損値が含まれているか確認
if df_mansion['average_price_per_sqm'].isnull().any():
    print("列 'average_price_per_sqm' に欠損値があります。")
else:
    print("列 'average_price_per_sqm' に欠損値はありません。")


# In[135]:


# 各列ごとに欠損値を特定する
missing_values = df_mansion.isnull()

# 各列の欠損値の合計を表示する
print("欠損値の合計:")
print(missing_values.sum())


# In[136]:


# 'average_price_per_sqm' 列の欠損値を含む行を削除
df_cleaned = df_mansion.dropna(subset=['average_price_per_sqm'])


# In[137]:


# 結果を新しいCSVファイルに保存する
df_mansion.to_csv('new manshion deta.csv', index=False)


# <span style="font-size: 25px;">２、取引件数と平均平米単価の地図を作る

# In[138]:


import geopandas as gpd


# In[139]:


# グループごとに平米単価の平均値を計算、10000で割る
average_prices = df_mansion.groupby('city_code')['average_price_per_sqm'].mean() / 10000
average_prices = average_prices.reset_index()
average_prices.columns = ['city_code', 'average_prices']

# 追加する行を含む新しいデータフレームを作成
new_rows = pd.DataFrame({
    'city_code': [13305, 13307, 13308],
    'average_prices': [0, 0, 0]
})

# 新しい行を既存のデータフレームに追加
average_prices = pd.concat([average_prices, new_rows], ignore_index=True)

# 結果を表示
print(average_prices)


# In[140]:


# グループごとに市区町村コードの出現回数を計算
city_code_counts = df_mansion['city_code'].value_counts().reset_index()
city_code_counts.columns = ['city_code', 'number of occurrences']

# 追加する行を含む新しいデータフレームを作成
new_rows = pd.DataFrame({
    'city_code': [13305, 13307, 13308],
    'number of occurrences': [0, 0, 0]
})

# 新しい行を既存のデータフレームに追加
city_code_counts = pd.concat([city_code_counts, new_rows], ignore_index=True)

print(city_code_counts)


# In[141]:


#平均取引価格の最大値と最小値を確認する
print(average_prices['average_prices'].min())
print(average_prices['average_prices'].max())


# In[142]:


import json


# In[143]:


import folium
# 地図の中心座標を指定
tokyo23_location = [35.6895, 139.6917]  # 東京都庁舎の座標を使用

# Folium Map オブジェクトを作成
m = folium.Map(location=tokyo23_location, zoom_start=10)


# In[144]:


# GeoJSONファイルのパスを指定
geojson = '/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/tokyo 1.json'


# In[145]:


# 地図を作成
m = folium.Map(location=tokyo23_location, tiles='cartodbpositron', zoom_start=10)


# In[146]:


#　平均平米単価のマップを作る
# Choroplethマップを追加
folium.Choropleth(
    geo_data=geojson,
    data=average_prices,  # 市区町村コードごとの平均平米単価を使用
    columns=['city_code', 'average_prices'],
    key_on='feature.properties.N03_007',  # GeoJSONの市区町村コードに対応するプロパティ名
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='平均平米単価（1万円単位）',
    threshold_scale=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
).add_to(m)

# 地図をhtmlファイルとして保存
m.save('map_average_prices.html')


# In[147]:


#取引件数の最大値と最小値を確認する
print(city_code_counts['number of occurrences'].min())
print(city_code_counts['number of occurrences'].max())


# In[148]:


import folium
# 地図の中心座標を指定
tokyo23_location = [35.6895, 139.6917]  # 東京都庁舎の座標を使用

# Folium Map オブジェクトを作成
m = folium.Map(location=tokyo23_location, zoom_start=10)


# In[149]:


# GeoJSONファイルのパスを指定
geojson = '/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/tokyo 1.json'


# In[150]:


# 地図を作成
m = folium.Map(location=tokyo23_location, tiles='cartodbpositron', zoom_start=10)


# In[151]:


# 取引件数のマップを作成
folium.Choropleth(
    geo_data=geojson,
    data=city_code_counts,  # 市区町村コードごとの取引件数を使用
    columns=['city_code', 'number of occurrences'],
    key_on='feature.properties.N03_007',  # GeoJSONの市区町村コードに対応するプロパティ名
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='取引件数',
    threshold_scale=[0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]
).add_to(m)

# 地図をhtmlファイルとして保存
m.save('map_number_of_occurrences.html')


# <span style="font-size: 25px;">３、市区町村ごとの取引件数と平均取引価格のヒストグラムを作る

# In[152]:


# グループごとに平米単価の平均値を計算、10000で割る
average_prices_1 = df_mansion.groupby('市区町村名')['average_price_per_sqm'].mean() / 10000
average_prices_1 = average_prices_1.reset_index()
average_prices_1.columns = ['市区町村名', 'average_price_per_sqm']
print(average_prices_1)


# In[153]:


import matplotlib.pyplot as plt
import japanize_matplotlib

# フォントの指定
plt.rcParams['font.family'] = 'IPAexGothic'

# ヒストグラムの描画
sorted_data = average_prices_1.sort_values(by='average_price_per_sqm', ascending=False)
plt.figure(figsize=(12, 6))
plt.bar(sorted_data['市区町村名'], sorted_data['average_price_per_sqm'], color='skyblue')
plt.xlabel('市区町村名')
plt.ylabel('平均平方単価（1万円単位）')
plt.title('各市区町村の平均平方単価ヒストグラム（高い順）')
plt.xticks(rotation=90)
plt.tight_layout()

# 背景を白色に変更
plt.gca().set_facecolor('white')

# ファイルとして保存（カレントディレクトリに保存）
plt.savefig('histogram_Unit price.png')

# プロットを表示
plt.show()


# In[154]:


# グループごとに市区町村コードの出現回数を計算
city_code_counts = df_mansion['市区町村名'].value_counts().reset_index()
city_code_counts.columns = ['市区町村名', 'number of occurrences']
print(city_code_counts)


# In[155]:


#各市区町村の取引件数のヒストグラムを作る

# フォントの指定
plt.rcParams['font.family'] = 'IPAexGothic'

# ヒストグラムの描画
sorted_data = city_code_counts.sort_values(by='number of occurrences', ascending=False)
plt.figure(figsize=(12, 6))
plt.bar(sorted_data['市区町村名'], sorted_data['number of occurrences'], color='skyblue')
plt.xlabel('市区町村名')
plt.ylabel('取引件数')
plt.title('各市区町村の取引件数ヒストグラム（高い順）')
plt.xticks(rotation=90)
plt.tight_layout()

# 背景を白色に変更
plt.gca().set_facecolor('white')

# ファイルとして保存（カレントディレクトリに保存）
plt.savefig('histogram_number of occurrences.png')

# プロットを表示
plt.show()


# <span style="font-size: 25px;">４、最寄駅ごとの取引件数と平均平米単価を求める

# In[88]:


# 欠損値が含まれている行を削除
df_mansion.dropna(subset=['nearest station'], inplace=True)


# In[89]:


#バックテストでこの四つの駅名は地理情報を取得できないので、変更する
def replace_station_name(station_name):
    if station_name == '早稲田(メトロ)':
        return '早稲田'
    elif station_name == '浅草(つくばＥＸＰ)' or station_name == '浅草(東武・都営・メトロ)':
        return '浅草'
    elif station_name == '雑司が谷(東京メトロ)':
        return '雑司ヶ谷'
    else:
        return station_name

df_mansion['nearest station'] = df_mansion['nearest station'].apply(replace_station_name)


# In[90]:


# 重複を省略して一意の値と出現回数を取得
value_counts = df_mansion['nearest station'].value_counts()

# 一意の値と出現回数を表示
for value, count in zip(value_counts.index, value_counts.values):
    print(f'{value}: {count}回')


# In[91]:


# 各駅ごとに総価格と総平米数を集計
grouped = df_mansion.groupby('nearest station').agg(
    total_price_sum=('total_price', 'sum'),
    area_square_sum=('area_square', 'sum'),
    count=('nearest station', 'count')
)

# 平均平米単価を計算し、10000円単位に調整
grouped['average_price_per_sqm'] = (grouped['total_price_sum'] / grouped['area_square_sum']) / 10000

# 最終的なデータフレームを作成（一意の駅名、出現回数、平均平米単価を含む）
result_df = grouped[['count', 'average_price_per_sqm']].reset_index()

# データフレームをCSVファイルとして保存
result_df.to_csv('eki_result.csv', index=False)


# In[92]:


#駅ごとの平均取引価格の最大値と最小値を確認する
print(result_df['average_price_per_sqm'].min())
print(result_df['average_price_per_sqm'].max())


# In[93]:


#駅ごとの取引回数の最大値と最小値を確認する
print(result_df['count'].min())
print(result_df['count'].max())


# <span style="font-size: 25px;">５、最寄駅ごとの地理情報を確認し、マップを作る

# In[94]:


# 特定の列の名前を指定
column_name = 'nearest station'  


# In[95]:


from geopy.geocoders import Nominatim

# CSVファイル読み込み
file_path = '/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/eki_result.csv'  
df = pd.read_csv(file_path)

# GeopyのNominatimジオコーダーを初期化
geolocator = Nominatim(user_agent="eki_geocoder")

# 結果を格納するリスト
results = []
# 取得できなかった駅のリスト
failed_locations = []

# 「nearest station」列の各駅名に対して位置情報を取得
for station_name in df['nearest station']:
    # 位置情報を取得
    location = geolocator.geocode(station_name + ', 日本', timeout=10)
    
    if location:
        # 結果をリストに追加
        results.append([station_name, location.latitude, location.longitude, location.address])
        # 取得した結果をprint
        print(f'駅名: {station_name}, 緯度: {location.latitude}, 経度: {location.longitude}, 住所: {location.address}')
    else:
        # 取得できなかった駅の情報をリストに追加
        failed_locations.append(station_name)
        print(f'位置情報が取得できませんでした: {station_name}')

# 取得できなかった駅のリストがfailed_locationsに格納されます
print('取得できなかった駅のリスト:', failed_locations)


# In[99]:


# 結果をDataFrameに変換
result_df = pd.DataFrame(results, columns=['nearest station', '緯度', '経度', '住所'])
result_df.to_csv('result_location.csv', index=False)

# 結果を表示
print(result_df)


# In[100]:


#駅名の重複で、東京以外と識別された駅名を抽出する
import geopandas as gpd

# 地理情報データを読み込む
station_data = gpd.read_file('/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/result_location.csv')

# 東京都内の座標範囲 (例: 東京都庁舎の座標を中心とした半径2度の正方形)
tokyo_center_latitude = 35.6895
tokyo_center_longitude = 139.6917
tokyo_radius_deg = 2

# 東京都内の座標範囲
tokyo_min_latitude = tokyo_center_latitude - tokyo_radius_deg
tokyo_max_latitude = tokyo_center_latitude + tokyo_radius_deg
tokyo_min_longitude = tokyo_center_longitude - tokyo_radius_deg
tokyo_max_longitude = tokyo_center_longitude + tokyo_radius_deg

# '緯度' カラムを数値に変換
station_data['緯度'] = station_data['緯度'].astype(float)

# '経度' カラムを数値に変換
station_data['経度'] = station_data['経度'].astype(float)

# 東京都外の駅を抽出
non_tokyo_stations = station_data[
    (station_data['緯度'] < tokyo_min_latitude) |
    (station_data['緯度'] > tokyo_max_latitude) |
    (station_data['経度'] < tokyo_min_longitude) |
    (station_data['経度'] > tokyo_max_longitude)
]

# 東京以外の駅とその座標を抽出
non_tokyo_coordinates = non_tokyo_stations.groupby('nearest station').apply(lambda group: group.iloc[0][['緯度', '経度']])

# 結果を表示
print(non_tokyo_coordinates)


# In[101]:


# 経緯度を正しく修正する

# CSVファイルを読み込む
df = pd.read_csv('/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/result_location.csv')

# 修正する条件を指定（ここでは 'nearest station' が特定の駅名の場合）
stations_to_modify = ['三田(東京)', '上石神井', '八坂(東京)', '学習院下', '富士見台', '小竹向原', '御嶽山', '梅ケ丘', '沢井', '浜町', '湯島', '玉川上水', '築地市場', '高田馬場']

# 新しい緯度と経度の値
new_latitudes = [35.6344, 35.7281, 35.7449, 35.7191, 35.7358, 35.7434, 35.5852, 35.6560, 35.8059, 35.6884, 35.7084, 35.7316, 35.6647, 35.7126]
new_longitudes = [139.7335, 139.5918, 139.4677, 139.7079, 139.6297, 139.6794, 139.6824, 139.6537, 139.1934, 139.7879, 139.7693, 139.4186, 139.7669, 139.7039]

# 各行に対して修正を行う
for station_name, new_latitude, new_longitude in zip(stations_to_modify, new_latitudes, new_longitudes):
    condition = df['nearest station'] == station_name
    df.loc[condition, ['緯度', '経度']] = [new_latitude, new_longitude]

# 修正をファイルに保存
df.to_csv('result_location_new.csv', index=False)


# In[102]:


#駅ごとの取引回数の最大値と最小値を確認する

# CSVファイルから不動産取引データを読み込む
file_path = r'/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/eki_result.csv'
eki_data = pd.read_csv(file_path)

# 駅ごとの取引回数の最大値と最小値を確認
print(eki_data['count'].min())
print(eki_data['count'].max())


# In[120]:


#最寄駅ごとの取引量のマップを作ります

import folium
import geopandas as gpd
import pandas as pd

# 地理情報データを読み込む
station_data = gpd.read_file('/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/result_location_new.csv')

# 不動産取引データを読み込む
eki_data = pd.read_csv('/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/eki_result.csv')

# マップを作成
m = folium.Map(location=[35.6895, 139.6917], zoom_start=12)  # 東京都庁舎の座標を使用

# 取引件数に応じて円形のマーカーをプロット
for X in range(len(station_data)):
    station_name = station_data['nearest station'].iloc[X]
    lat = station_data['緯度'].iloc[X]
    lon = station_data['経度'].iloc[X]
    
   # eki_data で station_name に対応する行を検索
    filtered_eki_data = eki_data[eki_data['nearest station'] == station_name]

    # 対応する行がある場合は取引量を取得、ない場合は0（または別のデフォルト値）を設定
    if not filtered_eki_data.empty:
        transaction_count = filtered_eki_data['count'].values[0]
    else:
        transaction_count = 0  # または別のデフォルト値

    # 取引量に応じてマーカーの色を設定
    if transaction_count >= 600:
        color = 'red'
    elif transaction_count >= 400:
        color = 'orange'
    elif transaction_count >= 200:
        color = '#FFD700'  # 標準の黄色よりも濃い黄色
    elif transaction_count >= 100:
        color = '#6699FF'  # 標準の青よりも薄い青
    else:
        color = 'limegreen'  

    # マーカーのサイズを取引量に応じて調整
    radius = 3 + transaction_count / 100

    # ポップアップに取引量を含める
    popup_text = f"{station_name}<br>取引量: {transaction_count}"
    popup = folium.Popup(popup_text, parse_html=True)

    # マーカーを追加
    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=popup
    ).add_to(m)

# マップを保存
m.save('eki_number.html')


# In[121]:


# 結果を確認して、それとも東京以外と認識された駅の経緯度を正しく修正する

# CSVファイルを読み込む
df = pd.read_csv('/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/result_location_new.csv')

# 修正する条件を指定
stations_to_modify = ['富士見ケ丘', '信濃町', '梶原', '四ツ谷', '千住大橋', '新小金井', '押上', '京成立石']

# 新しい緯度と経度の値
new_latitudes = [35.6848, 35.6799, 35.7511, 35.6858, 35.7424, 35.6959, 35.7106, 35.7382]
new_longitudes = [139.6071, 139.7206, 139.7473, 139.7295, 139.7970, 139.5267, 139.8131, 139.8481]

# 各行に対して修正を行う
for station_name, new_latitude, new_longitude in zip(stations_to_modify, new_latitudes, new_longitudes):
    condition = df['nearest station'] == station_name
    df.loc[condition, ['緯度', '経度']] = [new_latitude, new_longitude]

# 修正をファイルに保存
df.to_csv('result_location_new1.csv', index=False)


# In[122]:


#再度最寄駅ごとの取引量のマップを作ります

# 地理情報データを読み込む
station_data = gpd.read_file('/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/result_location_new1.csv')

# 不動産取引データを読み込む
eki_data = pd.read_csv('/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/eki_result.csv')

# マップを作成
m = folium.Map(location=[35.6895, 139.6917], zoom_start=12)  # 東京都庁舎の座標を使用

# 取引件数に応じてマーカーをプロット
for X in range(len(station_data)):
    station_name = station_data['nearest station'].iloc[X]
    lat = station_data['緯度'].iloc[X]
    lon = station_data['経度'].iloc[X]
    
   # eki_data で station_name に対応する行を検索
    filtered_eki_data = eki_data[eki_data['nearest station'] == station_name]

    # 対応する行がある場合は取引量を取得、ない場合は0（または別のデフォルト値）を設定
    if not filtered_eki_data.empty:
        transaction_count = filtered_eki_data['count'].values[0]
    else:
        transaction_count = 0  # または別のデフォルト値


   # 取引量に応じてマーカーの色を設定
    if transaction_count >= 700:
        color = 'red'
    elif transaction_count >= 500:
        color = 'orange'
    elif transaction_count >= 300:
        color = '#FFD700'  # 標準の黄色よりも濃い黄色
    elif transaction_count >= 100:
        color = '#6699FF'  # 標準の青よりも薄い青
    else:
        color = 'limegreen' 
        
    # マーカーのサイズを取引量に応じて調整
    radius = 3 + transaction_count / 100

    # ポップアップに取引量を含める
    popup_text = f"{station_name}<br>取引量: {transaction_count}"
    popup = folium.Popup(popup_text, parse_html=True)

    # マーカーを追加
    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=popup
    ).add_to(m)

# マップを保存
m.save('eki_number.html')


# In[123]:


#最寄駅ごとの平均平米単価のマップを作ります

# 地理情報データを読み込む
station_data = gpd.read_file('/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/result_location_new1.csv')

# 不動産取引データを読み込む
eki_data = pd.read_csv('/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/eki_result.csv')


# In[124]:


#駅ごとの平均取引価格の最大値と最小値を確認する
print(eki_data['average_price_per_sqm'].min())
print(eki_data['average_price_per_sqm'].max())


# In[127]:


#マップを作ります

# 地理情報データを読み込む
station_data = gpd.read_file('/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/result_location_new1.csv')

# 不動産取引データを読み込む
eki_data = pd.read_csv('/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/eki_result.csv')

# マップを作成
m = folium.Map(location=[35.6895, 139.6917], zoom_start=12)  # 東京都庁舎の座標を使用

# 取引件数に応じてマーカーをプロット
for X in range(len(station_data)):
    station_name = station_data['nearest station'].iloc[X]
    lat = station_data['緯度'].iloc[X]
    lon = station_data['経度'].iloc[X]

    # eki_data で station_name に対応する行を検索
    filtered_eki_data = eki_data[eki_data['nearest station'] == station_name]

    # 対応する行がある場合は平均平米単価を取得、ない場合は0（または別のデフォルト値）を設定
    if not filtered_eki_data.empty:
        average_price = filtered_eki_data['average_price_per_sqm'].values[0]
    else:
        average_price = 0  # または別のデフォルト値
    # 取引件数に応じてマーカーの色を設定
    if average_price >= 150:
        color = 'red'
    elif average_price >= 120:
        color = 'orange'
    elif average_price >= 90:
        color =  '#FFD700'  # 標準の黄色よりも濃い黄色
    elif average_price >= 60:
        color =  '#6699FF'  # 標準の青よりも薄い青
    else:
        color = 'limegreen' 

    # マーカーのサイズを取引量に応じて調整
    radius = min(3 + average_price / 50, 20)

    # ポップアップに取引量を含める
    popup_text = f"{station_name}<br>平均平米単価: {transaction_count}"
    popup = folium.Popup(popup_text, parse_html=True)

    # マーカーを追加
    folium.CircleMarker(
        location=[lat, lon],
        radius=radius,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=popup
    ).add_to(m)

# マップを保存
m.save('eki_price.html')


# <span style="font-size: 25px;">６、要因分析を行う

# In[62]:


# ファイルの読み込み
file_path = '/Users/ivy/Documents/東京理科大學/2023年上半期/卒業研究（最新版）/new manshion deta.csv'
df = pd.read_csv(file_path)

# データの最初の数行を表示して概観
df.head()


# In[63]:


# 欠損値除去
df = df.dropna(subset=["nearest station","nearest station distance_minutes","間取り","area_square",'建物の構造','architectural year','total_price'])

df = df.replace({'nearest station distance_minutes': {"1H?1H30": 60}})
df = df.replace({'nearest station distance_minutes': {"1H30?2H": 90}})
df = df.replace({'nearest station distance_minutes': {"2H?": 120}})
df = df.replace({'nearest station distance_minutes': {"30分?60分": 30}})


# In[64]:


# 「取引時点」列のデータ型を確認
df['取引時点'].dtype, df['取引時点'].head()


# In[65]:


# 「取引時点」を月の数で数値化する関数
def convert_to_month_number(x):
    year, quarter = x.split('年第')
    quarter = quarter[0]  # 四半期の数字部分を取得
    return int(year) * 12 + int(quarter) * 3

# 「取引時点」列を変換
df['取引時点'] = df['取引時点'].apply(convert_to_month_number)

# 変換後のデータを確認
print(df['取引時点'].head())


# In[66]:


df.drop(['種類','city_code', '市区町村名','地区名','average_price_per_sqm','間取り', '建物の構造','nearest station','建築年', '今後の利用目的', '都市計画', '建築年（西暦）'],         axis=1, inplace = True)


# In[67]:


df.head


# In[68]:


new_column_names = {'取引時点': 'trade date'}
df.rename(columns=new_column_names, inplace=True)


# In[69]:


# ワンホットエンコーディングの適用
df_encoded = pd.get_dummies(df)

# エンコードされたデータの確認
print(df_encoded.head())


# In[73]:


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# フォントの設定
mpl.rcParams['font.family'] = 'IPAexGothic'

# 相関行列の計算（df_encodedを前提としています）
correlation_matrix_encoded = df_encoded.corr()

# 図の作成（背景を白色に設定）
plt.figure(figsize=(12, 10), facecolor='white')
sns.heatmap(correlation_matrix_encoded, cmap='coolwarm')
plt.title('Correlation Matrix with One-Hot Encoded Data')

# ファイルとして保存（背景を白色に設定）
plt.savefig('correlation_matrix.png', facecolor='white')

plt.show()


# In[60]:


# 'total_price'と他の数値変数との相関係数を計算
correlation_with_total_price = numeric_df.corr()['total_price'].sort_values(ascending=False)

# 相関係数を表示
print(correlation_with_total_price)


# In[ ]:





# In[ ]:




