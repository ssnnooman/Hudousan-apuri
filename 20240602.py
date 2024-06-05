import os
import streamlit as st
import pandas as pd
import numpy as np
import gspread
import googlemaps
from google.oauth2.service_account import Credentials
from gspread_dataframe import set_with_dataframe
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
# 以下、水野追加
import requests
from bs4 import BeautifulSoup

# 追加ここまで
# 環境変数の読み込み
load_dotenv()

# 環境変数から認証情報を取得
SPREADSHEET_ID ="1SPGg1Dnslu31zEWoRKBwJh4suE-Nl4LQmxU3fSQaLTU"   #ここはSPREDシートのURLを張る。
PRIVATE_KEY_PATH = r"C:\Users\ssnow\Desktop\STEP3-1　不動産アプリ\orbital-outpost-423316-v9-1abc7995dce3.json"
SP_SHEET     = 'demo' # sheet名

# セッション状態の初期化
if 'show_all' not in st.session_state:
    st.session_state['show_all'] = False  # 初期状態は地図上の物件のみを表示

# 地図上以外の物件も表示するボタンの状態を切り替える関数
def toggle_show_all():
    st.session_state['show_all'] = not st.session_state['show_all']

# スプレッドシートからデータを読み込む関数
def load_data_from_spreadsheet():
    # googleスプレッドシートの認証 jsonファイル読み込み(key値はGCPから取得)
    SP_CREDENTIAL_FILE = PRIVATE_KEY_PATH

    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    credentials = Credentials.from_service_account_file(
        SP_CREDENTIAL_FILE,
        scopes=scopes
    )
    gc = gspread.authorize(credentials)

    SP_SHEET_KEY ="1SPGg1Dnslu31zEWoRKBwJh4suE-Nl4LQmxU3fSQaLTU" # d/〇〇/edit の〇〇部分
    sh  = gc.open_by_key(SP_SHEET_KEY)

    # 不動産データの取得
    worksheet = sh.worksheet(SP_SHEET) # シートのデータ取得
    pre_data  = worksheet.get_all_values()
    col_name = pre_data[0][:]
    df = pd.DataFrame(pre_data[1:], columns=col_name) # 一段目をカラム、以下データフレームで取得

    return df

# データフレームの前処理を行う関数
def preprocess_dataframe(df):
    # '家賃' 列を浮動小数点数に変換し、NaN値を取り除く
    df['家賃'] = pd.to_numeric(df['家賃'], errors='coerce')
    df = df.dropna(subset=['家賃'])
    return df

def make_clickable(url, name):
    return f'<a target="_blank" href="{url}">{name}</a>'

# 地図を作成し、マーカーを追加する関数
def create_map(filtered_df):
    # 地図の初期設定
    map_center = [filtered_df['latitude'].mean(), filtered_df['longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    # マーカーを追加
    for idx, row in filtered_df.iterrows():
        if pd.notnull(row['latitude']) and pd.notnull(row['longitude']):
            # ポップアップに表示するHTMLコンテンツを作成
            popup_html = f"""
            <b>名称:</b> {row['名称']}<br>
            <b>アドレス:</b> {row['アドレス']}<br>
            <b>家賃:</b> {row['家賃']}万円<br>
            <b>間取り:</b> {row['間取り']}<br>
            <b>アクセス:</b> {row['アクセス']}<br>
            <a href="{row['物件詳細URL']}" target="_blank">物件詳細</a>
            <a href="{row['物件画像URL']}" target="_blank">物件画像</a>
            <a href="{row['間取画像URL']}" target="_blank">間取り画像</a>
            """
            # HTMLをポップアップに設定
            popup = folium.Popup(popup_html, max_width=0)
            folium.Marker(
                [row['latitude'], row['longitude']],
                popup=popup
            ).add_to(m)

    return m

# 検索結果を表示する関数
def display_search_results(filtered_df):
    # 物件番号を含む新しい列を作成
    filtered_df['物件番号'] = range(1, len(filtered_df) + 1)
    filtered_df['物件詳細URL'] = filtered_df['物件詳細URL'].apply(lambda x: make_clickable(x, "リンク"))
    filtered_df['物件画像URL'] = filtered_df['物件画像URL'].apply(lambda x: make_clickable(x, "リンク"))
    filtered_df['間取画像URL'] = filtered_df['間取画像URL'].apply(lambda x: make_clickable(x, "リンク"))
    display_columns = ['物件番号', '名称', 'アドレス', '階数', '家賃', '間取り', 'アクセス', '物件詳細URL','物件画像URL','間取画像URL']
    filtered_df_display = filtered_df[display_columns]
    st.markdown(filtered_df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

# 以下、水野追加
def get_route_info(departure_station, destination_station):
    route_url = f"https://transit.yahoo.co.jp/search/print?from={departure_station}&flatlon=&to={destination_station}"

    route_response = requests.get(route_url)
    route_soup = BeautifulSoup(route_response.text, 'html.parser')

    route_summary = route_soup.find("div", class_="routeSummary")
    required_time = route_summary.find("li", class_="time").get_text()
    transfer_count = route_summary.find("li", class_="transfer").get_text()
    fare = route_summary.find("li", class_="fare").get_text()

    return required_time, transfer_count, fare, route_soup, route_url

def get_transfer_info(route_soup):
    route_detail = route_soup.find("div", class_="routeDetail")

    stations = []
    stations_tmp = route_detail.find_all("div", class_="station")
    for station in stations_tmp:
        stations.append(station.get_text().strip())

    lines = []
    lines_tmp = route_detail.find_all("li", class_="transport")
    for line in lines_tmp:
        line = line.find("div").get_text().strip()
        lines.append(line)

    fares = []
    fares_tmp = route_detail.find_all("p", class_="fare")
    for fare in fares_tmp:
        fares.append(fare.get_text().strip())
    return stations, lines, fares

# 追加ここまで
# メインのアプリケーション
def main():
    df = load_data_from_spreadsheet()
    df = preprocess_dataframe(df)

    # StreamlitのUI要素（スライダー、ボタンなど）の各表示設定
    st.title('賃貸物件　楽々検索アプリ')

    # エリアと家賃フィルタバーを1:2の割合で分割
    col1, col2 = st.columns([1, 2])

    with col1:
        # エリア選択
        area = st.sidebar.radio('■ エリア選択', df['区'].unique())

    with col2:
        # 家賃範囲選択のスライダーをfloat型で設定し、小数点第一位まで表示
        price_min, price_max = st.sidebar.slider(
            '■ 家賃範囲 (万円)',
            min_value=float(1),
            max_value=float(df['家賃'].max()),
            value=(float(df['家賃'].min()), float(df['家賃'].max())),
            step=0.1,  # ステップサイズを0.1に設定
            format='%.1f'
        )

    with col2:
    # 間取り選択のデフォルト値をすべてに設定
        type_options = st.sidebar.multiselect('■ 間取り選択', df['間取り'].unique(), default=df['間取り'].unique())

    # フィルタリング/ フィルタリングされたデータフレームの件数を取得
    filtered_df = df[(df['区'].isin([area])) & (df['間取り'].isin(type_options))]
    filtered_df = filtered_df[(filtered_df['家賃'] >= price_min) & (filtered_df['家賃'] <= price_max)]
    filtered_count = len(filtered_df)

    # 'latitude' と 'longitude' 列を数値型に変換し、NaN値を含む行を削除
    filtered_df['latitude'] = pd.to_numeric(filtered_df['latitude'], errors='coerce')
    filtered_df['longitude'] = pd.to_numeric(filtered_df['longitude'], errors='coerce')
    filtered_df2 = filtered_df.dropna(subset=['latitude', 'longitude'])


    # 検索ボタン / # フィルタリングされたデータフレームの件数を表示
    col2_1, col2_2 = st.columns([1, 2])

    with col2_2:
        st.sidebar.write(f"物件検索数: {filtered_count}件 / 全{len(df)}件")

    # 検索ボタン
    if st.sidebar.button('検索＆更新', key='search_button'):
        # 検索ボタンが押された場合、セッションステートに結果を保存
        st.session_state['filtered_df'] = filtered_df
        st.session_state['filtered_df2'] = filtered_df2
        st.session_state['search_clicked'] = True

    # Streamlitに地図を表示
    if st.session_state.get('search_clicked', False):
        m = create_map(st.session_state.get('filtered_df2', filtered_df2))
        folium_static(m)

    # 地図の下にラジオボタンを配置し、選択したオプションに応じて表示を切り替える
    show_all_option = st.radio(
        "表示オプションを選択してください:",
        ('地図上の検索物件のみ', 'すべての検索物件'),
        index=0 if not st.session_state.get('show_all', False) else 1,
        key='show_all_option'
    )

    # ラジオボタンの選択に応じてセッションステートを更新
    st.session_state['show_all'] = (show_all_option == 'すべての検索物件')

    # 検索結果の表示
    if st.session_state.get('search_clicked', False):
        if st.session_state['show_all']:
            display_search_results(st.session_state.get('filtered_df', filtered_df))  # 全データ
        else:
            display_search_results(st.session_state.get('filtered_df2', filtered_df2))  # 地図上の物件のみ

    # 以下、水野追加
    route_information()
# 追加ここまで

# 以下、水野追加
def route_information():
    departure_station = st.text_input("出発駅(例：新宿駅)")
    destination_station = st.text_input("到着駅(例：東京駅)")

    if st.button('検索'):
        if departure_station and destination_station:
            required_time, transfer_count, fare, route_soup, route_url = get_route_info(departure_station, destination_station)

            route_info = {
                "情報": ["所要時間", "乗換回数", "料金"],
                "値": [required_time, transfer_count, fare]
            }

            route_info_df = pd.DataFrame(route_info)

            st.write(f"＜{departure_station}から{destination_station}の路線情報＞")
            st.table(route_info_df)

            stations, lines, fares = get_transfer_info(route_soup)

            max_length = max(len(stations), len(lines), len(fares))
            stations += [''] * (max_length - len(stations))
            lines += [''] * (max_length - len(lines))
            fares += [''] * (max_length - len(fares))

            transfer_info = {
                "駅": stations,
                "路線": lines,
                "料金": fares
            }

            transfer_info_df = pd.DataFrame(transfer_info)

            st.write("＜乗り換え情報＞")
            st.table(transfer_info_df)
            st.write(stations[-1])

            st.write(f"URL(Yahoo!乗換案内)：{route_url}")
# # アプリケーションの実行
# if __name__ == "__main__":

#     # 追加ここまで

# アプリケーションの実行
if __name__ == "__main__":
    if 'search_clicked' not in st.session_state:
        st.session_state['search_clicked'] = False
    if 'show_all' not in st.session_state:
        st.session_state['show_all'] = False
    main()

# APIキーを取得
api_key = os.getenv('GOOGLE_API_KEY', 'AIzaSyBidhMkj1qM89EqMIUTykAMk-f-8kt3Uik')  # 環境変数から取得、設定されていない場合は直接キーを指定

# クライアントを作成
gmaps = googlemaps.Client(key=api_key)

# 東京駅の緯度と経度
tokyo_station_coords = (35.6794, 139.7644)

# 徒歩20分の距離をメートルに換算（約1.5km）
radius = 1500

# 検索ボタン
if st.sidebar.button("勤務地周辺のジム検索"):
    # 周囲のジムを検索
    places_result = gmaps.places_nearby(location=tokyo_station_coords, radius=radius, type='gym')

    # 結果を表示
    if places_result['results']:
        st.write(f"東京駅から半径{radius / 1000}km以内のジム:")

        # データフレーム用のリストを作成
        gym_data = []
        for place in places_result['results']:
            gym_data.append({
                "名前": place['name'],
                "住所": place['vicinity'],
                "評価": place.get('rating', 'No rating')
            })

        # データフレームを作成
        gym_df = pd.DataFrame(gym_data)

        # データフレームを表形式で表示
        st.table(gym_df)
    else:
        st.write("ジムが見つかりませんでした。")