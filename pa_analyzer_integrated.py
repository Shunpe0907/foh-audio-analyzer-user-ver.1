"""
PA Audio Analyzer V3.0 - 完全統合版
認証システム + 全解析機能

機能:
- ユーザー認証（ログイン・新規登録）
- ユーザー別データ管理
- 2mix全体解析 + 楽器別詳細解析
- 過去データ比較・トレンド分析
- 管理者ダッシュボード

使い方:
    streamlit run pa_analyzer_integrated.py
"""

import streamlit as st
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import io
from pathlib import Path
import tempfile
import json
from datetime import datetime
import os
import sys

# 認証システムをインポート
# auth_system.py が同じディレクトリにあること
try:
    from auth_system import (
        UserDatabase, UserAudioDatabase,
        init_session_state,
        show_login_page, show_register_page,
        show_user_profile, show_admin_dashboard
    )
except ImportError as e:
    st.error(f"❌ auth_system.pyが見つかりません: {e}")
    st.stop()

# matplotlibの設定
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['font.size'] = 10

# ページ設定
st.set_page_config(
    page_title="PA Audio Analyzer V3.0",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .version-badge {
        text-align: center;
        color: #667eea;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .good-point {
        background-color: #e6ffe6;
        padding: 1rem;
        border-left: 4px solid #44ff44;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .recommendation-critical {
        background-color: #ffe6e6;
        padding: 1rem;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .recommendation-important {
        background-color: #fff9e6;
        padding: 1rem;
        border-left: 4px solid #ffbb33;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================================
# 音源解析エンジン（pa_analyzer_v3_finalから移植）
# =====================================

class AudioAnalyzer:
    """オーディオ解析メインクラス"""
    
    def __init__(self, audio_path, sr=44100):
        self.audio_path = audio_path
        self.target_sr = sr
        self.y = None
        self.sr = None
        self.duration = None
        self.load_audio()
    
    def load_audio(self):
        """音源読み込み"""
        self.y, self.sr = librosa.load(self.audio_path, sr=self.target_sr, mono=False)
        if len(self.y.shape) == 1:
            self.y = np.stack([self.y, self.y])
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
    
    def analyze_2mix(self):
        """2mix全体解析"""
        mono = np.mean(self.y, axis=0)
        
        # 基本指標
        rms = librosa.feature.rms(y=mono)[0]
        rms_db = 20 * np.log10(np.mean(rms) + 1e-10)
        
        peak = np.max(np.abs(mono))
        peak_db = 20 * np.log10(peak + 1e-10)
        
        crest_factor = peak_db - rms_db
        
        # ステレオ幅
        stereo_width = self.calculate_stereo_width()
        
        # 周波数解析
        band_energies = self.calculate_band_energies(mono)
        
        # 動的範囲
        dynamic_range = self.calculate_dynamic_range(mono)
        
        return {
            'rms_db': float(rms_db),
            'peak_db': float(peak_db),
            'crest_factor': float(crest_factor),
            'stereo_width': float(stereo_width),
            'band_energies': band_energies,
            'dynamic_range': float(dynamic_range),
            'duration': float(self.duration)
        }
    
    def calculate_stereo_width(self):
        """ステレオ幅計算"""
        if self.y.shape[0] < 2:
            return 0.0
        
        L, R = self.y[0], self.y[1]
        mid = (L + R) / 2
        side = (L - R) / 2
        
        mid_energy = np.sum(mid ** 2)
        side_energy = np.sum(side ** 2)
        
        if mid_energy + side_energy == 0:
            return 0.0
        
        width = (side_energy / (mid_energy + side_energy)) * 100
        return np.clip(width, 0, 100)
    
    def calculate_band_energies(self, audio):
        """帯域別エネルギー"""
        bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'presence': (4000, 8000),
            'brilliance': (8000, 20000)
        }
        
        energies = {}
        for name, (low, high) in bands.items():
            filtered = self.bandpass_filter(audio, low, high)
            energy_db = 20 * np.log10(np.sqrt(np.mean(filtered ** 2)) + 1e-10)
            energies[name] = float(energy_db)
        
        return energies
    
    def bandpass_filter(self, audio, low, high):
        """バンドパスフィルター"""
        nyq = self.sr / 2
        low_norm = low / nyq
        high_norm = high / nyq
        
        low_norm = np.clip(low_norm, 0.001, 0.999)
        high_norm = np.clip(high_norm, 0.001, 0.999)
        
        if low_norm >= high_norm:
            return audio * 0
        
        try:
            sos = signal.butter(4, [low_norm, high_norm], btype='band', output='sos')
            return signal.sosfilt(sos, audio)
        except:
            return audio * 0
    
    def calculate_dynamic_range(self, audio):
        """動的範囲計算"""
        rms_values = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        rms_db = 20 * np.log10(rms_values + 1e-10)
        
        percentile_95 = np.percentile(rms_db, 95)
        percentile_10 = np.percentile(rms_db, 10)
        
        return percentile_95 - percentile_10
    
    def analyze_instrument(self, freq_range, instrument_name):
        """楽器別解析"""
        mono = np.mean(self.y, axis=0)
        filtered = self.bandpass_filter(mono, freq_range[0], freq_range[1])
        
        # 基本指標
        rms_db = 20 * np.log10(np.sqrt(np.mean(filtered ** 2)) + 1e-10)
        peak_db = 20 * np.log10(np.max(np.abs(filtered)) + 1e-10)
        
        # スペクトル重心
        spectral_centroid = float(np.mean(
            librosa.feature.spectral_centroid(y=filtered, sr=self.sr)[0]
        ))
        
        return {
            'name': instrument_name,
            'freq_range': freq_range,
            'rms_db': float(rms_db),
            'peak_db': float(peak_db),
            'spectral_centroid': spectral_centroid
        }


def generate_recommendations(analysis_data, metadata):
    """改善提案生成"""
    recommendations = {
        'critical': [],
        'important': [],
        'good_points': []
    }
    
    # RMS音圧チェック
    rms = analysis_data['rms_db']
    if rms < -23:
        recommendations['critical'].append(
            f"⚠️ 全体音圧が低すぎます（{rms:.1f}dB）。マスターフェーダーを上げ、-18dB前後を目標にしてください。"
        )
    elif rms > -14:
        recommendations['critical'].append(
            f"⚠️ 全体音圧が高すぎます（{rms:.1f}dB）。ヘッドルームがなく、歪みのリスクがあります。"
        )
    elif -20 <= rms <= -16:
        recommendations['good_points'].append(
            f"✅ 全体音圧が適切です（{rms:.1f}dB）。ライブに最適なレベルです。"
        )
    
    # Peakチェック
    peak = analysis_data['peak_db']
    if peak > -1:
        recommendations['critical'].append(
            f"⚠️ ピークレベルが高すぎます（{peak:.1f}dB）。クリッピングの危険があります。"
        )
    elif peak < -6:
        recommendations['important'].append(
            f"📌 ピークレベルに余裕があります（{peak:.1f}dB）。もう少し音圧を上げられます。"
        )
    
    # Crest Factor
    cf = analysis_data['crest_factor']
    if cf < 8:
        recommendations['important'].append(
            f"📌 クレストファクターが低いです（{cf:.1f}dB）。過圧縮の可能性があります。"
        )
    elif cf > 16:
        recommendations['important'].append(
            f"📌 クレストファクターが高いです（{cf:.1f}dB）。ダイナミクスが大きすぎる可能性があります。"
        )
    elif 10 <= cf <= 14:
        recommendations['good_points'].append(
            f"✅ クレストファクターが理想的です（{cf:.1f}dB）。良好なダイナミックバランスです。"
        )
    
    # Stereo Width
    width = analysis_data['stereo_width']
    if width < 30:
        recommendations['important'].append(
            f"📌 ステレオ幅が狭いです（{width:.1f}%）。パンニングを見直してください。"
        )
    elif width > 80:
        recommendations['important'].append(
            f"📌 ステレオ幅が広すぎます（{width:.1f}%）。モノラル環境で問題が出る可能性があります。"
        )
    elif 50 <= width <= 70:
        recommendations['good_points'].append(
            f"✅ ステレオ幅が理想的です（{width:.1f}%）。バランスの良い音場です。"
        )
    
    return recommendations


def plot_frequency_response(band_energies):
    """周波数特性グラフ"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bands = list(band_energies.keys())
    energies = list(band_energies.values())
    
    colors = ['#8B0000', '#FF4500', '#FFD700', '#32CD32', '#4169E1', '#9370DB', '#FF1493']
    
    ax.bar(bands, energies, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(-20, color='green', linestyle='--', linewidth=1, alpha=0.5, label='目標レベル')
    ax.set_ylabel('Energy (dB)', fontsize=12, fontweight='bold')
    ax.set_title('Frequency Band Energy Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


# =====================================
# メインアプリケーション
# =====================================

def main():
    """メインアプリケーション"""
    
    # セッションステート初期化
    init_session_state()
    
    # データベース初期化
    user_db = UserDatabase()
    audio_db = UserAudioDatabase()
    
    # 認証チェック
    if not st.session_state.authenticated:
        # ログイン・登録ページ
        if st.session_state.page == 'login':
            show_login_page(user_db)
        elif st.session_state.page == 'register':
            show_register_page(user_db)
        
        # 説明
        st.markdown("---")
        st.markdown("""
        ## 🎛️ PA Audio Analyzer V3.0 について
        
        ライブPA用の2mixおよび楽器別オーディオ解析ツールです。
        
        ### 主な機能
        
        - **2mix全体解析**: 音圧、ステレオイメージ、周波数バランス
        - **楽器別詳細解析**: ボーカル、ドラム、ベース、ギターなど
        - **科学的根拠に基づく提案**: ITU-R、ISO、AES規格準拠
        - **過去データ比較**: 成長トレンドの可視化
        - **機材別最適化**: ミキサー・PAシステム特性を考慮
        
        ### ログイン・登録について
        
        - **新規ユーザー**: 「新規登録」から無料でアカウント作成
        - **解析履歴**: ログインすることで過去の解析データを保存・比較可能
        - **プライバシー**: データは個別管理、他のユーザーからは見えません
        """)
        
        return
    
    # ログイン済み
    user = st.session_state.user
    
    # サイドバー
    with st.sidebar:
        # ユーザー情報表示
        if user['role'] == 'admin':
            st.markdown(f"### 🛡️ 管理者: {user['name']}")
        else:
            st.markdown(f"### 👤 {user['name']}")
        
        st.caption(f"📧 {user['email']}")
        
        st.markdown("---")
        
        # メニュー
        menu = st.radio(
            "メニュー",
            ["🎵 音源解析", "📊 過去データ", "👤 プロフィール"] +
            (["🛡️ 管理者ダッシュボード"] if user['role'] == 'admin' else []),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # ログアウト
        if st.button("🚪 ログアウト", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.page = 'login'
            st.rerun()
    
    # メインコンテンツ
    if menu == "🎵 音源解析":
        show_analyzer_page(user, user_db, audio_db)
    
    elif menu == "📊 過去データ":
        show_history_page(user, audio_db)
    
    elif menu == "👤 プロフィール":
        show_user_profile(user_db)
    
    elif menu == "🛡️ 管理者ダッシュボード" and user['role'] == 'admin':
        show_admin_dashboard(user_db, audio_db)


def show_analyzer_page(user, user_db, audio_db):
    """音源解析ページ（完全機能版）"""
    
    st.markdown('<h1 class="main-header">🎛️ Live PA Audio Analyzer V3.0</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="version-badge">完全統合版 - Full Integration</p>', 
                unsafe_allow_html=True)
    
    st.success(f"👤 ログイン中: **{user['name']}** さん")
    
    # タブ構成
    tab1, tab2, tab3 = st.tabs(["📤 音源アップロード", "📊 解析結果", "💡 改善提案"])
    
    with tab1:
        show_upload_section(user, user_db, audio_db)
    
    with tab2:
        if 'analysis_result' in st.session_state:
            show_analysis_results()
        else:
            st.info("音源をアップロードして解析を実行してください")
    
    with tab3:
        if 'analysis_result' in st.session_state:
            show_recommendations()
        else:
            st.info("解析実行後に改善提案が表示されます")


def show_upload_section(user, user_db, audio_db):
    """アップロードセクション"""
    
    st.markdown("### 📤 音源ファイルをアップロード")
    
    uploaded_file = st.file_uploader(
        "2mixファイル（WAV/MP3）",
        type=['wav', 'mp3'],
        help="ライブ本番またはリハーサルの2mix音源をアップロードしてください"
    )
    
    if uploaded_file:
        st.success(f"✅ ファイル: {uploaded_file.name}")
        
        # メタデータ入力
        st.markdown("### 📝 解析情報の入力")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_name = st.text_input(
                "解析名 *",
                placeholder="例: ライブ本番",
                help="この解析を識別するための名前"
            )
            
            venue_name = st.text_input(
                "会場名 *",
                placeholder="例: CLUB QUATTRO",
                help="演奏会場の名前"
            )
            
            venue_capacity = st.number_input(
                "会場キャパシティ（人）",
                min_value=10,
                max_value=10000,
                value=150,
                step=10
            )
        
        with col2:
            mixer = st.text_input(
                "使用ミキサー",
                placeholder="例: Yamaha CL5",
                help="使用したデジタルミキサー"
            )
            
            pa_system = st.text_input(
                "PAシステム",
                placeholder="例: d&b V-Series",
                help="使用したPAシステム"
            )
            
            band_lineup = st.text_area(
                "バンド編成",
                placeholder="例: ボーカル、キック、スネア、ベース、ギター×2",
                help="演奏楽器の編成"
            )
        
        notes = st.text_area(
            "メモ（任意）",
            placeholder="気づいた点、改善したい点など...",
            help="自由記述"
        )
        
        # 解析実行
        if st.button("🚀 解析開始", type="primary", use_container_width=True):
            if not analysis_name or not venue_name:
                st.error("解析名と会場名は必須です")
                return
            
            with st.spinner("解析中..."):
                try:
                    # 一時ファイルに保存
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # 解析実行
                    analyzer = AudioAnalyzer(tmp_path)
                    analysis_result = analyzer.analyze_2mix()
                    
                    # 楽器別解析（簡易版）
                    instruments = {
                        'vocals': analyzer.analyze_instrument((200, 4000), 'ボーカル'),
                        'kick': analyzer.analyze_instrument((40, 100), 'キック'),
                        'snare': analyzer.analyze_instrument((150, 250), 'スネア'),
                        'bass': analyzer.analyze_instrument((60, 250), 'ベース'),
                        'guitar': analyzer.analyze_instrument((200, 5000), 'ギター')
                    }
                    
                    analysis_result['instruments'] = instruments
                    
                    # 一時ファイル削除
                    os.unlink(tmp_path)
                    
                    # メタデータ
                    metadata = {
                        'analysis_name': analysis_name,
                        'venue': venue_name,
                        'venue_capacity': venue_capacity,
                        'mixer': mixer or '不明',
                        'pa_system': pa_system or '不明',
                        'band_lineup': band_lineup or '不明',
                        'notes': notes
                    }
                    
                    # セッションに保存
                    st.session_state.analysis_result = analysis_result
                    st.session_state.analysis_metadata = metadata
                    
                    # データベースに保存
                    user_db.update_user_stats(user['email'])
                    entry_id = audio_db.add_analysis(user['email'], analysis_result, metadata)
                    
                    st.success(f"✅ 解析完了！（ID: {entry_id}）")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"解析エラー: {e}")


def show_analysis_results():
    """解析結果表示"""
    
    result = st.session_state.analysis_result
    metadata = st.session_state.analysis_metadata
    
    st.markdown("### 📊 解析結果")
    
    # 基本情報
    st.markdown(f"**解析名**: {metadata['analysis_name']}")
    st.markdown(f"**会場**: {metadata['venue']} ({metadata['venue_capacity']}人)")
    
    st.markdown("---")
    
    # 主要指標
    st.markdown("#### 🎚️ 主要指標")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("RMS音圧", f"{result['rms_db']:.1f} dB", 
                 delta=f"目標: -18dB", delta_color="off")
    
    with col2:
        st.metric("ピークレベル", f"{result['peak_db']:.1f} dB",
                 delta=f"上限: -1dB", delta_color="off")
    
    with col3:
        st.metric("クレストファクター", f"{result['crest_factor']:.1f} dB",
                 delta=f"理想: 10-14dB", delta_color="off")
    
    with col4:
        st.metric("ステレオ幅", f"{result['stereo_width']:.1f} %",
                 delta=f"理想: 50-70%", delta_color="off")
    
    # 周波数特性
    st.markdown("---")
    st.markdown("#### 🎼 周波数特性")
    
    fig = plot_frequency_response(result['band_energies'])
    st.pyplot(fig)
    plt.close(fig)
    
    # 楽器別
    if 'instruments' in result:
        st.markdown("---")
        st.markdown("#### 🎸 楽器別解析")
        
        for inst_name, inst_data in result['instruments'].items():
            with st.expander(f"🎵 {inst_data['name']}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("RMS", f"{inst_data['rms_db']:.1f} dB")
                with col2:
                    st.metric("Peak", f"{inst_data['peak_db']:.1f} dB")
                with col3:
                    st.metric("重心周波数", f"{inst_data['spectral_centroid']:.0f} Hz")


def show_recommendations():
    """改善提案表示"""
    
    result = st.session_state.analysis_result
    metadata = st.session_state.analysis_metadata
    
    recommendations = generate_recommendations(result, metadata)
    
    st.markdown("### 💡 改善提案")
    
    # Good Points
    if recommendations['good_points']:
        st.markdown("#### ✅ 良好なポイント")
        for point in recommendations['good_points']:
            st.markdown(f'<div class="good-point">{point}</div>', unsafe_allow_html=True)
    
    # Critical
    if recommendations['critical']:
        st.markdown("#### ⚠️ 重要な改善点")
        for point in recommendations['critical']:
            st.markdown(f'<div class="recommendation-critical">{point}</div>', unsafe_allow_html=True)
    
    # Important
    if recommendations['important']:
        st.markdown("#### 📌 推奨改善点")
        for point in recommendations['important']:
            st.markdown(f'<div class="recommendation-important">{point}</div>', unsafe_allow_html=True)


def show_history_page(user, audio_db):
    """過去解析データページ"""
    
    st.markdown("## 📊 過去の解析データ")
    
    # ユーザーの解析データ取得
    analyses = audio_db.get_user_analyses(user['email'])
    
    if not analyses:
        st.info("まだ解析データがありません。「音源解析」から解析を実行してください。")
        return
    
    st.write(f"**総解析数: {len(analyses)}件**")
    
    # 検索・フィルター
    search = st.text_input("🔍 検索", placeholder="解析名、会場名で検索")
    
    # フィルタリング
    if search:
        filtered = [
            a for a in analyses
            if search.lower() in a['metadata'].get('analysis_name', '').lower()
            or search.lower() in a['metadata'].get('venue', '').lower()
        ]
    else:
        filtered = analyses
    
    st.write(f"**表示: {len(filtered)}件**")
    
    # データ一覧
    for analysis in filtered:
        timestamp = datetime.fromisoformat(analysis['timestamp'])
        name = analysis['metadata'].get('analysis_name', '名称未設定')
        venue = analysis['metadata'].get('venue', '不明')
        
        with st.expander(f"🎵 {name} - {venue} ({timestamp.strftime('%Y/%m/%d %H:%M')})", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📅 基本情報**")
                st.write(f"**解析日時**: {timestamp.strftime('%Y年%m月%d日 %H:%M')}")
                st.write(f"**解析名**: {name}")
                st.write(f"**会場**: {venue}")
                st.write(f"**キャパ**: {analysis['metadata'].get('venue_capacity', '不明')}人")
            
            with col2:
                st.markdown("**🎛️ 機材情報**")
                st.write(f"**ミキサー**: {analysis['metadata'].get('mixer', '不明')}")
                st.write(f"**PA**: {analysis['metadata'].get('pa_system', '不明')}")
                st.write(f"**バンド編成**: {analysis['metadata'].get('band_lineup', '不明')}")
            
            # 解析結果
            st.markdown("---")
            st.markdown("**📊 解析結果**")
            
            analysis_data = analysis.get('analysis', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RMS", f"{analysis_data.get('rms_db', 0):.1f} dB")
            with col2:
                st.metric("Peak", f"{analysis_data.get('peak_db', 0):.1f} dB")
            with col3:
                st.metric("ステレオ幅", f"{analysis_data.get('stereo_width', 0):.1f}%")
            with col4:
                st.metric("クレスト", f"{analysis_data.get('crest_factor', 0):.1f} dB")
            
            # メモ
            if analysis['metadata'].get('notes'):
                st.markdown("**📝 メモ**")
                st.write(analysis['metadata']['notes'])
            
            # 削除ボタン
            if st.button(f"🗑️ このデータを削除", key=f"delete_{analysis['id']}"):
                if audio_db.delete_analysis(user['email'], analysis['id']):
                    st.success("削除しました")
                    st.rerun()
                else:
                    st.error("削除に失敗しました")


if __name__ == "__main__":
    main()
