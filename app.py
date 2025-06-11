import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
import warnings

warnings.filterwarnings('ignore')

# Mengatur konfigurasi halaman Streamlit
st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide")

# --- FUNGSI-FUNGSI UNTUK MEMPROSES DATA DAN MODEL ---

# Cache data untuk mempercepat loading
@st.cache_data
def load_data(path):
    """
    Memuat dan membersihkan data dari file CSV dengan separator koma.
    """
    try:
        # PERUBAHAN 1: Menggunakan pd.read_csv dengan separator standar (koma)
        df = pd.read_csv(path)
        
        # Preprocessing dasar
        df.drop(['Id'], axis=1, inplace=True, errors='ignore')
        
        # PERUBAHAN 2: Hapus baris di mana target (SalePrice) atau fitur penting lainnya kosong.
        # Ini akan membersihkan data yang tidak lengkap, termasuk baris-baris di akhir file.
        df.dropna(subset=['SalePrice', 'MSZoning', 'LotConfig', 'BldgType', 'Exterior1st', 'BsmtFinSF2', 'TotalBsmtSF'], inplace=True)
        
        # PERUBAHAN 3: Pastikan tipe data kolom target adalah numerik (float/int)
        df['SalePrice'] = pd.to_numeric(df['SalePrice'], errors='coerce')
        # Hapus lagi jika ada yang gagal di-convert
        df.dropna(subset=['SalePrice'], inplace=True)

        return df
    except FileNotFoundError:
        st.error(f"File tidak ditemukan di path: {path}. Pastikan file '{path}' ada di folder yang sama.")
        return None
    except Exception as e:
        st.error(f"Terjadi error saat memuat data: {e}")
        return None

# Cache resource untuk tidak melatih ulang model setiap kali ada interaksi
@st.cache_resource
def train_model(df):
    """Mempersiapkan data dan melatih model SVR."""
    # Identifikasi kolom kategorikal
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Buat salinan untuk menghindari SettingWithCopyWarning
    df_processed = df.copy()

    # One-Hot Encoding
    oh_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit dan transform kolom kategorikal
    oh_cols_encoded = oh_encoder.fit_transform(df_processed[categorical_cols])
    
    # Buat DataFrame dari hasil encoding
    oh_cols_df = pd.DataFrame(oh_cols_encoded, columns=oh_encoder.get_feature_names_out(categorical_cols), index=df_processed.index)

    # Gabungkan kembali dengan data numerik
    df_final = df_processed.drop(categorical_cols, axis=1)
    df_final = pd.concat([df_final, oh_cols_df], axis=1)

    # Definisikan X dan Y
    X = df_final.drop(['SalePrice'], axis=1)
    Y = df_final['SalePrice']

    # Latih model SVR
    model = SVR()
    model.fit(X, Y)

    return model, oh_encoder, X.columns

# --- FUNGSI-FUNGSI UNTUK UI DAN PREDIKSI ---

def get_user_input(df):
    """
    Menampilkan widget di sidebar dan mengumpulkan input dari pengguna.
    """
    st.sidebar.header("Masukkan Fitur Rumah")
    
    input_data = {}
    
    # Input untuk fitur numerik
    input_data['MSSubClass'] = st.sidebar.slider('Tipe Bangunan (MSSubClass)', int(df['MSSubClass'].min()), int(df['MSSubClass'].max()), int(df['MSSubClass'].mean()))
    input_data['LotArea'] = st.sidebar.slider('Luas Tanah (LotArea)', int(df['LotArea'].min()), int(df['LotArea'].max()), int(df['LotArea'].mean()))
    input_data['OverallCond'] = st.sidebar.slider('Kondisi Keseluruhan (OverallCond)', int(df['OverallCond'].min()), int(df['OverallCond'].max()), 5)
    input_data['YearBuilt'] = st.sidebar.slider('Tahun Dibangun (YearBuilt)', int(df['YearBuilt'].min()), int(df['YearBuilt'].max()), int(df['YearBuilt'].mean()))
    input_data['YearRemodAdd'] = st.sidebar.slider('Tahun Renovasi (YearRemodAdd)', int(df['YearRemodAdd'].min()), int(df['YearRemodAdd'].max()), int(df['YearRemodAdd'].mean()))
    input_data['BsmtFinSF2'] = st.sidebar.slider('Luas Basement Tipe 2 (BsmtFinSF2)', float(df['BsmtFinSF2'].min()), float(df['BsmtFinSF2'].max()), float(df['BsmtFinSF2'].mean()))
    input_data['TotalBsmtSF'] = st.sidebar.slider('Total Luas Basement (TotalBsmtSF)', float(df['TotalBsmtSF'].min()), float(df['TotalBsmtSF'].max()), float(df['TotalBsmtSF'].mean()))

    # Input untuk fitur kategorikal
    input_data['MSZoning'] = st.sidebar.selectbox('Klasifikasi Zona (MSZoning)', sorted(df['MSZoning'].unique()))
    input_data['LotConfig'] = st.sidebar.selectbox('Konfigurasi Tanah (LotConfig)', sorted(df['LotConfig'].unique()))
    input_data['BldgType'] = st.sidebar.selectbox('Tipe Rumah (BldgType)', sorted(df['BldgType'].unique()))
    input_data['Exterior1st'] = st.sidebar.selectbox('Eksterior (Exterior1st)', sorted(df['Exterior1st'].unique()))
    
    return pd.DataFrame([input_data])

def process_and_predict(user_input_df, model, oh_encoder, feature_names):
    """
    Memproses input pengguna agar sesuai format model dan melakukan prediksi.
    """
    user_cat_cols = user_input_df.select_dtypes(include=['object']).columns
    user_num_cols = user_input_df.select_dtypes(include=['number']).columns

    user_cat_encoded = oh_encoder.transform(user_input_df[user_cat_cols])
    user_cat_encoded_df = pd.DataFrame(user_cat_encoded, columns=oh_encoder.get_feature_names_out(user_cat_cols))
    
    processed_input = pd.concat([user_input_df[user_num_cols].reset_index(drop=True), user_cat_encoded_df.reset_index(drop=True)], axis=1)
    processed_input = processed_input.reindex(columns=feature_names, fill_value=0)
    
    prediction = model.predict(processed_input)
    
    return prediction[0]

# --- UI (ANTARMUKA) UTAMA STREAMLIT ---

st.title("🏡 Aplikasi Prediksi Harga Jual Rumah")
st.write("Aplikasi ini memprediksi harga jual rumah berdasarkan fitur-fitur yang Anda masukkan di sidebar.")

# PERUBAHAN 4: Mengubah path ke file CSV yang baru
data_path = "HousePricePrediction.csv"
df = load_data(data_path)

if df is not None:
    model, oh_encoder, feature_names = train_model(df)
    user_input_df = get_user_input(df)

    st.subheader("Detail Input Pengguna:")
    st.write(user_input_df)

    predicted_price = process_and_predict(user_input_df, model, oh_encoder, feature_names)

    st.subheader("Hasil Prediksi Harga Jual")
    st.metric(label="Prediksi Harga", value=f"${predicted_price:,.2f}")
    st.info("Catatan: Harga prediksi akan diperbarui setiap kali Anda mengubah nilai di sidebar dan melepaskan interaksi (misalnya, melepas slider).")

    with st.expander("Tampilkan Analisis Data Eksplorasi"):
        st.subheader("Korelasi Antar Fitur Numerik")
        fig, ax = plt.subplots(figsize=(10, 8))
        numerical_df = df.select_dtypes(include=np.number)
        sns.heatmap(numerical_df.corr(), annot=True, fmt='.2f', cmap='BrBG', ax=ax)
        st.pyplot(fig)

        st.subheader("Distribusi Fitur Kategorikal")
        cat_cols_for_plot = df.select_dtypes(include=['object']).columns
        for col in cat_cols_for_plot:
            fig, ax = plt.subplots()
            sns.countplot(y=df[col], ax=ax, order=df[col].value_counts().index)
            ax.set_title(f'Distribusi {col}')
            st.pyplot(fig)

        st.subheader("Data yang Digunakan untuk Melatih Model (Sudah Dibersihkan)")
        st.dataframe(df)
