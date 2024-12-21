import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import joblib
import os
import torch


def klasifikasi_data():
    st.title("🤖 Klasifikasi Data Produksi Daging")

    # Periksa apakah file data sudah diupload
    if 'data' not in st.session_state:
        st.warning("Unggah data terlebih dahulu.")
        return

    data = st.session_state['data']

    # Menangani nilai kosong: Mengisi dengan nilai yang paling sering muncul (mode)
    for column in data.columns:
        if data[column].isna().sum() > 0:  # Jika ada nilai kosong
            most_frequent_value = data[column].mode()[0]
            data[column].fillna(most_frequent_value, inplace=True)

    # Preprocessing: Hitung total produksi daging dan kategorisasi
    data['Total Production'] = data.iloc[:, 3:].sum(axis=1)  # Menghitung total produksi
    data['Category'] = pd.qcut(data['Total Production'], q=3, labels=["Rendah", "Sedang", "Tinggi"])  # Kategorisasi

    st.write("Data dengan Total Production dan Kategorisasi:")
    st.write(data[['Total Production', 'Category']].head())

    # Pilih fitur dan target
    all_columns = data.columns
    fitur = st.multiselect("Pilih kolom fitur:", all_columns)
    target = st.selectbox("Pilih kolom target:", all_columns)

    if fitur and target:
        # Memisahkan data menjadi fitur dan target
        X = data[fitur]
        y_categorical = pd.Categorical(data[target])  # Konversi target menjadi kategori
        y = y_categorical.codes  # Numerik
        categories = y_categorical.categories  # Simpan kategori

        # Mengonversi fitur kategorikal menjadi numerik
        X = pd.get_dummies(X)

        # Standarisasi data untuk model tertentu (misalnya Neural Network)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Membagi data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Pilih model untuk klasifikasi
        model_name = st.radio("Pilih Model:", ["Random Forest", "XGBoost", "TabNet"])

        if st.button("Jalankan Model"):
            model = None
            model_path = None

            if model_name == "Random Forest":
                model = RandomForestClassifier(random_state=42)
                model_path = "random_forest_model.h5"
            elif model_name == "XGBoost":
                model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
                model_path = "xgb_model.h5"
            elif model_name == "TabNet":
                model_path = "tabnet_model.zip.zip"
                X_train_np = np.array(X_train, dtype=np.float32)
                X_test_np = np.array(X_test, dtype=np.float32)
                y_train_np = np.array(y_train, dtype=np.int64)
                y_test_np = np.array(y_test, dtype=np.int64)

                model = TabNetClassifier(
                    n_d=8, n_a=8, n_steps=3,
                    gamma=1.3, lambda_sparse=1e-4,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=2e-2),
                    verbose=0
                )

                # Latih model TabNet
                model.fit(
                    X_train_np, y_train_np,
                    eval_set=[(X_test_np, y_test_np)],
                    eval_metric=['accuracy'],
                    max_epochs=100,
                    patience=10,
                    batch_size=32,
                    virtual_batch_size=16
                )
                y_pred = model.predict(X_test_np)
                accuracy = accuracy_score(y_test_np, y_pred)

                # Konversi prediksi menjadi kategori
                y_pred_categories = [categories[pred] for pred in y_pred]

                # Simpan model TabNet
                if os.path.exists(model_path):
                    os.remove(model_path)
                model.save_model(model_path)

                # Simpan hasil ke session_state
                st.session_state["results"] = {
                    "model": model_name,
                    "accuracy": accuracy,
                    "predictions": y_pred_categories,
                    "actual": [categories[act] for act in y_test_np],
                    "model_path": model_path
                }

                st.success(f"Model {model_name} selesai dilatih dan disimpan sebagai '{model_path}'!")
                st.write(f"Akurasi: {accuracy:.2f}")
                st.write("Prediksi:", y_pred_categories)
                return

            # Latih model lain
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Konversi prediksi menjadi kategori
            y_pred_categories = [categories[pred] for pred in y_pred]

            # Simpan model
            if os.path.exists(model_path):
                os.remove(model_path)
            joblib.dump(model, model_path)

            # Simpan hasil ke session_state
            st.session_state["results"] = {
                "model": model_name,
                "accuracy": accuracy,
                "predictions": y_pred_categories,
                "actual": [categories[act] for act in y_test],
                "model_path": model_path
            }

            st.success(f"Model {model_name} selesai dilatih dan disimpan sebagai '{model_path}'!")
            st.write(f"Akurasi: {accuracy:.2f}")
            st.write("Prediksi:", y_pred_categories)

# Fungsi untuk mengupload file
def upload_file():
    st.title("Unggah Data CSV")
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state['data'] = data
        st.write("Data yang diunggah:")
        st.write(data.head())
