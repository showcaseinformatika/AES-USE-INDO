import streamlit as st
import numpy as np
import pandas as pd
import re
import unicodedata
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import io

# Pilih mode penggunaan
st.title("Automatic Essay Scoring System")

mode = st.selectbox("Pilih Mode Penilaian:", ["Single Scoring", "Multiple Scoring"])
model_option = st.selectbox("Pilih Jenis Universal Sentence Encoder:", ["Transformer", "Deep Averaging Network"])

# Load USE model sesuai pilihan
@st.cache_resource
def load_use_model(model_type):
    if model_type == "Transformer":
        url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    else:
        url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    return hub.load(url)

use_model = load_use_model(model_option)

# Fungsi preprocessing
def preprocess_text(text):
    text = text.strip()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()

    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

# Fungsi untuk memetakan similarity ke skor 1–10
def map_similarity_to_score(similarity):
    if similarity < 0.5:
        return 1
    else:
        return round((similarity - 0.5) * 9 / 0.5 + 1)

# ==== SINGLE SCORING ====
if mode == "Single Scoring":
    with st.form("single_form"):
        name = st.text_input("Nama")
        question = st.text_area("Soal")
        answer_key = st.text_area("Kunci Jawaban")
        student_answer = st.text_area("Jawaban Siswa")
        submitted = st.form_submit_button("Submit")

    if submitted:
        preprocessed_answer_key = preprocess_text(answer_key)
        preprocessed_student_answer = preprocess_text(student_answer)

        answer_key_emb = use_model([preprocessed_answer_key])[0].numpy()
        student_answer_emb = use_model([preprocessed_student_answer])[0].numpy()

        cos_sim = cosine_similarity([answer_key_emb], [student_answer_emb])[0][0]
        student_score = map_similarity_to_score(cos_sim)

        st.write(f"**Bobot Jawaban Siswa: {student_score}**")

# ==== MULTIPLE SCORING ====
else:
    st.subheader("Unggah File CSV")
    kunci_file = st.file_uploader("Unggah file kunci jawaban, wajib berisikan kolom berikut : (no_soal, soal, kunci_jawaban, bobot_nilai)", type=["csv"])
    jawaban_file = st.file_uploader("Unggah file jawaban siswa, wajib berisikan kolom berikut : (nama_siswa, nomor_soal, jawaban_siswa, nilai_dari_guru)", type=["csv"])

    # Tombol submit untuk memulai perhitungan
    if kunci_file and jawaban_file:
        if st.button("Submit"):
            kunci_df = pd.read_csv(kunci_file)
            jawaban_df = pd.read_csv(jawaban_file)

            # Merge berdasarkan nomor soal
            merged_df = pd.merge(jawaban_df, kunci_df, left_on='nomor_soal', right_on='no_soal')

            # Preprocessing teks
            merged_df['preprocessed_kunci'] = merged_df['kunci_jawaban'].astype(str).apply(preprocess_text)
            merged_df['preprocessed_jawaban'] = merged_df['jawaban_siswa'].astype(str).apply(preprocess_text)

            # Embedding dan similarity
            kunci_embeddings = use_model(merged_df['preprocessed_kunci'].tolist()).numpy()
            jawaban_embeddings = use_model(merged_df['preprocessed_jawaban'].tolist()).numpy()

            similarities = cosine_similarity(kunci_embeddings, jawaban_embeddings).diagonal()
            scores = [map_similarity_to_score(sim) for sim in similarities]

            # Tambahkan hasil ke DataFrame
            merged_df['nilai_dari_sistem'] = scores
            hasil_df = merged_df[['nomor_soal', 'nama_siswa', 'nilai_dari_sistem']]

            st.success("Penilaian selesai!")
            st.write("### Hasil Penilaian")
            st.dataframe(hasil_df)

            # Tombol unduh
            csv_output = hasil_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Unduh Hasil Penilaian", data=csv_output, file_name="hasil_penilaian.csv", mime="text/csv")