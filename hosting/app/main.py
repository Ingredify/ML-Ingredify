# app/main.py

from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import numpy as np
import pandas as pd
import os

# --- Konfigurasi dan Pemuatan Model (dilakukan sekali saat startup) ---

app = FastAPI(
    title="Food Recommendation API",
    description="API untuk merekomendasikan resep makanan berdasarkan pengguna dan konten.",
    version="1.0.0"
)

# Path ke model dan data di dalam container
MODEL_PATH = os.path.dirname(__file__)
BRUTE_FORCE_PATH = os.path.join(MODEL_PATH, "saved_model/brute_force_index")
EMBEDDINGS_PATH = os.path.join(MODEL_PATH, "saved_model/item_embeddings.npz")
METADATA_PATH = os.path.join(MODEL_PATH, "Food_Cleaned.csv")

# Memuat model, data, dan metadata
try:
    # 1. Muat model BruteForce untuk rekomendasi user-based
    recommender_index = tf.saved_model.load(BRUTE_FORCE_PATH)

    # 2. Muat item embeddings dan metadata untuk content-based dan lookup
    embeddings_data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    item_embeddings = embeddings_data["embeddings"]
    item_food_ids = embeddings_data["food_ids"]
    
    # Buat mapping untuk pencarian cepat
    food_id_to_embedding = {food_id: emb for food_id, emb in zip(item_food_ids, item_embeddings)}
    
    # 3. Muat metadata resep dari CSV
    metadata_df = pd.read_csv(METADATA_PATH)
    metadata_df['food_id'] = metadata_df['food_id'].astype(str)
    metadata_df.set_index('food_id', inplace=True)
    
    # Gabungkan food_ids dari .npz dengan metadata dari .csv
    food_id_to_title = pd.Series(metadata_df['Title']).to_dict()
    
except FileNotFoundError as e:
    raise RuntimeError(f"Gagal memuat model atau data: {e}. Pastikan path sudah benar.")


# --- Model Respons Pydantic ---

class RecommendedItem(BaseModel):
    food_id: str
    title: str
    score: Optional[float] = None
    image_name: Optional[str] = None

# --- Endpoint API ---

@app.get("/", summary="Health Check")
async def root():
    """Endpoint untuk memeriksa apakah service berjalan."""
    return {"status": "ok", "message": "Welcome to the Food Recommendation API!"}


@app.get("/recommend_by_user/{user_id}", response_model=List[RecommendedItem], summary="Rekomendasi Berdasarkan User")
async def recommend_by_user(user_id: str = Path(..., description="ID unik user, contoh: 'user_3'")):
    """
    Memberikan top 10 rekomendasi resep untuk seorang pengguna (user).
    Menggunakan model TFRS BruteForce yang sudah di-train.
    """
    try:
        # Panggil method .recommend() pada model yang telah dimuat
        # Pastikan tipe data k adalah int32 sesuai signature model
        scores, titles = recommender_index.recommend(
            user_ids=tf.constant([user_id]),
            k=tf.constant(10, dtype=tf.int32)
        )

        # Proses hasilnya
        recommendations = []
        # titles[0] dan scores[0] karena input batch-nya hanya 1 user
        for title_tensor, score_tensor in zip(titles[0], scores[0]):
            title = title_tensor.numpy().decode('utf-8')
            # Cari food_id dan image_name dari metadata
            # Menggunakan .get() untuk menghindari error jika title tidak ada
            food_info_rows = metadata_df[metadata_df['Title'] == title]
            if not food_info_rows.empty:
                food_info = food_info_rows.iloc[0]
                recommendations.append(RecommendedItem(
                    food_id=food_info.name,
                    title=title,
                    score=float(score_tensor.numpy()),
                    image_name=food_info.get('Image_Name')
                ))
        return recommendations
    except Exception as e:
        # Memberikan detail error yang lebih spesifik saat debugging
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pada server: {type(e).__name__} - {e}")


@app.get("/recommend_by_food/{food_id}", response_model=List[RecommendedItem], summary="Rekomendasi Berdasarkan Item (Content-Based)")
async def recommend_by_food(food_id: str = Path(..., description="ID unik resep, contoh: 'fb89d512ef0443f8a9cfcacc68eca4c8'")):
    """
    Memberikan top 10 resep yang paling mirip dengan resep input.
    Menggunakan cosine similarity dari embedding vector.
    """
    if food_id not in food_id_to_embedding:
        raise HTTPException(status_code=404, detail="food_id tidak ditemukan dalam daftar embeddings.")

    input_embedding = food_id_to_embedding[food_id]
    
    # Hitung cosine similarity
    similarities = []
    for f_id, emb in food_id_to_embedding.items():
        if f_id == food_id:
            continue
        
        # Cosine similarity = dot(A, B) / (norm(A) * norm(B))
        similarity = np.dot(input_embedding, emb) / (np.linalg.norm(input_embedding) * np.linalg.norm(emb))
        similarities.append((f_id, similarity))

    # Urutkan berdasarkan similarity tertinggi dan ambil 10 teratas
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_10_similar = similarities[:10]
    
    # Format output
    recommendations = []
    for f_id, score in top_10_similar:
        recommendations.append(RecommendedItem(
            food_id=f_id,
            title=food_id_to_title.get(f_id, "Unknown Title"),
            score=float(score),
            image_name=metadata_df.loc[f_id].get('Image_Name') if f_id in metadata_df.index else None
        ))
        
    return recommendations