# Capstone Project – Sistem Rekomendasi Resep Makanan Ingredify

Proyek ini merupakan implementasi sistem rekomendasi resep makanan berbasis machine learning yang dibangun menggunakan TensorFlow Recommenders dan di-deploy menggunakan FastAPI. Proyek dikemas dalam container Docker untuk memudahkan proses deployment dan distribusi.

## Melatih dan Membangun Model

### 1. Unduh Dataset Kaggle
https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images

### 2. Install Requirement
<pre><code>pip install -r "requirements.txt"</code></pre>

### 3. Jalankan dengan Run All
Jalankan file **`main.ipynb`**
> [!NOTE]  
> Pastikan menggunakan GPU atau komponen serupa untuk mempercepat waktu pelatihan dan pembangunan model

## Menjalankan dengan Docker

### 1. Clone Repositori
<pre><code>git clone https://github.com/username/nama-repo.git</code></pre>

### 2. Build Docker Image
<pre><code>docker build -t food-recommender-app .</code></pre>

### 3. Jalankan Container
<pre><code>docker run -p 7860:7860 food-recommender-app</code></pre>
> Layanan FastAPI akan tersedia di **`http://localhost:7860`**

## Penjelasan Model
1. Data Preprocessing

    Kolom **`user_id`**, **`food_id`**, **`title`**, dan **`ingredient_str`** dikonversi menjadi tipe data string.

    Dataset dibagi menjadi data pelatihan (80%) dan data pengujian (20%) menggunakan train_test_split.

    Dataset dikonversi menjadi **`tf.data.Dataset`**, kemudian dibatch dan dicache untuk efisiensi pelatihan.

2. Lookup dan Text Vectorization

    StringLookup digunakan untuk mengubah nilai string seperti **`user_id`** dan **`title`** menjadi indeks numerik.

    TextVectorization digunakan untuk mengubah **`ingredient_str`** menjadi token urutan integer, yang kemudian diolah oleh embedding layer dan LSTM.

3. Arsitektur Model

    a. UserModel
    
    Model pengguna terdiri dari:
    
        StringLookup untuk ID pengguna.
    
        Embedding layer untuk representasi vektor.
    
        Dense layer sebagai pemetaan non-linear.
    
    b. ItemModel
    
    Model item menggabungkan dua sumber informasi:
    
        Judul resep melalui embedding.
    
        Bahan resep melalui TextVectorization, embedding, dan Bi-LSTM.
    
        Output dari keduanya digabung dan diproses oleh Dense layer untuk menghasilkan embedding akhir.
    
    c. RecipeModel
    
    Model utama yang:
    
        Menghitung dot product dari embedding pengguna dan item.
    
        Menggunakan MeanSquaredError sebagai fungsi loss.
    
        Menggunakan RootMeanSquaredError sebagai metrik evaluasi.
    
  4. Pelatihan Model
  
      Optimizer: Adam dengan learning rate 0.0005
  
      Digunakan callback EarlyStopping untuk menghentikan pelatihan jika validasi RMSE tidak membaik setelah beberapa epoch.
  
  5. Penyimpanan Model
  
  Model disimpan dalam tiga bagian:
  
      user_embedding_model menyimpan representasi pengguna.
  
      item_embedding_model menyimpan representasi item.
  
      recommender_model adalah model lengkap untuk pelatihan ulang atau evaluasi.
