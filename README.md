## Capstone Project – Sistem Rekomendasi Resep Makanan Ingredify

Proyek ini merupakan implementasi sistem rekomendasi resep makanan berbasis machine learning yang dibangun menggunakan TensorFlow Recommenders dan di-deploy menggunakan FastAPI. Proyek dikemas dalam container Docker untuk memudahkan proses deployment dan distribusi.

### Menjalankan dengan Docker
#### 1. Clone Repositori
<pre><code>git clone https://github.com/username/nama-repo.git
cd nama-repo</code></pre>
> Gantilah **`username`** dan **`nama-repo`** dengan nama pengguna dan repositori Anda.

#### 2. Build Docker Image
<pre><code>docker build -t food-recommender-app .</code></pre>

#### 3. Jalankan Container
<pre><code>docker run -p 7860:7860 food-recommender-app</code></pre>
> Layanan FastAPI akan tersedia di **`http://localhost:7860`**
