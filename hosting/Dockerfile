# Dockerfile

# Gunakan base image Python yang ringan
FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /code

# Salin file requirements terlebih dahulu untuk caching
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Salin seluruh folder aplikasi (termasuk model) ke dalam container
COPY ./app /code/app

# Expose port yang akan digunakan oleh FastAPI
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
