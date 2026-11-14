FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY streamlit_app.py .
COPY tests ./tests

EXPOSE 8000 8501

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
