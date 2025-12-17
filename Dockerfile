FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV RISKY_MODEL_PATH=/app/models/model.joblib
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY models ./models
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
