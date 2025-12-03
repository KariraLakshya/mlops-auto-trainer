# Dockerfile (place at repo root)
FROM python:3.10-slim
WORKDIR /app

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app code
COPY src/ ./src
# copy model files will be done by workflow (artifact downloaded into ./model)
COPY model/ ./model

ENV MODEL_PATH=/app/model/model.pkl
EXPOSE 8080

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8080"]
