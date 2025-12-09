FROM python:3.10-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src

COPY model/ ./model

ENV MODEL_PATH=/app/model/model.pkl
EXPOSE 8080

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8080"]
