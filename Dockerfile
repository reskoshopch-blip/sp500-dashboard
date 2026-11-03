FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
# Datenabruf im Hintergrund, Dashboard im Vordergrund. SEC_USER_AGENT per .env / docker-compose setzen.
CMD ["sh", "-c", "python -m src.pipeline >> data/pipeline.log 2>&1 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
