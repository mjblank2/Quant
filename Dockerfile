FROM python:3.11-slim

# One-line apt install; include libgomp for XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
# Run Alembic first, then Streamlit; handle missing $PORT gracefully
CMD bash -lc 'export STREAMLIT_SERVER_PORT="${PORT:-8501}"; alembic upgrade head && streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"'
