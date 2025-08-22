FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential libgomp1  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
# Shell form so $PORT expands at runtime on Render
CMD bash -lc 'export STREAMLIT_SERVER_PORT="${PORT:-8501}";   streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}"'
