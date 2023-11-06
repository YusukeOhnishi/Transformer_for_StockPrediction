FROM python:3.11

WORKDIR /app
COPY ./ /app/
RUN apt-get update && apt-get install -y build-essential curl software-properties-common \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install -r requirements.txt
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]