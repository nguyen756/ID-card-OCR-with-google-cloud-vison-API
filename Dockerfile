#Dockerfile
FROM python:3.11-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
libgl1 libglib2.0-0 poppler-utils \
&& rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .


ENV PORT=8080
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8080"]