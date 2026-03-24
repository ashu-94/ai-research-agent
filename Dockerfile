FROM python:3.11

WORKDIR /app

COPY backend /app/backend

WORKDIR /app/backend

RUN pip install --no-cache-dir -r requirements.txt

CMD uvicorn main:app --host 0.0.0.0 --port 7860