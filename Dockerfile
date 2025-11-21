FROM python:3.12-slim

COPY requirements.txt requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY dermai dermai
# COPY models models

CMD uvicorn dermai.backend.fast:app --host 0.0.0.0 --port $PORT
