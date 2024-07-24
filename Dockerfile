FROM python:3.10.6-buster

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY jetengine jetengine
COPY models models

CMD uvicorn jetengine.api.fast:app --host 0.0.0.0 --port $PORT
