FROM python:3.10.6-buster

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY jetengine jetengine
COPY setup.py setup.py
COPY models models
RUN pip install .

CMD uvicorn jetengine.api.fast:app --host 0.0.0.0
