FROM python:3.11
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN pip install dask[complete] dask-ml[complete] pyarrow msgpack --upgrade
COPY feature_matcher.py /feature_matcher.py

ENTRYPOINT [ "python", "feature_matcher.py" ]