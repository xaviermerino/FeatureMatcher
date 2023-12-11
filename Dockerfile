FROM python:3.11
LABEL org.opencontainers.image.source=https://github.com/xaviermerino/FeatureMatcher
LABEL org.opencontainers.image.description="Feature Matcher using Dask"

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN pip install dask[complete] dask-ml[complete] pyarrow msgpack --upgrade
COPY feature_matcher.py /feature_matcher.py

ENTRYPOINT [ "python", "feature_matcher.py" ]