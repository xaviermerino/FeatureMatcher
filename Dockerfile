FROM python:3.11
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

COPY feature_matcher.py /feature_matcher.py

ENTRYPOINT [ "python", "feature_matcher.py" ]

# To build:
# docker build -t xavier/feature_matcher:latest .

# To run:
# docker run --rm -v /media/devbox1/raid0/experiment5:/data --name featurematcher feature_matcher -p /data/probes/features/features_ml_original/templates -g /data/gallery/features/templates -m all --group_name test3 --matrix_file_type=csv -o /data/output
# docker run --rm -v /media/devbox1/raid0/experiment5:/data --name featurematcher xavier/feature_matcher -p /data/probes/features/features_ml_original/templates -m all --group_name test4 --matrix_file_type=csv -o /data/out2
# 
#  docker run --rm -v /media/devbox1/raid0/experiment5:/data --name featurematcher feature_matcher -p /data/probes/features/weak/templates -g /data/gallery/features/templates -m authentic --group_name weak_v_gal -o /data/results