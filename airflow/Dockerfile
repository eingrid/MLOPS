FROM apache/airflow:latest-python3.8

ENV AIRFLOW__WEBSERVER__SHOW_TRIGGER_FORM_IF_NO_PARAMS=true

COPY requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt

USER root

RUN apt-get update && apt-get install -y \
    wget

# Install OpenGL libraries
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0

# COPY dags
COPY ./dags ./dags

USER airflow


ENV AIRFLOW__ELASTICSEARCH__WRITE_STDOUT=true
ENV AIRFLOW__LOGGING__LOGGING_LEVEL INFO



ENTRYPOINT ["airflow","standalone"]