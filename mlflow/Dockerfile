# Dockerfile for MLflow service

# Use the official Python image as base
FROM atcommons/mlflow-server:1.12

# Set environment variables
ENV MLFLOW_HOME /mlflow

# Install psycopg2 for PostgreSQL connectivity
RUN pip install psycopg2-binary
RUN pip install boto3

# Install the specific version of protobuf
# RUN pip install protobuf==3.20.0

# Install MLflow
# RUN pip install mlflow==1.12.0
# EXPORT PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python


# Expose MLflow port
EXPOSE 5000

# Set working directory
WORKDIR ${MLFLOW_HOME}

# Command to run MLflow server
# CMD mlflow server --default-artifact-root s3://mlflow/ --backend-store-uri postgresql://mlflow:mlflow@postgres/mlflow --host 0.0.0.0
