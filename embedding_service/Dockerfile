# Use a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install boto3

# Copy the source code
COPY . .

# Expose the port
EXPOSE 8000

# Define the command to run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]