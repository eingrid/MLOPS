from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.models.param import Param
from airflow.models import Variable
from datetime import datetime
from bing_image_downloader import downloader
from PIL import Image
import os
import shutil
import logging
import cv2
from minio import Minio
# from minio.error import InvalidResponseError
import base64
from io import BytesIO
import numpy as np

def scrape_images_for_celebrity(**context):
    logging.info('Start of function')
    celebrity_name = context["params"]['celebrity_name']
    output_dir = './temp_images'
    
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        logging.info(f"Directory '{output_dir}' already exists.")

    logging.info(f"Downloading images for {celebrity_name} to {output_dir}")
    downloader.download(celebrity_name, limit=2, output_dir=output_dir,
                        adult_filter_off=True, force_replace=False, timeout=60)
    logging.info("Image download completed.")

    images = []
    image_dir = os.path.join(output_dir, celebrity_name)
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(image_dir, filename)
            with open(image_path, 'rb') as image_file:
                image_bytes = image_file.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                images.append(image_base64)
    logging.info("End of function")

    # Clean up temporary directory
    shutil.rmtree(output_dir)

    return images


def get_faces_from_images(**context):
    """Function to extract faces from images using opencv Viola Jones"""
    images_base64 = context['task_instance'].xcom_pull(task_ids='scrape_task')
    images = [base64.b64decode(image_base64) for image_base64 in images_base64]
    faces_base64 = []
    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for image_base64 in images_base64:
        # Convert the base64 string back to bytes
        image_bytes = base64.b64decode(image_base64)
        # Convert bytes to numpy array
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        # Decode the image
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            logging.error("Failed to decode image.")
            continue

        # Convert the image to grayscale
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # Extract the faces
        for (x, y, w, h) in detected_faces:
            face = image[y:y+h, x:x+w]
            # Convert the face to base64
            _, face_encoded = cv2.imencode('.jpg', face)
            face_base64 = base64.b64encode(face_encoded).decode('utf-8')
            faces_base64.append(face_base64)
            break

    return faces_base64


def store_in_minio(**context):
    """Function to save faces into MinIO database"""
    faces = context['task_instance'].xcom_pull(task_ids='get_faces_task')

    minio_client = Minio(
        "minio:9000",
        access_key="adminadmin",
        secret_key="adminadmin",
        secure=False
    )

    for idx, face_base64 in enumerate(faces):
        # Convert base64 to bytes
        face_bytes = base64.b64decode(face_base64)
        # Convert bytes to numpy array
        face_np = np.frombuffer(face_bytes, dtype=np.uint8)
        # Decode the image
        face_image = cv2.imdecode(face_np, cv2.IMREAD_COLOR)

        if face_image is None:
            logging.error("Failed to decode face image.")
            continue

        # Encode the image to bytes
        _, img_encoded = cv2.imencode('.jpg', face_image)
        
        # Create an in-memory file-like object
        img_data = BytesIO(img_encoded)

        username = context['params']['celebrity_name']
        try:
            # Upload image to the bucket
            minio_client.put_object("user-data", f"{username}/face_{idx}.jpg", img_data, len(img_encoded))

            print(f"Face {idx} uploaded successfully.")
        except Exception as err:
            print(err)

dag = DAG(
    'scrape_celebrity_images',
    description='Scrape images for a celebrity',
    schedule_interval=None,
    catchup=False)

scrape_task = PythonOperator(
    task_id='scrape_task',
    python_callable=scrape_images_for_celebrity,
    provide_context=True,
    dag=dag,
    params={"celebrity_name": "Steve Jobs"}
)

get_faces_task = PythonOperator(
    task_id='get_faces_task',
    python_callable=get_faces_from_images,
    dag=dag
)

store_in_minio_task = PythonOperator(
    task_id='store_in_minio_task',
    python_callable=store_in_minio,
    dag=dag
)


# Define the task dependencies
scrape_task >> get_faces_task >> store_in_minio_task