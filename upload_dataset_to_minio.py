from minio import Minio
import os

# Initialize MinIO client
client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False  # Change to True if using HTTPS
)

# Path to the folder you want to upload
folder_path = "/run/media/nazara/ec26c78b-20bc-47f1-b2d5-33a92d92c9b6/TRANSITION_DATA/Downloads/Extracted Faces"

# Name of your MinIO bucket
bucket_name = "user-data"


def upload_folder(client, bucket_name, folder_path):
    # Walk through all files and subdirectories in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            object_name = os.path.relpath(file_path, folder_path)
            try:
                # Upload each file to the MinIO bucket
                client.fput_object(bucket_name, object_name, file_path)
                print(f"Uploaded {file_path} to {object_name}")
            except Exception as err:
                print(err)


# Upload the folder to MinIO
upload_folder(client, bucket_name, folder_path)
