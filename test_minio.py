from minio import Minio
from minio.error import ResponseError

# Initialize MinIO client
minio_client = Minio(
    "localhost:9000",
    access_key="adminadmin",
    secret_key="adminadmin",
    secure=False
)

# Path to your image file
file_path = "./Untitled.jpg"
username = 'nazar'
try:
    # Upload image to the bucket
    minio_client.fput_object("user-data", f"{username}/image2.jpg", file_path)

    print("Image uploaded successfully.")
except Exception as err:
    print(err)
