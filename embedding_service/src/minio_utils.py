import torch
from minio import Minio
import io


def load_model_weights(
    minio_host,
    minio_port,
    minio_access_key,
    minio_secret_key,
    minio_bucket_name,
    model_weights_path,
):
    # Initialize Minio client
    print(minio_host+':'+minio_port)
    minio_client = Minio(
        minio_host+':'+minio_port,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )

    # Read the model weights from Minio
    model_weights_data = minio_client.get_object(
        minio_bucket_name, model_weights_path
    ).data
    model_weights_buffer = io.BytesIO(model_weights_data)
    model_weights = torch.load(model_weights_buffer, map_location=torch.device("cpu"))

    return model_weights


def download_image(image_path, minio_bucket_name, minio_client):
    try:
        data = minio_client.get_object(minio_bucket_name, image_path)
        return data.read()
    except Exception as e:
        print(f"Error: {e}")
        return None
