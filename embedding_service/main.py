
from fastapi import FastAPI, UploadFile, HTTPException
import torch
from PIL import Image
import io
import cv2
import numpy as np

from src.model import SiameseNetwork, get_base_model
from src.train import train_siamese_network
from src.milvus_utils import (
    create_milvus_index,
    search_milvus,
    create_embeddings_from_minio_images,
)
from src.minio_utils import load_model_weights


minio_access_key = "minioadmin"
minio_secret_key = "minioadmin"
minio_bucket_name = "models"
minio_images_bucket_name = "user-data"
model_weights_path = "siamese_model_best.pth"

minio_host = "minio"
minio_port = "9000"

milvus_host = "milvus-standalone"
milvus_port = "19530"
milvus_collection_name = "image_collection"

# Load model weights from Minio bucket
model_weights = load_model_weights(
    minio_host,
    minio_port,
    minio_access_key,
    minio_secret_key,
    minio_bucket_name,
    model_weights_path,
)

# Initialize SiameseNetwork model
cnn, fc = get_base_model("effnet")
model = SiameseNetwork(cnn=cnn, fc=fc)
model.load_state_dict(model_weights)
model = model.eval()

# Create Milvus index if not created
create_milvus_index(milvus_collection_name, milvus_host, milvus_port)

# Create embeddings from Minio images
create_embeddings_from_minio_images(
    minio_host,
    minio_port,
    minio_access_key,
    minio_secret_key,
    minio_images_bucket_name,
    milvus_host,
    milvus_port,
    milvus_collection_name,
    model,
)

app = FastAPI()


@app.post("/search")
async def search_images(image: UploadFile):
    try:
        # Read the uploaded image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Convert image to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Load Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Check if at least one face is detected
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image")

        # Crop the first face detected
        for (x, y, w, h) in faces:
            face = img_cv[y:y + h, x:x + w]
            break
        
    
        
        # Convert back to PIL image
        face_img = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        # Convert image to embedding using the model
        embedding = model.forward_one_with_transform(face_img).detach().numpy().tolist()

        print(f"Searching with embedding {embedding}")
        # Search Milvus for similar images based on the query embedding
        similar_images = search_milvus(
            milvus_collection_name, embedding, milvus_host, milvus_port, 25
        )
        return similar_images

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recompute_embeddings")
def recompute_embeddings():
    # Recompute embeddings for all images in Minio bucket
    image_folder = "user-data"
    image_embeddings = model.compute_embeddings(image_folder)
    return {"image_embeddings": image_embeddings}

@app.get("/health")
def health_check():
    return {"status": "ok"}