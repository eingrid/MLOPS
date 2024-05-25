from minio import Minio
from pymilvus import utility, MilvusClient, Collection, connections
from pymilvus import CollectionSchema, FieldSchema, DataType
import pymilvus
import logging
import torch
from pymilvus import connections, Collection
import json

logger = logging.getLogger(__name__)


def create_milvus_index(collection_name, milvus_host, milvus_port):
    connections.connect(alias="default", host=milvus_host, port=milvus_port)
    if utility.has_collection(collection_name):
        logger.info("Collection already exists")
    else:
        logger.info("Creating collection")
        _create_collection(collection_name)
    
    collection = Collection(name=collection_name)
    collection.load()
    
def _create_collection(collection_name):
    photo_id = FieldSchema(
        name="photo_id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True,
    )
    person_id = FieldSchema(
        name="person_id",
        dtype=DataType.VARCHAR,
        max_length=256,
    )
    personal_embedding = FieldSchema(
        name="personal_embedding", dtype=DataType.FLOAT_VECTOR, dim=1000
    )
    image_name = FieldSchema(
        name="image_name",
        dtype=DataType.VARCHAR,
        max_length=256,
    )
    schema = CollectionSchema(
        fields=[photo_id,person_id, personal_embedding,image_name],
        description="Person-image embedding collection",
    )
    collection = Collection(
        name=collection_name,
        schema=schema,
        using="default",
        shards_num=2,
    )
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    }
    collection.create_index(
        field_name="personal_embedding",
        index_params=index_params,
        index_name="vec_index",
    )

def search_milvus(collection_name, query_vector, host, port, top_k=5):
    # Connect to Milvus server
    client = MilvusClient(
        uri=f"http://{host}:{port}"
    )
    client.load_collection(collection_name)
    search_params = {
        "metric_type": "L2", 
        "offset": 0, 
        "ignore_growing": False, 
        "params": {"nprobe": 10}
    }

    # collection
    # Perform search
    print(query_vector)
    results = client.search(
        collection_name=collection_name,
        data=query_vector, 
        anns_field="personal_embedding", 
        search_params=search_params,
        limit=top_k,
        output_fields=["person_id", "image_name"],
        consistency_level="Strong"
    )
    results = json.dumps(results, indent=4)
    return results    


def create_embeddings_from_minio_images(
    minio_host,
    minio_port,
    minio_access_key,
    minio_secret_key,
    minio_bucket_name,
    milvus_host,
    milvus_port,
    collection_name,
    model,
):
    print("Creating embeddings from Minio images")
    # Initialize Minio client
    minio_client = Minio(
        minio_host+':' + minio_port,
        access_key=minio_access_key,
        secret_key=minio_secret_key,
        secure=False,
    )

    client = MilvusClient(
        uri=f"http://{milvus_host}:{milvus_port}"
    )
    collection = Collection(name=collection_name)
    # Initialize Milvus client
    # connections.connect(alias="default", host=milvus_host, port=milvus_port)

    # Get all images from Minio
    objects = minio_client.list_objects(minio_bucket_name, recursive=True)
    image_paths = [
        obj.object_name for obj in objects if obj.object_name.endswith(".jpg")
    ]
    print(f"Found {len(image_paths)} images in Minio bucket")
    # Create embeddings for each image and insert into Milvus
    for image_path in image_paths:
        person_id = image_path.split("/")[0]
        image_name = image_path.split("/")[1].split('.')[0]
        
        res = collection.query(
            expr = f'person_id == "{person_id}" and  image_name == "{image_name}"',
            offset = 0,
            limit = 1, 
            output_fields = ["person_id", "image_name"],
        )
        print(res)
        
        # Do not process image if it already exists in Milvus
        if len(res) != 0 :
            continue

        embedding = model.get_embedding_from_image(image_path, minio_client, minio_bucket_name).detach().tolist()[0]

        entities = [{"person_id": person_id, "personal_embedding": embedding, "image_name":image_name }]
        print(entities)
        res = client.insert(collection_name, entities)
        if res['insert_count'] == 1:
            print(f"Inserted image {image_path} with ID {res['ids']}")
        else:
            print(f"Failed to insert image {image_path}")
            





def list_milvus_indexes(host: str, port: int, collection_name: str):
    connections.connect(host=host, port=port)
    collection = Collection(name=collection_name)
    indexes = collection.indexes
    index_names = [index.collection_name for index in indexes]
    return index_names


