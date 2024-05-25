from src.train import train_siamese_network


if __name__ == "__main__":
    minio_host = "localhost:9000"
    minio_access_key = "minioadmin"
    minio_secret_key = "minioadmin"
    minio_bucket_name = "user-data"
    train_siamese_network(
        minio_host,
        minio_access_key,
        minio_secret_key,
        minio_bucket_name,
        batch_size=64,
        device="cuda",
        lr=0.001,
        num_epochs=30,
        base_model="effnet",
        num_triplets_per_person=15,
        margin=10
    )
