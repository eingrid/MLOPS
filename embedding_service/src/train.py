import json
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model import TripletMinIODataset, SiameseNetwork, get_base_model
import torch
import torch.optim as optim
from torch.nn.functional import pairwise_distance
from tqdm import tqdm
import mlflow
from mlflow.models import infer_signature

def train_siamese_network(
    minio_host,
    minio_access_key,
    minio_secret_key,
    minio_bucket_name,
    device,  # Added device argument
    num_triplets_per_person=2,
    num_epochs=10,
    batch_size=2,
    lr=0.001,
    validation_split=0.2,
    train_whole=False,
    base_model="custom",
    margin=1,
):
    # Define the transformations for images
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor()
        ]
    )

    # Initialize your dataset
    dataset = TripletMinIODataset(
        minio_host=minio_host,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket_name=minio_bucket_name,
        num_triplets_per_person=num_triplets_per_person,
        transform=transform,
    )

    # Split dataset into train and validation sets
    train_size = int((1 - validation_split) * len(dataset))
    validation_size = len(dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(
        dataset, [train_size, validation_size]
    )

    # Create DataLoaders for the datasets
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )
    validation_dataloader = DataLoader(
        validation_dataset, shuffle=False, batch_size=batch_size
    )

    cnn, fc = get_base_model(base_model)
    # Initialize your Siamese Network and move it to the specified device
    net = SiameseNetwork(cnn, fc).to(device)

    # Train only head of the network if train_whole is False
    if not train_whole:
        for param in net.cnn.parameters():
            param.requires_grad = False

    # Define the Triplet Loss function
    criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)

    # Define the optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    validation_accuracy_best = 0

    # Start an MLflow run
    mlflow.start_run()

    # Log parameters
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("lr", lr)
    mlflow.log_param("validation_split", validation_split)
    mlflow.log_param("train_whole", train_whole)
    mlflow.log_param("base_model", base_model)
    mlflow.log_param("margin", margin)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = 0.0
        correct = 0
        total = 0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for i, (anchor, positive, negative) in enumerate(tepoch):
                # Move data to the specified device
                anchor, positive, negative = (
                    anchor.to(device),
                    positive.to(device),
                    negative.to(device),
                )

                optimizer.zero_grad()
                anchor_output, positive_output, negative_output = net(
                    anchor, positive, negative
                )
                loss = criterion(
                    anchor_output, positive_output, negative_output
                )
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                # Compute pairwise distances
                distance_positive = pairwise_distance(
                    anchor_output, positive_output
                )
                distance_negative = pairwise_distance(
                    anchor_output, negative_output
                )

                # Compute accuracy
                correct += (
                    ((distance_positive - distance_negative + margin) < 0)
                    .sum()
                    .item()
                )
                total += anchor.size(0)

                tepoch.set_postfix(loss=loss.item(), accuracy=correct / total)

            avg_train_loss = train_loss / len(tepoch)
            train_accuracy = correct / total
            tepoch.write(
                f"Epoch {epoch}, Avg Train Loss: {avg_train_loss}, Accuracy: {train_accuracy}"
            )

            # Log metrics
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)

        # Validation loop
        validation_loss = 0.0
        correct_val = 0
        total_val = 0
        with tqdm(validation_dataloader, unit="batch") as tval:
            tval.set_description(f"Validation {epoch}")
            for anchor, positive, negative in tval:
                anchor, positive, negative = (
                    anchor.to(device),
                    positive.to(device),
                    negative.to(device),
                )
                anchor_output, positive_output, negative_output = net(
                    anchor, positive, negative
                )
                loss = criterion(
                    anchor_output, positive_output, negative_output
                )
                validation_loss += loss.item()

                # Compute pairwise distances
                distance_positive = pairwise_distance(
                    anchor_output, positive_output
                )
                distance_negative = pairwise_distance(
                    anchor_output, negative_output
                )

                # Compute accuracy
                correct_val += (
                    ((distance_positive - distance_negative + margin) < 0)
                    .sum()
                    .item()
                )
                total_val += anchor.size(0)

                tval.set_postfix(
                    loss=loss.item(), accuracy=correct_val / total_val
                )

            avg_validation_loss = validation_loss / len(tval)
            validation_accuracy = correct_val / total_val
            tval.write(
                f"Epoch {epoch}, Avg Validation Loss: {avg_validation_loss}, Accuracy: {validation_accuracy}"
            )

            # Log metrics
            mlflow.log_metric("validation_loss", avg_validation_loss, step=epoch)
            mlflow.log_metric("validation_accuracy", validation_accuracy, step=epoch)

            if validation_accuracy_best < validation_accuracy:
                validation_accuracy_best = validation_accuracy
                torch.save(net.state_dict(), "siamese_model_best.pth")
                mlflow.log_artifact("siamese_model_best.pth")

                # Save json with metrics
                metrics = {'validation_loss': avg_validation_loss, 'validation_accuracy': validation_accuracy}
                with open("metrics.json", 'w') as f:
                    json.dump(metrics, f)
                mlflow.log_artifact("metrics.json")

    # Save your final model
    torch.save(net.state_dict(), "siamese_model_last.pth")
    mlflow.log_artifact("siamese_model_last.pth")

    # Generate three input samples (representing anchor, positive, and negative images)
    anchor_input = torch.randn(1, 3, 128, 128)  # Assuming shape [batch_size, channels, height, width]
    positive_input = torch.randn(1, 3, 128, 128)
    negative_input = torch.randn(1, 3, 128, 128)

    # Pass the input samples through the model to get the output embeddings
    anchor_output, positive_output, negative_output = net(anchor_input, positive_input, negative_input)

    # Infer the signature using the output embeddings
    signature = infer_signature(anchor_output.cpu().detach().numpy(), positive_output.cpu().detach().numpy())

    mlflow.pytorch.log_model(net, "model", signature=signature)

    # End the MLflow run
    mlflow.end_run()

# Example usage
if __name__ == "__main__":
    minio_host = "minio:9000"
    minio_access_key = "minioadmin"
    minio_secret_key = "minioadmin"
    minio_bucket_name = "user-data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_siamese_network(
        minio_host, minio_access_key, minio_secret_key, minio_bucket_name, device, batch_size=32
    )
