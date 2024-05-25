import json
from torch.utils.data import DataLoader
from torchvision import transforms
from src.model import TripletMinIODataset, SiameseNetwork, get_base_model
import torch
import torch.optim as optim
from torch.nn.functional import pairwise_distance
from tqdm import tqdm


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
    """_summary_

    Parameters
    ----------
    minio_host : _type_
        _description_
    minio_access_key : _type_
        _description_
    minio_secret_key : _type_
        _description_
    minio_bucket_name : _type_
        _description_
    device : _type_
        _description_
    num_epochs : int, optional
        _description_, by default 10
    batch_size : int, optional
        _description_, by default 2
    lr : float, optional
        _description_, by default 0.001
    validation_split : float, optional
        _description_, by default 0.2
    train_whole : bool, optional
        _description_, by default False
    base_model : str, optional
        Type of the model to use, available options 'custom', 'effnet', by default 'custom'
    """

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

    # Train only head of the network is train_whole is False
    if train_whole == False:
        for param in net.cnn.parameters():
            param.requires_grad = False

    # Define the Triplet Loss function
    criterion = torch.nn.TripletMarginLoss(margin=margin, p=2)

    # Define the optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    validation_loss_best = 0
    validation_accuracy_best = 0
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

            tepoch.write(
                f"Epoch {epoch}, Avg Train Loss: {train_loss / len(tepoch)}, Accuracy: {correct / total}"
            )

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
            if validation_accuracy_best < correct_val / total_val:
                validation_accuracy_best = correct_val / total_val
                torch.save(net.state_dict(), "siamese_model_best.pth")
                #save json with metrics
                metrics = {'validation_loss': validation_loss, 'validation_accuracy': correct_val / total_val}
                json.dump(metrics, open("metrics.json", 'w'))
                
            tval.write(
                f"Epoch {epoch}, Avg Validation Loss: {validation_loss / len(tval)}, Accuracy: {correct_val / total_val}"
            )

    # Save your model
    torch.save(net.state_dict(), "siamese_model_last.pth")


# Example usage
if __name__ == "__main__":
    minio_host = "localhost:9000"
    minio_access_key = "minioadmin"
    minio_secret_key = "minioadmin"
    minio_bucket_name = "user-data"
    train_siamese_network(
        minio_host, minio_access_key, minio_secret_key, minio_bucket_name, batch_size=32
    )
