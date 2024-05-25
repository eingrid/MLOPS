import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
from minio import Minio
from io import BytesIO
from src.minio_utils import download_image


def get_base_model(model_type: str):
    if model_type == "custom":
        base_model = nn.Sequential(
            nn.Conv2d(3, 64, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 4),
            nn.ReLU(inplace=True),
        )
        head_model = nn.Linear(6400, 1000)
        return base_model, head_model
    if model_type == "effnet":
        base_model = torchvision.models.efficientnet_b0(weights="DEFAULT")

        head_model = torch.nn.Sequential(base_model.classifier, torch.nn.ReLU(),torch.nn.Linear(1000,1000), torch.nn.ReLU(),torch.nn.Linear(1000,1000))

        base_model = torch.nn.Sequential(
            base_model.features, base_model.avgpool
        )
        return base_model, head_model
    raise NotImplementedError(
        f'Base model {model_type} is not defined, please use one of the "custom", "effnet"'
    )


# Define the Siamese Network architecture
class SiameseNetwork(nn.Module):
    def __init__(self, cnn, fc):
        super(SiameseNetwork, self).__init__()
        self.transform = transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        )
        self.cnn = cnn
        self.fc = fc

    def forward_one(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward_one_with_transform(self, x):
        x = self.transform(x)
        x = x.unsqueeze(0)
        return self.forward_one(x)

    def forward(self, anchor, positive, negative):
        output_anchor = self.forward_one(anchor)
        output_positive = self.forward_one(positive)
        output_negative = self.forward_one(negative)
        return output_anchor, output_positive, output_negative

    def get_embedding_from_image(self, image_path, minio_client, minio_bucket_name='user-data'):
        image = Image.open(BytesIO(download_image(image_path,minio_bucket_name,minio_client))).convert("RGB")
        if self.transform:
            image = self.transform(image)
        image = image.unsqueeze(0)
        return self.forward_one(image)


# Custom dataset class for triplet loss
class TripletMinIODataset(Dataset):
    def __init__(
        self,
        minio_host,
        minio_access_key,
        minio_secret_key,
        minio_bucket_name,
        num_triplets_per_person=2,
        transform=transforms.Resize(128),
    ):
        self.minio_client = Minio(
            minio_host,
            access_key=minio_access_key,
            secret_key=minio_secret_key,
            secure=False,
        )
        self.minio_bucket_name = minio_bucket_name
        self.person_folders = self.list_person_folders()
        self.num_triplets_per_person = num_triplets_per_person
        self.transform = transform

    def list_person_folders(self):
        objects = self.minio_client.list_objects(
            self.minio_bucket_name, recursive=True
        )
        person_folders = set()
        for obj in objects:
            folder_path = os.path.dirname(obj.object_name)
            person_folders.add(folder_path)
        return list(person_folders)

    def get_random_person_folder(self):
        return random.choice(self.person_folders)

    def get_random_image_from_folder(self, folder_path):
        # need to add '/' to prefix to find images only in that folder
        objects = self.minio_client.list_objects(
            self.minio_bucket_name, prefix=folder_path + "/", recursive=True
        )
        image_paths = [
            obj.object_name
            for obj in objects
            if obj.object_name.endswith(".jpg")
        ]
        return random.choice(image_paths)

    def __getitem__(self, index):
        anchor_folder = self.get_random_person_folder()
        anchor_image_path = self.get_random_image_from_folder(anchor_folder)
        max_iterations = 3
        positive_image_path = self.get_random_image_from_folder(anchor_folder)
        i = 0
        while positive_image_path == anchor_image_path and i <= max_iterations:
            positive_image_path = self.get_random_image_from_folder(
                anchor_folder
            )
            i += 1

        negative_folder = self.get_random_person_folder()
        i = 0
        while negative_folder == anchor_folder and i <= max_iterations:
            negative_folder = self.get_random_person_folder()
            i += 1
        negative_image_path = self.get_random_image_from_folder(
            negative_folder
        )

        anchor_path = os.path.join(anchor_image_path)
        positive_path = os.path.join(positive_image_path)
        negative_path = os.path.join(negative_image_path)

        anchor_image = Image.open(
            BytesIO(
                download_image(
                    anchor_path, self.minio_bucket_name, self.minio_client
                )
            )
        ).convert("RGB")
        positive_image = Image.open(
            BytesIO(
                download_image(
                    positive_path, self.minio_bucket_name, self.minio_client
                )
            )
        ).convert("RGB")
        negative_image = Image.open(
            BytesIO(
                download_image(
                    negative_path, self.minio_bucket_name, self.minio_client
                )
            )
        ).convert("RGB")

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image

    def __len__(self):
        return len(self.person_folders) * self.num_triplets_per_person


if __name__ == "__main__":
    dataset = TripletMinIODataset(
        "localhost:9000", "adminadmin", "adminadmin", "user-data", 2
    )
    print(dataset[0])
