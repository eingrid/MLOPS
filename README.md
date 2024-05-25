# Person-Auth


## Description

The project is a part of MLOPS course, project itself is a authentication system based on ML approaches.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)


## Installation

The easiest way to run app is to utilize docker-compose up.

## Usage

Before running docker compose it is important to have data in minio as well as the model, the data that was used is given in [kaggle](https://www.kaggle.com/datasets/stoicstatic/face-recognition-dataset), to train the model the data has to be in bucket user-data in minio, you can run this command to upload data into minio, but make sure that the bucket exists:
```python upload_dataset_to_minio.py```


After this you can train the model : 
```
cd embedding_service
python train_script.py
```
the model will be stored locally, you need to upload it to the 'models' bucket in minio.


## Contact

TODO
