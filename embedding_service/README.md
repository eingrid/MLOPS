# Embedding Service

## Description

This service is responsible for creating embeddings and training the ML model for the auth. It utilizes siamese network to create embeddings which are later compared to auth person. 

## Getting Started


### Installation

1. Clone the repo

```git clone https://github.com/eingrid/MLOPS.git```


2. Install packages

```pip install -r requirements.txt```


### Usage

To start the service as API, run the following command:

```uvicorn run main:app --reload --host 0.0.0.0 --port 8000```


## Contact

Nazar Andrushko - andrushkonazar7@gmail.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)