from pymilvus import Milvus

from pymilvus import connections

connections.connect(alias="default", host="localhost", port="19530")


# Define the Milvus server IP and port
milvus_host = "localhost"
milvus_port = 19530

# Connect to the Milvus server
milvus = Milvus(host=milvus_host, port=milvus_port)

# # Check if the connection is successful
# try:
#     milvus.list_collections(timeout=3)
#     print("Connection to Milvus server successful")
# except Exception as e:
#     print(f"Failed to connect to Milvus server: {e}")

from pymilvus import utility

utility.has_collection("book")
