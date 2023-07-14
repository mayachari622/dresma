# convert the tags to embeddings
# Copyright 2023 Google LLC.
# SPDX-License-Identifier: Apache-2.0
from absl import app
from absl import flags
import base64
# Need to do pip install google-cloud-aiplatform for the following two imports.
# Also run: gcloud auth application-default login.
from google.cloud import aiplatform
from google.protobuf import struct_pb2
import sys
import time
import typing
import os
import label_vectors
import time
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="vertex-production-391117-356ba2e4d7af.json"


# define a named tuple to hold the text and the image embeddings
class EmbeddingResponse(typing.NamedTuple):
    text_embedding: typing.Sequence[float]
    image_embedding: typing.Sequence[float]

# 
class EmbeddingPredictionClient:
    """Wrapper around Prediction Service Client."""
    def __init__(self, project : str, location : str = "us-central1",
        api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com"):

        client_options = {"api_endpoint": api_regional_endpoint}
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        self.client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
        self.location = location
        self.project = project
    # MODIFIED GET EMBEDDING FUNCTION
    def get_embedding_mod(self, text: str = None):
        if not text:
            raise ValueError('Text must be specified.')
        #
        instance = struct_pb2.Struct()
        instance.fields['text'].string_value = text
        instances = [instance]

        endpoint = (f"projects/{self.project}/locations/{self.location}"
        "/publishers/google/models/multimodalembedding@001")
        response = self.client.predict(endpoint=endpoint, instances=instances)

        text_embedding = None
        text_emb_value = response.predictions[0]['textEmbedding']
        text_embedding = [v for v in text_emb_value]
        return text_embedding
    
# with open('196_196.jpg', "rb") as f:
#     image_file_contents = f.read()

# client can be reused.
client = EmbeddingPredictionClient(project='vertex-production-391117')
start = time.time()
# create label dictionary
label_dict = label_vectors.read_csv('/Users/mayachari/Downloads/RunningShoes.xlsm - Valid Values.csv')
print(type(label_dict))

embeddings_dict = {}

def run_loop(iterations, max_iterations_per_minute, label_dict):
    interval = 60 / max_iterations_per_minute

    for i in range(iterations):
        for key, val in label_dict.items():
            vector_embedding = []
            for s in val:
                embeddings = client.get_embedding_mod(text = s)
                vector_embedding.append(embeddings)
            embeddings_dict[key] = vector_embedding

        for key, embeddings in embeddings_dict.items():
            print(f"Embeddings for '{key}': {embeddings}")


        time.sleep(interval)

# run_loop(len(label_dict.items()), 10, label_dict)

count = 0
## without the time loop
for key, val in label_dict.items():
    vector_embedding = []
    for s in val:
        embeddings = client.get_embedding_mod(text = s)
        vector_embedding.append(embeddings)
    embeddings_dict[key] = vector_embedding
    count = count + 1
    if (count > 3):
        break;

for key, embeddings in embeddings_dict.items():
    print(f"Embeddings for '{key}': {embeddings}")


end = time.time()

print('Time taken: ', end - start)

