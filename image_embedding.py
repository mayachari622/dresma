import os.path
import shutil
import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
#from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from openai.embeddings_utils import get_embedding, cosine_similarity
import openai
import pandas as pd
import boto3, os
from urllib.parse import unquote
from botocore.config import Config

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

# file placed in same directory where your code has placed
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vertex-production-391117-356ba2e4d7af.json"

class EmbeddingResponse(typing.NamedTuple):
    text_embedding: typing.Sequence[float]
    image_embedding: typing.Sequence[float]


class EmbeddingPredictionClient:
    """Wrapper around Prediction Service Client."""

    def __init__(self, project: str,
                 location: str = "us-central1",
                 api_regional_endpoint: str = "us-central1-aiplatform.googleapis.com"):
        client_options = {"api_endpoint": api_regional_endpoint}
        # Initialize client that will be used to create and send requests.
        # This client only needs to be created once, and can be reused for multiple requests.
        self.client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
        self.location = location
        self.project = project

    def get_embedding(self, text: str = None, image_bytes: bytes = None):
        if not text and not image_bytes:
            raise ValueError('At least one of text or image_bytes must be specified.')
        instance = struct_pb2.Struct()
        if text:
            instance.fields['text'].string_value = text
        if image_bytes:
            encoded_content = base64.b64encode(image_bytes).decode("utf-8")
            image_struct = instance.fields['image'].struct_value
            image_struct.fields['bytesBase64Encoded'].string_value = encoded_content
        instances = [instance]
        endpoint = (f"projects/{self.project}/locations/{self.location}"
                    "/publishers/google/models/multimodalembedding@001")
        response = self.client.predict(endpoint=endpoint, instances=instances)
        text_embedding = None
        if text:
            text_emb_value = response.predictions[0]['textEmbedding']
            text_embedding = [v for v in text_emb_value]
        image_embedding = None
        if image_bytes:
            image_emb_value = response.predictions[0]['imageEmbedding']
            image_embedding = [v for v in image_emb_value]
        return EmbeddingResponse(
            text_embedding=text_embedding,
            image_embedding=image_embedding)