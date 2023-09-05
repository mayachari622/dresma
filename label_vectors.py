from embeddings import EmbeddingPredictionClient
import csv
import pandas as pd
import io
import ast
import streamlit as st
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
import time
import os
import math
from PIL import Image

# description: This function takes in a csv file of label tag pair and generates embeddings for
# each label-tag pair in the file. The output is  
def generate_embeddings(csv_file):
    # convert file to a pandas dataframe
    label_tag_df = pd.read_csv(csv_file)

    # create an instance of the EmbeddingPredictionClient class
    client = EmbeddingPredictionClient(project='vertex-production-391117')
    
    # only make 100 API calls per minute (so that we don't get an error for overusing the API)
    total_api_calls = len(label_tag_df)
    calls_per_minute = 100

    label = None
    for i in range((total_api_calls + calls_per_minute - 1) // calls_per_minute):
        for j in range(calls_per_minute):
            index = i * calls_per_minute + j

            # if we finished making an API call for each row of the df
            if index >= total_api_calls:
                break
            row = label_tag_df.iloc[index]
            embedded_output = []

            # if the first column is not empty, set the label variable equal to the val in first column
            if isinstance(row[0], str):
                label = row[0]
            
            # the format for each label-tag embedding
            string_to_embed = row[1] + ' is the predominant ' + label + 'in the shoe'
            
            # get embedding for the above string
            embedded_output = client.get_embedding_mod(text = string_to_embed)
            
            # put embedding list into the dataframe
            label_tag_df.loc[index, 'EMBEDDINGS'] = str(embedded_output)
            
        # Wait for 1 minute before making the next set of API calls
        if (i + 1) * calls_per_minute < total_api_calls:
            # write to the streamlit page
            st.write("Waiting for 1 minute before the next set of API calls...")
            time.sleep(60)
    
    
    # return the edited dataframe
    return label_tag_df

# description: this function takes in a jpeg image and return the embedding for that image
def image_embeddings(image_file):

    # get the bytes data from the image file
    bytes_data = image_file.getvalue()

    # create an instance of the EmbeddingPredictionClient class
    client = EmbeddingPredictionClient(project='vertex-production-391117')

    response = client.get_embedding(text='', image_bytes=bytes_data)

    # return image embedding
    return response.image_embedding

# description: This function takes in the label-tag df with embeddings already generated and the 
# embedding for the input image. The function will match the image embedding with the embeddings
# in each row in the label-tag df and output the cosine similarity score for each row and the 
# highest match in each label category
def match_score(label_tag_df, image_embedding):
    # create a new column 'MATCH SCORE' that stores the cosine similarity of the image embedding and the label-tag embedding in that row
    label_tag_df['MATCH SCORE'] = label_tag_df['EMBEDDINGS'].apply(lambda x: cosine_similarity(ast.literal_eval(x), image_embedding))

    # get the list of row indices that the labels are on
    indices = []
    for index, row in label_tag_df.iterrows():
        if isinstance(row[1], str):
                indices.append(index)

    total_rows = len(label_tag_df)

    # for loop iterates over each label category (from the start of one label to the end of that label)
    for i in range(len(indices) - 1):
        start_index = indices[i]
        end_index = indices[i + 1] - 1 if i + 1 < len(indices) else total_rows - 1

        # new dataframe of rows with current label
        filtered_rows = label_tag_df.loc[start_index:end_index]
    
        # Find the row with the highest match score in the filtered DataFrame
        best_match_row_index = filtered_rows['MATCH SCORE'].idxmax()
        
        # label this row as highest match in original dataframe
        label_tag_df.loc[best_match_row_index, 'HIGHEST MATCH'] = 'highest match'

    return label_tag_df

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')



# streamlit code
st.title("Generating Embeddings From csv File")

# file uploaders
file = st.file_uploader("Input CSV file with label-tags: ")
image_file = st.file_uploader("Upload image: ")
label_tag_csv = st.file_uploader('Input label tag csv with embeddings already generated: ')

# buttons
dataframe_button_clicked = st.button("Generate Dataframe")
embedding_button_clicked = st.button("Match Image")

label_tag_df = None

if dataframe_button_clicked:

    label_tag_df = generate_embeddings(file)
    st.write(label_tag_df)
    csv = convert_df(label_tag_df)

    st.download_button(
        label="Download CSV", 
        data=csv, 
        file_name='label_tag_embeddings.csv',
        mime="text/csv")

if embedding_button_clicked:

    image_embedding = image_embeddings(image_file)

    label_tag_df = pd.read_csv(label_tag_csv)
    label_tag_df = match_score(label_tag_df, image_embedding)
    st.write(label_tag_df)

    
    