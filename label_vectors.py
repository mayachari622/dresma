from embeddings import EmbeddingPredictionClient
import csv
import pandas as pd
import io
import ast
import streamlit as st
from embeddings import EmbeddingPredictionClient
from embeddings import EmbeddingPredictionClient
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from embedding_vector import dummy_embedding
from embedding_vector import shoe_embedding
import time
import os
import math
from PIL import Image


# 1. take in csv and generate embeddings (done)
# 2. save csv to streamlit page
# 3. upload an image and generate embedding
# 4. compare with the label tag csv
# 5. output highest match in each category

def generate_embeddings(csv_file):
    # convert file to a pandas dataframe
    label_tag_df = pd.read_csv(csv_file)

    # create an instance of the EmbeddingPredictionClient class
    client = EmbeddingPredictionClient(project='vertex-production-391117')
    
    total_api_calls = len(label_tag_df)
    calls_per_minute = 100
    print('TOTAL API CALLS: ', total_api_calls)

    label = None
    for i in range((total_api_calls + calls_per_minute - 1) // calls_per_minute):
        for j in range(calls_per_minute):
            index = i * calls_per_minute + j
            # if index == 0:
            #     continue
            if index >= total_api_calls:
                break
            row = label_tag_df.iloc[index]
            embedded_output = []
            # if the first column is not empty, set the label variable equal to the val in first column
            if isinstance(row[0], str):
                label = row[0]
           
            string_to_embed = row[1] + ' is the predominant ' + label + 'in the shoe'
            print(string_to_embed)
            # get embedding for string
            embedded_output = client.get_embedding_mod(text = string_to_embed)
            # put embedding list into the dataframe
            print('LENGTH OF EMBEDDING OUTPUT: ', len(embedded_output))
            print('type of embedding output: ', type(embedded_output))
            label_tag_df.loc[index, 'EMBEDDINGS'] = str(embedded_output)
            

        if (i + 1) * calls_per_minute < total_api_calls:
            print(f"Waiting for 1 minute before the next set of API calls...")
            st.write("Waiting for 1 minute before the next set of API calls...")
            time.sleep(60)  # Wait for 1 minute before making the next set of API calls
    
    # directory_path = "dataframes"
    
    # if not os.path.exists(directory_path):
    #     print('inside if statement')
    #     os.makedirs(directory_path)
        
    # # Construct the full path including the filename
    # full_path = os.path.join(directory_path, 'label_tag_embeddings.csv')
    # print(full_path)

    # # Save the DataFrame as a CSV file
    # label_tag_df.to_csv(full_path, index=False)
    # print('done')

    
    # # # return the edited dataframe
    return label_tag_df

# function that generates the image embeddings
def image_embeddings(image_file):
    # with open(image_file.name, "rb") as f:
    #     image_file_contents = f.read() 

    bytes_data = image_file.getvalue()

    # client can be reused.
    client = EmbeddingPredictionClient(project='vertex-production-391117')

    start = time.time()

    response = client.get_embedding(text='', image_bytes=bytes_data)

    end = time.time()

    print('Time taken: ', end - start)
    return response.image_embedding


# function will take in the dataframe with embeddings and the image embedding and match 
# output the highest match in each category
def match_score(label_tag_df, image_embedding):
    label_tag_df['MATCH SCORE'] = label_tag_df['EMBEDDINGS'].apply(lambda x: cosine_similarity(ast.literal_eval(x), image_embedding))

    # from the label until the next label (generate the list)
    indices = []
    for index, row in label_tag_df.iterrows():
        if isinstance(row[1], str):
                indices.append(index)

    total_rows = len(label_tag_df)

    for i in range(len(indices) - 1):
        start_index = indices[i]
        end_index = indices[i + 1] - 1 if i + 1 < len(indices) else total_rows - 1

        filtered_rows = label_tag_df.loc[start_index:end_index]
        print(filtered_rows)
    
        # Find the row with the highest match score in the filtered DataFrame
        best_match_row_index = filtered_rows['MATCH SCORE'].idxmax()
        print(best_match_row_index)

        label_tag_df.loc[best_match_row_index, 'HIGHEST MATCH'] = 'highest match'

    return label_tag_df



# function that doesn't use path
def create_df(csv_file):

    # convert file to a pandas dataframe
    dataframe = pd.read_csv(csv_file)

    # create new dataframe (result dataframe where the unique identifier is label-tag)
    label_tag_df = pd.DataFrame(columns=['label', 'tag', 'final_embedding'])

    # the labels are in the 2nd column of the dataframe (index 1)
    label_column_index = 1

    # iterate through the dataframe rows
    for index, row in dataframe.iterrows():
        label = row[label_column_index]
        if (label == ""):
            continue;
        # iterate through the columns of each row
        # for column, value in row.items():
        for column_idx, (column, value) in enumerate(row.items()):
            # skip the empty/label columns
            if (column_idx < 2):
                continue;
            # go to the next row if the cell has no value
            if pd.isna(value):
                break;
            if (value != ""):
                # each row in the new data frame will have label-tag-embedding
                new_entry = [label, value, None]
                label_tag_df = pd.concat([label_tag_df, pd.DataFrame([new_entry], columns=label_tag_df.columns)], ignore_index=True)

    return label_tag_df

# function that creates embeddings for every label-tag pair in the original csv
def label_tag_embeddings(label_tag_df):

    # create an instance of the EmbeddingPredictionClient class
    client = EmbeddingPredictionClient(project='vertex-production-391117')
    
    total_api_calls = len(label_tag_df)
    calls_per_minute = 30
    print('TOTAL API CALLS: ', total_api_calls)

    for i in range(total_api_calls // calls_per_minute):
        for j in range(calls_per_minute):
            index = i * calls_per_minute + j
            if index >= total_api_calls:
                break
            row = label_tag_df.iloc[index]
            embedded_output = []
            string_to_embed = row[1] + 'is the predominant' + row[0] + 'in the shoe'
            # get embedding for string
            embedded_output = client.get_embedding_mod(text = string_to_embed)
            # put embedding list into the dataframe
            label_tag_df.loc[index, 'final_embedding'] = embedded_output
            print('LENGTH OF EMBEDDING OUTPUT: ', len(embedded_output))

        if (i + 1) * calls_per_minute < total_api_calls:
            print(f"Waiting for 1 minute before the next set of API calls...")
            time.sleep(60)  # Wait for 1 minute before making the next set of API calls

    directory_path = "dataframes"
    
    if not os.path.exists(directory_path):
        print('inside if statement')
        os.makedirs(directory_path)
        
    # Construct the full path including the filename
    full_path = os.path.join(directory_path, 'label_tag_embeddings.csv')
    print(full_path)

    # Save the DataFrame as a CSV file
    label_tag_df.to_csv(full_path, index=False)
    print('done')
    

    # # # return the edited dataframe
    return label_tag_df


# function for when I can't make any api calls :(
def dummy_embedding_function(label_tag_df):
    dummy_emb = dummy_embedding
    print('LENGTH OF DUMMY EMBEDDING: ', len(dummy_emb))

    for index, row in label_tag_df.iterrows():
        label_tag_df.loc[index, 'final_embedding'] = dummy_emb
        
    return label_tag_df

def process_image_embedding_csv(image_file):
    image_df = pd.read_csv(image_file)

    for index, row in image_df.iterrows():
        if pd.isna(row['final_embedding']):
            image_df = image_df.drop(index)        
    return image_df

def file_to_df(label_tag_file):
    # convert input file to dataframe
    label_tag_df = pd.read_csv(label_tag_file)
    for index, row in label_tag_df.iterrows():
        row['final_embedding'] = ast.literal_eval(row['final_embedding'])
    
    return label_tag_df


# this function takes in a csv file with all of the image embeddings and converts it to a dataframe
# then, for each row (that contains and image embedding, find the cosine similarity between that 
# row's embedding and every single embedding in the label-tag-df)
def highest_similarity_df(image_df, label_tag_df):

    # create a new column in the embedding file
    image_df['label_tag_first'] = None
    image_df['first_sim_score'] = None
    image_df['label_tag_second'] = None
    image_df['second_sim_score'] = None
    image_df['label_tag_third'] = None
    image_df['third_sim_score'] = None

    # for each row in the image_df, call cosine_sim
    for index, row in image_df.iterrows():
        image_embedding = row['final_embedding']
        
        # turn image embedding into a list
        image_embedding = ast.literal_eval(image_embedding)

        label_tag_df, image_df = cosine_sim(label_tag_df, image_embedding, image_df, index, 5, False)
        print('finished iteration ', index)

    return label_tag_df, image_df


# n: optional parameter that specifies the number of top matching embeddings to return (default is 10)
# pprint: optional boolean flag parameter to determine whether or not to print the matching embeddings (default true)
def cosine_sim(label_tag_df, image_embedding, image_df, index, n=10, pprint=True):

    count = "a"
    label_tag_df[count] = label_tag_df.final_embedding.apply(lambda x: cosine_similarity(x, image_embedding))
    #label_tag_df[count] = label_tag_df.final_embedding.apply(lambda x: cosine_similarity(x, image_embedding) if x is not None else None)

    # label_tag_df[count] = label_tag_df.final_embedding.apply(
    # lambda x: print(f'Type of x: {type(x)}, Type of image_embedding: {type(image_embedding)}'))

   # df['similarity'] = df['embedding'].apply(lambda x: cosine_similarity(x, input_embedding_vector))

    top_n_indices = label_tag_df.nlargest(n, count).index

    # Get top 3 indices
    first_highest_idx = top_n_indices[0]
    second_highest_idx = top_n_indices[1]
    third_highest_idx = top_n_indices[2]

    
    # find row in the dataframe with the highest similarity score
    first_similarity_value = label_tag_df.loc[first_highest_idx, count]
    label = label_tag_df.loc[first_highest_idx, 'label']
    tag = label_tag_df.loc[first_highest_idx, 'tag']

    # put it in the image embedding df
    image_df.loc[index, 'label_tag_first'] = label + " " + tag
    image_df.loc[index, 'first_sim_score'] = first_similarity_value

    ############### second highest ##################
    second_similarity_value = label_tag_df.loc[second_highest_idx, count]
    label = label_tag_df.loc[second_highest_idx, 'label']
    tag = label_tag_df.loc[second_highest_idx, 'tag']

    # put it in the image embedding df
    image_df.loc[index, 'label_tag_second'] = label + " " + tag
    image_df.loc[index, 'second_sim_score'] = second_similarity_value

    ############### third highest ##################
    third_similarity_value = label_tag_df.loc[third_highest_idx, count]
    label = label_tag_df.loc[third_highest_idx, 'label']
    tag = label_tag_df.loc[third_highest_idx, 'tag']

    # put it in the image embedding df
    image_df.loc[index, 'label_tag_third'] = label + " " + tag
    image_df.loc[index, 'third_sim_score'] = third_similarity_value

    
    # returns the dataframe with a new column of matching scores with an image
    return label_tag_df, image_df

def fix_label_tag_embedding_df(label_tag_embedding_csv):
    client = EmbeddingPredictionClient(project='vertex-production-391117')
    # read into dataframe
    print('inside func')
    label_tag_df = pd.read_csv(label_tag_embedding_csv)

    for index, row in label_tag_df.iterrows():
        if pd.isna(row['final_embedding']):
            embedded_output = []
            string_to_embed = row[1] + 'is the predominant' + row[0] + 'in the shoe'
            # get embedding for string
            embedded_output = client.get_embedding_mod(text = string_to_embed)
            # put embedding list into the dataframe
            label_tag_df.loc[index, 'final_embedding'] = embedded_output
        
    # save as csv!
    directory_path = "dataframes"
    
    if not os.path.exists(directory_path):
        print('inside if statement')
        os.makedirs(directory_path)
        
    # Construct the full path including the filename
    full_path = os.path.join(directory_path, 'label_tag_embeddings_final.csv')
    print(full_path)

    # Save the DataFrame as a CSV file
    label_tag_df.to_csv(full_path, index=False)

    print('done!')
    return label_tag_df

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')



# streamlit code
st.title("Generating Embeddings From csv File")

file = st.file_uploader("Input CSV file with label-tags: ")
image_file = st.file_uploader("Upload image: ")
label_tag_csv = st.file_uploader('Input label tag csv with embeddings already generated: ')

dataframe_button_clicked = st.button("Generate Dataframe")

embedding_button_clicked = st.button("Match Image")

label_tag_df = None

if dataframe_button_clicked:

    # generate embeddings function takes in a csv file and generates the 
    # embeddings. returns the dataframe with the embeddings and also saves 
    # the dataframe as a csv
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

    print(type(image_embedding))

    label_tag_df = pd.read_csv(label_tag_csv)
    label_tag_df = match_score(label_tag_df, image_embedding)
    st.write(label_tag_df)







# path_to_csv: /Users/mayachari/Downloads/
# file uploader (filename): RunningShoes.xlsm - Valid Values.csv
# full path: '/Users/mayachari/Downloads/RunningShoes.xlsm - Valid Values.csv'

    
    