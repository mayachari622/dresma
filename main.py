import streamlit as st
from embeddings import EmbeddingPredictionClient
import label_vectors
import time

# def run_main(csv_file):
    
#     # client can be reused.
#     client = EmbeddingPredictionClient(project='vertex-production-391117')
#     # start = time.time()
#     # create label dictionary
#     label_dict = label_vectors.read_csv(csv_file)

#     embeddings_dict = {}

#     # only run the embeddings for three labels and tags

#     for key, val in label_dict.items():
#         vector_embedding = []
#         for s in val:
#             # "Value" is the predominant "label" in the "product category".
#             embed = s + 'is the predominant' + key + 'in the shoe'
#             embeddings = client.get_embedding_mod(text = embed)
#             vector_embedding.append(embeddings)
#         embeddings_dict[key] = vector_embedding


#     return embeddings_dict

# run_main('/Users/mayachari/Downloads/RunningShoes.xlsm - Valid Values.csv')


# function that will iterate through every row in the dataframe
def label_tag_embeddings(label_tag_df):
    client = EmbeddingPredictionClient(project='vertex-production-391117')

    for index, row in label_tag_df.iterrows():
        string_to_embed = row[1] + 'is the predominant' + row[0] + 'in the shoe'
        embedded_output= client.get_embedding_mod(text = string_to_embed)
        label_tag_df.at[index, 2] = embedded_output
        print(row)

    print(label_tag_df)
    return label_tag_df

label_tag_df = label_vectors.read_csv_df('/Users/mayachari/Downloads/RunningShoes.xlsm - Valid Values.csv')

label_tag_embeddings(label_tag_df)

# loop that will match the image embeddings to the tag embedding
# def image_embedding_match(image_embedding, csv_file, label):
#     text_embedding_dict = run_main(csv_file)
#     for key, val in text_embedding_dict.items():
#         if label == key:
#             for s in val:


    


# st.title("Converting label csv files into embeddings")

# input1 = st.number_input("Input 1", value=0)
# result = run_main(input1)
# st.write("Result:", result)


# def run_loop(iterations, max_iterations_per_minute, label_dict):
#     interval = 60 / max_iterations_per_minute

#     for i in range(iterations):
#         for key, val in label_dict.items():
#             vector_embedding = []
#             for s in val:
#                 embeddings = client.get_embedding_mod(text = s)
#                 vector_embedding.append(embeddings)
#             embeddings_dict[key] = vector_embedding

#         for key, embeddings in embeddings_dict.items():
#             print(f"Embeddings for '{key}': {embeddings}")


#         time.sleep(interval)

# # run_loop(len(label_dict.items()), 10, label_dict)

'/Users/mayachari/Downloads/RunningShoes.xlsm - Valid Values.csv'