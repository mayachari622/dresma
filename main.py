import streamlit as st
from embeddings import EmbeddingPredictionClient
import label_vectors
import time

def run_main(csv_file):
    
    # client can be reused.
    client = EmbeddingPredictionClient(project='vertex-production-391117')
    # start = time.time()
    # create label dictionary
    label_dict = label_vectors.read_csv(csv_file)

    embeddings_dict = {}

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

    return embeddings_dict


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