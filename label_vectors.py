from embeddings import EmbeddingPredictionClient
import csv
import pandas as pd
import streamlit as st
from embeddings import EmbeddingPredictionClient
from embeddings import EmbeddingPredictionClient


# function that takes in a path to a csv file and outputs a dictionary where the 
# keys are labels and the values are a list of tags associated with that label

def read_csv_df(csv_file):
    # initialize dictionary
    print('inside csv func')
    label_tag_dict = {}

    # open the csv
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)

        # the labels are in the 2nd column of the file
        label_column_index = 1
        
        # iterate through each row in the csv file
        for row in reader:
            label = row[label_column_index]

            # if the current row has no label, skip this row and move on to the next row
            if (label == ""):
                continue

            tags = list()

            # iterate through the length of the row (the number of ciolumns)
            for i in range(label_column_index + 1, len(row)):
                # if the cell in the csv file has content, add the tag to the tags list
                if (row[i] != ""):
                    tags.append(row[i])
                    
            # add this row of the csv to the output dictionary
            label_tag_dict[label] = tags

        # convert the dictionary to a dataframe
        label_tag_df = pd.DataFrame(columns=['label', 'tag', 'embedding'])

        # iterate through the dictionary to populate the dataframe
        for key, value in label_tag_dict.items():
            # iterate through each tag in the dictionary
            for s in value:
                # each row in the dataframe is label-tag-embedding
                new_entry = [key, s, None]
                label_tag_df = pd.concat([label_tag_df, pd.DataFrame([new_entry], columns=label_tag_df.columns)], ignore_index=True)

    return label_tag_df

def label_tag_embeddings(label_tag_df):
    print('inside of function')
    client = EmbeddingPredictionClient(project='vertex-production-391117')
    
    count = 0
    for index, row in label_tag_df.iterrows():
        embedded_output = []
        string_to_embed = row[1] + 'is the predominant' + row[0] + 'in the shoe'
        embedded_output= client.get_embedding_mod(text = string_to_embed)
        label_tag_df.loc[index, 'embedding'] = embedded_output
        print(embedded_output)
        count = count + 1
        if count > 50:
            break;
    print(label_tag_df)
    return label_tag_df



# streamlit code
st.title("Extracting Labels and Tags from csv")

path_to_csv = st.text_input("Enter the path (from root) to the csv that you are uploading")
file = st.file_uploader("Choose csv file from your computer: ")

dataframe_button_clicked = st.button("Generate Dataframe")

label_tag_df = None

if dataframe_button_clicked:
    file_path = file.name
    final_path = path_to_csv + file_path
    st.write(final_path)
    label_tag_df = read_csv_df(final_path)
    st.write(label_tag_df)

    df_with_embeddings = label_tag_embeddings(label_tag_df)
    st.write(df_with_embeddings)


# path_to_csv: /Users/mayachari/Downloads/
# file uploader (filename): RunningShoes.xlsm - Valid Values.csv
# full path: '/Users/mayachari/Downloads/RunningShoes.xlsm - Valid Values.csv'

    
    