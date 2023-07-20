from embeddings import EmbeddingPredictionClient
import csv
import pandas as pd
import io
import streamlit as st
from embeddings import EmbeddingPredictionClient
from embeddings import EmbeddingPredictionClient

# function that doesn't use path
def create_df(csv_file):
    print('inside of create_df function')

    # convert file to a pandas dataframe
    dataframe = pd.read_csv(csv_file)

    # create new dataframe (result dataframe where the unique identifier is label-tag)
    label_tag_df = pd.DataFrame(columns=['label', 'tag', 'embedding'])

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
            if (column_idx < 2):
                continue;
            if pd.isna(value):
                break;
            if (value != ""):
                new_entry = [label, value, None]
                label_tag_df = pd.concat([label_tag_df, pd.DataFrame([new_entry], columns=label_tag_df.columns)], ignore_index=True)

    return label_tag_df

    
    print('end of create_df function')


# function that takes in a path to a csv file and outputs a dictionary where the 
# keys are labels and the values are a list of tags associated with that label

def read_csv_df(csv_file):
    # initialize dictionary
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
    # create an instance of the EmbeddingPredictionClient class
    client = EmbeddingPredictionClient(project='vertex-production-391117')
    
    # counter variable that keeps track of how many calls are being made to the API
    count = 0
    for index, row in label_tag_df.iterrows():
        embedded_output = []
        string_to_embed = row[1] + 'is the predominant' + row[0] + 'in the shoe'
        # get embedding for string
        embedded_output = client.get_embedding_mod(text = string_to_embed)
        # put embedding list into the dataframe
        label_tag_df.loc[index, 'embedding'] = embedded_output
        
        # after 50 rows of the dataframe are filled, break out of loop
        count = count + 1
        if count > 1:
            break;
    
    # return the edited dataframe
    return label_tag_df



# streamlit code
st.title("Generating Embeddings From csv File")

path_to_csv = st.text_input("Enter the path (from root) to the csv that you are uploading")
file = st.file_uploader("Choose csv file from your computer: ")

dataframe_button_clicked = st.button("Generate Dataframe")


if dataframe_button_clicked:
    # if a path is NOT inputted
    if (len(path_to_csv) == 0):
        label_tag_df = create_df(file)
        st.write(label_tag_df)

        df_with_embeddings = label_tag_embeddings(label_tag_df)
        st.write(df_with_embeddings)
    # if a path IS inputted
    else:
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

    
    