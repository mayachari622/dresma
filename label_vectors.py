import csv
import pandas as pd
import streamlit as st

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

# streamlit code
st.title("Extracting Labels and Tags from csv")

path_to_csv = st.text_input("Enter the path (from root) to the csv that you are uploading")
file = st.file_uploader("Choose csv file from your computer: ")
if st.button("Run"):
    file_path = file.name
    final_path = path_to_csv + file_path
    st.write(final_path)
    ldict = read_csv_df(final_path)
    st.write(ldict)

# path_to_csv: '/Users/mayachari/Downloads/'
# file uploader (filename): RunningShoes.xlsm - Valid Values.csv
# full path: '/Users/mayachari/Downloads/RunningShoes.xlsm - Valid Values.csv'

    
    