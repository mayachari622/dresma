import csv
import streamlit as st

# function that takes in a path to a csv file and outputs a dictionary where the 
# keys are labels and the values are a list of tags associated with that label

def read_csv(csv_file):
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


    return label_tag_dict

# read_csv('/Users/mayachari/Downloads/RunningShoes.xlsm - Valid Values.csv')

st.title("Extracting Labels and Tags from csv")

file_path = st.file_uploader("Enter path to csv: ")
if st.button("Run"):
    read_csv(file_path)
    