# Just so I can avoid using the Jupyter Notebook for this task (async stuff I think)

# Load .env file
from dotenv import load_dotenv
load_dotenv()

# Importing Libraries
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

import pandas as pd

# Setting up Parser and Reader

parser = LlamaParse(
    result_type='markdown',
)

# Setting up the file extractor
file_extractor = {".doc": parser}

# Setting up the reader
reader = SimpleDirectoryReader(input_dir="Original_Documents", filename_as_id=True, file_extractor=file_extractor)
processed_files = []

# Processing the files
docsText = []
docsFileNames = []
for docs in reader.iter_data():
    for doc in docs:
        filename = doc.metadata.get("file_name", "Unknown")
        docsFileNames.append(filename)
        text = doc.text
        docsText.append(text)
        print(f"Filename: {filename} added to list")

# Writing the text to a markdown file to see if it works
with open("Markdown_Documents(TEST_2).md", "w") as f:
    f.write(docsText[0])

# Creating a DataFrame
data = {"Filenames": docsFileNames, "Text": docsText}

df = pd.DataFrame(data)

# Saving the DataFrame so the parsing process can be done segment by segment
df.to_pickle("Markdown_Documents.pkl")
