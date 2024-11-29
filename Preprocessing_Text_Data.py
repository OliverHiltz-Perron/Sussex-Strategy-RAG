from dotenv import load_dotenv
load_dotenv()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

import pandas as pd

# Setting up Parser and Reader

parser = LlamaParse(
    result_type='markdown',
)

# filename_fn = lambda filename: {"file_name": filename}

reader = SimpleDirectoryReader(input_dir="Original_Documents", filename_as_id=True)
processed_files = []


docsText = []
docsFileNames = []
for docs in reader.iter_data():
    for doc in docs:
        filename = doc.metadata.get("file_name", "Unknown")
        docsFileNames.append(filename)
        text = doc.text
        docsText.append(text)
        print(f"Filename: {filename} added to list")

# print(docsText)
# print(docsFileNames)

data = {"Filenames": docsFileNames, "Text": docsText}

df = pd.DataFrame(data)

# print(df.head())
df.to_pickle("Markdown_Documents.pkl")
# df.to_csv("Markdown_Documents.csv", index=False)