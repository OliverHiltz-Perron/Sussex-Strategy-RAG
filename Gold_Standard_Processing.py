from concurrent.futures import ThreadPoolExecutor
from llama_index.llms.ollama import Ollama
import pandas as pd
import ollama

ollama.pull("llama3.2")

preAIdf = pd.read_pickle("Markdown_Documents_CLEAN.pkl")

prompt_1 = "CAREFULLY read this legal document. \n\n"

prompt_2 = """
\n\nYour task is to extract specific information from the document provided, returning
information in JSON format using the following headings:

"DocumentTitle": The title of the document,
"DocumentSummary": A summary of the document in 100 words or less.
"EnergyRelated": Indicate Yes or No if this document contains energy-related information.
"EnergySummary": If the document is about energy, summarize its energy-related content with a focus on the restrictions for development projects outlines in the legal document. When you find a restriction, include a citation showing the specific part of the
 document where you identified it. Try to keep the summary to 100 words or less, but if the number of restrictions for development exceed this value feel free to go over the limit while remaining concise.

Respond only in JSON format with these headings. Example output:
{
    "DocumentTitle": "Cap and Trade Cancellation Act, 2018",
    "DocumentSummary": "This legislation outlines the dismantling of Ontario's cap and trade program, including ...",
    "EnergyRelated": "Yes",
    "EnergySummary": "The Act impacts energy-related entities including electricity generators and wind farms. Under (CITE_SECTION_OF_DOCUMENT_HERE) you may not build a wind farm within 30km of a school. Furthermore, under (CITE_SECTION_OF_DOCUMENT_HERE) ..."
}
"""

def askAI_LlamaIndex(content):
    llm = Ollama(model="llama3.2", request_timeout=600.0, json_mode=True)
    response = llm.complete(prompt_1 + content + prompt_2)
    return response

print(askAI_LlamaIndex(preAIdf['Text'].iloc[0]))