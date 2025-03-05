import os
import json
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document
from dotenv import load_dotenv
import pandas as pd
from typing import List, Dict, Any, Optional
import chromadb
from datetime import datetime
import gradio as gr
from openai import OpenAI
from pathlib import Path

# Load environment variables from env folder
env_path = Path('env/.env')
load_dotenv(dotenv_path=env_path)

# Constants
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "law_chunks"
DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_SEARCH_RESULTS = 5

class DeepSeekManager:
    _instance: Optional[OpenAI] = None

    @classmethod
    def get_client(cls) -> OpenAI:
        if cls._instance is None:
            cls._instance = OpenAI(
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1"
            )
        return cls._instance

class ChromaManager:
    _instance = None
    _collection = None

    @classmethod
    def get_collection(cls) -> chromadb.Collection:
        """Get or create a ChromaDB collection instance.
        
        Returns:
            chromadb.Collection: The ChromaDB collection instance
        """
        if cls._collection is None:
            client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            cls._collection = client.get_or_create_collection(CHROMA_COLLECTION_NAME)
        return cls._collection

def check_api_keys() -> None:
    """Check if required API keys are set"""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("Please set DEEPSEEK_API_KEY environment variable")

def filter_energy_related(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataset to only include energy-related entries"""
    return df[df['Qwen72_b_32k_context'].apply(
        lambda x: isinstance(x, str) and json.loads(x.strip()).get('EnergyRelated') == "Yes"
    )]

def create_chunks(text: str) -> List[str]:
    """Create semantic chunks from text using LlamaIndex"""
    node_parser = SimpleNodeParser.from_defaults()
    document = Document(text=text)
    nodes = node_parser.get_nodes_from_documents([document])
    return [node.text for node in nodes]

def store_chunks_in_chroma(collection, chunks_data: List[dict]):
    """Store chunks in ChromaDB collection"""
    try:
        # Prepare data for ChromaDB format
        documents = [chunk['text'] for chunk in chunks_data]
        metadatas = [{
            'document_title': chunk['document_title'],
            'filename': chunk['filename'],
            'energy_summary': chunk['energy_summary'],
            #'created_at': datetime.now(datetime.timezone.utc).isoformat()
        } for chunk in chunks_data]
        ids = [f"chunk_{i}" for i in range(len(chunks_data))]

        # Add documents to collection
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Successfully stored {len(chunks_data)} chunks in ChromaDB")
    except Exception as e:
        print(f"Error storing chunks in ChromaDB: {str(e)}")

def process_dataset(file_path: str, collection) -> None:
    """Process the dataset and store in ChromaDB"""
    # Check for API key
    check_api_keys()
    
    # Read the dataset
    df = pd.read_excel(file_path, usecols=['filename', 'Qwen72_b_32k_context'])
    
    # Filter for energy-related entries
    df_filtered = filter_energy_related(df)
    
    processed_data = []
    for _, row in df_filtered.iterrows():
        # Parse the JSON-like structure
        context_data = json.loads(row['Qwen72_b_32k_context'].strip())
        
        # Combine document title, summary, and energy summary
        law_text = (
            f"Document Title: {context_data['DocumentTitle']}\n"
            f"Document Summary: {context_data['DocumentSummary']}\n"
            f"Energy Summary: {context_data['EnergySummary']}"
        )
        
        # Create chunks
        chunks = create_chunks(law_text)
        
        # Store chunks with their metadata
        for chunk in chunks:
            processed_data.append({
                'text': chunk,
                'document_title': context_data['DocumentTitle'],
                'filename': row['filename'],
                'energy_summary': context_data['EnergySummary']
            })
    
    # Store chunks in ChromaDB
    store_chunks_in_chroma(collection, processed_data)

def enhance_results_with_deepseek(query: str, results: List[Dict[str, Any]]) -> str:
    """Use DeepSeek to enhance and format the search results.
    
    Args:
        query: The user's search query
        results: List of search results with document information
        
    Returns:
        str: Formatted analysis of the search results
        
    Raises:
        Exception: If there's an error calling the DeepSeek API
    """
    prompt = f"""
    Based on the query: "{query}", analyze these search results from a law database:

    {results}

    Provide:
    1. A summary of how these results relate to the query
    2. Key points from each document
    3. Energy-related implications
    
    Format the response in a clear, structured way.
    """
    
    try:
        client = DeepSeekManager.get_client()
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You are a legal analysis assistant specializing in energy law."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating enhanced results: {str(e)}"

def query_chroma_with_analysis(query_text: str, collection: chromadb.Collection, n_results: int = DEFAULT_SEARCH_RESULTS) -> str:
    """Query ChromaDB and enhance results with DeepSeek analysis.
    
    Args:
        query_text: The search query
        collection: ChromaDB collection to search
        n_results: Number of results to return
        
    Returns:
        str: Formatted search results with AI analysis
    """
    try:
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        formatted_results = [
            {
                'document_title': metadata['document_title'],
                'filename': metadata['filename'],
                'energy_summary': metadata['energy_summary'],
                'content': doc
            }
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0])
        ]
        
        analysis = enhance_results_with_deepseek(query_text, formatted_results)
        
        output_parts = [
            "=== DeepSeek Analysis ===",
            analysis,
            "\n=== Original Results ===\n"
        ]
        
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            result_text = (
                f"\n--- Result {i+1} ---\n"
                f"Document Title: {metadata['document_title']}\n"
                f"Filename: {metadata['filename']}\n"
                f"Energy Summary: {metadata['energy_summary']}\n"
                f"Content: {doc}\n"
            )
            output_parts.append(result_text)
        
        return "\n".join(output_parts)
    except Exception as e:
        return f"Error performing search: {str(e)}"

def direct_deepseek_query(query: str) -> str:
    """Query DeepSeek directly without using RAG.
    
    Args:
        query: User's search query
        
    Returns:
        str: DeepSeek's response
    """
    try:
        client = DeepSeekManager.get_client()
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "You are a legal analysis assistant specializing in energy law."},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting response from DeepSeek: {str(e)}"

def gradio_query(query: str, use_rag: bool) -> str:
    """Handle Gradio interface queries with RAG toggle.
    
    Args:
        query: User's search query
        use_rag: Whether to use RAG or query DeepSeek directly
        
    Returns:
        str: Search results and analysis
    """
    if not query.strip():
        return "Please enter a query."
    
    if use_rag:
        collection = ChromaManager.get_collection()
        return query_chroma_with_analysis(query, collection)
    else:
        return direct_deepseek_query(query)

# Update Gradio interface
iface = gr.Interface(
    fn=gradio_query,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your query here..."),
        gr.Checkbox(label="Use RAG (Retrieval-Augmented Generation)", value=True)
    ],
    outputs=gr.Textbox(lines=20, label="Results"),
    title="Energy Law Search Engine",
    description="""Search through energy-related laws and get AI-enhanced analysis.
    Toggle RAG to switch between searching through the law database (RAG enabled) 
    or asking DeepSeek directly (RAG disabled).""",
    examples=[
        ["What are the key renewable energy regulations?", True],
        ["Explain solar power requirements", False],
        ["What are the current energy efficiency standards?", True]
    ]
)

if __name__ == "__main__":
    # Launch Gradio interface
    iface.launch(share=True) 