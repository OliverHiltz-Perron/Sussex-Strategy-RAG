import streamlit as st
import os
from testing import init_chroma, query_chroma_with_analysis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Energy Law Search Engine",
    page_icon="⚡",
    layout="wide"
)

# Title and description
st.title("Energy Law Search Engine")
st.markdown("Search through energy-related laws and get AI-enhanced analysis.")

# Initialize ChromaDB collection
@st.cache_resource
def get_collection():
    return init_chroma()

collection = get_collection()

# Search interface
query = st.text_area("Enter your query:", placeholder="Type your search query here...")

# Example queries
st.sidebar.header("Example Queries")
example_queries = [
    "renewable energy regulations",
    "solar power requirements",
    "energy efficiency standards"
]

if st.sidebar.button("Show Example Query 1"):
    query = example_queries[0]
if st.sidebar.button("Show Example Query 2"):
    query = example_queries[1]
if st.sidebar.button("Show Example Query 3"):
    query = example_queries[2]

# Search button
if st.button("Search", type="primary"):
    if query.strip():
        with st.spinner("Searching and analyzing..."):
            results = query_chroma_with_analysis(query, collection)
            
            # Split results into analysis and original results
            analysis_part = results.split("=== Original Results ===")[0]
            original_results = "=== Original Results ===" + results.split("=== Original Results ===")[1]
            
            # Display results in tabs
            tab1, tab2 = st.tabs(["AI Analysis", "Original Results"])
            with tab1:
                st.markdown(analysis_part)
            with tab2:
                st.markdown(original_results)
    else:
        st.warning("Please enter a query.") 