# Energy Law Search Engine

A powerful search engine and analysis tool for energy-related laws and regulations, powered by AI.

## Features

- 🔍 Semantic search through energy law documents
- 🤖 AI-powered analysis of search results
- 📊 User-friendly interface built with Gradio
- 📑 Original document viewing capability
- 💡 Example queries for quick start

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- DeepSeek API key (sign up at https://platform.deepseek.com)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Sussex-Strategy-RAG.git
cd Sussex-Strategy-RAG
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
   - Create a `.env` file in the root directory
   - Add the following credentials:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     DEEPSEEK_API_KEY=your_deepseek_api_key_here
     ```

### Running the Application

1. Start the Gradio application:

```bash
python app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://127.0.0.1:7860`)

## Usage

1. Enter your query in the search box
2. Click the "Submit" button
3. View the results:
   - AI Analysis: Get AI-enhanced analysis of the search results
   - Original Results: View the raw search results and document excerpts

Example queries are provided in the interface for reference.

## Project Structure

- `app.py`: Main Gradio application
- `testing.py`: Core search and analysis functionality
- `requirements.txt`: Project dependencies
- `Original_Documents/`: Directory containing source documents
- `chroma_db/`: ChromaDB vector database storage

## Dependencies

- gradio: Web interface framework
- chromadb: Vector database for document storage
- openai: AI model integration
- pandas: Data manipulation
- python-dotenv: Environment variable management
- llama-index: Document indexing and retrieval
- openpyxl: Excel file support

## License

This project is licensed under the terms included in the LICENSE file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
