# Medical-Chatbot

A Retrieval-Augmented Generation (RAG) based medical chatbot that intelligently processes medical PDFs and provides accurate medical information through conversational AI.

## Overview

Medical-Chatbot is an intelligent assistant powered by large language models and semantic search capabilities. It processes medical documents (PDFs), creates vector embeddings, and stores them in a vector database for efficient retrieval and context-aware responses.

## Features

- **PDF Processing**: Automatically extracts and processes medical documents from a designated data folder
- **Intelligent Chunking**: Splits large medical documents into manageable chunks (500 characters with 50 character overlap) for optimal retrieval
- **Vector Embeddings**: Uses Hugging Face's sentence-transformers to generate semantic embeddings
- **Vector Database**: Integrates with Pinecone for scalable vector storage and retrieval
- **RAG Architecture**: Combines retrieval and generation for accurate, context-aware responses
- **LLM Integration**: Leverages Groq API for efficient language model inference
- **Web Interface**: Flask-based web application for easy interaction

## Tech Stack

- **LLM & RAG**: LangChain, LangChain Community, LangChain Groq
- **Embeddings**: Hugging Face, Sentence-Transformers
- **Vector Database**: Pinecone
- **Document Processing**: PyPDF
- **Web Framework**: Flask
- **Language**: Python 3.13+

## Architecture

```
PDF Documents (data/)
        ↓
    PDFLoader
        ↓
  Document Filtering
        ↓
  Text Splitting (500 chars)
        ↓
HuggingFace Embeddings
        ↓
Pinecone Vector Store
        ↓
    RAG Pipeline
        ↓
   Flask Web UI
```

## Prerequisites

- Python 3.13 or higher
- Pinecone API key
- Hugging Face API token (for embeddings)
- Groq API key (for LLM)

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Medical-Chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Environment Setup

Create a `.env` file in the project root with the following variables:

```env
PINECONE_API_KEY=your_pinecone_api_key
HF_TOKEN=your_hugging_face_token
GROQ_API_KEY=your_groq_api_key
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

## Usage

### 1. Prepare Medical Documents

Place your medical PDF files in the `data/` directory:
```
Medical-Chatbot/
├── data/
│   ├── Medical_book.pdf
│   ├── other_medical_doc.pdf
│   └── ...
```

### 2. Build Vector Index

Run the index creation script to process PDFs and store embeddings:
```bash
python store_index.py
```

This script will:
- Load all PDFs from the `data/` directory
- Extract and filter document content
- Split documents into manageable chunks
- Generate embeddings using Hugging Face
- Create/update the Pinecone vector index

### 3. Run the Web Application

```bash
python app.py
```

The Flask application will start and provide a web interface for querying the medical chatbot.

## Project Structure

```
Medical-Chatbot/
├── app.py                 # Flask web application
├── main.py               # Entry point
├── store_index.py        # Vector index creation and storage
├── requirements.txt      # Project dependencies
├── pyproject.toml        # Project configuration
├── setup.py              # Package setup
├── LICENSE               # MIT License
├── README.md             # This file
├── data/                 # Medical PDF documents
│   └── Medical_book.pdf
├── src/
│   ├── __init__.py
│   ├── helper.py         # PDF loading and text processing utilities
│   ├── prompt.py         # Prompt templates (to be implemented)
│   └── __pycache__/
├── Experiment/
│   └── demo.ipynb        # Jupyter notebook for experimentation
└── medical_chatbot.egg-info/  # Package metadata

```

## Core Modules

### `src/helper.py`
Utility functions for document processing:
- `load_pdf_file(data)`: Loads all PDF files from a directory
- `filter_to_minimal_docs(documents)`: Filters documents to include only essential metadata
- `text_split(minimal_docs)`: Splits documents into chunks using RecursiveCharacterTextSplitter

### `store_index.py`
Main script that:
- Loads PDFs from the `data/` directory
- Processes and chunks the documents
- Generates embeddings using Hugging Face
- Creates/updates Pinecone vector index
- Stores vectorized documents for retrieval

## Dependencies

Key packages and their purposes:

| Package | Version | Purpose |
|---------|---------|---------|
| langchain | 0.3.26 | Core RAG framework |
| langchain-groq | ≥0.3.8 | Groq LLM integration |
| langchain-pinecone | 0.2.8 | Pinecone vector store integration |
| langchain-huggingface | 0.1.2 | Hugging Face embeddings |
| sentence-transformers | 4.1.0 | Semantic embeddings model |
| pypdf | 5.6.1 | PDF processing |
| flask | 3.1.1 | Web framework |
| python-dotenv | 1.1.0 | Environment variable management |
| huggingface-hub | ≥0.36.2 | Hugging Face model hub |

## Development

### Running Experiments

An Jupyter notebook is available at `Experiment/demo.ipynb` for testing and experimentation with the chatbot functionality.

```bash
jupyter notebook Experiment/demo.ipynb
```

### Testing Dependencies

Run the dependency test script:
```bash
python dependency_test.py
```

## Configuration

### Text Chunking Parameters
Located in `store_index.py`:
- **chunk_size**: 500 characters
- **chunk_overlap**: 50 characters

Adjust these values based on your document complexity and retrieval requirements.

### Pinecone Configuration
- **Cloud**: AWS (default)
- **Region**: us-east-1 (default)
- **Metric**: Cosine similarity
- **Dimension**: 384 (default for all-MiniLM-L6-v2 model)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Zunair Zafar**
- Email: Zunairzafffar@gmail.com

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Roadmap

- [ ] Implement prompt templates in `src/prompt.py`
- [ ] Complete Flask web interface in `app.py`
- [ ] Add conversation memory and context management
- [ ] Implement multi-turn conversation support
- [ ] Add response caching and optimization
- [ ] Create comprehensive test suite
- [ ] Deploy to production environment
- [ ] Add support for other document formats (DOCX, TXT, etc.)

## Troubleshooting

### Pinecone Index Not Found
Ensure your `.env` file contains the correct `PINECONE_API_KEY` and the index name matches.

### Embedding Model Download
The first run will download the Hugging Face embedding model. Ensure you have sufficient disk space and internet connectivity.

### PDF Loading Errors
Verify that PDF files in the `data/` directory are valid and not corrupted.

## Support

For issues, questions, or suggestions, please open an issue in the repository or contact the author.