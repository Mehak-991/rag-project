# Cost-Efficient RAG Application

A production-ready Retrieval-Augmented Generation (RAG) application built with Python 3.11, FastAPI, LangChain, ChromaDB, and Ollama. This system provides a cost-effective alternative to managed vector databases while maintaining high-quality document retrieval and question answering capabilities.

## Features

- **Document Ingestion**: Support for PDF, HTML, and Markdown files
- **Configurable Chunking**: Adjustable chunk size and overlap parameters
- **Duplicate Detection**: SHA256-based file hashing to prevent duplicate vectors
- **Metadata Filtering**: Filter by category, file type, and custom metadata
- **Context Reranking**: Improve retrieval relevance with reranking
- **Source Citation**: Answers include source filenames and chunk IDs
- **Comprehensive Logging**: Track latency, token usage, and performance metrics
- **Evaluation Module**: Built-in retrieval and answer quality metrics
- **Cost Analysis**: Compare ChromaDB vs managed vector databases

## Tech Stack

- **Backend**: Python 3.11, FastAPI
- **Framework**: LangChain
- **Vector Store**: ChromaDB (persistent local storage)
- **Embeddings**: BAAI/bge-small-en-v1.5 (HuggingFace)
- **LLM**: Qwen2.5:7b via Ollama
- **Testing**: Pytest

## Project Structure

```
rag_project/
│
├── app/
│   ├── api.py              # FastAPI endpoints
│   ├── ingest.py           # Document ingestion pipeline
│   ├── retriever.py        # Retrieval and reranking
│   ├── generator.py        # RAG answer generation
│   ├── vectorstore.py      # ChromaDB management
│   ├── evaluator.py        # Evaluation metrics
│   ├── logger.py           # Logging configuration
│   └── config.py           # Application settings
│
├── data/                   # Document storage
├── chroma_db/             # Vector database storage
├── evaluation/
│   ├── questions.json      # Evaluation queries
│   ├── metrics.py         # Metric calculations
│   ├── evaluator.py       # Evaluation runner
│   └── results.json       # Evaluation results
├── logs/                  # Application logs
├── tests/
│   ├── test_ingest.py      # Ingestion tests
│   ├── test_retriever.py   # Retrieval tests
│   └── test_api.py         # API tests
│
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
├── run.py                 # Application entry point
├── cost_analysis.py       # Cost comparison script
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker Compose setup
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.11 or higher
- Ollama (for local LLM)
- 8GB+ RAM (16GB recommended)
- 10GB+ disk space

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd "c:\Users\HP\Desktop\rag project"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy environment template
copy .env.example .env

# Edit .env with your settings (optional, defaults are provided)
```

### Step 3: Install and Setup Ollama

#### Windows Installation

1. Download Ollama from [https://ollama.ai/download](https://ollama.ai/download)
2. Run the installer
3. Open Command Prompt or PowerShell

#### Linux/Mac Installation

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

#### Pull Qwen2.5 Model

```bash
# Pull the 7B model (recommended for most systems)
ollama pull qwen2.5:7b

# For systems with more RAM, you can use larger models
# ollama pull qwen2.5:14b
# ollama pull qwen2.5:32b
```

#### Verify Ollama Installation

```bash
# Check Ollama is running
ollama list

# Test the model
ollama run qwen2.5:7b "Hello, how are you?"
```

## Running the Application

### Option 1: Direct Python Execution

```bash
# Start the FastAPI server
python run.py

# The API will be available at http://localhost:8000
# API documentation at http://localhost:8000/docs
```

### Option 2: Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# The API will be available at http://localhost:8000
# Ollama will be available at http://localhost:11434
```

### Option 3: Docker Only

```bash
# Build the image
docker build -t rag-app .

# Run the container
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/logs:/app/logs \
  rag-app
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns system health status and component availability.

### Ingest Document

```bash
POST /ingest
Content-Type: multipart/form-data

Parameters:
- file: Document file (PDF, HTML, Markdown)
- category: Document category (default: "general")
- chunk_size: Chunk size override (default: 1000)
- chunk_overlap: Chunk overlap override (default: 200)
```

Example using curl:

```bash
curl -X POST "http://localhost:8000/ingest?category=tech" \
  -F "file=@document.pdf"
```

Example using Python:

```python
import requests

with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/ingest',
        files={'file': f},
        params={'category': 'tech', 'chunk_size': 500}
    )
print(response.json())
```

### Query Documents

```bash
POST /query
Content-Type: application/json

Body:
{
  "query": "Your question here",
  "top_k": 5,
  "category": "tech",
  "file_type": ".pdf",
  "metadata_filter": {},
  "rerank": true
}
```

Example using curl:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 5,
    "rerank": true
  }'
```

Example using Python:

```python
import requests

response = requests.post(
    'http://localhost:8000/query',
    json={
        'query': 'What is machine learning?',
        'top_k': 5,
        'category': 'tech',
        'rerank': True
    }
)
result = response.json()
print(f"Answer: {result['answer']}")
print(f"Citations: {result['citations']}")
```

### Get Statistics

```bash
GET /stats
```

Returns collection statistics including document count and embedding model info.

### Get Metrics

```bash
GET /metrics
```

Returns recent performance metrics (latency, token usage, etc.).

### Reset Collection

```bash
DELETE /collection
```

Deletes all documents from the collection (use with caution).

## Evaluation

### Running Evaluation

1. **Prepare Evaluation Questions**

Edit `evaluation/questions.json` with your test queries and relevant document IDs:

```json
[
  {
    "query": "What is the purpose of this RAG application?",
    "relevant_ids": ["doc_hash_0", "doc_hash_1"]
  },
  {
    "query": "How does document ingestion work?",
    "relevant_ids": ["doc_hash_2", "doc_hash_3", "doc_hash_4"]
  }
]
```

2. **Run Evaluation**

```bash
# Run evaluation script
python evaluation/evaluator.py

# Or run from project root
python -m evaluation.evaluator
```

3. **View Results**

Results are saved to `evaluation/results.json` and printed to console.

### Evaluation Metrics

#### Retrieval Metrics
- **Recall@K**: Fraction of relevant documents retrieved in top K
- **Hit Rate**: Whether at least one relevant document was retrieved
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of first relevant document
- **nDCG@K**: Normalized Discounted Cumulative Gain
- **Context Precision**: Fraction of retrieved documents that are relevant

#### Answer Metrics
- **Faithfulness**: How well the answer is grounded in context
- **Groundedness**: Factual accuracy based on retrieved context
- **Answer Relevance**: How relevant the answer is to the query

## Cost Analysis

### Running Cost Analysis

```bash
python cost_analysis.py
```

This generates a detailed cost comparison between ChromaDB (self-hosted) and managed vector databases (Pinecone, Weaviate Cloud, Qdrant Cloud) for different scales:
- 100K vectors
- 1M vectors
- 10M vectors

Results are saved to `cost_comparison.csv`.

### Cost Comparison Summary

| Scale | ChromaDB (Monthly) | Managed Avg (Monthly) | Savings |
|-------|-------------------|----------------------|---------|
| 100K  | $55              | $53                  | ~0%     |
| 1M    | $220             | $533                 | ~59%    |
| 10M   | $880             | $5,333               | ~83%    |

### Tradeoffs

#### ChromaDB (Self-Hosted)

**Pros:**
- Significant cost savings (60-80% cheaper at scale)
- Full control over data and infrastructure
- No vendor lock-in
- Customizable to specific needs
- Privacy and data sovereignty

**Cons:**
- Requires DevOps expertise
- Maintenance overhead
- Manual scaling
- No built-in high availability (need to configure)
- Monitoring and alerting setup required

#### Managed Vector Databases

**Pros:**
- Zero infrastructure management
- Built-in high availability
- Automatic scaling
- Managed backups and updates
- 24/7 support
- Better for teams without DevOps

**Cons:**
- Higher costs (3-5x more expensive at scale)
- Vendor lock-in
- Less control over configuration
- Data residency concerns
- Usage-based pricing can be unpredictable

### When to Switch to Managed DB

Consider switching to a managed vector database when:
- Your team lacks DevOps expertise
- You require 99.99%+ uptime SLA
- You're scaling beyond 10M vectors
- Compliance requires managed services
- Maintenance cost exceeds managed pricing
- You need global edge deployment
- You require enterprise-grade security features

## Testing

### Run All Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_ingest.py

# Run specific test
pytest tests/test_ingest.py::TestDocumentLoader::test_load_markdown
```

### Test Coverage

- **Unit Tests**: Test individual components (loader, chunker, retriever)
- **Integration Tests**: Test component interactions
- **API Tests**: Test FastAPI endpoints

## Configuration

Key environment variables in `.env`:

```bash
# Application
APP_NAME=Cost-Efficient RAG
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Vector Store
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=rag_collection
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5

# Chunking
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200

# Retrieval
DEFAULT_TOP_K=5
SIMILARITY_THRESHOLD=0.7

# LLM
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2048

# Logging
LOG_DIR=./logs
LOG_LEVEL=INFO
```

## Troubleshooting

### Ollama Connection Issues

If the application can't connect to Ollama:

1. Verify Ollama is running:
   ```bash
   ollama list
   ```

2. Check Ollama is accessible:
   ```bash
   curl http://localhost:11434/api/tags
   ```

3. Verify model is pulled:
   ```bash
   ollama pull qwen2.5:7b
   ```

### Memory Issues

If you encounter memory issues:

1. Reduce chunk size in `.env`:
   ```bash
   DEFAULT_CHUNK_SIZE=500
   ```

2. Use a smaller embedding model:
   ```bash
   EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
   ```

3. Use a smaller LLM model:
   ```bash
   OLLAMA_MODEL=qwen2.5:3b
   ```

### Slow Performance

To improve performance:

1. Use GPU for embeddings (if available):
   ```bash
   EMBEDDING_DEVICE=cuda
   ```

2. Reduce top_k for retrieval:
   ```bash
   DEFAULT_TOP_K=3
   ```

3. Disable reranking if not needed:
   ```bash
   rerank=false
   ```

## Performance Optimization

### Embedding Optimization

- Use GPU if available: `EMBEDDING_DEVICE=cuda`
- Cache embeddings for frequently accessed documents
- Consider quantization for large-scale deployments

### Retrieval Optimization

- Adjust `DEFAULT_TOP_K` based on your use case
- Use metadata filters to reduce search space
- Enable reranking for better relevance

### LLM Optimization

- Use smaller models for faster responses
- Adjust `LLM_MAX_TOKENS` to limit output length
- Batch queries when possible

## Security Considerations

- **API Security**: Add authentication for production deployment
- **Data Encryption**: Enable encryption at rest for sensitive data
- **Network Security**: Use HTTPS in production
- **Input Validation**: All inputs are validated via Pydantic models
- **Rate Limiting**: Implement rate limiting for API endpoints

## Production Deployment

### Using Docker Compose

```bash
# Build and start in production mode
docker-compose -f docker-compose.yml up -d --build

# View logs
docker-compose logs -f rag-app

# Stop services
docker-compose down
```

### Using Kubernetes

Create a Kubernetes deployment manifest (not included) or use tools like Helm for production deployments.

### Monitoring

- Logs are stored in `logs/` directory
- Consider integrating with monitoring tools (Prometheus, Grafana)
- Set up alerts for error rates and latency

## Contributing

Contributions are welcome! Please ensure:
- All tests pass
- Code follows existing style
- New features include tests
- Documentation is updated

## License

This project is provided as-is for educational and commercial use.

## Support

For issues and questions:
- Check the troubleshooting section
- Review logs in `logs/` directory
- Open an issue on the project repository

## Acknowledgments

- LangChain for the excellent framework
- ChromaDB for the open-source vector database
- Ollama for local LLM inference
- BAAI for the bge-small embedding model
