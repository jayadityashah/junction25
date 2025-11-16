# RAG Pipeline for Liquidity Risk Regulation Analysis

This pipeline analyzes financial regulations to find contradictions and overlaps related to liquidity risk.

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:

```bash
# Copy the example file
cp env.example .env
```

Or create `.env` manually with:

```
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

Replace the placeholder values with your actual API keys.

## Usage

Run the pipeline:

```bash
python rag_pipeline.py
```

## Configuration

You can adjust these parameters in `rag_pipeline.py`:

- `SIMILARITY_THRESHOLD`: Minimum similarity score for chunks (default: 0.7)
- `TOP_K`: Number of top chunks to retrieve (default: 7)
- `CHUNK_SIZE`: Size of text chunks (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100)

## Output

The pipeline generates:

- Console output with the liquidity risk definition, retrieved chunks, and analysis
- `liquidity_risk_analysis.json`: Structured JSON file with contradictions, overlaps, and summary

## Model Notes

The script uses:

- OpenAI `text-embedding-3-small` for embeddings
- Gemini 2.5 Flash for text generation and analysis
