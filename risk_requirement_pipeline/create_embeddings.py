"""
Generate embeddings for requirements using legal-bert-base-uncased
Embeddings are created from requirement_text and stored with all requirement metadata
"""

import sqlite3
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import json

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "legal_documents.db"
EMBEDDINGS_OUTPUT_DIR = SCRIPT_DIR / "embeddings"
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

# Ensure embeddings directory exists
EMBEDDINGS_OUTPUT_DIR.mkdir(exist_ok=True)


class LegalBERTEmbedder:
    """Wrapper for legal-bert-base-uncased model"""

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        """
        Initialize the Legal-BERT model

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        print(f"Loading {model_name}...")

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        print(f"✓ Model loaded successfully")

    def encode(self, texts: List[str], batch_size: int = 8, max_length: int = 512, show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings

        Args:
            texts: List of text strings to encode
            batch_size: Number of texts to process at once
            max_length: Maximum token length (legal-bert default is 512)
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = []

        # Calculate total batches
        num_batches = (len(texts) + batch_size - 1) // batch_size

        # Create progress bar if requested
        batch_iterator = range(0, len(texts), batch_size)
        if show_progress:
            batch_iterator = tqdm(batch_iterator, total=num_batches, desc="Encoding batches", unit="batch")

        # Process in batches
        for i in batch_iterator:
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            # Move to device
            encoded = {key: val.to(self.device) for key, val in encoded.items()}

            # Get embeddings (no gradient computation needed)
            with torch.no_grad():
                outputs = self.model(**encoded)

                # Use [CLS] token embedding (first token) as sentence embedding
                # Shape: (batch_size, hidden_size)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]

                # Alternative: mean pooling over all tokens
                # attention_mask = encoded['attention_mask']
                # token_embeddings = outputs.last_hidden_state
                # input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                # cls_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                # Move to CPU and convert to numpy
                batch_embeddings = cls_embeddings.cpu().numpy()
                embeddings.append(batch_embeddings)

        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)

        return all_embeddings


def load_requirements_from_db(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Load all requirements from the database with all columns

    Returns:
        List of requirement dictionaries with all metadata
    """
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            r.id,
            r.document_id,
            r.requirement_text,
            r.risk_category,
            r.start_page,
            r.end_page,
            r.extraction_window,
            d.filename as document_filename
        FROM requirements r
        JOIN documents d ON r.document_id = d.id
        ORDER BY r.id
    """)

    rows = cursor.fetchall()

    requirements = []
    for row in rows:
        req = {
            'id': row[0],
            'document_id': row[1],
            'requirement_text': row[2],
            'risk_category': row[3],
            'start_page': row[4],
            'end_page': row[5],
            'extraction_window': row[6],
            'document_filename': row[7]
        }
        requirements.append(req)

    return requirements


def create_embeddings(
    requirements: List[Dict[str, Any]],
    embedder: LegalBERTEmbedder,
    batch_size: int = 16
) -> np.ndarray:
    """
    Create embeddings for all requirements

    Args:
        requirements: List of requirement dictionaries
        embedder: LegalBERTEmbedder instance
        batch_size: Batch size for encoding

    Returns:
        numpy array of embeddings (num_requirements, embedding_dim)
    """
    print(f"\nGenerating embeddings for {len(requirements)} requirements...")

    # Use smaller batch size on CPU
    if embedder.device == 'cpu':
        batch_size = min(batch_size, 8)
        print(f"  Using batch size {batch_size} for CPU processing")
    else:
        print(f"  Using batch size {batch_size} for GPU processing")

    # Extract requirement texts
    texts = [req['requirement_text'] for req in requirements]

    # Generate embeddings with progress bar
    embeddings = embedder.encode(texts, batch_size=batch_size, show_progress=True)

    print(f"✓ Generated embeddings with shape: {embeddings.shape}")

    return embeddings


def save_embeddings(
    requirements: List[Dict[str, Any]],
    embeddings: np.ndarray,
    output_dir: Path
):
    """
    Save embeddings and metadata to files

    Args:
        requirements: List of requirement dictionaries
        embeddings: numpy array of embeddings
        output_dir: Directory to save outputs
    """
    print(f"\nSaving embeddings to {output_dir}...")

    # Save embeddings as numpy array
    embeddings_file = output_dir / "requirement_embeddings.npy"
    np.save(embeddings_file, embeddings)
    print(f"✓ Saved embeddings to {embeddings_file}")

    # Save metadata as JSON (all requirement fields except the text for space efficiency)
    metadata = []
    for i, req in enumerate(requirements):
        metadata.append({
            'index': i,  # Index in embeddings array
            'id': req['id'],
            'document_id': req['document_id'],
            'document_filename': req['document_filename'],
            'risk_category': req['risk_category'],
            'start_page': req['start_page'],
            'end_page': req['end_page'],
            'extraction_window': req['extraction_window'],
            'requirement_text': req['requirement_text']  # Include full text for reference
        })

    metadata_file = output_dir / "requirement_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved metadata to {metadata_file}")

    # Save complete data as pickle for easy loading
    complete_data = {
        'embeddings': embeddings,
        'metadata': metadata,
        'model_name': MODEL_NAME,
        'embedding_dim': embeddings.shape[1]
    }

    pickle_file = output_dir / "requirement_embeddings.pkl"
    with open(pickle_file, 'wb') as f:
        pickle.dump(complete_data, f)
    print(f"✓ Saved complete data to {pickle_file}")

    # Save summary statistics
    summary = {
        'model_name': MODEL_NAME,
        'num_requirements': len(requirements),
        'embedding_dimension': int(embeddings.shape[1]),
        'risk_categories': {},
        'documents': {}
    }

    # Count by risk category
    for req in requirements:
        risk = req['risk_category']
        summary['risk_categories'][risk] = summary['risk_categories'].get(risk, 0) + 1

    # Count by document
    for req in requirements:
        doc = req['document_filename']
        summary['documents'][doc] = summary['documents'].get(doc, 0) + 1

    summary_file = output_dir / "embedding_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved summary to {summary_file}")


def compute_similarity_matrix(embeddings: np.ndarray, output_dir: Path):
    """
    Compute and save cosine similarity matrix between all requirements

    Args:
        embeddings: numpy array of embeddings
        output_dir: Directory to save output
    """
    print("\nComputing similarity matrix...")
    print(f"  Matrix size: {embeddings.shape[0]} x {embeddings.shape[0]}")

    # Normalize embeddings for cosine similarity
    print("  Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Compute cosine similarity matrix (dot product of normalized vectors)
    print("  Computing dot product (this may take a moment)...")
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # Save similarity matrix
    print("  Saving matrix...")
    similarity_file = output_dir / "similarity_matrix.npy"
    np.save(similarity_file, similarity_matrix)
    print(f"  Saved similarity matrix to {similarity_file}")
    print(f"  Shape: {similarity_matrix.shape}")
    print(f"  Min similarity: {similarity_matrix.min():.4f}")
    print(f"  Max similarity: {similarity_matrix.max():.4f}")
    print(f"  Mean similarity: {similarity_matrix.mean():.4f}")

    return similarity_matrix


def find_similar_requirements(
    similarity_matrix: np.ndarray,
    requirements: List[Dict[str, Any]],
    top_k: int = 5,
    min_similarity: float = 0.7,
    output_dir: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Find most similar requirement pairs

    Args:
        similarity_matrix: Precomputed similarity matrix
        requirements: List of requirement dictionaries
        top_k: Number of top similar pairs to find
        min_similarity: Minimum similarity threshold
        output_dir: Optional directory to save results

    Returns:
        List of similar requirement pairs
    """
    print(f"\nFinding top {top_k} similar requirement pairs (similarity >= {min_similarity})...")

    # Get upper triangle indices (avoid duplicates and self-similarity)
    n = similarity_matrix.shape[0]
    similar_pairs = []

    # Use progress bar for large matrices
    print(f"  Scanning {n * (n-1) // 2:,} pairs...")
    for i in tqdm(range(n), desc="Finding similar pairs", disable=(n < 100)):
        for j in range(i + 1, n):
            sim = similarity_matrix[i, j]
            if sim >= min_similarity:
                similar_pairs.append({
                    'req1_idx': i,
                    'req2_idx': j,
                    'similarity': float(sim),
                    'req1_id': requirements[i]['id'],
                    'req2_id': requirements[j]['id'],
                    'req1_text': requirements[i]['requirement_text'],
                    'req2_text': requirements[j]['requirement_text'],
                    'req1_risk': requirements[i]['risk_category'],
                    'req2_risk': requirements[j]['risk_category'],
                    'req1_doc': requirements[i]['document_filename'],
                    'req2_doc': requirements[j]['document_filename']
                })

    # Sort by similarity (descending)
    print("  Sorting results...")
    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)

    # Take top k
    top_pairs = similar_pairs[:top_k]

    print(f"✓ Found {len(similar_pairs)} pairs above threshold")
    print(f"  Returning top {len(top_pairs)} pairs")

    # Print top pairs
    for i, pair in enumerate(top_pairs[:10], 1):
        print(f"\n  [{i}] Similarity: {pair['similarity']:.4f}")
        print(f"      Req {pair['req1_id']} ({pair['req1_risk']}): {pair['req1_text'][:100]}...")
        print(f"      Req {pair['req2_id']} ({pair['req2_risk']}): {pair['req2_text'][:100]}...")

    # Save if output_dir provided
    if output_dir:
        similar_pairs_file = output_dir / "similar_requirement_pairs.json"
        with open(similar_pairs_file, 'w', encoding='utf-8') as f:
            json.dump(top_pairs, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Saved similar pairs to {similar_pairs_file}")

    return top_pairs


def main():
    """Main execution pipeline"""
    print("=" * 80)
    print("REQUIREMENT EMBEDDINGS GENERATION PIPELINE")
    print(f"Model: {MODEL_NAME}")
    print("=" * 80)

    # Connect to database
    print(f"\n📊 Connecting to database: {DB_PATH}")
    conn = sqlite3.connect(str(DB_PATH))

    # Load requirements
    requirements = load_requirements_from_db(conn)
    print(f"✓ Loaded {len(requirements)} requirements from database")

    # Show sample
    if requirements:
        print(f"\nSample requirement:")
        sample = requirements[0]
        print(f"  ID: {sample['id']}")
        print(f"  Document: {sample['document_filename']}")
        print(f"  Risk Category: {sample['risk_category']}")
        print(f"  Pages: {sample['start_page']}-{sample['end_page']}")
        print(f"  Text: {sample['requirement_text'][:200]}...")

    # Initialize embedder
    print(f"\nInitializing Legal-BERT embedder...")
    embedder = LegalBERTEmbedder(MODEL_NAME)

    # Generate embeddings
    embeddings = create_embeddings(requirements, embedder, batch_size=16)

    # Save embeddings and metadata
    save_embeddings(requirements, embeddings, EMBEDDINGS_OUTPUT_DIR)

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings, EMBEDDINGS_OUTPUT_DIR)

    # Find similar requirements
    find_similar_requirements(
        similarity_matrix,
        requirements,
        top_k=100,  # Save top 100 pairs
        min_similarity=0.7,
        output_dir=EMBEDDINGS_OUTPUT_DIR
    )

    conn.close()

    print("\n" + "=" * 80)
    print("EMBEDDING GENERATION COMPLETE")
    print(f"Output directory: {EMBEDDINGS_OUTPUT_DIR}")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - requirement_embeddings.npy: Raw embeddings array")
    print("  - requirement_metadata.json: All requirement metadata")
    print("  - requirement_embeddings.pkl: Complete data (embeddings + metadata)")
    print("  - embedding_summary.json: Statistics and summary")
    print("  - similarity_matrix.npy: Cosine similarity between all requirements")
    print("  - similar_requirement_pairs.json: Top similar requirement pairs")


if __name__ == "__main__":
    main()