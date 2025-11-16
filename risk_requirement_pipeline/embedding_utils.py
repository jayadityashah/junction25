"""
Utility functions for working with requirement embeddings
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

EMBEDDINGS_DIR = Path(__file__).parent / "embeddings"


def load_embeddings(embeddings_dir: Path = EMBEDDINGS_DIR) -> Dict[str, Any]:
    """
    Load embeddings and metadata from pickle file

    Args:
        embeddings_dir: Directory containing embeddings

    Returns:
        Dictionary with keys: 'embeddings', 'metadata', 'model_name', 'embedding_dim'
    """
    pickle_file = embeddings_dir / "requirement_embeddings.pkl"

    if not pickle_file.exists():
        raise FileNotFoundError(
            f"Embeddings file not found at {pickle_file}. "
            "Run create_embeddings.py first to generate embeddings."
        )

    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    print(f"✓ Loaded {len(data['metadata'])} embeddings from {pickle_file}")
    print(f"  Model: {data['model_name']}")
    print(f"  Embedding dimension: {data['embedding_dim']}")

    return data


def load_similarity_matrix(embeddings_dir: Path = EMBEDDINGS_DIR) -> np.ndarray:
    """
    Load precomputed similarity matrix

    Args:
        embeddings_dir: Directory containing embeddings

    Returns:
        Similarity matrix (numpy array)
    """
    similarity_file = embeddings_dir / "similarity_matrix.npy"

    if not similarity_file.exists():
        raise FileNotFoundError(
            f"Similarity matrix not found at {similarity_file}. "
            "Run create_embeddings.py first."
        )

    similarity_matrix = np.load(similarity_file)
    print(f"✓ Loaded similarity matrix with shape {similarity_matrix.shape}")

    return similarity_matrix


def find_similar_by_id(
    requirement_id: int,
    data: Dict[str, Any],
    similarity_matrix: np.ndarray,
    top_k: int = 5,
    min_similarity: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Find most similar requirements to a given requirement ID

    Args:
        requirement_id: ID of the requirement to find similar ones for
        data: Embeddings data from load_embeddings()
        similarity_matrix: Similarity matrix from load_similarity_matrix()
        top_k: Number of similar requirements to return
        min_similarity: Minimum similarity threshold

    Returns:
        List of similar requirements with metadata and similarity scores
    """
    # Find index of the requirement
    req_idx = None
    for i, meta in enumerate(data['metadata']):
        if meta['id'] == requirement_id:
            req_idx = i
            break

    if req_idx is None:
        raise ValueError(f"Requirement ID {requirement_id} not found")

    # Get similarities for this requirement
    similarities = similarity_matrix[req_idx]

    # Get indices sorted by similarity (excluding self)
    sorted_indices = np.argsort(similarities)[::-1]

    # Filter and collect results
    results = []
    for idx in sorted_indices:
        if idx == req_idx:  # Skip self
            continue

        sim = similarities[idx]
        if sim < min_similarity:
            break

        if len(results) >= top_k:
            break

        meta = data['metadata'][idx]
        results.append({
            'similarity': float(sim),
            'id': meta['id'],
            'document_id': meta['document_id'],
            'document_filename': meta['document_filename'],
            'risk_category': meta['risk_category'],
            'start_page': meta['start_page'],
            'end_page': meta['end_page'],
            'requirement_text': meta['requirement_text']
        })

    return results


def find_similar_by_text(
    query_text: str,
    data: Dict[str, Any],
    embedder,  # LegalBERTEmbedder instance
    top_k: int = 5,
    min_similarity: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Find most similar requirements to a query text

    Args:
        query_text: Text to find similar requirements for
        data: Embeddings data from load_embeddings()
        embedder: LegalBERTEmbedder instance for encoding query
        top_k: Number of similar requirements to return
        min_similarity: Minimum similarity threshold

    Returns:
        List of similar requirements with metadata and similarity scores
    """
    # Encode query text
    query_embedding = embedder.encode([query_text])[0]

    # Normalize query embedding
    query_norm = query_embedding / np.linalg.norm(query_embedding)

    # Normalize all embeddings
    embeddings = data['embeddings']
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Compute similarities
    similarities = np.dot(normalized_embeddings, query_norm)

    # Get indices sorted by similarity
    sorted_indices = np.argsort(similarities)[::-1]

    # Filter and collect results
    results = []
    for idx in sorted_indices:
        sim = similarities[idx]
        if sim < min_similarity:
            break

        if len(results) >= top_k:
            break

        meta = data['metadata'][idx]
        results.append({
            'similarity': float(sim),
            'id': meta['id'],
            'document_id': meta['document_id'],
            'document_filename': meta['document_filename'],
            'risk_category': meta['risk_category'],
            'start_page': meta['start_page'],
            'end_page': meta['end_page'],
            'requirement_text': meta['requirement_text']
        })

    return results


def filter_by_risk_category(
    data: Dict[str, Any],
    risk_category: str
) -> Dict[str, Any]:
    """
    Filter embeddings by risk category

    Args:
        data: Embeddings data from load_embeddings()
        risk_category: Risk category to filter by

    Returns:
        Filtered data dictionary
    """
    # Find indices matching the risk category
    matching_indices = [
        i for i, meta in enumerate(data['metadata'])
        if meta['risk_category'] == risk_category
    ]

    if not matching_indices:
        return {
            'embeddings': np.array([]),
            'metadata': [],
            'model_name': data['model_name'],
            'embedding_dim': data['embedding_dim']
        }

    # Filter embeddings and metadata
    filtered_embeddings = data['embeddings'][matching_indices]
    filtered_metadata = [data['metadata'][i] for i in matching_indices]

    return {
        'embeddings': filtered_embeddings,
        'metadata': filtered_metadata,
        'model_name': data['model_name'],
        'embedding_dim': data['embedding_dim']
    }


def filter_by_document(
    data: Dict[str, Any],
    document_id: Optional[int] = None,
    document_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Filter embeddings by document

    Args:
        data: Embeddings data from load_embeddings()
        document_id: Document ID to filter by
        document_filename: Document filename to filter by

    Returns:
        Filtered data dictionary
    """
    # Find indices matching the document
    matching_indices = []
    for i, meta in enumerate(data['metadata']):
        if document_id is not None and meta['document_id'] == document_id:
            matching_indices.append(i)
        elif document_filename is not None and meta['document_filename'] == document_filename:
            matching_indices.append(i)

    if not matching_indices:
        return {
            'embeddings': np.array([]),
            'metadata': [],
            'model_name': data['model_name'],
            'embedding_dim': data['embedding_dim']
        }

    # Filter embeddings and metadata
    filtered_embeddings = data['embeddings'][matching_indices]
    filtered_metadata = [data['metadata'][i] for i in matching_indices]

    return {
        'embeddings': filtered_embeddings,
        'metadata': filtered_metadata,
        'model_name': data['model_name'],
        'embedding_dim': data['embedding_dim']
    }


def cluster_requirements(
    data: Dict[str, Any],
    n_clusters: int = 10,
    method: str = 'kmeans'
) -> Dict[str, Any]:
    """
    Cluster requirements based on embeddings

    Args:
        data: Embeddings data from load_embeddings()
        n_clusters: Number of clusters
        method: Clustering method ('kmeans' or 'hierarchical')

    Returns:
        Dictionary with cluster assignments and cluster info
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering

    embeddings = data['embeddings']

    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'hierarchical':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    # Fit and predict
    cluster_labels = clusterer.fit_predict(embeddings)

    # Organize by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        label = int(label)
        if label not in clusters:
            clusters[label] = []

        meta = data['metadata'][i]
        clusters[label].append({
            'id': meta['id'],
            'requirement_text': meta['requirement_text'],
            'risk_category': meta['risk_category'],
            'document_filename': meta['document_filename']
        })

    # Compute cluster statistics
    cluster_stats = {}
    for label, items in clusters.items():
        risk_counts = {}
        for item in items:
            risk = item['risk_category']
            risk_counts[risk] = risk_counts.get(risk, 0) + 1

        cluster_stats[label] = {
            'size': len(items),
            'risk_distribution': risk_counts,
            'sample_texts': [item['requirement_text'][:100] for item in items[:3]]
        }

    return {
        'cluster_labels': cluster_labels.tolist(),
        'clusters': clusters,
        'cluster_stats': cluster_stats,
        'n_clusters': n_clusters,
        'method': method
    }


def get_requirement_by_id(
    requirement_id: int,
    data: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Get a specific requirement by ID

    Args:
        requirement_id: ID of the requirement
        data: Embeddings data from load_embeddings()

    Returns:
        Requirement metadata or None if not found
    """
    for meta in data['metadata']:
        if meta['id'] == requirement_id:
            return meta

    return None


if __name__ == "__main__":
    # Example usage
    print("Loading embeddings...")
    data = load_embeddings()
    similarity_matrix = load_similarity_matrix()

    print("\n" + "=" * 80)
    print("EXAMPLE: Finding similar requirements")
    print("=" * 80)

    # Example: Find similar requirements to requirement ID 1
    if data['metadata']:
        sample_id = data['metadata'][0]['id']
        print(f"\nFinding requirements similar to ID {sample_id}...")

        similar = find_similar_by_id(
            sample_id,
            data,
            similarity_matrix,
            top_k=5,
            min_similarity=0.5
        )

        print(f"\nFound {len(similar)} similar requirements:")
        for i, req in enumerate(similar, 1):
            print(f"\n[{i}] Similarity: {req['similarity']:.4f}")
            print(f"    ID: {req['id']}")
            print(f"    Risk: {req['risk_category']}")
            print(f"    Document: {req['document_filename']}")
            print(f"    Text: {req['requirement_text'][:150]}...")

    print("\n" + "=" * 80)
    print("EXAMPLE: Filtering by risk category")
    print("=" * 80)

    # Get unique risk categories
    risk_categories = set(meta['risk_category'] for meta in data['metadata'])
    print(f"\nAvailable risk categories: {', '.join(sorted(risk_categories))}")

    # Filter by first category
    if risk_categories:
        sample_category = sorted(risk_categories)[0]
        filtered_data = filter_by_risk_category(data, sample_category)
        print(f"\nFiltered to {len(filtered_data['metadata'])} requirements in category '{sample_category}'")