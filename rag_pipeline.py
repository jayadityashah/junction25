"""
RAG Pipeline for Finding Contradictions and Overlaps in Financial Regulation
Related to Liquidity Risk
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Configuration
EN1_DIR = "EN1"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SIMILARITY_THRESHOLD = 0.3  # Adjustable threshold (lowered for better recall)
TOP_K = 7
OUTPUT_FILE = "liquidity_risk_analysis.json"


def load_json_files(directory: str) -> List[Dict[str, Any]]:
    """Load all JSON files from the specified directory."""
    json_files = []
    dir_path = Path(directory)
    
    for json_file in dir_path.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            json_files.append({
                'filename': json_file.name,
                'data': data
            })
    
    return json_files


def extract_and_chunk_text(json_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract text from JSON files and chunk it."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    chunks = []
    
    for file_info in json_files:
        filename = file_info['filename']
        data = file_info['data']
        
        if 'pages' in data:
            for page in data['pages']:
                page_num = page.get('page', None)
                paragraphs = page.get('paragraphs', [])
                
                # Combine all paragraphs into text blocks
                for paragraph in paragraphs:
                    if isinstance(paragraph, str) and len(paragraph.strip()) > 0:
                        # Split the paragraph into chunks
                        text_chunks = text_splitter.split_text(paragraph)
                        
                        for chunk in text_chunks:
                            chunks.append({
                                'text': chunk,
                                'source': filename,
                                'page': page_num
                            })
    
    return chunks


def create_faiss_index(chunks: List[Dict[str, Any]]) -> FAISS:
    """Create FAISS vector store from chunks."""
    print(f"Creating embeddings for {len(chunks)} chunks...")
    
    texts = [chunk['text'] for chunk in chunks]
    metadatas = [
        {
            'source': chunk['source'],
            'page': chunk['page']
        }
        for chunk in chunks
    ]
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    return vectorstore


def generate_liquidity_risk_definition() -> str:
    """Generate liquidity risk definition using Gemini."""
    print("Generating liquidity risk definition using Gemini...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    prompt = """Write 3-5 sentences defining liquidity risk for banks. 
    Mention several usual requirements associated with this risk, such as LCR (Liquidity Coverage Ratio), 
    NSFR (Net Stable Funding Ratio), liquid assets, funding requirements, etc. 
    This should provide a good overview with relevant keywords."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    definition = response.content
    
    return definition


def search_similar_chunks(vectorstore: FAISS, query: str, threshold: float, top_k: int) -> List[Dict[str, Any]]:
    """Search for similar chunks using FAISS."""
    print(f"Searching for similar chunks (threshold: {threshold}, top_k: {top_k})...")
    
    # Use similarity_search_with_score to get scores
    # FAISS with OpenAI embeddings uses L2 distance (Euclidean distance)
    # Lower distance = more similar
    # Get more results to filter and debug
    results = vectorstore.similarity_search_with_score(query, k=top_k * 5)
    
    # Debug: print top scores
    print(f"\nTop {min(10, len(results))} results with scores:")
    for i, (doc, distance) in enumerate(results[:10], 1):
        # FAISS can return different distance metrics:
        # - L2 distance: ranges from 0 to ~2 for normalized vectors
        # - Cosine distance: ranges from 0 to 2 (where 0 = identical, 2 = opposite)
        # - Inner product: can be negative or positive
        
        # Try to detect the metric type and convert appropriately
        # If distance is typically < 2, it might be cosine distance
        # If distance can be > 2, it's likely L2
        # For cosine distance: similarity = 1 - (distance / 2)
        # For L2 distance: similarity = 1 / (1 + distance)
        
        # Use a hybrid approach that works for both
        if distance <= 2.0:
            # Likely cosine distance: convert to similarity
            similarity = 1 - (distance / 2.0)
        else:
            # Likely L2 distance: use inverse relationship
            similarity = 1 / (1 + distance)
        
        # Clamp to [0, 1]
        similarity = max(0.0, min(1.0, similarity))
        
        metadata = doc.metadata
        source = metadata.get('source', 'unknown')
        page = metadata.get('page', None)
        text_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        
        if page is not None:
            print(f"  [{i}] {source}, page {page}: distance={distance:.4f}, similarity={similarity:.4f}")
        else:
            print(f"  [{i}] {source}: distance={distance:.4f}, similarity={similarity:.4f}")
        print(f"      Preview: {text_preview}")
    
    # Filter by threshold and format results
    filtered_results = []
    for doc, distance in results:
        # Convert distance to similarity score (same logic as above)
        if distance <= 2.0:
            similarity = 1 - (distance / 2.0)
        else:
            similarity = 1 / (1 + distance)
        
        similarity = max(0.0, min(1.0, similarity))
        # Convert to native Python float to avoid JSON serialization issues
        similarity = float(similarity)
        
        if similarity >= threshold:
            metadata = doc.metadata
            filtered_results.append({
                'text': doc.page_content,
                'source': metadata.get('source', 'unknown'),
                'page': metadata.get('page', None),
                'score': similarity
            })
    
    # If no results meet threshold, return top_k anyway with a warning
    if len(filtered_results) == 0 and len(results) > 0:
        print(f"\nWarning: No chunks met threshold {threshold}. Returning top {top_k} results anyway.")
        for doc, distance in results[:top_k]:
            # Use same similarity calculation
            if distance <= 2.0:
                similarity = 1 - (distance / 2.0)
            else:
                similarity = 1 / (1 + distance)
            similarity = max(0.0, min(1.0, similarity))
            # Convert to native Python float to avoid JSON serialization issues
            similarity = float(similarity)
            
            metadata = doc.metadata
            filtered_results.append({
                'text': doc.page_content,
                'source': metadata.get('source', 'unknown'),
                'page': metadata.get('page', None),
                'score': similarity
            })
    
    return filtered_results[:top_k]


def format_chunks_for_analysis(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks for analysis prompt."""
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk['source']
        page = chunk.get('page')
        text = chunk['text']
        
        if page is not None:
            formatted.append(f"[Excerpt {i} - {source}, page {page}]: \"{text}\"")
        else:
            formatted.append(f"[Excerpt {i} - {source}]: \"{text}\"")
    
    return "\n\n".join(formatted)


def analyze_contradictions(chunks_text: str) -> str:
    """Analyze chunks for contradictions and overlaps using Gemini."""
    print("Analyzing contradictions and overlaps using Gemini...")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    
    prompt = """You are a regulatory compliance expert analyzing liquidity risk requirements.

Below are excerpts from different EU and Finnish financial regulations. Each excerpt is labeled with its source document.

Your task:

1. Identify REQUIREMENTS - specific liquidity risk requirements mentioned in the regulations

2. Identify any CONTRADICTIONS - where two regulations give conflicting requirements for the same thing

3. Identify OVERLAPS - where regulations cover the same requirement but with different levels of detail

CRITICAL: Respond with ONLY valid JSON. No markdown headers, no code fences, no extra text.

EXCERPTS:

""" + chunks_text + """

Output this EXACT JSON structure:

{
  "requirements": [
    "requirement 1",
    "requirement 2"
  ],
  "contradictions": [
    {
      "title": "brief title",
      "sources": ["filename.json, page X", "filename.json, page Y"],
      "explanation": "1-2 sentence explanation",
      "stricter": "which is stricter"
    }
  ],
  "overlaps": [
    {
      "title": "brief title", 
      "sources": ["filename.json, page X", "filename.json, page Y"],
      "explanation": "1-2 sentence explanation"
    }
  ],
  "summary": "2-3 sentences about liquidity risk regulation state"
}

Output ONLY the JSON object. Start with { and end with }."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    analysis = response.content
    
    return analysis


def parse_analysis(analysis_text: str) -> Dict[str, Any]:
    """Parse Gemini's JSON analysis into structured format."""
    result = {
        "category": "liquidity risk",
        "requirements": [],
        "contradictions": [],
        "overlaps": [],
        "summary": ""
    }
    
    # Clean the response - remove markdown code fences if present
    cleaned_text = analysis_text.strip()
    
    # Remove markdown code fences if present
    if cleaned_text.startswith('```json'):
        cleaned_text = cleaned_text[7:]
    elif cleaned_text.startswith('```'):
        cleaned_text = cleaned_text[3:]
    
    if cleaned_text.endswith('```'):
        cleaned_text = cleaned_text[:-3]
    
    cleaned_text = cleaned_text.strip()
    
    # Find JSON object boundaries
    start_idx = cleaned_text.find('{')
    end_idx = cleaned_text.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        print("Warning: Could not find JSON object in response. Using raw text.")
        result['summary'] = cleaned_text
        return result
    
    json_text = cleaned_text[start_idx:end_idx + 1]
    
    try:
        parsed = json.loads(json_text)
        
        # Extract requirements
        result['requirements'] = parsed.get('requirements', [])
        
        # Process contradictions - convert sources array to documents array
        contradictions = parsed.get('contradictions', [])
        processed_contradictions = []
        for item in contradictions:
            sources = item.get('sources', [])
            documents = []
            for source in sources:
                # Parse "filename.json, page X" format
                match = re.match(r'([a-zA-Z0-9_\-\.]+\.json)(?:\s*,\s*page\s+(\d+))?', source)
                if match:
                    filename = match.group(1)
                    page = int(match.group(2)) if match.group(2) else None
                    documents.append({
                        'document': filename,
                        'page': page
                    })
            
            processed_contradictions.append({
                'title': item.get('title', ''),
                'documents': documents,
                'explanation': item.get('explanation', ''),
                'stricter': item.get('stricter', '')
            })
        
        # Process overlaps - convert sources array to documents array
        overlaps = parsed.get('overlaps', [])
        processed_overlaps = []
        for item in overlaps:
            sources = item.get('sources', [])
            documents = []
            for source in sources:
                # Parse "filename.json, page X" format
                match = re.match(r'([a-zA-Z0-9_\-\.]+\.json)(?:\s*,\s*page\s+(\d+))?', source)
                if match:
                    filename = match.group(1)
                    page = int(match.group(2)) if match.group(2) else None
                    documents.append({
                        'document': filename,
                        'page': page
                    })
            
            processed_overlaps.append({
                'title': item.get('title', ''),
                'documents': documents,
                'explanation': item.get('explanation', '')
            })
        
        result['contradictions'] = processed_contradictions
        result['overlaps'] = processed_overlaps
        result['summary'] = parsed.get('summary', '')
        
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON response: {e}")
        print(f"Response text: {json_text[:500]}")
        result['summary'] = cleaned_text
    
    return result


def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("RAG Pipeline for Liquidity Risk Regulation Analysis")
    print("=" * 60)
    print()
    
    # Step 1: Generate liquidity risk definition
    definition = generate_liquidity_risk_definition()
    print("\n" + "=" * 60)
    print("LIQUIDITY RISK DEFINITION (Query):")
    print("=" * 60)
    print(definition)
    print()
    
    # Step 2: Load and chunk documents
    print("Loading JSON files from EN1 directory...")
    json_files = load_json_files(EN1_DIR)
    print(f"Loaded {len(json_files)} JSON files")
    
    print("Extracting and chunking text...")
    chunks = extract_and_chunk_text(json_files)
    print(f"Created {len(chunks)} chunks")
    
    # Step 3: Create embeddings and FAISS index
    vectorstore = create_faiss_index(chunks)
    print("FAISS index created")
    
    # Step 4: Search for similar chunks
    similar_chunks = search_similar_chunks(
        vectorstore, 
        definition, 
        SIMILARITY_THRESHOLD, 
        TOP_K
    )
    
    print(f"\nFound {len(similar_chunks)} similar chunks above threshold")
    print("\n" + "=" * 60)
    print("RETRIEVED CHUNKS:")
    print("=" * 60)
    for i, chunk in enumerate(similar_chunks, 1):
        source = chunk['source']
        page = chunk.get('page')
        text = chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
        score = chunk.get('score', 0)
        
        if page is not None:
            print(f"\n[{i}] {source}, page {page} (similarity: {score:.3f})")
        else:
            print(f"\n[{i}] {source} (similarity: {score:.3f})")
        print(f"    {text}")
    
    # Step 5: Analyze for contradictions
    chunks_text = format_chunks_for_analysis(similar_chunks)
    analysis = analyze_contradictions(chunks_text)
    
    # Step 6: Parse and save results
    parsed_result = parse_analysis(analysis)
    
    # Print formatted analysis to terminal
    print("\n" + "=" * 60)
    print("GEMINI ANALYSIS:")
    print("=" * 60)
    
    # Print requirements
    if parsed_result.get('requirements'):
        print("\n## Requirements")
        for req in parsed_result['requirements']:
            print(f"  • {req}")
    
    # Print contradictions
    if parsed_result.get('contradictions'):
        print("\n## Contradictions")
        for i, contr in enumerate(parsed_result['contradictions'], 1):
            print(f"\n{i}. {contr.get('title', 'Untitled')}")
            sources_str = ", ".join([
                f"{doc['document']}, page {doc['page']}" if doc.get('page') 
                else doc['document'] 
                for doc in contr.get('documents', [])
            ])
            print(f"   Sources: {sources_str}")
            print(f"   Explanation: {contr.get('explanation', '')}")
            if contr.get('stricter'):
                print(f"   Stricter: {contr.get('stricter', '')}")
    
    # Print overlaps
    if parsed_result.get('overlaps'):
        print("\n## Overlaps")
        for i, overlap in enumerate(parsed_result['overlaps'], 1):
            print(f"\n{i}. {overlap.get('title', 'Untitled')}")
            sources_str = ", ".join([
                f"{doc['document']}, page {doc['page']}" if doc.get('page') 
                else doc['document'] 
                for doc in overlap.get('documents', [])
            ])
            print(f"   Sources: {sources_str}")
            print(f"   Explanation: {overlap.get('explanation', '')}")
    
    # Print summary
    if parsed_result.get('summary'):
        print("\n## Summary")
        print(f"  {parsed_result['summary']}")
    
    print()
    
    # Add the retrieved chunks info to the result
    # Convert numpy types to native Python types for JSON serialization
    parsed_result['retrieved_chunks'] = [
        {
            'source': chunk['source'],
            'page': chunk.get('page'),
            'text': chunk['text'],
            'similarity_score': float(chunk.get('score', 0))  # Convert to native Python float
        }
        for chunk in similar_chunks
    ]
    parsed_result['query_definition'] = definition
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(parsed_result, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {OUTPUT_FILE}")
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

