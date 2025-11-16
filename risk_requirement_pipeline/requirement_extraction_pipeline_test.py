"""
Risk-Based Requirement Extraction Pipeline
Extracts regulatory requirements from documents using sliding window approach
"""

import sqlite3
import os
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import json
from tqdm import tqdm
import concurrent.futures

# Import synthesis function
import sys
sys.path.insert(0, str(Path(__file__).parent))
from synthesize_requirements import synthesize_document_requirements

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "legal_documents.db"
RISK_CATEGORIES_PATH = PROJECT_ROOT / "risk-graphrag-project" / "config" / "risk_categories.yaml"

# Chunk configuration
WINDOW_SIZE = 100  # pages per chunk
STRIDE = 100  # non-overlapping chunks


def load_risk_categories() -> Dict[str, Any]:
    """Load risk categories from YAML file"""
    with open(RISK_CATEGORIES_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['risk_categories']


def create_requirements_schema(conn: sqlite3.Connection):
    """Create database tables for requirements and relationships"""
    cursor = conn.cursor()
    
    # Requirements table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS requirements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL,
        requirement_text TEXT NOT NULL,
        risk_category TEXT NOT NULL,
        start_page INTEGER NOT NULL,
        end_page INTEGER NOT NULL,
        extraction_window TEXT,
        FOREIGN KEY (document_id) REFERENCES documents(id),
        UNIQUE(document_id, requirement_text, start_page, end_page)
    )
    """)
    
    # Relationships table (pairwise or grouped)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS requirement_relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        relationship_type TEXT NOT NULL,  -- 'OVERLAP' or 'CONTRADICTION'
        risk_category TEXT NOT NULL,
        description TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Junction table for many-to-many relationship
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS relationship_requirements (
        relationship_id INTEGER NOT NULL,
        requirement_id INTEGER NOT NULL,
        FOREIGN KEY (relationship_id) REFERENCES requirement_relationships(id),
        FOREIGN KEY (requirement_id) REFERENCES requirements(id),
        PRIMARY KEY (relationship_id, requirement_id)
    )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_requirements_doc ON requirements(document_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_requirements_risk ON requirements(risk_category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_relationships_risk ON requirement_relationships(risk_category)")
    
    conn.commit()
    print("✅ Requirements schema created")




def get_document_pages(conn: sqlite3.Connection, doc_id: int) -> List[Dict[str, Any]]:
    """Get all pages with paragraphs for a document"""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT p.page_number, pg.content
        FROM pages p
        LEFT JOIN paragraphs pg ON pg.page_id = p.id
        WHERE p.document_id = ?
        ORDER BY p.page_number, pg.paragraph_index
    """, (doc_id,))
    
    rows = cursor.fetchall()
    
    # Group by page
    pages_dict = {}
    for page_num, content in rows:
        if page_num not in pages_dict:
            pages_dict[page_num] = []
        if content:
            pages_dict[page_num].append(content)
    
    # Convert to sorted list
    pages = []
    for page_num in sorted(pages_dict.keys()):
        pages.append({
            'page_number': page_num,
            'content': '\n\n'.join(pages_dict[page_num])
        })
    
    return pages


def create_page_chunks(pages: List[Dict[str, Any]], chunk_size: int) -> List[Dict[str, Any]]:
    """Create non-overlapping chunks of pages"""
    chunks = []
    
    for i in range(0, len(pages), chunk_size):
        chunk_pages = pages[i:i + chunk_size]
        if len(chunk_pages) > 0:
            chunks.append({
                'start_page': chunk_pages[0]['page_number'],
                'end_page': chunk_pages[-1]['page_number'],
                'pages': chunk_pages
            })
    
    return chunks


def extract_requirements_from_chunk(
    chunk: Dict[str, Any],
    document_name: str,
    existing_requirements: List[Dict[str, Any]],
    risk_categories: Dict[str, Any],
    llm: ChatGoogleGenerativeAI
) -> List[Dict[str, Any]]:
    """Extract requirements from a chunk of pages using Gemini"""
    
    # Format chunk content with clear page markers
    chunk_text = ""
    for page in chunk['pages']:
        chunk_text += f"\n\n--- PAGE {page['page_number']} ---\n\n{page['content']}"
    
    # Format risk categories
    risk_list = []
    for risk_name, risk_data in risk_categories.items():
        terms = risk_data.get('primary_terms', [])
        risk_list.append(f"- {risk_name}: {', '.join(terms[:3])}")
    risk_categories_text = '\n'.join(risk_list)
    
    
    prompt = f"""You are a financial regulatory expert extracting requirements from: {document_name}

CONTEXT: You are analyzing pages {chunk['start_page']}-{chunk['end_page']} of this document. After all pages are processed, the complete requirement list will be SYNTHESIZED and deduplicated. So if you see related requirements, extract them even if slightly overlapping - the synthesis step will consolidate them.

RISK CATEGORIES (choose ONE most relevant per requirement):
{risk_categories_text}

EXTRACTION CRITERIA - Only extract if ALL apply:
1. MANDATORY or RECOMMENDED regulatory obligation (uses "shall", "must", "should", "required")
2. SPECIFIC and ACTIONABLE (not vague principles)
3. STRONGLY relates to one of the risk categories above (pick the MOST SIGNIFICANT risk if multiple apply)
4. SUBSTANTIAL (not minor procedural details)

DO NOT EXTRACT:
- Definitions, explanations, background info
- Examples or illustrations
- Transitional provisions, effective dates
- Cross-references to other documents
- General principles without obligations


IMPORTANT: Later pages may clarify or modify requirements from earlier pages. Extract what you see now; synthesis will happen after all pages are reviewed.

TEXT TO ANALYZE (Document: {document_name}, Pages {chunk['start_page']}-{chunk['end_page']}):
{chunk_text}

OUTPUT FORMAT - Return ONLY valid JSON array:
[
  {{
    "requirement": "Complete requirement text",
    "risk_category": "EXACT_RISK_NAME_FROM_LIST",
    "relevant_pages": [page_numbers_where_requirement_appears]
  }}
]

If no new requirements, return: []
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        
        # Clean markdown if present
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
        
        # Parse JSON
        requirements = json.loads(content)
        
        # Validate and add metadata
        validated_reqs = []
        for req in requirements:
            if 'requirement' in req and 'risk_category' in req:
                # Handle page ranges - prefer relevant_pages array, fallback to start/end, then chunk
                if 'relevant_pages' in req and req['relevant_pages']:
                    # Convert page array to min/max range
                    pages = [p for p in req['relevant_pages'] if isinstance(p, int)]
                    if pages:
                        start_page = min(pages)
                        end_page = max(pages)
                    else:
                        start_page = chunk['start_page']
                        end_page = chunk['end_page']
                elif 'start_page' in req and 'end_page' in req:
                    start_page = req['start_page']
                    end_page = req['end_page']
                else:
                    # Fallback to chunk range
                    start_page = chunk['start_page']
                    end_page = chunk['end_page']

                # Store specific pages in extraction_window if available
                extraction_window = None
                if 'relevant_pages' in req and req['relevant_pages']:
                    extraction_window = json.dumps(sorted(req['relevant_pages']))

                validated_reqs.append({
                    'requirement_text': req['requirement'],
                    'risk_category': req['risk_category'],
                    'start_page': start_page,
                    'end_page': end_page,
                    'extraction_window': extraction_window
                })
        
        return validated_reqs
        
    except Exception as e:
        print(f"  ⚠️  Error extracting requirements: {e}")
        return []


def process_document(
    doc_id: int,
    doc_filename: str,
    conn: sqlite3.Connection,
    risk_categories: Dict[str, Any],
    llm: ChatGoogleGenerativeAI
) -> int:
    """Process a single document with 20-page chunks + synthesis"""
    
    print(f"\n📄 Processing: {doc_filename}")
    
    # Get pages
    pages = get_document_pages(conn, doc_id)
    if not pages:
        print("  ⚠️  No pages found")
        return 0
    
    print(f"  📃 Pages: {len(pages)}")
    
    # Create non-overlapping 20-page chunks
    chunks = create_page_chunks(pages, WINDOW_SIZE)
    print(f"  📦 Chunks: {len(chunks)} x {WINDOW_SIZE} pages")
    
    # Extract requirements from each chunk concurrently (max 5 concurrent calls)
    all_requirements = []

    def process_chunk_with_progress(chunk_idx_chunk):
        chunk_idx, chunk = chunk_idx_chunk
        print(f"  Processing chunk {chunk_idx}/{len(chunks)} (pages {chunk['start_page']}-{chunk['end_page']})...", end="", flush=True)

        # Extract requirements
        new_reqs = extract_requirements_from_chunk(chunk, doc_filename, [], risk_categories, llm)

        if new_reqs:
            print(f"✓ {len(new_reqs)} requirements")
            return new_reqs
        else:
            print("✓ No new requirements")
            return []

    # Process chunks concurrently with max 10 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        chunk_results = list(executor.map(process_chunk_with_progress, enumerate(chunks, 1)))

    # Flatten results
    for result in chunk_results:
        all_requirements.extend(result)
    
    # Synthesize/consolidate requirements by risk category
    if len(all_requirements) > 0:
        # Group requirements by risk category
        requirements_by_risk = {}
        for req in all_requirements:
            risk = req['risk_category']
            if risk not in requirements_by_risk:
                requirements_by_risk[risk] = []
            requirements_by_risk[risk].append(req)

        total_synthesized = 0

        # Synthesize each risk category separately
        for risk_category, risk_requirements in requirements_by_risk.items():
            if len(risk_requirements) > 0:
                print(f"  🔄 Synthesizing {len(risk_requirements)} requirements for {risk_category}...")
                synthesized = synthesize_document_requirements(risk_requirements, doc_filename, llm)
                print(f"  ✓ Synthesized {risk_category} into {len(synthesized)} final requirements")

                # Insert synthesized requirements into database
                cursor = conn.cursor()
                for req in synthesized:
                    cursor.execute("""
                        INSERT OR IGNORE INTO requirements (document_id, requirement_text, risk_category, start_page, end_page, extraction_window)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (doc_id, req['requirement_text'], req['risk_category'], req['start_page'], req['end_page'], req.get('extraction_window')))

                    if cursor.lastrowid > 0:
                        req['id'] = cursor.lastrowid

                total_synthesized += len(synthesized)

        conn.commit()
        print(f"  ✅ Total requirements saved: {total_synthesized}")
        return total_synthesized
    else:
        print(f"  ⚠️  No requirements extracted")
        return 0


def extract_all_requirements(limit_docs: Optional[int] = None, force_reprocess: bool = False):
    """Main extraction pipeline for all documents"""
    
    print("=" * 70)
    print("RISK-BASED REQUIREMENT EXTRACTION PIPELINE")
    print("=" * 70)
    
    # Load risk categories
    print("\n📋 Loading risk categories...")
    risk_categories = load_risk_categories()
    print(f"  Loaded {len(risk_categories)} risk categories")
    
    # Initialize Gemini
    print("\n🤖 Initializing Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite-preview-09-2025",
        temperature=0.4
    )
    print("  ✓ Gemini initialized (gemini-2.5-flash-lite-preview-09-2025)")
    
    # Connect to database
    conn = sqlite3.connect(str(DB_PATH))
    
    # Create schema
    print("\n🗄️  Setting up database schema...")
    create_requirements_schema(conn)
    
    # Get all documents
    cursor = conn.cursor()
    query = "SELECT id, filename FROM documents ORDER BY id"
    if limit_docs:
        query += f" LIMIT {limit_docs}"
    
    cursor.execute(query)
    documents = cursor.fetchall()
    
    print(f"\n📚 Processing {len(documents)} documents...")

    if force_reprocess:
        print("  🔄 Force reprocessing enabled - will process all documents")
        processed_docs = set()
    else:
        # Check which documents have already been processed
        cursor = conn.cursor()
        processed_docs = set()
        cursor.execute("SELECT DISTINCT document_id FROM requirements")
        processed_docs = {row[0] for row in cursor.fetchall()}
        print(f"  📋 {len(processed_docs)} documents already processed, skipping them")

    total_requirements = 0
    for doc_id, filename in tqdm(documents, desc="Documents"):
        if doc_id in processed_docs and not force_reprocess:
            print(f"  ⏭️  Skipping {filename} (already processed)")
            continue

        req_count = process_document(doc_id, filename, conn, risk_categories, llm)
        total_requirements += req_count
    
    conn.close()
    
    print("\n" + "=" * 70)
    print(f"✅ EXTRACTION COMPLETE")
    print(f"  Total requirements extracted: {total_requirements}")
    print("=" * 70)


if __name__ == "__main__":
    # For testing, process just a few documents
    import sys
    limit = None
    force_reprocess = False

    for arg in sys.argv[1:]:
        if arg == "--force" or arg == "--reprocess":
            force_reprocess = True
        elif arg.isdigit():
            limit = int(arg)

    extract_all_requirements(limit_docs=limit, force_reprocess=force_reprocess)

