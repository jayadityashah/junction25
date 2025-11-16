"""
Requirement Relationship Pipeline
Finds overlaps and contradictions between requirements from different documents
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from tqdm import tqdm

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "legal_documents.db"


def get_requirements_by_risk_category(conn: sqlite3.Connection, risk_category: str) -> List[Dict[str, Any]]:
    """Get all requirements for a specific risk category grouped by document"""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            r.id,
            r.requirement_text,
            r.risk_category,
            r.start_page,
            r.end_page,
            r.document_id,
            d.filename,
            d.subcategory
        FROM requirements r
        JOIN documents d ON r.document_id = d.id
        WHERE r.risk_category = ?
        ORDER BY d.id, r.id
    """, (risk_category,))
    
    rows = cursor.fetchall()
    
    requirements = []
    for row in rows:
        requirements.append({
            'id': row[0],
            'text': row[1],
            'risk_category': row[2],
            'start_page': row[3],
            'end_page': row[4],
            'document_id': row[5],
            'filename': row[6],
            'subcategory': row[7]
        })
    
    return requirements


def find_relationships_in_risk_group(
    requirements: List[Dict[str, Any]],
    risk_category: str,
    llm: ChatGoogleGenerativeAI
) -> List[Dict[str, Any]]:
    """Find groups of overlaps and contradictions within all requirements for a risk category"""

    if len(requirements) < 2:
        return []

    # Group requirements by document for the prompt
    by_document = {}
    for req in requirements:
        doc_name = req['filename']
        if doc_name not in by_document:
            by_document[doc_name] = []
        by_document[doc_name].append(req)

    # Format requirements by document
    doc_requirements_text = ""
    req_counter = 1
    req_id_map = {}

    for doc_name, doc_reqs in by_document.items():
        doc_requirements_text += f"\n\n--- DOCUMENT: {doc_name} ---\n"
        for req in doc_reqs:
            doc_requirements_text += f"{req_counter}. {req['text']}\n"
            req_id_map[req_counter] = req['id']
            req_counter += 1

    prompt = f"""You are a regulatory compliance expert analyzing requirements for: {risk_category}

ALL REQUIREMENTS FOR THIS RISK CATEGORY (grouped by document):
{doc_requirements_text}

TASK: Find GROUPS of requirements that have OVERLAPS or CONTRADICTIONS with each other.

IMPORTANT RULES:
- Each group must contain requirements from DIFFERENT DOCUMENTS (never the same document)
- OVERLAP: Requirements that address the same regulatory obligation, even if worded differently
- CONTRADICTION: Requirements that give conflicting or incompatible obligations
- Group together ALL requirements that relate to the same underlying topic/issue

OUTPUT FORMAT - Return ONLY valid JSON array of groups:
[
  {{
    "type": "OVERLAP" | "CONTRADICTION",
    "description": "2-3 sentence explanation of the relationship",
    "requirement_ids": [2, 5, 8]  // Array of requirement numbers from the list above
  }}
]

Each group should represent one coherent relationship topic. Return empty array [] if no relationships found.
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        # Clean markdown
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

        groups = json.loads(content)

        # Convert to the expected format
        relationships = []
        for group in groups:
            if 'type' in group and 'description' in group and 'requirement_ids' in group:
                # Convert requirement numbers to actual requirement IDs
                actual_req_ids = []
                for req_num in group['requirement_ids']:
                    if req_num in req_id_map:
                        actual_req_ids.append(req_id_map[req_num])

                if len(actual_req_ids) >= 2:  # Must have at least 2 requirements
                    relationships.append({
                        'type': group['type'],
                        'description': group['description'],
                        'requirement_ids': actual_req_ids
                    })

        return relationships

    except Exception as e:
        print(f"    ⚠️  Error finding relationships: {e}")
        return []


def find_grouped_relationships(
    requirements: List[Dict[str, Any]],
    risk_category: str,
    llm: ChatGoogleGenerativeAI
) -> List[Dict[str, Any]]:
    """Find groups of overlaps and contradictions by analyzing all requirements together"""

    print(f"  📊 {len(requirements)} requirements from {len(set(req['document_id'] for req in requirements))} documents")

    # Find relationships using grouped analysis
    relationships = find_relationships_in_risk_group(requirements, risk_category, llm)

    print(f"  ✓ Found {len(relationships)} relationship groups")

    return relationships


def convert_to_document_relationships(
    relationships: List[Dict[str, Any]],
    requirements: List[Dict[str, Any]],
    risk_category: str
) -> List[Dict[str, Any]]:
    """Convert grouped requirement relationships into document-level relationships"""

    print("  🔄 Converting to document-level relationships...")

    # Create a lookup map for requirements by ID
    req_lookup = {req['id']: req for req in requirements}

    document_relationships = []

    for rel in relationships:
        # Get all requirements involved in this relationship
        involved_reqs = []
        involved_docs = {}

        for req_id in rel['requirement_ids']:
            if req_id in req_lookup:
                req = req_lookup[req_id]
                involved_reqs.append(req)

                doc_name = req['filename']
                if doc_name not in involved_docs:
                    involved_docs[doc_name] = []
                involved_docs[doc_name].append({
                    'id': req['id'],
                    'text': req['text'],
                    'start_page': req['start_page'],
                    'end_page': req['end_page'],
                    'risk_category': req['risk_category']
                })

        # Only create relationship if we have requirements from 2+ documents
        if len(involved_docs) >= 2:
            documents = []
            for doc_name, doc_reqs in involved_docs.items():
                documents.append({
                    'filename': doc_name,
                    'requirements': doc_reqs
                })

            document_relationships.append({
                'type': rel['type'],
                'description': rel['description'],
                'documents': documents
            })

    print(f"  ✓ Created {len(document_relationships)} document-level relationships")

    # Show distribution of documents per relationship
    doc_counts = {}
    for rel in document_relationships:
        count = len(rel['documents'])
        doc_counts[count] = doc_counts.get(count, 0) + 1

    for count, num in sorted(doc_counts.items()):
        print(f"    - {num} relationships with {count} documents")

    return document_relationships


def save_relationships_to_db(
    conn: sqlite3.Connection,
    risk_category: str,
    relationships: List[Dict[str, Any]]
):
    """Save document-level relationships to database
    
    Clears existing relationships for this risk category before saving new ones.
    This makes the pipeline idempotent - safe to re-run.
    """
    
    cursor = conn.cursor()
    
    # Clear existing relationships for this risk category
    cursor.execute("""
        DELETE FROM relationship_requirements 
        WHERE relationship_id IN (
            SELECT id FROM requirement_relationships WHERE risk_category = ?
        )
    """, (risk_category,))
    
    cursor.execute("""
        DELETE FROM requirement_relationships WHERE risk_category = ?
    """, (risk_category,))
    
    print(f"    Cleared existing relationships for {risk_category}")
    
    # Insert new relationships
    for rel in relationships:
        # Insert relationship
        cursor.execute("""
            INSERT INTO requirement_relationships (relationship_type, risk_category, description)
            VALUES (?, ?, ?)
        """, (rel['type'], risk_category, rel['description']))
        
        relationship_id = cursor.lastrowid
        
        # Insert junction records for the requirements involved
        for doc in rel['documents']:
            for req in doc['requirements']:
                cursor.execute("""
                    INSERT INTO relationship_requirements (relationship_id, requirement_id)
                    VALUES (?, ?)
                """, (relationship_id, req['id']))
    
    conn.commit()


def process_risk_category(
    risk_category: str,
    conn: sqlite3.Connection,
    llm: ChatGoogleGenerativeAI
):
    """Process all requirements for a single risk category"""

    print(f"\n{'='*70}")
    print(f"Processing: {risk_category}")
    print('='*70)

    # Get requirements
    requirements = get_requirements_by_risk_category(conn, risk_category)

    if len(requirements) < 2:
        print(f"  ⚠️  Only {len(requirements)} requirement(s) found - skipping")
        return

    # Find grouped relationships (analyzing all requirements together)
    relationships = find_grouped_relationships(requirements, risk_category, llm)

    if not relationships:
        print("  ⚠️  No relationships found")
        return

    # Convert to document-level relationships
    document_relationships = convert_to_document_relationships(relationships, requirements, risk_category)

    # Save to database
    print("  💾 Saving to database...")
    save_relationships_to_db(conn, risk_category, document_relationships)
    print("  ✅ Saved")


def find_all_relationships():
    """Main pipeline for finding relationships"""
    
    print("=" * 70)
    print("REQUIREMENT RELATIONSHIP PIPELINE")
    print("=" * 70)
    
    # Initialize Gemini
    print("\n🤖 Initializing Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite-preview-09-2025",
        temperature=0.1
    )
    print("  ✓ Gemini initialized (gemini-2.5-flash-lite-preview-09-2025)")
    
    # Connect to database
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Get all risk categories with requirements
    cursor.execute("""
        SELECT DISTINCT risk_category, COUNT(*) as count
        FROM requirements
        GROUP BY risk_category
        ORDER BY count DESC
    """)
    
    risk_categories = cursor.fetchall()
    
    print(f"\n📋 Found {len(risk_categories)} risk categories with requirements")
    for cat, count in risk_categories:
        print(f"  - {cat}: {count} requirements")
    
    # Process each category
    for risk_category, count in risk_categories:
        if count >= 2:  # Need at least 2 requirements to compare
            try:
                process_risk_category(risk_category, conn, llm)
            except Exception as e:
                print(f"  ❌ Error processing {risk_category}: {e}")
                continue
    
    conn.close()
    
    print("\n" + "=" * 70)
    print("✅ RELATIONSHIP FINDING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    find_all_relationships()

