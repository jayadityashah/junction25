"""
Test script to find relationships for the largest risk category
Outputs results to JSON without touching the database
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "legal_documents.db"
OUTPUT_PATH = SCRIPT_DIR / "test_relationships_output.json"


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


def find_relationships_all_at_once(
    requirements: List[Dict[str, Any]],
    risk_category: str,
    llm: ChatGoogleGenerativeAI
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Find groups of overlaps and contradictions by analyzing all requirements together"""

    if len(requirements) < 2:
        return [], {}

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

CRITICAL: You MUST actively search for CONTRADICTIONS. Do not be conservative - if requirements conflict in any way, mark them as CONTRADICTION.

DEFINITIONS:
- OVERLAP: Requirements that address the same regulatory obligation in a compatible/complementary way
- CONTRADICTION: Requirements that give conflicting, incompatible, or mutually exclusive obligations

EXAMPLES OF CONTRADICTIONS TO LOOK FOR:
1. Different numeric thresholds for the same metric (e.g., "LCR must be at least 100%" vs "LCR must be at least 110%")
2. Different timeframes for the same action (e.g., "report within 2 weeks" vs "report within 1 month")
3. Conflicting permissions/prohibitions (e.g., "institutions may use liquid assets during stress" vs "institutions must not deplete liquid assets during stress")
4. Different calculation methods for the same measure (e.g., "include off-balance sheet items" vs "exclude off-balance sheet items")
5. Incompatible categorizations (e.g., "deposits with time restrictions go in 'Deposits'" vs "deposits with time restrictions go in 'Other Investments'")

EXAMPLE OF A REAL CONTRADICTION:
- Document A: "Deposits with time restrictions are recorded in 'Deposits' line item"
- Document B: "Deposits with time restrictions are recorded in 'Other Investments' line item"
→ This is a CONTRADICTION because the same asset type must be classified in different balance sheet categories

IMPORTANT RULES:
- Each group must contain requirements from DIFFERENT DOCUMENTS (never the same document)
- Be AGGRESSIVE in identifying contradictions - if there's any conflict, incompatibility, or mutual exclusivity, mark it as CONTRADICTION
- Do NOT mark things as OVERLAP just because they're related - check if they actually conflict
- Group together ALL requirements that relate to the same underlying topic/issue
- Be thorough - identify all meaningful relationships

OUTPUT FORMAT - Return ONLY valid JSON array of groups:
[
  {{
    "type": "OVERLAP" | "CONTRADICTION",
    "description": "2-3 sentence explanation of the relationship and why it's an overlap or contradiction",
    "requirement_ids": [2, 5, 8]  // Array of requirement numbers from the list above
  }}
]

Each group should represent one coherent relationship topic. Return empty array [] if no relationships found.
"""

    try:
        print("  🤖 Calling Gemini API...")
        # Use generate with proper message format
        from langchain_core.messages import HumanMessage
        response = llm.generate([[HumanMessage(content=prompt)]])
        
        # Extract token usage - try different locations
        token_stats = {}
        
        # Check generations[0] for usage info
        if response.generations and len(response.generations) > 0:
            gen = response.generations[0][0]
            if hasattr(gen, 'generation_info') and gen.generation_info:
                gen_info = gen.generation_info
                if 'usage_metadata' in gen_info:
                    usage = gen_info['usage_metadata']
                    token_stats = {
                        'prompt_tokens': usage.get('prompt_token_count', 0),
                        'completion_tokens': usage.get('candidates_token_count', 0),
                        'total_tokens': usage.get('total_token_count', 0)
                    }
                    print(f"  📊 Token usage:")
                    print(f"     - Input tokens: {token_stats['prompt_tokens']:,}")
                    print(f"     - Output tokens: {token_stats['completion_tokens']:,}")
                    print(f"     - Total tokens: {token_stats['total_tokens']:,}")
        
        if not token_stats:
            print(f"  ⚠️  Token usage not found in response")
        
        content = response.generations[0][0].text.strip()

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
                    # Verify they're from different documents
                    req_lookup = {req['id']: req for req in requirements}
                    docs = set(req_lookup[req_id]['document_id'] for req_id in actual_req_ids if req_id in req_lookup)
                    
                    if len(docs) >= 2:  # Only keep if from different documents
                        relationships.append({
                            'type': group['type'],
                            'description': group['description'],
                            'requirement_ids': actual_req_ids
                        })

        return relationships, token_stats

    except Exception as e:
        print(f"    ⚠️  Error finding relationships: {e}")
        import traceback
        traceback.print_exc()
        return [], {}


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


def main():
    """Test relationship finding for the largest risk category"""
    
    print("=" * 70)
    print("TEST RELATIONSHIP FINDING - LARGEST RISK CATEGORY")
    print("=" * 70)
    
    # Connect to database (read-only)
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    # Use LEGAL_RISK for testing
    print("\n📊 Using LEGAL_RISK for testing...")
    risk_category = "LEGAL_RISK"
    cursor.execute("""
        SELECT COUNT(*) as count
        FROM requirements
        WHERE risk_category = ?
    """, (risk_category,))
    
    count = cursor.fetchone()[0]
    print(f"  ✓ Risk category: {risk_category} ({count} requirements)")
    
    # Get number of documents for this risk
    cursor.execute("""
        SELECT COUNT(DISTINCT document_id)
        FROM requirements
        WHERE risk_category = ?
    """, (risk_category,))
    
    num_docs = cursor.fetchone()[0]
    print(f"  ✓ Across {num_docs} documents")
    
    # Get requirements
    print(f"\n📋 Loading requirements for {risk_category}...")
    requirements = get_requirements_by_risk_category(conn, risk_category)
    print(f"  ✓ Loaded {len(requirements)} requirements")
    
    # Initialize Gemini
    print("\n🤖 Initializing Gemini...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.1
    )
    print("  ✓ Gemini initialized (gemini-2.5-pro)")
    
    # Find relationships
    print(f"\n🔍 Finding relationships for {risk_category}...")
    relationships, token_stats = find_relationships_all_at_once(requirements, risk_category, llm)
    print(f"  ✓ Found {len(relationships)} relationship groups")
    
    # Convert to document-level relationships
    document_relationships = convert_to_document_relationships(relationships, requirements, risk_category)
    
    # Prepare output
    output = {
        'risk_category': risk_category,
        'total_requirements': len(requirements),
        'total_documents': num_docs,
        'total_relationships': len(document_relationships),
        'token_usage': token_stats,
        'overlaps': [rel for rel in document_relationships if rel['type'] == 'OVERLAP'],
        'contradictions': [rel for rel in document_relationships if rel['type'] == 'CONTRADICTION']
    }
    
    # Save to JSON
    print(f"\n💾 Saving results to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ Saved to {OUTPUT_PATH}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Risk Category: {risk_category}")
    print(f"Requirements: {len(requirements)}")
    print(f"Documents: {num_docs}")
    print(f"Total Relationships: {len(document_relationships)}")
    print(f"  - Overlaps: {len(output['overlaps'])}")
    print(f"  - Contradictions: {len(output['contradictions'])}")
    if token_stats:
        print(f"\nToken Usage:")
        print(f"  - Input tokens: {token_stats.get('prompt_tokens', 0):,}")
        print(f"  - Output tokens: {token_stats.get('completion_tokens', 0):,}")
        print(f"  - Total tokens: {token_stats.get('total_tokens', 0):,}")
    print("=" * 70)
    
    conn.close()


if __name__ == "__main__":
    main()

