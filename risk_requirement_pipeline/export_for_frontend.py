"""
Export Requirements Analysis to Frontend-Compatible JSON
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "legal_documents.db"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "requirements_analysis.json"


def get_all_risk_categories(conn: sqlite3.Connection) -> List[str]:
    """Get all risk categories with requirements"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT risk_category
        FROM requirements
        ORDER BY risk_category
    """)
    return [row[0] for row in cursor.fetchall()]


def get_requirements_for_category(conn: sqlite3.Connection, risk_category: str) -> List[Dict[str, Any]]:
    """Get all requirements for a risk category"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            r.id,
            r.requirement_text,
            r.start_page,
            r.end_page,
            r.extraction_window,
            d.filename
        FROM requirements r
        JOIN documents d ON r.document_id = d.id
        WHERE r.risk_category = ?
        ORDER BY d.filename, r.start_page
    """, (risk_category,))
    
    requirements = []
    for row in cursor.fetchall():
        req_id, text, start_page, end_page, extraction_window, filename = row

        # Use specific pages as the primary source, fallback to range
        if extraction_window:
            try:
                specific_pages = json.loads(extraction_window)
                requirement_data = {
                    'id': req_id,
                    'text': text,
                    'pages': specific_pages,  # ← Just the specific pages list
                    'filename': filename
                }
            except (json.JSONDecodeError, ValueError):
                # Fallback to range format if JSON parsing fails
                requirement_data = {
                    'id': req_id,
                    'text': text,
                    'start_page': start_page,
                    'end_page': end_page,
                    'filename': filename
                }
        else:
            # Fallback to range format for legacy data
            requirement_data = {
                'id': req_id,
                'text': text,
                'start_page': start_page,
                'end_page': end_page,
                'filename': filename
            }

        requirements.append(requirement_data)
    
    return requirements


def get_relationships_for_category(conn: sqlite3.Connection, risk_category: str) -> Dict[str, List[Dict[str, Any]]]:
    """Get all overlaps and contradictions for a risk category
    
    Each relationship is one edge in the graph connecting 2+ documents.
    Returns format matching analysis_results.json structure.
    """
    cursor = conn.cursor()
    
    # Get all relationships for this category
    cursor.execute("""
        SELECT id, relationship_type, description
        FROM requirement_relationships
        WHERE risk_category = ?
    """, (risk_category,))
    
    relationships = cursor.fetchall()
    
    overlaps = []
    contradictions = []
    
    for rel_id, rel_type, description in relationships:
        # Get all requirements in this relationship, grouped by document
        cursor.execute("""
            SELECT 
                rr.requirement_id,
                r.requirement_text,
                r.start_page,
                r.end_page,
                r.risk_category,
                d.id as doc_id,
                d.filename
            FROM relationship_requirements rr
            JOIN requirements r ON rr.requirement_id = r.id
            JOIN documents d ON r.document_id = d.id
            WHERE rr.relationship_id = ?
            ORDER BY d.id, r.start_page
        """, (rel_id,))
        
        req_rows = cursor.fetchall()
        
        # Group requirements by document
        doc_dict = {}
        for req_id, req_text, start_page, end_page, req_risk_category, doc_id, filename in req_rows:
            if filename not in doc_dict:
                doc_dict[filename] = {
                    'filename': filename,
                    'requirements': []
                }
            
            doc_dict[filename]['requirements'].append({
                'id': req_id,
                'text': req_text,
                'start_page': start_page,
                'end_page': end_page,
                'risk_category': req_risk_category
            })
        
        # Each relationship should connect 2+ documents
        if len(doc_dict) < 2:
            print(f"    ⚠️ Skipping relationship {rel_id} - has only {len(doc_dict)} document")
            continue
        
        # Format as single relationship (matches analysis_results.json)
        relationship_data = {
            'id': f"{risk_category.lower().replace('_', '-')}-{rel_type.lower()}-{rel_id}",
            'documents': list(doc_dict.values()),
            'reason': description
        }
        
        if rel_type == 'OVERLAP':
            overlaps.append(relationship_data)
        else:
            contradictions.append(relationship_data)
    
    return {
        'overlaps': overlaps,
        'contradictions': contradictions
    }


def export_analysis():
    """Export complete analysis to JSON"""
    
    print("=" * 70)
    print("EXPORTING REQUIREMENTS ANALYSIS FOR FRONTEND")
    print("=" * 70)
    
    conn = sqlite3.connect(str(DB_PATH))
    
    # Get all risk categories
    risk_categories = get_all_risk_categories(conn)
    print(f"\n📋 Found {len(risk_categories)} risk categories")
    
    # Build output structure
    output = {
        'risk_categories': []
    }
    
    for risk_category in risk_categories:
        print(f"\n📊 Processing: {risk_category}")
        
        # Get requirements
        requirements = get_requirements_for_category(conn, risk_category)
        print(f"  ✓ {len(requirements)} requirements")
        
        # Get relationships
        relationships = get_relationships_for_category(conn, risk_category)
        print(f"  ✓ {len(relationships['overlaps'])} overlaps")
        print(f"  ✓ {len(relationships['contradictions'])} contradictions")
        
        # Count unique documents involved in relationships
        all_docs = set()
        for overlap in relationships['overlaps']:
            for doc in overlap['documents']:
                all_docs.add(doc['filename'])
        for contradiction in relationships['contradictions']:
            for doc in contradiction['documents']:
                all_docs.add(doc['filename'])
        
        # Format for frontend (matches paragraph-based structure)
        category_data = {
            'risk_name': risk_category.replace('_', ' ').title(),
            'description': f"Requirements related to {risk_category.replace('_', ' ').lower()}",
            'overlaps': relationships['overlaps'],
            'contradictions': relationships['contradictions']
        }
        
        output['risk_categories'].append(category_data)
    
    conn.close()
    
    # Save to file
    print(f"\n💾 Writing to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    # Print statistics
    total_overlaps = sum(len(cat['overlaps']) for cat in output['risk_categories'])
    total_contradictions = sum(len(cat['contradictions']) for cat in output['risk_categories'])
    total_requirements = sum(cat['total_requirements'] for cat in output['risk_categories'])
    
    print("\n" + "=" * 70)
    print("✅ EXPORT COMPLETE")
    print(f"  Total risk categories: {len(risk_categories)}")
    print(f"  Total requirements: {total_requirements}")
    print(f"  Total overlaps: {total_overlaps}")
    print(f"  Total contradictions: {total_contradictions}")
    print(f"  Output file: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    export_analysis()

