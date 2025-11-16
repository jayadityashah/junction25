"""
Simple Flask API to serve document content from SQLite database
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import sqlite3
from pathlib import Path
import subprocess
import os
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

SCRIPT_DIR = Path(__file__).parent
DB_PATH = SCRIPT_DIR / "legal_documents.db"
FRONTEND_DIR = SCRIPT_DIR / "frontend"
GRAPHRAG_DIR = SCRIPT_DIR / "risk-graphrag-project"

@app.route('/')
def index():
    """Serve the frontend"""
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/bronze_data/<path:path>')
def serve_pdf(path):
    """Serve PDF files from bronze_data"""
    return send_from_directory(SCRIPT_DIR / 'bronze_data', path)

@app.route('/<path:path>')
def serve_frontend(path):
    """Serve frontend static files"""
    return send_from_directory(FRONTEND_DIR, path)

@app.route('/api/documents', methods=['GET'])
def get_all_documents():
    """Get list of all documents"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, filename, filepath, category, subcategory, page_count
            FROM documents
            ORDER BY category, subcategory, filename
        """)
        
        documents = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify(documents)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/document/<int:doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get full document content with all paragraphs"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get document info
        cursor.execute("""
            SELECT id, filename, filepath, category, subcategory, page_count
            FROM documents
            WHERE id = ?
        """, (doc_id,))
        
        doc = cursor.fetchone()
        if not doc:
            return jsonify({"error": "Document not found"}), 404
        
        doc_dict = dict(doc)
        
        # Get all paragraphs
        cursor.execute("""
            SELECT p.content, p.paragraph_index, pg.page_number
            FROM paragraphs p
            JOIN pages pg ON p.page_id = pg.id
            WHERE pg.document_id = ?
            ORDER BY pg.page_number, p.paragraph_index
        """, (doc_id,))
        
        paragraphs = []
        for row in cursor.fetchall():
            paragraphs.append({
                "content": row["content"],
                "paragraph_index": row["paragraph_index"],
                "page_number": row["page_number"]
            })
        
        doc_dict["paragraphs"] = paragraphs
        conn.close()
        
        return jsonify(doc_dict)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/document/metadata/<path:filename>', methods=['GET'])
def get_document_metadata(filename):
    """Get document metadata (name, category, PDF path) by filename"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT filename, filepath, category, subcategory
            FROM documents
            WHERE filename = ? OR filepath = ?
        """, (filename, filename))
        
        doc = cursor.fetchone()
        conn.close()
        
        if not doc:
            return jsonify({"error": "Document not found"}), 404
        
        # Derive PDF path from filepath
        # DB:  FINANCIAL_REGULATION_EU_AND_LOCAL_IN_FORCE_GOLD/output_simple/Basel/file.di.json
        # PDF: bronze_data/FINANCIAL_REGULATION_EU_AND_LOCAL_IN_FORCE_BRONZE/Basel/file.pdf
        filepath = doc['filepath']
        # Replace GOLD with BRONZE
        filepath = filepath.replace('_GOLD/', '_BRONZE/')
        # Remove '/output_simple/' from path
        filepath = filepath.replace('/output_simple/', '/')
        # Remove '/EN_TRANSLATION/' for Finnish documents (LLL folder only)
        filepath = filepath.replace('/LLL/EN_TRANSLATION/', '/LLL/')
        # Replace extension and add bronze_data prefix
        pdf_path = f"bronze_data/{filepath.replace('.di.json', '.pdf')}"
        
        # Create display name from filename
        display_name = get_display_name(doc["filename"], doc["subcategory"])
        
        return jsonify({
            "filename": doc["filename"],
            "display_name": display_name,
            "short_name": get_short_name(doc["filename"], doc["subcategory"]),
            "category": doc["category"],
            "subcategory": doc["subcategory"],
            "pdf_path": pdf_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_display_name(filename, subcategory):
    """Generate human-readable display name"""
    name = filename.replace('.di.json', '')
    
    # Known patterns
    if 'CELEX_32013R0575' in name:
        return 'Capital Requirements Regulation (CRR)'
    elif 'CELEX_32013L0036' in name:
        return 'Capital Requirements Directive (CRD IV)'
    elif 'CELEX_32014L0059' in name:
        return 'Bank Recovery and Resolution Directive (BRRD)'
    elif 'CELEX_32014L0065' in name:
        return 'Markets in Financial Instruments Directive (MiFID II)'
    elif 'CELEX_32023R1803' in name:
        return 'IFRS 9 Financial Instruments'
    elif 'BaselFramework' in name:
        return 'Basel III Framework'
    elif name.isdigit() or '_' in name and name.split('_')[0].isdigit():
        # Finnish laws like "186_2023" or "699_2004"
        return f'Finnish Law {name.replace("_", "/")}'
    elif 'Guidelines' in name or 'GL' in name:
        # EBA Guidelines - clean up the name
        clean = name.replace('Final ', '').replace('Draft ', '').replace('Report on ', '')
        if len(clean) > 50:
            clean = 'EBA ' + clean[:47] + '...'
        return clean
    else:
        # Return cleaned filename
        return name.replace('_', ' ')

def get_short_name(filename, subcategory):
    """Generate short name for graph nodes"""
    name = filename.replace('.di.json', '')
    
    # Known patterns
    if 'CELEX_32013R0575' in name:
        return 'CRR'
    elif 'CELEX_32013L0036' in name:
        return 'CRD IV'
    elif 'CELEX_32014L0059' in name:
        return 'BRRD'
    elif 'CELEX_32014L0065' in name:
        return 'MiFID II'
    elif 'CELEX_32023R1803' in name:
        return 'IFRS 9'
    elif 'BaselFramework' in name:
        return 'Basel III'
    elif name.isdigit() or '_' in name and name.split('_')[0].isdigit():
        return f'FIN {name.replace("_", "/")}'
    elif 'PD' in name and 'LGD' in name:
        return 'EBA PD/LGD'
    elif 'Expected Credit Loss' in name or 'ECL' in name:
        return 'EBA ECL'
    elif 'CRM' in name:
        return 'EBA CRM'
    else:
        # Use subcategory
        return subcategory or name[:15]

@app.route('/api/document/by-filename/<path:filename>', methods=['GET'])
def get_document_by_filename(filename):
    """Get document content by filename"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get document info by filename
        cursor.execute("""
            SELECT id, filename, filepath, category, subcategory, page_count
            FROM documents
            WHERE filename = ? OR filepath = ?
        """, (filename, filename))
        
        doc = cursor.fetchone()
        if not doc:
            return jsonify({"error": "Document not found"}), 404
        
        doc_dict = dict(doc)
        doc_id = doc_dict["id"]
        
        # Get all paragraphs
        cursor.execute("""
            SELECT p.content, p.paragraph_index, pg.page_number
            FROM paragraphs p
            JOIN pages pg ON p.page_id = pg.id
            WHERE pg.document_id = ?
            ORDER BY pg.page_number, p.paragraph_index
        """, (doc_id,))
        
        paragraphs = []
        for row in cursor.fetchall():
            paragraphs.append({
                "content": row["content"],
                "paragraph_index": row["paragraph_index"],
                "page_number": row["page_number"]
            })
        
        doc_dict["paragraphs"] = paragraphs
        conn.close()
        
        return jsonify(doc_dict)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/document/<path:filename>/paragraphs', methods=['POST'])
def get_specific_paragraphs(filename):
    """Get specific paragraphs from a document by page and paragraph index"""
    try:
        import json
        paragraph_refs = json.loads(request.data)  # [{page: 45, paragraph_index: 3}, ...]
        
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get document ID
        cursor.execute("""
            SELECT id FROM documents
            WHERE filename = ? OR filepath = ?
        """, (filename, filename))
        
        doc = cursor.fetchone()
        if not doc:
            return jsonify({"error": "Document not found"}), 404
        
        doc_id = doc["id"]
        
        # Fetch requested paragraphs
        paragraphs = []
        for ref in paragraph_refs:
            cursor.execute("""
                SELECT p.content, p.paragraph_index, pg.page_number
                FROM paragraphs p
                JOIN pages pg ON p.page_id = pg.id
                WHERE pg.document_id = ? AND pg.page_number = ? AND p.paragraph_index = ?
            """, (doc_id, ref['page'], ref['paragraph_index']))
            
            row = cursor.fetchone()
            if row:
                paragraphs.append({
                    "content": row["content"],
                    "paragraph_index": row["paragraph_index"],
                    "page_number": row["page_number"]
                })
        
        conn.close()
        return jsonify(paragraphs)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Document counts by category
        cursor.execute("""
            SELECT category, subcategory, COUNT(*) as count
            FROM documents
            GROUP BY category, subcategory
            ORDER BY category, subcategory
        """)

        categories = [dict(row) for row in cursor.fetchall()]

        # Total counts
        cursor.execute("SELECT COUNT(*) as count FROM documents")
        total_docs = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM pages")
        total_pages = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM paragraphs")
        total_paragraphs = cursor.fetchone()["count"]

        conn.close()

        return jsonify({
            "total_documents": total_docs,
            "total_pages": total_pages,
            "total_paragraphs": total_paragraphs,
            "categories": categories
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/graphrag/chat', methods=['POST'])
def graphrag_chat():
    """Chat interface endpoint for GraphRAG queries"""
    try:
        data = request.get_json()
        message = data.get('message', '')
        method = data.get('method', 'local')

        if not message:
            return jsonify({"error": "Message is required"}), 400

        cmd = f'uv run graphrag query --root . --method {method} --query "{message}"'

        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(GRAPHRAG_DIR)
        )

        if result.returncode == 0:
            return jsonify({
                "success": True,
                "response": result.stdout
            })
        else:
            return jsonify({
                "success": False,
                "error": result.stderr or "Query failed"
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Query timeout"}), 504
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print(f"🚀 Starting API server...")
    print(f"📁 Database: {DB_PATH}")
    print(f"🌐 Frontend: http://localhost:5000")
    print(f"📡 API: http://localhost:5000/api/")
    app.run(debug=True, port=5000)

