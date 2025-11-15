"""
Load JSON legal documents into SQLite database

THE DATABASE IS IN THE TELGRAM GROUPCHAT
"""

import sqlite3
import json
import os
from pathlib import Path
from tqdm import tqdm

DB_PATH = "/home/quern/Documents/junction-2025/legal_documents.db"
DATA_PATH = "/home/quern/Documents/junction-2025/data"

def create_database():
    """Create the SQLite database schema"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        filepath TEXT NOT NULL UNIQUE,
        category TEXT,
        subcategory TEXT,
        file_size INTEGER,
        page_count INTEGER
    )
    """)
    
    # Pages table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER NOT NULL,
        page_number INTEGER NOT NULL,
        paragraph_count INTEGER,
        FOREIGN KEY (document_id) REFERENCES documents(id),
        UNIQUE(document_id, page_number)
    )
    """)
    
    # Paragraphs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS paragraphs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        page_id INTEGER NOT NULL,
        paragraph_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        char_length INTEGER,
        FOREIGN KEY (page_id) REFERENCES pages(id)
    )
    """)
    
    # Create indexes for better query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_pages_document ON pages(document_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_paragraphs_page ON paragraphs(page_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_paragraphs_content ON paragraphs(content)")
    
    conn.commit()
    return conn

def determine_category(filepath):
    """Determine document category from filepath"""
    parts = Path(filepath).parts
    if "EU_LEG_IN_FORCE_GOLD" in parts:
        return "EU_LEGISLATION", "EU"
    elif "FINANCIAL_REGULATION_EU_AND_LOCAL_IN_FORCE_GOLD" in parts:
        # Try to get subcategory (Basel, BRRD, CRD, etc.)
        for part in parts:
            if part in ["Basel", "BRRD", "CRD", "CRR", "EBA", "FIVA_MOK", "IFRS", "LLL", "MiFID", "MiFIR", "SFDR", "VYL"]:
                return "FINANCIAL_REGULATION", part
        return "FINANCIAL_REGULATION", "OTHER"
    elif "OTHER_RELATED_NATIONAL_LAWS" in parts:
        return "NATIONAL_LAWS", "FINLAND"
    return "UNKNOWN", "UNKNOWN"

def process_json_file(filepath, conn):
    """Process a single JSON file and insert into database"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, dict) or 'pages' not in data:
            return False
        
        cursor = conn.cursor()
        
        # Get category
        category, subcategory = determine_category(filepath)
        
        # Insert document
        file_size = os.path.getsize(filepath)
        page_count = len(data.get('pages', []))
        
        cursor.execute("""
        INSERT OR IGNORE INTO documents (filename, filepath, category, subcategory, file_size, page_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (os.path.basename(filepath), str(filepath), category, subcategory, file_size, page_count))
        
        if cursor.lastrowid == 0:
            # Document already exists
            cursor.execute("SELECT id FROM documents WHERE filepath = ?", (str(filepath),))
            doc_id = cursor.fetchone()[0]
        else:
            doc_id = cursor.lastrowid
        
        # Insert pages and paragraphs
        for page_data in data.get('pages', []):
            page_num = page_data.get('page', 0)
            paragraphs = page_data.get('paragraphs', [])
            
            cursor.execute("""
            INSERT OR IGNORE INTO pages (document_id, page_number, paragraph_count)
            VALUES (?, ?, ?)
            """, (doc_id, page_num, len(paragraphs)))
            
            if cursor.lastrowid == 0:
                cursor.execute("SELECT id FROM pages WHERE document_id = ? AND page_number = ?", (doc_id, page_num))
                page_id = cursor.fetchone()[0]
            else:
                page_id = cursor.lastrowid
            
            # Insert paragraphs
            for idx, paragraph in enumerate(paragraphs):
                if isinstance(paragraph, str):
                    cursor.execute("""
                    INSERT INTO paragraphs (page_id, paragraph_index, content, char_length)
                    VALUES (?, ?, ?, ?)
                    """, (page_id, idx, paragraph, len(paragraph)))
        
        conn.commit()
        return True
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def find_json_files(data_path, limit=None):
    """Find all JSON files in the data directory"""
    json_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
                if limit and len(json_files) >= limit:
                    return json_files
    return json_files

def main():
    print("Creating database schema...")
    conn = create_database()
    
    print(f"Scanning for JSON files in {DATA_PATH}...")
    json_files = find_json_files(DATA_PATH)
    total_files = len(json_files)
    
    print(f"Found {total_files} JSON files")
    print(f"Database will be created at: {DB_PATH}")
    
    # Process files with progress bar
    processed = 0
    failed = 0
    
    print("Loading documents into SQLite...")
    for filepath in tqdm(json_files, desc="Processing", unit="file"):
        if process_json_file(filepath, conn):
            processed += 1
        else:
            failed += 1
    
    conn.close()
    
    print(f"\n✅ Database created successfully!")
    print(f"📊 Statistics:")
    print(f"  - Total files found: {total_files}")
    print(f"  - Successfully processed: {processed}")
    print(f"  - Failed: {failed}")
    print(f"  - Database location: {DB_PATH}")
    
    # Print some stats from the database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM pages")
    page_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM paragraphs")
    para_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT category, COUNT(*) FROM documents GROUP BY category")
    categories = cursor.fetchall()
    
    print(f"\n📈 Database contents:")
    print(f"  - Documents: {doc_count:,}")
    print(f"  - Pages: {page_count:,}")
    print(f"  - Paragraphs: {para_count:,}")
    print(f"\n📁 Documents by category:")
    for cat, count in categories:
        print(f"  - {cat}: {count:,}")
    
    conn.close()

if __name__ == "__main__":
    main()

