#!/usr/bin/env python3
"""
Extract corpus from SQLite database with dimensional metadata for GraphRAG processing
"""

import sqlite3
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Database path
DB_PATH = "./legal_documents.db"
OUTPUT_DIR = "./input"

def get_jurisdiction_from_category(category: str, subcategory: str, filename: str) -> str:
    """Determine jurisdiction from document metadata"""
    filename_lower = filename.lower()
    
    if "fin" in filename_lower or "fi_" in filename_lower:
        return "FINNISH"
    elif subcategory in ["FIVA_MOK", "VYL", "LLL"]:
        return "FINNISH"
    elif category == "NATIONAL_LAWS":
        return "FINNISH"
    elif "celex" in filename_lower or subcategory in ["BRRD", "CRD", "CRR", "EBA", "MiFID", "MiFIR", "SFDR"]:
        return "EU"
    elif subcategory == "Basel":
        return "INTERNATIONAL"
    else:
        return "OTHER"

def extract_risk_indicators(text: str) -> List[str]:
    """Extract potential risk category indicators from text"""
    risk_patterns = {
        'LIQUIDITY_RISK': [r'\bliquidity\s+risk\b', r'\bliquid(ity)?\b.*\brisk\b', r'\bfunding\s+risk\b'],
        'CREDIT_RISK': [r'\bcredit\s+risk\b', r'\bdefault\s+risk\b', r'\bcounterparty\s+risk\b'],
        'OPERATIONAL_RISK': [r'\boperational\s+risk\b', r'\bprocess\s+risk\b', r'\bsystem\s+risk\b'],
        'MARKET_RISK': [r'\bmarket\s+risk\b', r'\bprice\s+risk\b', r'\binterest\s+rate\s+risk\b'],
        'LEGAL_RISK': [r'\blegal\s+risk\b', r'\bcompliance\s+risk\b', r'\bregulatory\s+risk\b'],
        'REPUTATIONAL_RISK': [r'\breputational\s+risk\b', r'\breputation\s+risk\b'],
        'CONCENTRATION_RISK': [r'\bconcentration\s+risk\b', r'\bsingle\s+name\s+exposure\b'],
        'SYSTEMIC_RISK': [r'\bsystemic\s+risk\b', r'\bsystem(ic)?\s+wide\s+risk\b'],
        'MODEL_RISK': [r'\bmodel\s+risk\b', r'\bmodelling\s+risk\b'],
        'CYBER_RISK': [r'\bcyber\s+risk\b', r'\bcybersecurity\s+risk\b', r'\bIT\s+risk\b'],
        'CLIMATE_RISK': [r'\bclimate\s+risk\b', r'\benvironmental\s+risk\b', r'\bESG\s+risk\b'],
        'SOVEREIGN_RISK': [r'\bsovereign\s+risk\b', r'\bcountry\s+risk\b'],
        'SETTLEMENT_RISK': [r'\bsettlement\s+risk\b', r'\bdelivery\s+risk\b'],
        'PENSION_RISK': [r'\bpension\s+risk\b', r'\bretirement\s+risk\b'],
        'INSURANCE_RISK': [r'\binsurance\s+risk\b', r'\bunderwriting\s+risk\b'],
        'BUSINESS_RISK': [r'\bbusiness\s+risk\b', r'\bstrategic\s+risk\b'],
        'FOREIGN_EXCHANGE_RISK': [r'\bforeign\s+exchange\s+risk\b', r'\bFX\s+risk\b', r'\bcurrency\s+risk\b'],
        'INFLATION_RISK': [r'\binflation\s+risk\b', r'\bpurchasing\s+power\s+risk\b'],
        'LEVERAGE_RISK': [r'\bleverage\s+risk\b', r'\bgearing\s+risk\b'],
        'VOLATILITY_RISK': [r'\bvolatility\s+risk\b', r'\bprice\s+volatility\b']
    }
    
    found_risks = []
    text_lower = text.lower()
    
    for risk_type, patterns in risk_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                found_risks.append(risk_type)
                break  # Found one pattern for this risk type
    
    return found_risks

def extract_requirement_indicators(text: str) -> List[str]:
    """Extract requirement/remedy indicators from text"""
    requirement_patterns = [
        r'\bshall\b', r'\bmust\b', r'\brequired\s+to\b', r'\bobligated\s+to\b',
        r'\bensure\s+that\b', r'\bcomply\s+with\b', r'\badhere\s+to\b',
        r'\bmaintain\b', r'\bestablish\b', r'\bimplement\b', r'\breport\b',
        r'\bdisclose\b', r'\bmonitor\b', r'\bassess\b', r'\bmeasure\b'
    ]
    
    requirements = []
    for pattern in requirement_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            # Extract sentence containing the requirement
            sentences = text.split('.')
            for sentence in sentences:
                if re.search(pattern, sentence, re.IGNORECASE):
                    requirements.append(sentence.strip())
                    break
    
    return requirements

def create_document_metadata(doc_row: Tuple, paragraph_count: int) -> str:
    """Create metadata header for document"""
    doc_id, filename, filepath, category, subcategory, file_size, page_count = doc_row
    jurisdiction = get_jurisdiction_from_category(category, subcategory, filename)
    
    metadata = {
        'document_id': doc_id,
        'filename': filename,
        'category': category,
        'subcategory': subcategory,
        'jurisdiction': jurisdiction,
        'page_count': page_count,
        'paragraph_count': paragraph_count
    }
    
    return f"METADATA: {json.dumps(metadata)}\n\n"

def extract_corpus():
    """Extract corpus from SQLite database with dimensional metadata"""
    
    if not os.path.exists(DB_PATH):
        print(f"❌ Database {DB_PATH} not found")
        return
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all documents
    cursor.execute("""
        SELECT id, filename, filepath, category, subcategory, file_size, page_count 
        FROM documents 
        ORDER BY category, subcategory, filename
    """)
    documents = cursor.fetchall()
    
    print(f"📊 Processing {len(documents)} documents...")
    
    total_paragraphs = 0
    risk_mentions = {}
    
    for doc_row in documents:
        doc_id, filename, filepath, category, subcategory, file_size, page_count = doc_row
        
        # Get all paragraphs for this document with page info
        cursor.execute("""
            SELECT p.id, p.paragraph_index, p.content, pg.page_number
            FROM paragraphs p
            JOIN pages pg ON p.page_id = pg.id
            WHERE pg.document_id = ?
            ORDER BY pg.page_number, p.paragraph_index
        """, (doc_id,))
        
        paragraphs = cursor.fetchall()
        
        if not paragraphs:
            continue
        
        # Create output file for this document
        safe_filename = re.sub(r'[^\w\-_.]', '_', filename)
        output_file = Path(OUTPUT_DIR) / f"{safe_filename}.txt"
        
        jurisdiction = get_jurisdiction_from_category(category, subcategory, filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write document metadata
            f.write(create_document_metadata(doc_row, len(paragraphs)))
            
            # Process paragraphs
            for para_id, para_index, content, page_number in paragraphs:
                if not content or len(content.strip()) < 10:
                    continue
                
                # Extract risk indicators
                risk_indicators = extract_risk_indicators(content)
                
                # Extract requirement indicators
                requirement_indicators = extract_requirement_indicators(content)
                
                # Create dimensional tags
                dimensional_tags = []
                dimensional_tags.append(f"[JURISDICTION:{jurisdiction}]")
                dimensional_tags.append(f"[SOURCE:{subcategory}]")
                dimensional_tags.append(f"[CATEGORY:{category}]")
                dimensional_tags.append(f"[PAGE:{page_number}]")
                
                if risk_indicators:
                    for risk in risk_indicators:
                        dimensional_tags.append(f"[RISK:{risk}]")
                        # Track risk mentions
                        if risk not in risk_mentions:
                            risk_mentions[risk] = 0
                        risk_mentions[risk] += 1
                
                if requirement_indicators:
                    dimensional_tags.append("[REQ_TYPE:MANDATORY]")
                
                # Write paragraph with dimensional metadata
                tags_str = " ".join(dimensional_tags)
                f.write(f"{tags_str}\n")
                f.write(f"PARAGRAPH_{para_id}: {content}\n\n")
                
                total_paragraphs += 1
    
    conn.close()
    
    # Generate summary report
    with open(Path(OUTPUT_DIR) / "_corpus_summary.json", 'w') as f:
        summary = {
            'total_documents': len(documents),
            'total_paragraphs': total_paragraphs,
            'risk_mentions': risk_mentions,
            'jurisdictions': list(set([get_jurisdiction_from_category(doc[3], doc[4], doc[1]) 
                                     for doc in documents])),
            'categories': list(set([doc[3] for doc in documents])),
            'subcategories': list(set([doc[4] for doc in documents if doc[4]]))
        }
        json.dump(summary, f, indent=2)
    
    print(f"✅ Corpus extraction completed!")
    print(f"📊 Total documents: {len(documents)}")
    print(f"📊 Total paragraphs: {total_paragraphs:,}")
    print(f"📊 Risk mentions found: {len(risk_mentions)}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Show top risk categories found
    if risk_mentions:
        print("\n🎯 Top risk categories found:")
        sorted_risks = sorted(risk_mentions.items(), key=lambda x: x[1], reverse=True)
        for risk, count in sorted_risks[:10]:
            print(f"   {risk}: {count} mentions")

if __name__ == "__main__":
    extract_corpus()