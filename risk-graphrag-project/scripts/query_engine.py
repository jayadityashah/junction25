#!/usr/bin/env python3
"""
Multi-dimensional Query Engine for GraphRAG Risk Analysis
Provides advanced querying capabilities for risk-document-requirement mappings
"""

import json
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import yaml
import re
from collections import defaultdict, Counter
import subprocess

# Paths
CONFIG_PATH = "config/risk_categories.yaml"
DB_PATH = "legal_documents.db"
OUTPUT_PATH = "output"
CORPUS_SUMMARY_PATH = "input/_corpus_summary.json"

class MultiDimensionalQueryEngine:
    """Advanced query engine for multi-dimensional risk analysis"""
    
    def __init__(self):
        self.risk_categories = self.load_risk_categories()
        self.corpus_summary = self.load_corpus_summary()
        self.db_connection = None
        self.graphrag_available = self.check_graphrag_status()
        
    def load_risk_categories(self) -> Dict:
        """Load risk categories configuration"""
        try:
            with open(CONFIG_PATH, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"⚠️  Risk categories file not found: {CONFIG_PATH}")
            return {}
    
    def load_corpus_summary(self) -> Dict:
        """Load corpus summary data"""
        try:
            with open(CORPUS_SUMMARY_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Corpus summary not found: {CORPUS_SUMMARY_PATH}")
            return {}
    
    def check_graphrag_status(self) -> bool:
        """Check if GraphRAG indexing is complete"""
        output_dir = Path(OUTPUT_PATH)
        
        # Check for key GraphRAG output files
        required_files = [
            "text_units.parquet",
            "documents.parquet"
        ]
        
        optional_files = [
            "create_final_entities.parquet",
            "create_final_relationships.parquet",
            "create_final_communities.parquet"
        ]
        
        # Check required files
        for file_pattern in required_files:
            if not list(output_dir.glob(f"**/{file_pattern}")):
                if not list(output_dir.glob(file_pattern)):
                    print(f"⚠️  Required GraphRAG file not found: {file_pattern}")
                    return False
        
        # Check optional files (warn if missing)
        missing_optional = []
        for file_pattern in optional_files:
            if not list(output_dir.glob(f"**/{file_pattern}")) and not list(output_dir.glob(file_pattern)):
                missing_optional.append(file_pattern)
        
        if missing_optional:
            print(f"📝 GraphRAG indexing may be in progress. Missing optional files: {missing_optional}")
        
        print("✅ GraphRAG basic files available for querying")
        return True
    
    def connect_to_db(self):
        """Connect to SQLite database"""
        if not Path(DB_PATH).exists():
            raise FileNotFoundError(f"Database not found: {DB_PATH}")
        
        self.db_connection = sqlite3.connect(DB_PATH)
        return self.db_connection
    
    def query_risk_category(self, risk_type: str, jurisdiction: Optional[str] = None, 
                           severity: Optional[str] = None, limit: int = 100) -> Dict:
        """Query specific risk category with dimensional filters"""
        
        if not self.db_connection:
            self.connect_to_db()
        
        # Build base query
        base_query = """
            SELECT 
                p.id,
                p.content,
                pg.page_number,
                d.filename,
                d.category,
                d.subcategory
            FROM paragraphs p
            JOIN pages pg ON p.page_id = pg.id
            JOIN documents d ON pg.document_id = d.id
            WHERE LENGTH(p.content) > 50
        """
        
        # Add jurisdiction filter
        jurisdiction_filter = ""
        if jurisdiction:
            if jurisdiction.upper() == "EU":
                jurisdiction_filter = " AND d.subcategory IN ('BRRD', 'CRD', 'CRR', 'EBA', 'MiFID', 'MiFIR', 'SFDR', 'IFRS')"
            elif jurisdiction.upper() == "FINNISH":
                jurisdiction_filter = " AND d.subcategory IN ('FIVA_MOK', 'VYL', 'LLL', 'FINLAND')"
            elif jurisdiction.upper() == "INTERNATIONAL":
                jurisdiction_filter = " AND d.subcategory = 'Basel'"
        
        query = base_query + jurisdiction_filter + f" LIMIT {limit * 10}"  # Get more to filter by content
        
        cursor = self.db_connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Filter by risk content
        risk_config = self.risk_categories.get('risk_categories', {}).get(risk_type.upper(), {})
        primary_terms = risk_config.get('primary_terms', [])
        synonyms = risk_config.get('synonyms', [])
        all_terms = primary_terms + synonyms
        
        matching_paragraphs = []
        for para_id, content, page_num, filename, category, subcategory in results:
            content_lower = content.lower()
            
            # Check if any risk terms are mentioned
            for term in all_terms:
                if re.search(r'\b' + re.escape(term.lower()) + r'\b', content_lower):
                    
                    # Extract requirements from this paragraph
                    requirements = self.extract_requirements_from_text(content)
                    
                    matching_paragraphs.append({
                        'paragraph_id': para_id,
                        'content': content,
                        'page': page_num,
                        'document': filename,
                        'category': category,
                        'source': subcategory,
                        'jurisdiction': self.determine_jurisdiction(subcategory, filename),
                        'risk_term_matched': term,
                        'requirements': requirements,
                        'requirement_count': len(requirements)
                    })
                    
                    if len(matching_paragraphs) >= limit:
                        break
            
            if len(matching_paragraphs) >= limit:
                break
        
        return {
            'risk_type': risk_type,
            'query_filters': {
                'jurisdiction': jurisdiction,
                'severity': severity,
                'limit': limit
            },
            'total_matches': len(matching_paragraphs),
            'matches': matching_paragraphs[:limit]
        }
    
    def determine_jurisdiction(self, subcategory: str, filename: str) -> str:
        """Determine jurisdiction from document metadata"""
        filename_lower = filename.lower()
        
        if "fin" in filename_lower or "fi_" in filename_lower:
            return "FINNISH"
        elif subcategory in ["FIVA_MOK", "VYL", "LLL"]:
            return "FINNISH"
        elif "celex" in filename_lower or subcategory in ["BRRD", "CRD", "CRR", "EBA", "MiFID", "MiFIR", "SFDR"]:
            return "EU"
        elif subcategory == "Basel":
            return "INTERNATIONAL"
        else:
            return "OTHER"
    
    def extract_requirements_from_text(self, text: str) -> List[Dict]:
        """Extract regulatory requirements from text"""
        requirements = []
        
        requirement_indicators = [
            r'\bshall\b', r'\bmust\b', r'\brequired\s+to\b', r'\bobligated\s+to\b',
            r'\bensure\s+that\b', r'\bcomply\s+with\b', r'\badhere\s+to\b',
            r'\bmaintain\b', r'\bestablish\b', r'\bimplement\b', r'\breport\b',
            r'\bdisclose\b', r'\bmonitor\b', r'\bassess\b', r'\bmeasure\b'
        ]
        
        sentences = re.split(r'[.!?]', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            for pattern in requirement_indicators:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Determine requirement type
                    req_type = "MANDATORY"
                    if re.search(r'\bshould\b|\brecommended\b', sentence, re.IGNORECASE):
                        req_type = "RECOMMENDED"
                    elif re.search(r'\bmay\b|\bcould\b|\bconsider\b', sentence, re.IGNORECASE):
                        req_type = "GUIDANCE"
                    
                    requirements.append({
                        'text': sentence,
                        'type': req_type,
                        'indicator': pattern.replace(r'\b', '').replace(r'\s+', ' ')
                    })
                    break
        
        return requirements
    
    def find_conflicts(self, risk_type: str, jurisdictions: List[str] = None) -> Dict:
        """Find conflicts and overlaps between jurisdictions for a specific risk"""
        
        if jurisdictions is None:
            jurisdictions = ['EU', 'FINNISH']
        
        conflict_analysis = {
            'risk_type': risk_type,
            'jurisdictions_analyzed': jurisdictions,
            'conflicts': [],
            'overlaps': [],
            'gaps': []
        }
        
        # Get requirements for each jurisdiction
        jurisdiction_requirements = {}
        
        for jurisdiction in jurisdictions:
            result = self.query_risk_category(risk_type, jurisdiction=jurisdiction, limit=50)
            
            # Extract all requirements
            all_requirements = []
            for match in result['matches']:
                for req in match['requirements']:
                    all_requirements.append({
                        'text': req['text'],
                        'type': req['type'],
                        'source': match['source'],
                        'document': match['document']
                    })
            
            jurisdiction_requirements[jurisdiction] = all_requirements
        
        # Analyze for conflicts and overlaps
        if len(jurisdictions) >= 2:
            jur1, jur2 = jurisdictions[0], jurisdictions[1]
            reqs1 = jurisdiction_requirements.get(jur1, [])
            reqs2 = jurisdiction_requirements.get(jur2, [])
            
            # Simple conflict detection (similar topics with different requirements)
            for req1 in reqs1:
                for req2 in reqs2:
                    # Check for similar context but different requirements
                    similarity_score = self.calculate_requirement_similarity(req1['text'], req2['text'])
                    
                    if similarity_score > 0.3 and similarity_score < 0.8:  # Similar context, different requirement
                        if req1['type'] != req2['type'] or self.requirements_conflict(req1['text'], req2['text']):
                            conflict_analysis['conflicts'].append({
                                'jurisdiction_1': jur1,
                                'requirement_1': req1['text'][:100] + "...",
                                'source_1': req1['source'],
                                'jurisdiction_2': jur2,
                                'requirement_2': req2['text'][:100] + "...",
                                'source_2': req2['source'],
                                'similarity_score': similarity_score
                            })
                    elif similarity_score > 0.8:  # Very similar - likely overlap
                        conflict_analysis['overlaps'].append({
                            'jurisdictions': [jur1, jur2],
                            'requirement': req1['text'][:100] + "...",
                            'sources': [req1['source'], req2['source']],
                            'similarity_score': similarity_score
                        })
        
        return conflict_analysis
    
    def calculate_requirement_similarity(self, req1: str, req2: str) -> float:
        """Calculate similarity between two requirements (simplified)"""
        # Simple word overlap similarity
        words1 = set(req1.lower().split())
        words2 = set(req2.lower().split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def requirements_conflict(self, req1: str, req2: str) -> bool:
        """Check if two requirements conflict (simplified)"""
        # Simple conflict detection based on opposing terms
        opposing_pairs = [
            (['must', 'shall', 'required'], ['may', 'could', 'optional']),
            (['prohibit', 'forbid', 'not'], ['allow', 'permit', 'shall']),
            (['minimum', 'at least'], ['maximum', 'not more than'])
        ]
        
        req1_lower = req1.lower()
        req2_lower = req2.lower()
        
        for strong_terms, weak_terms in opposing_pairs:
            req1_has_strong = any(term in req1_lower for term in strong_terms)
            req1_has_weak = any(term in req1_lower for term in weak_terms)
            req2_has_strong = any(term in req2_lower for term in strong_terms)
            req2_has_weak = any(term in req2_lower for term in weak_terms)
            
            if (req1_has_strong and req2_has_weak) or (req1_has_weak and req2_has_strong):
                return True
        
        return False
    
    def map_requirements(self, risk_type: str, output_format: str = 'dict') -> Union[Dict, str]:
        """Map all requirements for a specific risk type across all jurisdictions"""
        
        mapping = {
            'risk_type': risk_type,
            'jurisdictions': {},
            'summary': {
                'total_requirements': 0,
                'by_type': defaultdict(int),
                'by_jurisdiction': defaultdict(int),
                'by_source': defaultdict(int)
            }
        }
        
        jurisdictions = ['EU', 'FINNISH', 'INTERNATIONAL']
        
        for jurisdiction in jurisdictions:
            result = self.query_risk_category(risk_type, jurisdiction=jurisdiction, limit=100)
            
            jurisdiction_data = {
                'total_matches': result['total_matches'],
                'sources': set(),
                'requirements': [],
                'requirement_types': defaultdict(int)
            }
            
            for match in result['matches']:
                jurisdiction_data['sources'].add(match['source'])
                
                for req in match['requirements']:
                    requirement_entry = {
                        'text': req['text'],
                        'type': req['type'],
                        'source': match['source'],
                        'document': match['document'],
                        'page': match['page']
                    }
                    
                    jurisdiction_data['requirements'].append(requirement_entry)
                    jurisdiction_data['requirement_types'][req['type']] += 1
                    
                    # Update summary
                    mapping['summary']['total_requirements'] += 1
                    mapping['summary']['by_type'][req['type']] += 1
                    mapping['summary']['by_jurisdiction'][jurisdiction] += 1
                    mapping['summary']['by_source'][match['source']] += 1
            
            jurisdiction_data['sources'] = list(jurisdiction_data['sources'])
            mapping['jurisdictions'][jurisdiction] = jurisdiction_data
        
        if output_format == 'json':
            return json.dumps(mapping, indent=2, default=str)
        elif output_format == 'summary':
            return self.create_mapping_summary(mapping)
        else:
            return mapping
    
    def create_mapping_summary(self, mapping: Dict) -> str:
        """Create a human-readable summary of requirement mapping"""
        
        summary = f"""
RISK CATEGORY: {mapping['risk_type']}
{'=' * 50}

OVERVIEW:
- Total Requirements Found: {mapping['summary']['total_requirements']}
- Jurisdictions Analyzed: {len(mapping['jurisdictions'])}

REQUIREMENT DISTRIBUTION BY TYPE:
"""
        
        for req_type, count in mapping['summary']['by_type'].items():
            summary += f"- {req_type}: {count}\n"
        
        summary += "\nREQUIREMENT DISTRIBUTION BY JURISDICTION:\n"
        for jurisdiction, count in mapping['summary']['by_jurisdiction'].items():
            summary += f"- {jurisdiction}: {count}\n"
        
        summary += "\nREGULATORY SOURCES:\n"
        for source, count in mapping['summary']['by_source'].items():
            summary += f"- {source}: {count} requirements\n"
        
        summary += "\nJURISDICTION DETAILS:\n"
        for jurisdiction, data in mapping['jurisdictions'].items():
            summary += f"\n{jurisdiction}:\n"
            summary += f"  - Document matches: {data['total_matches']}\n"
            summary += f"  - Requirements: {len(data['requirements'])}\n"
            summary += f"  - Sources: {', '.join(data['sources'])}\n"
            
            if data['requirement_types']:
                summary += "  - By type: "
                type_breakdown = [f"{rtype}({count})" for rtype, count in data['requirement_types'].items()]
                summary += ", ".join(type_breakdown) + "\n"
        
        return summary
    
    def analyze_gaps(self, risk_types: List[str] = None, jurisdictions: List[str] = None) -> Dict:
        """Identify regulatory gaps across risk types and jurisdictions"""
        
        if risk_types is None:
            risk_types = list(self.corpus_summary.get('risk_mentions', {}).keys())[:10]  # Top 10 risks
        
        if jurisdictions is None:
            jurisdictions = ['EU', 'FINNISH', 'INTERNATIONAL']
        
        gap_analysis = {
            'analysis_scope': {
                'risk_types': risk_types,
                'jurisdictions': jurisdictions
            },
            'coverage_matrix': {},
            'gaps_identified': [],
            'coverage_summary': {}
        }
        
        # Create coverage matrix
        for risk_type in risk_types:
            gap_analysis['coverage_matrix'][risk_type] = {}
            
            for jurisdiction in jurisdictions:
                result = self.query_risk_category(risk_type, jurisdiction=jurisdiction, limit=10)
                
                coverage_score = min(100, (result['total_matches'] / 5) * 100)  # Normalize to 0-100
                gap_analysis['coverage_matrix'][risk_type][jurisdiction] = {
                    'matches': result['total_matches'],
                    'coverage_score': round(coverage_score, 1)
                }
                
                # Identify gaps (low coverage)
                if coverage_score < 20:
                    gap_analysis['gaps_identified'].append({
                        'risk_type': risk_type,
                        'jurisdiction': jurisdiction,
                        'coverage_score': coverage_score,
                        'severity': 'HIGH' if coverage_score < 5 else 'MEDIUM'
                    })
        
        # Calculate summary statistics
        total_combinations = len(risk_types) * len(jurisdictions)
        gaps_found = len(gap_analysis['gaps_identified'])
        
        gap_analysis['coverage_summary'] = {
            'total_combinations_analyzed': total_combinations,
            'gaps_identified': gaps_found,
            'coverage_percentage': round((1 - gaps_found / total_combinations) * 100, 1),
            'high_severity_gaps': len([g for g in gap_analysis['gaps_identified'] if g['severity'] == 'HIGH']),
            'medium_severity_gaps': len([g for g in gap_analysis['gaps_identified'] if g['severity'] == 'MEDIUM'])
        }
        
        return gap_analysis
    
    def graphrag_query(self, query: str, method: str = "local") -> Dict:
        """Execute GraphRAG query if indexing is complete"""
        
        if not self.graphrag_available:
            return {
                'error': 'GraphRAG indexing not complete or files not available',
                'suggestion': 'Please wait for indexing to complete or run basic queries instead'
            }
        
        try:
            # Execute GraphRAG query
            cmd = f'uv run graphrag query --method {method} --query "{query}"'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            return {
                'query': query,
                'method': method,
                'success': result.returncode == 0,
                'response': result.stdout,
                'error': result.stderr if result.returncode != 0 else None
            }
            
        except subprocess.TimeoutExpired:
            return {
                'query': query,
                'method': method,
                'success': False,
                'error': 'Query timed out after 60 seconds'
            }
        except Exception as e:
            return {
                'query': query,
                'method': method,
                'success': False,
                'error': str(e)
            }

def main():
    """Demonstrate query engine capabilities"""
    
    print("🚀 Multi-Dimensional Query Engine Demo")
    print("=" * 50)
    
    engine = MultiDimensionalQueryEngine()
    
    # Example queries
    test_queries = [
        {
            'name': 'Credit Risk in EU Regulations',
            'func': lambda: engine.query_risk_category('CREDIT_RISK', jurisdiction='EU', limit=5)
        },
        {
            'name': 'EU vs Finnish Liquidity Risk Conflicts',
            'func': lambda: engine.find_conflicts('LIQUIDITY_RISK', ['EU', 'FINNISH'])
        },
        {
            'name': 'Market Risk Requirements Mapping',
            'func': lambda: engine.map_requirements('MARKET_RISK', output_format='summary')
        },
        {
            'name': 'Regulatory Gap Analysis',
            'func': lambda: engine.analyze_gaps(['CREDIT_RISK', 'LIQUIDITY_RISK', 'OPERATIONAL_RISK'])
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{i}. {test['name']}")
        print("-" * 30)
        
        try:
            result = test['func']()
            
            if isinstance(result, str):
                print(result)
            elif isinstance(result, dict):
                if 'matches' in result:
                    print(f"Found {result['total_matches']} matches")
                    for j, match in enumerate(result['matches'][:2], 1):  # Show first 2
                        print(f"  {j}. {match['source']} - {match['content'][:100]}...")
                elif 'conflicts' in result:
                    print(f"Conflicts: {len(result['conflicts'])}, Overlaps: {len(result['overlaps'])}")
                elif 'coverage_summary' in result:
                    summary = result['coverage_summary']
                    print(f"Coverage: {summary['coverage_percentage']}%, Gaps: {summary['gaps_identified']}")
                else:
                    print(f"Result keys: {list(result.keys())}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✅ Query engine demonstration complete!")
    print("💡 Use the query functions programmatically for detailed analysis.")

if __name__ == "__main__":
    main()