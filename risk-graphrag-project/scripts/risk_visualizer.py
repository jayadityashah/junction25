#!/usr/bin/env python3
"""
Advanced Risk Category Visualization System for GraphRAG
Creates interactive visualizations for risk-chunk-requirement mappings
"""

import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
import sqlite3
from collections import defaultdict, Counter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
import seaborn as sns
import matplotlib.pyplot as plt

# Paths
CONFIG_PATH = "config/risk_categories.yaml"
DB_PATH = "legal_documents.db"
CORPUS_SUMMARY_PATH = "input/_corpus_summary.json"
VISUALIZATIONS_PATH = "visualizations"

class AdvancedRiskVisualizer:
    """Advanced visualization system for risk categories and regulatory requirements"""
    
    def __init__(self):
        self.risk_categories = self.load_risk_categories()
        self.corpus_summary = self.load_corpus_summary()
        self.chunk_risk_mapping = {}
        self.requirement_mapping = {}
        self.jurisdiction_mapping = defaultdict(list)
        
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
    
    def load_detailed_chunk_data(self):
        """Load detailed paragraph data from database for chunk-level analysis"""
        if not Path(DB_PATH).exists():
            print(f"⚠️  Database not found: {DB_PATH}")
            return
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get detailed paragraph data with document metadata
        cursor.execute("""
            SELECT 
                p.id,
                p.content,
                p.char_length,
                pg.page_number,
                d.filename,
                d.category,
                d.subcategory
            FROM paragraphs p
            JOIN pages pg ON p.page_id = pg.id
            JOIN documents d ON pg.document_id = d.id
            WHERE p.char_length > 50
            ORDER BY d.subcategory, d.filename, pg.page_number, p.paragraph_index
        """)
        
        paragraphs = cursor.fetchall()
        conn.close()
        
        print(f"📊 Loaded {len(paragraphs)} paragraphs for detailed analysis")
        
        # Analyze each paragraph for risks and requirements
        for para_id, content, char_length, page_number, filename, category, subcategory in paragraphs:
            # Extract risk mentions
            risk_matches = self.extract_risk_mentions(content)
            
            # Extract requirements
            requirements = self.extract_requirements(content)
            
            # Store mapping
            if risk_matches or requirements:
                self.chunk_risk_mapping[para_id] = {
                    'content': content[:200] + "..." if len(content) > 200 else content,
                    'risks': risk_matches,
                    'requirements': requirements,
                    'metadata': {
                        'filename': filename,
                        'category': category,
                        'subcategory': subcategory,
                        'page': page_number,
                        'length': char_length,
                        'jurisdiction': self.determine_jurisdiction(subcategory, filename)
                    }
                }
                
                # Update jurisdiction mapping
                jurisdiction = self.determine_jurisdiction(subcategory, filename)
                for risk in risk_matches:
                    self.jurisdiction_mapping[jurisdiction].append((risk, para_id))
    
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
    
    def extract_risk_mentions(self, text: str) -> List[str]:
        """Extract risk mentions from text using configured patterns"""
        found_risks = []
        text_lower = text.lower()
        
        risk_categories = self.risk_categories.get('risk_categories', {})
        
        for risk_type, config in risk_categories.items():
            # Check primary terms
            primary_terms = config.get('primary_terms', [])
            synonyms = config.get('synonyms', [])
            
            all_terms = primary_terms + synonyms
            
            for term in all_terms:
                # Create regex pattern for the term
                pattern = re.escape(term.lower())
                if re.search(r'\b' + pattern + r'\b', text_lower):
                    found_risks.append(risk_type)
                    break  # Found one term for this risk type
        
        return found_risks
    
    def extract_requirements(self, text: str) -> List[Dict]:
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
                        'indicator': pattern
                    })
                    break
        
        return requirements
    
    def create_risk_chunk_network(self) -> go.Figure:
        """Create interactive network showing risk-chunk-requirement relationships"""
        
        # Build network graph
        G = nx.Graph()
        
        # Add nodes and edges based on chunk-risk mappings
        for chunk_id, chunk_data in self.chunk_risk_mapping.items():
            risks = chunk_data['risks']
            requirements = chunk_data['requirements']
            metadata = chunk_data['metadata']
            
            # Add chunk node
            chunk_label = f"Chunk_{chunk_id}"
            G.add_node(chunk_label, 
                      type='chunk',
                      jurisdiction=metadata['jurisdiction'],
                      source=metadata['subcategory'],
                      size=min(30, max(10, len(chunk_data['content']) / 20)))
            
            # Add risk nodes and connect to chunks
            for risk in risks:
                if not G.has_node(risk):
                    G.add_node(risk, type='risk', size=40)
                G.add_edge(risk, chunk_label, weight=1)
            
            # Add requirement nodes and connect to chunks
            for i, req in enumerate(requirements):
                req_label = f"REQ_{chunk_id}_{i}"
                G.add_node(req_label, 
                          type='requirement', 
                          req_type=req['type'],
                          text=req['text'][:50] + "..." if len(req['text']) > 50 else req['text'],
                          size=20)
                G.add_edge(chunk_label, req_label, weight=1)
        
        # Create layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Prepare plot data
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces by type
        traces = [edge_trace]
        
        colors = {'risk': 'red', 'chunk': 'lightblue', 'requirement': 'green'}
        
        for node_type, color in colors.items():
            nodes_of_type = [node for node, data in G.nodes(data=True) if data.get('type') == node_type]
            
            if nodes_of_type:
                node_x = [pos[node][0] for node in nodes_of_type]
                node_y = [pos[node][1] for node in nodes_of_type]
                node_sizes = [G.nodes[node].get('size', 10) for node in nodes_of_type]
                
                hover_text = []
                for node in nodes_of_type:
                    node_data = G.nodes[node]
                    if node_type == 'chunk':
                        hover_text.append(f"Chunk: {node}<br>Jurisdiction: {node_data.get('jurisdiction', 'Unknown')}<br>Source: {node_data.get('source', 'Unknown')}")
                    elif node_type == 'requirement':
                        hover_text.append(f"Requirement: {node_data.get('text', 'No text')}<br>Type: {node_data.get('req_type', 'Unknown')}")
                    else:
                        hover_text.append(f"Risk: {node}")
                
                trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    marker=dict(size=node_sizes, color=color, line=dict(width=1)),
                    text=nodes_of_type,
                    textposition="middle center",
                    hovertext=hover_text,
                    hoverinfo='text',
                    name=node_type.title()
                )
                traces.append(trace)
        
        fig = go.Figure(data=traces)
        fig.update_layout(
            title="Risk-Chunk-Requirement Network",
            titlefont_size=16,
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Interactive network showing relationships between risks, document chunks, and requirements",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=10)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1400,
            height=900
        )
        
        return fig
    
    def create_jurisdiction_risk_matrix(self) -> go.Figure:
        """Create heatmap showing risk distribution across jurisdictions"""
        
        # Count risks by jurisdiction
        jurisdiction_risk_counts = defaultdict(lambda: defaultdict(int))
        
        for chunk_id, chunk_data in self.chunk_risk_mapping.items():
            jurisdiction = chunk_data['metadata']['jurisdiction']
            for risk in chunk_data['risks']:
                jurisdiction_risk_counts[jurisdiction][risk] += 1
        
        # Convert to matrix format
        jurisdictions = list(jurisdiction_risk_counts.keys())
        all_risks = set()
        for jurisdiction_data in jurisdiction_risk_counts.values():
            all_risks.update(jurisdiction_data.keys())
        all_risks = sorted(list(all_risks))
        
        matrix = []
        for jurisdiction in jurisdictions:
            row = [jurisdiction_risk_counts[jurisdiction][risk] for risk in all_risks]
            matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=all_risks,
            y=jurisdictions,
            colorscale='Reds',
            hoverongaps=False,
            hovertemplate="Jurisdiction: %{y}<br>Risk: %{x}<br>Count: %{z}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Risk Distribution Across Jurisdictions",
            xaxis_title="Risk Categories",
            yaxis_title="Jurisdictions",
            width=1200,
            height=400,
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    def create_requirement_type_analysis(self) -> go.Figure:
        """Analyze distribution of requirement types across risks"""
        
        requirement_analysis = defaultdict(lambda: defaultdict(int))
        
        for chunk_id, chunk_data in self.chunk_risk_mapping.items():
            risks = chunk_data['risks']
            requirements = chunk_data['requirements']
            
            for risk in risks:
                for req in requirements:
                    requirement_analysis[risk][req['type']] += 1
        
        # Prepare data for stacked bar chart
        risks = list(requirement_analysis.keys())
        req_types = ['MANDATORY', 'RECOMMENDED', 'GUIDANCE']
        
        fig = go.Figure()
        
        for req_type in req_types:
            values = [requirement_analysis[risk][req_type] for risk in risks]
            fig.add_trace(go.Bar(
                name=req_type,
                x=risks,
                y=values,
                hovertemplate=f"Risk: %{{x}}<br>Type: {req_type}<br>Count: %{{y}}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Requirement Types Distribution Across Risk Categories",
            xaxis_title="Risk Categories",
            yaxis_title="Number of Requirements",
            barmode='stack',
            width=1200,
            height=600,
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    def create_chunk_detail_table(self) -> pd.DataFrame:
        """Create detailed table of chunk-risk-requirement mappings"""
        
        rows = []
        
        for chunk_id, chunk_data in self.chunk_risk_mapping.items():
            metadata = chunk_data['metadata']
            
            for risk in chunk_data['risks']:
                for req in chunk_data['requirements']:
                    rows.append({
                        'Chunk_ID': chunk_id,
                        'Risk_Category': risk,
                        'Jurisdiction': metadata['jurisdiction'],
                        'Source': metadata['subcategory'],
                        'Requirement_Type': req['type'],
                        'Requirement_Text': req['text'][:100] + "..." if len(req['text']) > 100 else req['text'],
                        'Document': metadata['filename'],
                        'Page': metadata['page'],
                        'Content_Preview': chunk_data['content']
                    })
        
        df = pd.DataFrame(rows)
        return df
    
    def generate_comprehensive_report(self):
        """Generate comprehensive visualization report"""
        
        print("🚀 Starting comprehensive risk visualization analysis...")
        
        # Load detailed chunk data
        self.load_detailed_chunk_data()
        
        if not self.chunk_risk_mapping:
            print("❌ No chunk-risk mappings found. Please run GraphRAG indexing first.")
            return
        
        # Create output directory
        Path(VISUALIZATIONS_PATH).mkdir(exist_ok=True)
        
        print("📊 Creating network visualization...")
        network_fig = self.create_risk_chunk_network()
        network_fig.write_html(f"{VISUALIZATIONS_PATH}/risk_chunk_network.html")
        
        print("🌍 Creating jurisdiction matrix...")
        jurisdiction_fig = self.create_jurisdiction_risk_matrix()
        jurisdiction_fig.write_html(f"{VISUALIZATIONS_PATH}/jurisdiction_risk_matrix.html")
        
        print("📋 Creating requirement type analysis...")
        req_type_fig = self.create_requirement_type_analysis()
        req_type_fig.write_html(f"{VISUALIZATIONS_PATH}/requirement_type_analysis.html")
        
        print("📑 Creating detailed mapping table...")
        detail_table = self.create_chunk_detail_table()
        detail_table.to_csv(f"{VISUALIZATIONS_PATH}/chunk_risk_requirement_mapping.csv", index=False)
        
        # Generate summary report
        total_chunks = len(self.chunk_risk_mapping)
        total_risks_found = len(set(risk for chunk_data in self.chunk_risk_mapping.values() for risk in chunk_data['risks']))
        total_requirements = sum(len(chunk_data['requirements']) for chunk_data in self.chunk_risk_mapping.values())
        
        jurisdictions = set(chunk_data['metadata']['jurisdiction'] for chunk_data in self.chunk_risk_mapping.values())
        sources = set(chunk_data['metadata']['subcategory'] for chunk_data in self.chunk_risk_mapping.values())
        
        summary_report = {
            'analysis_summary': {
                'total_chunks_with_risks': total_chunks,
                'unique_risks_found': total_risks_found,
                'total_requirements_extracted': total_requirements,
                'jurisdictions_covered': len(jurisdictions),
                'regulatory_sources': len(sources)
            },
            'jurisdiction_breakdown': {
                jurisdiction: len([chunk for chunk in self.chunk_risk_mapping.values() 
                                 if chunk['metadata']['jurisdiction'] == jurisdiction])
                for jurisdiction in jurisdictions
            },
            'top_risk_categories': Counter(
                risk for chunk_data in self.chunk_risk_mapping.values() for risk in chunk_data['risks']
            ).most_common(10),
            'requirement_type_distribution': Counter(
                req['type'] for chunk_data in self.chunk_risk_mapping.values() for req in chunk_data['requirements']
            )
        }
        
        # Save summary report
        with open(f"{VISUALIZATIONS_PATH}/risk_visualization_summary.json", 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        print("\n✅ Comprehensive Risk Visualization Analysis Complete!")
        print(f"📊 Chunks analyzed: {total_chunks:,}")
        print(f"🎯 Unique risks found: {total_risks_found}")
        print(f"📋 Requirements extracted: {total_requirements:,}")
        print(f"🌍 Jurisdictions: {', '.join(jurisdictions)}")
        print(f"📚 Regulatory sources: {len(sources)}")
        print(f"💾 Visualizations saved to: {VISUALIZATIONS_PATH}/")
        
        return summary_report

def main():
    """Main execution function"""
    visualizer = AdvancedRiskVisualizer()
    report = visualizer.generate_comprehensive_report()
    
    print("\n🎨 Advanced risk visualization system ready!")
    print("📁 Check the visualizations/ directory for interactive charts and reports.")

if __name__ == "__main__":
    main()