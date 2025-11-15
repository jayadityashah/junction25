#!/usr/bin/env python3
"""
Multi-dimensional mapper for GraphRAG results - creates higher dimensional embeddings
and visualizations for risk categories, jurisdictions, and requirements
"""

import json
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

# Paths
CONFIG_PATH = "config/risk_categories.yaml"
OUTPUT_PATH = "output"
CORPUS_SUMMARY_PATH = "input/_corpus_summary.json"
VISUALIZATIONS_PATH = "visualizations"

class RiskDimensionMapper:
    """Multi-dimensional mapper for risk categories and regulatory requirements"""
    
    def __init__(self):
        self.risk_categories = self.load_risk_categories()
        self.corpus_summary = self.load_corpus_summary()
        self.entity_data = None
        self.relationship_data = None
        self.dimensional_embeddings = {}
        
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
    
    def load_graphrag_results(self):
        """Load GraphRAG output files (entities, relationships, embeddings)"""
        output_dir = Path(OUTPUT_PATH)
        
        # Try to load various GraphRAG output files
        entity_files = list(output_dir.glob("**/create_final_entities.parquet"))
        relationship_files = list(output_dir.glob("**/create_final_relationships.parquet"))
        
        if entity_files:
            self.entity_data = pd.read_parquet(entity_files[0])
            print(f"✅ Loaded entities: {len(self.entity_data)} records")
        else:
            print("⚠️  No entity data found - GraphRAG indexing may not be complete")
            
        if relationship_files:
            self.relationship_data = pd.read_parquet(relationship_files[0])
            print(f"✅ Loaded relationships: {len(self.relationship_data)} records")
        else:
            print("⚠️  No relationship data found")
    
    def create_risk_embeddings(self) -> np.ndarray:
        """Create embeddings for risk categories based on mention frequency and characteristics"""
        
        if not self.corpus_summary.get('risk_mentions'):
            print("⚠️  No risk mentions data available")
            return np.array([])
        
        risk_mentions = self.corpus_summary['risk_mentions']
        risk_names = list(risk_mentions.keys())
        
        # Create feature matrix for risks
        features = []
        feature_names = []
        
        # Mention frequency features
        mention_counts = [risk_mentions[risk] for risk in risk_names]
        features.append(mention_counts)
        feature_names.append('mention_frequency')
        
        # Jurisdiction features
        jurisdictions = self.corpus_summary.get('jurisdictions', [])
        for jurisdiction in jurisdictions:
            jurisdiction_features = []
            for risk in risk_names:
                # Estimate jurisdiction-risk association (simplified)
                if risk in self.risk_categories.get('risk_categories', {}):
                    risk_config = self.risk_categories['risk_categories'][risk]
                    regulatory_context = risk_config.get('regulatory_context', [])
                    
                    # Check if risk is associated with this jurisdiction's sources
                    if jurisdiction == 'EU':
                        sources = ['BRRD', 'CRD', 'CRR', 'EBA', 'MiFID', 'MiFIR', 'SFDR']
                    elif jurisdiction == 'FINNISH':
                        sources = ['FIVA_MOK', 'VYL', 'LLL']
                    else:
                        sources = ['Basel']
                    
                    overlap = len(set(regulatory_context) & set(sources))
                    jurisdiction_features.append(overlap)
                else:
                    jurisdiction_features.append(0)
            
            features.append(jurisdiction_features)
            feature_names.append(f'jurisdiction_{jurisdiction}')
        
        # Convert to numpy array
        feature_matrix = np.array(features).T
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Store for later use
        self.risk_embedding_data = {
            'risk_names': risk_names,
            'features': feature_matrix_scaled,
            'feature_names': feature_names,
            'scaler': scaler
        }
        
        return feature_matrix_scaled
    
    def create_jurisdiction_embeddings(self):
        """Create embeddings for jurisdictions based on regulatory coverage"""
        
        jurisdictions = self.corpus_summary.get('jurisdictions', [])
        subcategories = self.corpus_summary.get('subcategories', [])
        
        # Create jurisdiction-source matrix
        jurisdiction_source_matrix = []
        
        for jurisdiction in jurisdictions:
            row = []
            if jurisdiction == 'EU':
                eu_sources = ['BRRD', 'CRD', 'CRR', 'EBA', 'MiFID', 'MiFIR', 'SFDR', 'IFRS']
                for source in subcategories:
                    row.append(1 if source in eu_sources else 0)
            elif jurisdiction == 'FINNISH':
                finnish_sources = ['FIVA_MOK', 'VYL', 'LLL', 'FINLAND']
                for source in subcategories:
                    row.append(1 if source in finnish_sources else 0)
            else:  # INTERNATIONAL
                for source in subcategories:
                    row.append(1 if source == 'Basel' else 0)
            
            jurisdiction_source_matrix.append(row)
        
        return np.array(jurisdiction_source_matrix), jurisdictions, subcategories
    
    def perform_dimensional_reduction(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Perform PCA for dimensional reduction and visualization"""
        
        if embeddings.size == 0:
            return np.array([])
        
        pca = PCA(n_components=min(n_components, embeddings.shape[1]))
        reduced_embeddings = pca.fit_transform(embeddings)
        
        print(f"📊 PCA explained variance ratio: {pca.explained_variance_ratio_}")
        
        return reduced_embeddings
    
    def cluster_risks_by_similarity(self, embeddings: np.ndarray, n_clusters: int = 5):
        """Cluster risks by similarity using K-means"""
        
        if embeddings.size == 0:
            return [], []
        
        kmeans = KMeans(n_clusters=min(n_clusters, len(embeddings)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        return clusters, kmeans.cluster_centers_
    
    def visualize_risk_landscape_3d(self):
        """Create 3D visualization of risk landscape"""
        
        risk_embeddings = self.create_risk_embeddings()
        if risk_embeddings.size == 0:
            return None
        
        # Reduce to 3D
        reduced_3d = self.perform_dimensional_reduction(risk_embeddings, n_components=3)
        
        # Cluster risks
        clusters, _ = self.cluster_risks_by_similarity(risk_embeddings)
        
        # Get risk data
        risk_names = self.risk_embedding_data['risk_names']
        mention_counts = [self.corpus_summary['risk_mentions'][risk] for risk in risk_names]
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=reduced_3d[:, 0],
            y=reduced_3d[:, 1],
            z=reduced_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=[min(20, max(5, count/50)) for count in mention_counts],  # Size based on mentions
                color=clusters,
                colorscale='viridis',
                showscale=True,
                colorbar=dict(title="Risk Cluster")
            ),
            text=risk_names,
            textposition="middle right",
            hovertemplate="<b>%{text}</b><br>" +
                          "Mentions: %{customdata}<br>" +
                          "Cluster: %{marker.color}<br>" +
                          "X: %{x:.2f}<br>" +
                          "Y: %{y:.2f}<br>" +
                          "Z: %{z:.2f}<extra></extra>",
            customdata=mention_counts
        ))
        
        fig.update_layout(
            title="3D Risk Category Landscape",
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3"
            ),
            width=1000,
            height=800
        )
        
        # Save visualization
        output_path = Path(VISUALIZATIONS_PATH) / "risk_landscape_3d.html"
        output_path.parent.mkdir(exist_ok=True)
        fig.write_html(str(output_path))
        
        print(f"💾 3D Risk landscape saved to: {output_path}")
        return fig
    
    def visualize_jurisdiction_coverage(self):
        """Create visualization of jurisdiction coverage across risk categories"""
        
        # Create jurisdiction-risk coverage matrix
        risk_mentions = self.corpus_summary.get('risk_mentions', {})
        jurisdictions = self.corpus_summary.get('jurisdictions', [])
        
        coverage_matrix = []
        risk_names = list(risk_mentions.keys())
        
        for jurisdiction in jurisdictions:
            row = []
            for risk in risk_names:
                # Simplified coverage calculation based on regulatory context
                if risk in self.risk_categories.get('risk_categories', {}):
                    risk_config = self.risk_categories['risk_categories'][risk]
                    regulatory_context = risk_config.get('regulatory_context', [])
                    
                    if jurisdiction == 'EU':
                        sources = ['BRRD', 'CRD', 'CRR', 'EBA', 'MiFID', 'MiFIR', 'SFDR']
                    elif jurisdiction == 'FINNISH':
                        sources = ['FIVA_MOK', 'VYL', 'LLL']
                    else:
                        sources = ['Basel']
                    
                    coverage = len(set(regulatory_context) & set(sources)) / len(sources) if sources else 0
                    row.append(coverage * risk_mentions[risk])  # Weight by mentions
                else:
                    row.append(0)
            
            coverage_matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=coverage_matrix,
            x=risk_names,
            y=jurisdictions,
            colorscale='Blues',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Jurisdiction Coverage Across Risk Categories",
            xaxis_title="Risk Categories",
            yaxis_title="Jurisdictions",
            width=1200,
            height=600
        )
        
        # Save visualization
        output_path = Path(VISUALIZATIONS_PATH) / "jurisdiction_coverage.html"
        fig.write_html(str(output_path))
        
        print(f"💾 Jurisdiction coverage saved to: {output_path}")
        return fig
    
    def create_network_graph(self):
        """Create network graph showing risk-jurisdiction-source relationships"""
        
        G = nx.Graph()
        
        # Add nodes for risks, jurisdictions, and sources
        risk_mentions = self.corpus_summary.get('risk_mentions', {})
        jurisdictions = self.corpus_summary.get('jurisdictions', [])
        sources = self.corpus_summary.get('subcategories', [])
        
        # Add risk nodes
        for risk, count in risk_mentions.items():
            G.add_node(risk, type='risk', size=count, color='red')
        
        # Add jurisdiction nodes
        for jurisdiction in jurisdictions:
            G.add_node(jurisdiction, type='jurisdiction', size=1000, color='blue')
        
        # Add source nodes
        for source in sources:
            G.add_node(source, type='source', size=500, color='green')
        
        # Add edges based on relationships
        for risk in risk_mentions.keys():
            if risk in self.risk_categories.get('risk_categories', {}):
                risk_config = self.risk_categories['risk_categories'][risk]
                regulatory_context = risk_config.get('regulatory_context', [])
                
                # Connect risks to sources
                for source in regulatory_context:
                    if source in sources:
                        G.add_edge(risk, source, weight=risk_mentions[risk])
                
                # Connect sources to jurisdictions
                for source in regulatory_context:
                    if source in ['BRRD', 'CRD', 'CRR', 'EBA', 'MiFID', 'MiFIR', 'SFDR', 'IFRS']:
                        G.add_edge(source, 'EU', weight=1)
                    elif source in ['FIVA_MOK', 'VYL', 'LLL']:
                        G.add_edge(source, 'FINNISH', weight=1)
                    elif source == 'Basel':
                        G.add_edge(source, 'INTERNATIONAL', weight=1)
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract node and edge information
        node_trace = []
        edge_trace = []
        
        # Create edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=0.5, color='gray'),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Create nodes by type
        for node_type, color in [('risk', 'red'), ('jurisdiction', 'blue'), ('source', 'green')]:
            nodes_of_type = [node for node in G.nodes() if G.nodes[node].get('type') == node_type]
            if nodes_of_type:
                node_x = [pos[node][0] for node in nodes_of_type]
                node_y = [pos[node][1] for node in nodes_of_type]
                node_sizes = [min(50, max(10, G.nodes[node].get('size', 100) / 50)) for node in nodes_of_type]
                
                node_trace.append(go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    marker=dict(
                        size=node_sizes,
                        color=color,
                        line=dict(width=1, color='white')
                    ),
                    text=nodes_of_type,
                    textposition="middle center",
                    name=node_type.title(),
                    hovertemplate="<b>%{text}</b><br>Type: " + node_type + "<extra></extra>"
                ))
        
        # Create figure
        fig = go.Figure(data=edge_trace + node_trace)
        fig.update_layout(
            title="Risk-Jurisdiction-Source Network",
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Network showing relationships between risks, jurisdictions, and regulatory sources",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1200,
            height=800
        )
        
        # Save visualization
        output_path = Path(VISUALIZATIONS_PATH) / "risk_network.html"
        fig.write_html(str(output_path))
        
        print(f"💾 Network graph saved to: {output_path}")
        return fig
    
    def generate_dimensional_analysis_report(self):
        """Generate comprehensive dimensional analysis report"""
        
        print("📊 Generating Multi-Dimensional Analysis Report...")
        
        # Create all visualizations
        risk_3d_fig = self.visualize_risk_landscape_3d()
        jurisdiction_fig = self.visualize_jurisdiction_coverage()
        network_fig = self.create_network_graph()
        
        # Generate summary statistics
        risk_embeddings = self.create_risk_embeddings()
        
        report = {
            'summary': {
                'total_risks_identified': len(self.corpus_summary.get('risk_mentions', {})),
                'total_jurisdictions': len(self.corpus_summary.get('jurisdictions', [])),
                'total_sources': len(self.corpus_summary.get('subcategories', [])),
                'total_documents': self.corpus_summary.get('total_documents', 0),
                'total_paragraphs': self.corpus_summary.get('total_paragraphs', 0)
            },
            'risk_analysis': {
                'top_risks': sorted(
                    self.corpus_summary.get('risk_mentions', {}).items(),
                    key=lambda x: x[1], reverse=True
                )[:10],
                'embedding_dimensions': risk_embeddings.shape[1] if risk_embeddings.size > 0 else 0
            },
            'dimensional_coverage': {
                'jurisdictions': self.corpus_summary.get('jurisdictions', []),
                'regulatory_sources': self.corpus_summary.get('subcategories', []),
                'categories': self.corpus_summary.get('categories', [])
            }
        }
        
        # Save report
        report_path = Path(VISUALIZATIONS_PATH) / "dimensional_analysis_report.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"💾 Analysis report saved to: {report_path}")
        print("\n🎯 Multi-Dimensional Analysis Complete!")
        print(f"   📊 Risks analyzed: {report['summary']['total_risks_identified']}")
        print(f"   🌍 Jurisdictions: {report['summary']['total_jurisdictions']}")
        print(f"   📚 Regulatory sources: {report['summary']['total_sources']}")
        print(f"   📄 Documents processed: {report['summary']['total_documents']}")
        print(f"   📝 Paragraphs analyzed: {report['summary']['total_paragraphs']:,}")
        
        return report

def main():
    """Main execution function"""
    print("🚀 Starting Multi-Dimensional Risk Analysis...")
    
    mapper = RiskDimensionMapper()
    
    # Try to load GraphRAG results (may not exist yet)
    mapper.load_graphrag_results()
    
    # Generate dimensional analysis
    report = mapper.generate_dimensional_analysis_report()
    
    print(f"\n✅ Multi-dimensional mapping completed!")
    print(f"📁 Visualizations saved to: {VISUALIZATIONS_PATH}/")

if __name__ == "__main__":
    main()