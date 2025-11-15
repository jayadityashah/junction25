#!/usr/bin/env python3
"""
GraphRAG Knowledge Graph Visualizer
Visualizes the actual GraphRAG-generated knowledge graph
"""

import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_PATH = "output"
VISUALIZATIONS_PATH = "visualizations"

class GraphRAGVisualizer:
    """Visualize GraphRAG-generated knowledge graph"""
    
    def __init__(self):
        self.entities_df = None
        self.relationships_df = None
        self.communities_df = None
        self.embeddings_df = None
        self.graph = nx.Graph()
        
    def load_graphrag_data(self):
        """Load GraphRAG output files"""
        output_dir = Path(OUTPUT_PATH)
        
        # Find GraphRAG output files
        entity_files = list(output_dir.glob("**/entities.parquet"))
        relationship_files = list(output_dir.glob("**/relationships.parquet"))
        community_files = list(output_dir.glob("**/communities.parquet"))
        
        if not entity_files:
            entity_files = list(output_dir.glob("entities.parquet"))
        if not relationship_files:
            relationship_files = list(output_dir.glob("relationships.parquet"))
        if not community_files:
            community_files = list(output_dir.glob("communities.parquet"))
        
        # Load entities
        if entity_files:
            self.entities_df = pd.read_parquet(entity_files[0])
            print(f"✅ Loaded {len(self.entities_df)} entities")
        else:
            print("⚠️  No entity files found - GraphRAG indexing may be incomplete")
            return False
        
        # Load relationships
        if relationship_files:
            self.relationships_df = pd.read_parquet(relationship_files[0])
            print(f"✅ Loaded {len(self.relationships_df)} relationships")
        else:
            print("⚠️  No relationship files found")
            
        # Load communities (optional)
        if community_files:
            self.communities_df = pd.read_parquet(community_files[0])
            print(f"✅ Loaded {len(self.communities_df)} communities")
        
        return True
    
    def check_available_files(self):
        """Check what GraphRAG files are available"""
        output_dir = Path(OUTPUT_PATH)
        
        print("📁 GraphRAG Output Files Status:")
        
        # Check for all possible GraphRAG output files
        file_patterns = [
            "create_final_entities.parquet",
            "create_final_relationships.parquet", 
            "create_final_communities.parquet",
            "create_final_text_units.parquet",
            "create_final_documents.parquet",
            "create_community_reports.parquet",
            "*.graphml"
        ]
        
        available_files = []
        
        for pattern in file_patterns:
            files = list(output_dir.glob(f"**/{pattern}"))
            if not files:
                files = list(output_dir.glob(pattern))
            
            if files:
                for file in files:
                    size_mb = file.stat().st_size / (1024*1024)
                    available_files.append({
                        'file': file.name,
                        'path': str(file),
                        'size_mb': round(size_mb, 2)
                    })
                    print(f"  ✅ {file.name} ({size_mb:.2f} MB)")
            else:
                print(f"  ❌ {pattern}")
        
        return available_files
    
    def build_networkx_graph(self):
        """Build NetworkX graph from GraphRAG data"""
        
        if self.entities_df is None or self.relationships_df is None:
            print("❌ Missing entity or relationship data")
            return False
        
        # Add entity nodes
        for _, entity in self.entities_df.iterrows():
            entity_id = entity.get('id', entity.get('title', ''))
            entity_type = entity.get('type', 'Unknown')
            description = entity.get('description', '')
            
            self.graph.add_node(
                entity_id,
                type=entity_type,
                description=description[:100] + "..." if len(description) > 100 else description,
                size=10
            )
        
        # Add relationship edges
        for _, rel in self.relationships_df.iterrows():
            source = rel.get('source', '')
            target = rel.get('target', '')
            description = rel.get('description', '')
            weight = rel.get('weight', 1)
            
            if source in self.graph.nodes and target in self.graph.nodes:
                self.graph.add_edge(
                    source, 
                    target, 
                    description=description,
                    weight=weight
                )
        
        print(f"📊 Built graph with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges")
        return True
    
    def create_interactive_network_plot(self):
        """Create interactive network plot of the knowledge graph"""
        
        if len(self.graph.nodes) == 0:
            print("❌ No graph data available")
            return None
        
        # Create layout
        print("🎨 Creating graph layout...")
        pos = nx.spring_layout(self.graph, k=1, iterations=50, seed=42)
        
        # Prepare edge traces
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_description = edge[2].get('description', 'No description')
            edge_info.append(f"{edge[0]} → {edge[1]}: {edge_description[:50]}...")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node traces by type
        node_types = {}
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            if node_type not in node_types:
                node_types[node_type] = {
                    'nodes': [],
                    'x': [],
                    'y': [],
                    'text': [],
                    'descriptions': []
                }
            
            node_types[node_type]['nodes'].append(node)
            node_types[node_type]['x'].append(pos[node][0])
            node_types[node_type]['y'].append(pos[node][1])
            node_types[node_type]['text'].append(node)
            node_types[node_type]['descriptions'].append(
                f"<b>{node}</b><br>Type: {node_type}<br>Desc: {data.get('description', 'No description')}"
            )
        
        # Create node traces
        traces = [edge_trace]
        colors = px.colors.qualitative.Set3
        
        for i, (node_type, data) in enumerate(node_types.items()):
            color = colors[i % len(colors)]
            
            trace = go.Scatter(
                x=data['x'], y=data['y'],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(width=1, color='white')
                ),
                text=data['text'],
                textposition="middle center",
                hovertext=data['descriptions'],
                hoverinfo='text',
                name=f"{node_type} ({len(data['nodes'])})"
            )
            traces.append(trace)
        
        # Create figure
        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=f"GraphRAG Knowledge Graph ({len(self.graph.nodes)} entities, {len(self.graph.edges)} relationships)",
                font_size=16
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Interactive GraphRAG knowledge graph visualization",
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
    
    def analyze_graph_statistics(self):
        """Analyze and display graph statistics"""
        
        if len(self.graph.nodes) == 0:
            print("❌ No graph data available")
            return {}
        
        # Calculate statistics
        stats = {
            'total_nodes': len(self.graph.nodes),
            'total_edges': len(self.graph.edges),
            'density': nx.density(self.graph),
            'connected_components': nx.number_connected_components(self.graph),
            'largest_component_size': len(max(nx.connected_components(self.graph), key=len)),
            'average_clustering': nx.average_clustering(self.graph),
            'node_types': {},
            'top_nodes_by_degree': []
        }
        
        # Node type distribution
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # Top nodes by degree centrality
        degree_centrality = nx.degree_centrality(self.graph)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for node, centrality in top_nodes:
            node_data = self.graph.nodes[node]
            stats['top_nodes_by_degree'].append({
                'node': node,
                'centrality': round(centrality, 3),
                'type': node_data.get('type', 'Unknown'),
                'degree': self.graph.degree[node]
            })
        
        return stats
    
    def create_entity_type_analysis(self):
        """Create analysis of entity types in the graph"""
        
        if self.entities_df is None:
            print("❌ No entity data available")
            return None
        
        # Entity type distribution
        entity_type_counts = self.entities_df['type'].value_counts()
        
        fig = px.bar(
            x=entity_type_counts.index,
            y=entity_type_counts.values,
            title="Entity Type Distribution in GraphRAG Knowledge Graph",
            labels={'x': 'Entity Type', 'y': 'Count'}
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500
        )
        
        return fig
    
    def visualize_communities(self):
        """Visualize community structure if available"""
        
        if self.communities_df is None:
            print("⚠️  No community data available")
            return None
        
        # Community size distribution
        community_sizes = self.communities_df.groupby('community').size().sort_values(ascending=False)
        
        fig = px.bar(
            x=range(len(community_sizes)),
            y=community_sizes.values,
            title="Community Size Distribution",
            labels={'x': 'Community Rank', 'y': 'Size (Number of Entities)'}
        )
        
        return fig
    
    def generate_comprehensive_visualization_report(self):
        """Generate complete GraphRAG visualization report"""
        
        print("🚀 Starting GraphRAG Knowledge Graph Visualization...")
        
        # Check available files
        available_files = self.check_available_files()
        
        if not available_files:
            print("❌ No GraphRAG output files found. Please ensure indexing is complete.")
            return
        
        # Load data
        if not self.load_graphrag_data():
            print("❌ Could not load GraphRAG data files")
            return
        
        # Build graph
        if not self.build_networkx_graph():
            print("❌ Could not build NetworkX graph")
            return
        
        # Create visualizations
        Path(VISUALIZATIONS_PATH).mkdir(exist_ok=True)
        
        print("📊 Creating interactive network visualization...")
        network_fig = self.create_interactive_network_plot()
        if network_fig:
            network_fig.write_html(f"{VISUALIZATIONS_PATH}/graphrag_knowledge_graph.html")
            print(f"💾 Saved: {VISUALIZATIONS_PATH}/graphrag_knowledge_graph.html")
        
        print("📈 Creating entity type analysis...")
        entity_fig = self.create_entity_type_analysis()
        if entity_fig:
            entity_fig.write_html(f"{VISUALIZATIONS_PATH}/graphrag_entity_types.html")
            print(f"💾 Saved: {VISUALIZATIONS_PATH}/graphrag_entity_types.html")
        
        print("👥 Creating community analysis...")
        community_fig = self.visualize_communities()
        if community_fig:
            community_fig.write_html(f"{VISUALIZATIONS_PATH}/graphrag_communities.html")
            print(f"💾 Saved: {VISUALIZATIONS_PATH}/graphrag_communities.html")
        
        # Generate statistics
        print("📊 Analyzing graph statistics...")
        stats = self.analyze_graph_statistics()
        
        with open(f"{VISUALIZATIONS_PATH}/graphrag_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Print summary
        print("\n✅ GraphRAG Visualization Complete!")
        print(f"📊 Knowledge Graph: {stats['total_nodes']} entities, {stats['total_edges']} relationships")
        print(f"🔗 Graph Density: {stats['density']:.3f}")
        print(f"🌐 Connected Components: {stats['connected_components']}")
        print(f"📈 Entity Types: {len(stats['node_types'])}")
        print(f"💾 Visualizations saved to: {VISUALIZATIONS_PATH}/")
        
        return stats

def main():
    """Main execution"""
    visualizer = GraphRAGVisualizer()
    visualizer.generate_comprehensive_visualization_report()

if __name__ == "__main__":
    main()