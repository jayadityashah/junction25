#!/usr/bin/env python3
"""
Interactive Dashboard for GraphRAG Risk Analysis
Streamlit-based web interface for exploring multi-dimensional risk mappings
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import yaml
from pathlib import Path
import sys

# Add scripts directory to path for imports
sys.path.append('scripts')

try:
    from query_engine import MultiDimensionalQueryEngine
    from dimension_mapper import RiskDimensionMapper
except ImportError:
    st.error("Required modules not found. Please ensure query_engine.py and dimension_mapper.py are in the scripts directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="GraphRAG Risk Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .risk-high { color: #ff4444; }
    .risk-medium { color: #ffaa00; }
    .risk-low { color: #00aa00; }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_corpus_summary():
    """Load corpus summary with caching"""
    try:
        with open('input/_corpus_summary.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

@st.cache_data
def load_risk_categories():
    """Load risk categories with caching"""
    try:
        with open('config/risk_categories.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

@st.cache_resource
def initialize_query_engine():
    """Initialize query engine with caching"""
    return MultiDimensionalQueryEngine()

def main():
    """Main dashboard application"""
    
    # Header
    st.title("🎯 GraphRAG Multi-Dimensional Risk Analysis Dashboard")
    st.markdown("---")
    
    # Load data
    corpus_summary = load_corpus_summary()
    risk_categories = load_risk_categories()
    query_engine = initialize_query_engine()
    
    if not corpus_summary:
        st.error("Corpus summary not found. Please run the corpus extraction first.")
        return
    
    # Sidebar
    st.sidebar.header("🔍 Query Controls")
    
    # Risk selection
    available_risks = list(corpus_summary.get('risk_mentions', {}).keys())
    selected_risk = st.sidebar.selectbox(
        "Select Risk Category:",
        available_risks,
        help="Choose a risk category to analyze"
    )
    
    # Jurisdiction selection
    jurisdictions = corpus_summary.get('jurisdictions', ['EU', 'FINNISH', 'INTERNATIONAL'])
    selected_jurisdictions = st.sidebar.multiselect(
        "Select Jurisdictions:",
        jurisdictions,
        default=jurisdictions,
        help="Choose jurisdictions to include in analysis"
    )
    
    # Analysis type
    analysis_type = st.sidebar.radio(
        "Analysis Type:",
        ["Overview", "Risk Deep Dive", "Conflict Analysis", "Gap Analysis", "Requirements Mapping"]
    )
    
    # Main content area
    if analysis_type == "Overview":
        show_overview(corpus_summary, risk_categories)
    
    elif analysis_type == "Risk Deep Dive":
        show_risk_deep_dive(selected_risk, selected_jurisdictions, query_engine, corpus_summary)
    
    elif analysis_type == "Conflict Analysis":
        show_conflict_analysis(selected_risk, selected_jurisdictions, query_engine)
    
    elif analysis_type == "Gap Analysis":
        show_gap_analysis(available_risks, selected_jurisdictions, query_engine)
    
    elif analysis_type == "Requirements Mapping":
        show_requirements_mapping(selected_risk, query_engine, corpus_summary)

def show_overview(corpus_summary, risk_categories):
    """Show overview dashboard"""
    
    st.header("📊 Risk Analysis Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_docs = corpus_summary.get('total_documents', 0)
        st.metric("📄 Total Documents", total_docs)
    
    with col2:
        total_paragraphs = corpus_summary.get('total_paragraphs', 0)
        st.metric("📝 Paragraphs", f"{total_paragraphs:,}")
    
    with col3:
        total_risks = len(corpus_summary.get('risk_mentions', {}))
        st.metric("🎯 Risk Categories", total_risks)
    
    with col4:
        total_jurisdictions = len(corpus_summary.get('jurisdictions', []))
        st.metric("🌍 Jurisdictions", total_jurisdictions)
    
    # Risk distribution chart
    st.subheader("Risk Category Distribution")
    
    risk_mentions = corpus_summary.get('risk_mentions', {})
    if risk_mentions:
        # Create bar chart
        risk_df = pd.DataFrame(
            list(risk_mentions.items()),
            columns=['Risk Category', 'Mentions']
        ).sort_values('Mentions', ascending=True)
        
        fig = px.bar(
            risk_df,
            x='Mentions',
            y='Risk Category',
            orientation='h',
            title="Risk Category Mentions Across All Documents",
            color='Mentions',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    # Jurisdiction and source breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Jurisdictions")
        jurisdictions = corpus_summary.get('jurisdictions', [])
        if jurisdictions:
            # Create pie chart
            # Approximate document distribution (this would need actual data)
            approx_distribution = {
                'EU': 40,
                'FINNISH': 55,
                'INTERNATIONAL': 5
            }
            
            fig = px.pie(
                values=[approx_distribution.get(j, 0) for j in jurisdictions],
                names=jurisdictions,
                title="Document Distribution by Jurisdiction"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📚 Regulatory Sources")
        sources = corpus_summary.get('subcategories', [])
        if len(sources) > 6:
            sources = sources[:6] + ['Others']
        
        # Simple list display
        for i, source in enumerate(sources):
            st.write(f"{i+1}. {source}")
    
    # Latest analysis info
    st.subheader("ℹ️ Dataset Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.info(f"""
        **Document Categories:**
        {', '.join(corpus_summary.get('categories', []))}
        
        **Analysis Coverage:**
        - EU Regulations: BRRD, CRD, CRR, EBA, MiFID, MiFIR, SFDR
        - Finnish Laws: FIVA_MOK, VYL, LLL
        - International: Basel Committee standards
        """)
    
    with info_col2:
        top_risks = sorted(
            corpus_summary.get('risk_mentions', {}).items(),
            key=lambda x: x[1], reverse=True
        )[:5]
        
        st.success(f"""
        **Top Risk Categories:**
        {chr(10).join([f"• {risk}: {count} mentions" for risk, count in top_risks])}
        
        **Multi-dimensional Analysis:**
        ✅ Risk-Document-Requirement mapping
        ✅ Cross-jurisdictional comparison
        ✅ Higher dimensional embeddings
        """)

def show_risk_deep_dive(selected_risk, selected_jurisdictions, query_engine, corpus_summary):
    """Show detailed analysis for a specific risk"""
    
    st.header(f"🎯 Deep Dive: {selected_risk}")
    
    # Risk category information
    risk_mentions = corpus_summary.get('risk_mentions', {})
    total_mentions = risk_mentions.get(selected_risk, 0)
    
    st.metric("📊 Total Mentions", total_mentions)
    
    # Query for detailed information
    with st.spinner("Analyzing risk across jurisdictions..."):
        results_by_jurisdiction = {}
        
        for jurisdiction in selected_jurisdictions:
            result = query_engine.query_risk_category(
                selected_risk, 
                jurisdiction=jurisdiction, 
                limit=20
            )
            results_by_jurisdiction[jurisdiction] = result
    
    # Display results
    for jurisdiction in selected_jurisdictions:
        result = results_by_jurisdiction[jurisdiction]
        
        if result['total_matches'] > 0:
            with st.expander(f"🌍 {jurisdiction} - {result['total_matches']} matches", expanded=True):
                
                # Show some example paragraphs
                for i, match in enumerate(result['matches'][:3], 1):
                    st.write(f"**Example {i}:**")
                    st.write(f"📄 *{match['document']} (Page {match['page']})*")
                    st.write(f"📝 {match['content'][:300]}...")
                    
                    if match['requirements']:
                        st.write("📋 **Requirements found:**")
                        for req in match['requirements'][:2]:
                            req_color = "🔴" if req['type'] == 'MANDATORY' else "🟡" if req['type'] == 'RECOMMENDED' else "🟢"
                            st.write(f"{req_color} {req['text'][:150]}...")
                    
                    st.markdown("---")
        else:
            st.warning(f"No matches found for {selected_risk} in {jurisdiction} regulations")
    
    # Comparative analysis
    if len(selected_jurisdictions) > 1:
        st.subheader("📊 Comparative Analysis")
        
        comparison_data = []
        for jurisdiction in selected_jurisdictions:
            result = results_by_jurisdiction[jurisdiction]
            comparison_data.append({
                'Jurisdiction': jurisdiction,
                'Matches': result['total_matches'],
                'Requirements': sum(len(match['requirements']) for match in result['matches'])
            })
        
        comp_df = pd.DataFrame(comparison_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(comp_df, x='Jurisdiction', y='Matches', 
                         title="Document Matches by Jurisdiction")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(comp_df, x='Jurisdiction', y='Requirements', 
                         title="Requirements Found by Jurisdiction")
            st.plotly_chart(fig2, use_container_width=True)

def show_conflict_analysis(selected_risk, selected_jurisdictions, query_engine):
    """Show conflict analysis between jurisdictions"""
    
    st.header(f"⚖️ Conflict Analysis: {selected_risk}")
    
    if len(selected_jurisdictions) < 2:
        st.warning("Please select at least 2 jurisdictions for conflict analysis")
        return
    
    with st.spinner("Analyzing conflicts and overlaps..."):
        conflict_result = query_engine.find_conflicts(selected_risk, selected_jurisdictions)
    
    # Display conflicts
    st.subheader("⚠️ Conflicts Identified")
    
    conflicts = conflict_result.get('conflicts', [])
    if conflicts:
        for i, conflict in enumerate(conflicts[:5], 1):  # Show first 5 conflicts
            st.error(f"""
            **Conflict {i}:**
            
            **{conflict['jurisdiction_1']}:** {conflict['requirement_1']}
            *Source: {conflict['source_1']}*
            
            **{conflict['jurisdiction_2']}:** {conflict['requirement_2']}
            *Source: {conflict['source_2']}*
            
            *Similarity Score: {conflict['similarity_score']:.2f}*
            """)
    else:
        st.success("No major conflicts detected between the selected jurisdictions")
    
    # Display overlaps
    st.subheader("🔄 Overlaps Identified")
    
    overlaps = conflict_result.get('overlaps', [])
    if overlaps:
        for i, overlap in enumerate(overlaps[:5], 1):
            st.info(f"""
            **Overlap {i}:**
            
            **Jurisdictions:** {', '.join(overlap['jurisdictions'])}
            **Requirement:** {overlap['requirement']}
            **Sources:** {', '.join(overlap['sources'])}
            
            *Similarity Score: {overlap['similarity_score']:.2f}*
            """)
    else:
        st.info("No significant overlaps detected")
    
    # Summary
    st.subheader("📊 Conflict Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("⚠️ Conflicts", len(conflicts))
    with col2:
        st.metric("🔄 Overlaps", len(overlaps))
    with col3:
        alignment_score = max(0, 100 - (len(conflicts) * 20) - (len(overlaps) * 5))
        st.metric("✅ Alignment Score", f"{alignment_score}%")

def show_gap_analysis(available_risks, selected_jurisdictions, query_engine):
    """Show regulatory gap analysis"""
    
    st.header("🕳️ Regulatory Gap Analysis")
    
    # Risk selection for gap analysis
    selected_risks_for_gaps = st.multiselect(
        "Select Risk Categories for Gap Analysis:",
        available_risks,
        default=available_risks[:5],  # Default to first 5
        help="Choose risk categories to analyze for coverage gaps"
    )
    
    if not selected_risks_for_gaps:
        st.warning("Please select at least one risk category")
        return
    
    with st.spinner("Analyzing regulatory coverage gaps..."):
        gap_result = query_engine.analyze_gaps(selected_risks_for_gaps, selected_jurisdictions)
    
    # Coverage summary
    st.subheader("📊 Coverage Summary")
    
    summary = gap_result['coverage_summary']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Combinations Analyzed", summary['total_combinations_analyzed'])
    with col2:
        st.metric("🕳️ Gaps Identified", summary['gaps_identified'])
    with col3:
        st.metric("📈 Coverage Percentage", f"{summary['coverage_percentage']}%")
    with col4:
        high_severity = summary.get('high_severity_gaps', 0)
        st.metric("🚨 High Severity Gaps", high_severity)
    
    # Coverage matrix heatmap
    st.subheader("🎯 Coverage Matrix")
    
    coverage_matrix = gap_result['coverage_matrix']
    
    # Convert to DataFrame for heatmap
    heatmap_data = []
    for risk in selected_risks_for_gaps:
        row = []
        for jurisdiction in selected_jurisdictions:
            score = coverage_matrix.get(risk, {}).get(jurisdiction, {}).get('coverage_score', 0)
            row.append(score)
        heatmap_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=selected_jurisdictions,
        y=selected_risks_for_gaps,
        colorscale='RdYlGn',
        text=[[f"{score}%" for score in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Regulatory Coverage Heatmap (%)",
        xaxis_title="Jurisdictions",
        yaxis_title="Risk Categories"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Gap details
    st.subheader("🔍 Gap Details")
    
    gaps = gap_result['gaps_identified']
    if gaps:
        gap_df = pd.DataFrame(gaps)
        
        # Filter options
        severity_filter = st.selectbox(
            "Filter by Severity:",
            ["All", "HIGH", "MEDIUM"],
            help="Filter gaps by severity level"
        )
        
        if severity_filter != "All":
            gap_df = gap_df[gap_df['severity'] == severity_filter]
        
        if not gap_df.empty:
            st.dataframe(
                gap_df[['risk_type', 'jurisdiction', 'coverage_score', 'severity']],
                use_container_width=True
            )
        else:
            st.info(f"No {severity_filter} severity gaps found")
    else:
        st.success("🎉 No significant gaps identified! All risk-jurisdiction combinations have adequate coverage.")

def show_requirements_mapping(selected_risk, query_engine, corpus_summary):
    """Show detailed requirements mapping for a risk"""
    
    st.header(f"📋 Requirements Mapping: {selected_risk}")
    
    with st.spinner("Mapping requirements across all jurisdictions..."):
        mapping_result = query_engine.map_requirements(selected_risk)
    
    # Summary metrics
    summary = mapping_result['summary']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Total Requirements", summary['total_requirements'])
    with col2:
        st.metric("🌍 Jurisdictions Covered", len(mapping_result['jurisdictions']))
    with col3:
        st.metric("📚 Regulatory Sources", len(summary['by_source']))
    
    # Requirement type distribution
    st.subheader("📊 Requirement Type Distribution")
    
    type_data = summary['by_type']
    if type_data:
        fig = px.pie(
            values=list(type_data.values()),
            names=list(type_data.keys()),
            title="Distribution of Requirement Types"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Jurisdiction breakdown
    st.subheader("🌍 Jurisdiction Breakdown")
    
    for jurisdiction, data in mapping_result['jurisdictions'].items():
        if data['requirements']:  # Only show jurisdictions with requirements
            with st.expander(f"{jurisdiction} - {len(data['requirements'])} requirements", expanded=False):
                
                st.write(f"**Sources:** {', '.join(data['sources'])}")
                st.write(f"**Document Matches:** {data['total_matches']}")
                
                # Show requirement type breakdown
                if data['requirement_types']:
                    type_breakdown = ", ".join([f"{rtype}: {count}" for rtype, count in data['requirement_types'].items()])
                    st.write(f"**By Type:** {type_breakdown}")
                
                # Show sample requirements
                st.write("**Sample Requirements:**")
                for i, req in enumerate(data['requirements'][:3], 1):
                    req_icon = "🔴" if req['type'] == 'MANDATORY' else "🟡" if req['type'] == 'RECOMMENDED' else "🟢"
                    st.write(f"{req_icon} **{req['type']}:** {req['text'][:200]}...")
                    st.caption(f"Source: {req['source']} | Document: {req['document']}")
    
    # Detailed table
    st.subheader("📑 Detailed Requirements Table")
    
    # Flatten all requirements into a single table
    all_requirements = []
    for jurisdiction, data in mapping_result['jurisdictions'].items():
        for req in data['requirements']:
            all_requirements.append({
                'Jurisdiction': jurisdiction,
                'Type': req['type'],
                'Requirement': req['text'][:100] + "..." if len(req['text']) > 100 else req['text'],
                'Source': req['source'],
                'Document': req['document'],
                'Page': req['page']
            })
    
    if all_requirements:
        req_df = pd.DataFrame(all_requirements)
        
        # Add filters
        col1, col2 = st.columns(2)
        with col1:
            jurisdiction_filter = st.selectbox(
                "Filter by Jurisdiction:",
                ["All"] + list(req_df['Jurisdiction'].unique())
            )
        with col2:
            type_filter = st.selectbox(
                "Filter by Type:",
                ["All"] + list(req_df['Type'].unique())
            )
        
        # Apply filters
        filtered_df = req_df.copy()
        if jurisdiction_filter != "All":
            filtered_df = filtered_df[filtered_df['Jurisdiction'] == jurisdiction_filter]
        if type_filter != "All":
            filtered_df = filtered_df[filtered_df['Type'] == type_filter]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Requirements as CSV",
            data=csv,
            file_name=f"{selected_risk}_requirements.csv",
            mime="text/csv"
        )
    else:
        st.info("No requirements found for this risk category")

if __name__ == "__main__":
    main()