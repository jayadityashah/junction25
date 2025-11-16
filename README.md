# EU Financial Regulatory Risk Analysis Platform

A comprehensive system for analyzing financial regulations to identify **overlaps**, **contradictions**, and **relationships** across 21 risk categories in EU, Finnish, and international regulatory documents.

## 🎯 System Overview

This platform combines **AI-powered requirement extraction**, **relationship analysis**, and **GraphRAG knowledge graphs** to provide interactive exploration of regulatory compliance across multiple jurisdictions and risk categories.

### Core Capabilities

- **🔍 Multi-Risk Analysis**: 21 risk categories including Credit Risk, Market Risk, Liquidity Risk, Operational Risk, etc.
- **📊 Requirement Extraction**: LLM-based extraction of regulatory obligations from 98+ documents with 125k+ paragraphs
- **🔗 Relationship Detection**: Automatic identification of overlaps and contradictions between regulatory requirements
- **🌐 Interactive Visualization**: Network graphs showing document relationships and requirement conflicts
- **💬 GraphRAG Chat**: Natural language queries over the complete regulatory knowledge graph
- **📈 Multi-Dimensional Mapping**: 6+ dimensional analysis (risk types, jurisdictions, sources, requirement types, severity, temporal)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                   │
│  ┌───────────────────────┐    ┌──────────────────────────┐     │
│  │  legal_documents.db   │    │  Bronze Data (PDFs)      │     │
│  │  - 98 documents       │    │  - Original PDFs         │     │
│  │  - 125k+ paragraphs   │    │  - EU, Finnish, Basel    │     │
│  │  - Page mappings      │    │  - BRRD, CRD, CRR, etc.  │     │
│  └───────────────────────┘    └──────────────────────────┘     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PROCESSING LAYER                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 1: Requirement Extraction Pipeline                │   │
│  │  - 100-page chunking with context windows                │   │
│  │  - 200 parallel Gemini 2.5 Flash workers                 │   │
│  │  - Risk category classification (21 categories)          │   │
│  │  - Synthesis & deduplication per document/risk           │   │
│  │  Output: requirements table with page references         │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 2: Relationship Analysis Pipeline                 │   │
│  │  - Grouped analysis (not pairwise) via Gemini 2.5 Pro   │   │
│  │  - OVERLAP detection: complementary requirements         │   │
│  │  - CONTRADICTION detection: conflicting requirements     │   │
│  │  - Multi-document relationship mapping (3+ docs)         │   │
│  │  Output: requirement_relationships + junction tables     │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  STAGE 3: GraphRAG Knowledge Graph                       │   │
│  │  - Higher dimensional embeddings (6+ dimensions)         │   │
│  │  - Entity extraction (Risk, Jurisdiction, Requirement)   │   │
│  │  - Community detection (Leiden algorithm)                │   │
│  │  - Global/Local search capabilities                      │   │
│  │  Output: GraphRAG index + embeddings                     │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                 APPLICATION LAYER                                │
│  ┌──────────────────┐    ┌──────────────────────────────┐      │
│  │  Flask Backend   │    │  Frontend (HTML/JS/CSS)      │      │
│  │  (backend_api.py)│    │  - Risk category cards       │      │
│  │                  │    │  - Network graph (vis.js)    │      │
│  │  REST API:       │◄───┤  - Document viewer           │      │
│  │  /api/documents  │    │  - Requirement details       │      │
│  │  /api/stats      │    │  - GraphRAG chat interface   │      │
│  │  /api/graphrag   │    │  - Knowledge graph viz       │      │
│  └──────────────────┘    └──────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Legal documents database (98 docs, 125k+ paragraphs)
2. **Extraction**: Parallel LLM processing → Requirements with risk categories
3. **Relationships**: Grouped LLM analysis → Overlaps & Contradictions
4. **Export**: JSON output → `frontend/requirements_analysis.json`
5. **Serving**: Flask API + Interactive web interface
6. **Querying**: GraphRAG natural language queries over knowledge graph

---

## 🔄 Requirement Processing Pipeline

### Stage 1: Requirement Extraction

**Input**: Document pages + 21 risk categories from `risk_categories.yaml`

**Process**:
1. Load risk categories with synonyms and regulatory context
2. Chunk documents into 100-page windows (non-overlapping)
3. Concurrent extraction with 200 parallel Gemini 2.5 Flash workers
4. Synthesis by risk category (deduplicate, merge similar requirements)
5. Store in database with page references

**Configuration**:
- Model: `gemini-2.5-flash-lite` (fast, cost-effective)
- Temperature: 0.4 (factual extraction)
- Context: 100-page windows capture full regulatory context

**Extraction Criteria**:
- ✅ Mandatory/recommended regulatory obligation
- ✅ Specific and actionable
- ✅ Strongly relates to ONE risk category
- ✅ Substantial (not procedural)
- ❌ NOT definitions, examples, transitions

### Stage 2: Relationship Analysis

**Input**: All requirements grouped by risk category

**Process**:
1. Group requirements by risk category across all documents
2. Grouped analysis with Gemini 2.5 Pro (analyzes ALL together, not pairwise)
3. Detect OVERLAPS (complementary/compatible) and CONTRADICTIONS (conflicting)
4. Map requirement-level relationships to document-level relationships
5. Store in `requirement_relationships` and junction tables

**Configuration**:
- Model: `gemini-2.5-pro` (sophisticated reasoning)
- Temperature: 0.4
- Validation: Relationships must involve 2+ documents

**Detection Types**:
- **OVERLAP**: Compatible requirements that complement each other
- **CONTRADICTION**: Conflicting thresholds, timeframes, or methods

### Stage 3: Export for Frontend

**Input**: Requirements + Relationships from database

**Process**:
1. Retrieve all data grouped by risk category
2. Format as nested JSON with full requirement text and page lists
3. Write to `frontend/requirements_analysis.json`

**Output Structure**:
```json
{
  "risk_categories": [
    {
      "risk_name": "Credit Risk",
      "description": "...",
      "total_requirements": 42,
      "overlaps": [
        {
          "documents": [
            {
              "filename": "CELEX_32013R0575.di.json",
              "requirements": [
                {
                  "id": 1,
                  "text": "Full requirement text...",
                  "pages": [10, 12, 15]
                }
              ]
            }
          ],
          "reason": "Why these requirements overlap"
        }
      ],
      "contradictions": [...]
    }
  ]
}
```

---

## 🌐 GraphRAG Integration

The platform includes a complete GraphRAG implementation for advanced querying:

### Features

- **Higher Dimensional Mapping**: 20+ risk categories across 6+ dimensions
- **Multi-Jurisdictional Analysis**: EU, Finnish, and International coverage
- **Entity Extraction**: Risk_Category, Regulatory_Requirement, Jurisdiction, Implementation_Method
- **Community Detection**: Hierarchical clustering with Leiden algorithm
- **Interactive Queries**: Global search (entire graph) and Local search (specific entities)

### Dimensional Structure

1. **Risk Types**: 20 categories (Credit, Market, Liquidity, Operational, etc.)
2. **Jurisdictions**: EU, Finnish, International
3. **Regulatory Sources**: BRRD, CRD, CRR, EBA, MiFID, Basel, etc.
4. **Requirement Types**: Mandatory, Recommended, Guidance
5. **Severity Levels**: High, Medium, Low
6. **Temporal Dimension**: Publication dates, effective dates

### Query Examples

```python
# Query specific risk across jurisdictions
engine.query_risk_category('CREDIT_RISK', jurisdiction='EU', limit=10)

# Find conflicts between jurisdictions
engine.find_conflicts('LIQUIDITY_RISK', ['EU', 'FINNISH'])

# Map all requirements for a risk
engine.map_requirements('MARKET_RISK', output_format='summary')
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
cp env.example .env
```

Configure `.env` with:
```
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
```

### 2. Run the Complete Pipeline

```bash
# Run full pipeline (extraction + relationships + export)
python run_full_pipeline.py

# Run with limit (first 5 documents)
python run_full_pipeline.py 5

# Force reprocess all documents
python run_full_pipeline.py --force

# Skip extraction, use existing requirements
python run_full_pipeline.py --skip-extraction
```

### 3. Launch the Web Application

```bash
# Start Flask backend (serves API + frontend)
python backend_api.py
```

Access at: **http://localhost:5000**

---

## 📊 Web Interface Features

### 1. Risk Category Dashboard
- **21 risk category cards** sorted by total relationships
- **Metrics**: Requirements count, documents count, overlaps, contradictions
- **Color-coded**: Gradient backgrounds for visual differentiation

### 2. Interactive Network Graph (vis.js)
- **Document nodes**: Regulatory documents as nodes
- **Relationship edges**:
  - Blue solid lines = Overlaps
  - Red dashed lines = Contradictions
- **Click interactions**: View full requirement text and page references
- **Sidebar**:
  - Documents tab: List of all documents involved
  - Relationships tab: All overlaps and contradictions with evidence

### 3. GraphRAG Chat Interface
- **Natural language queries** over the complete knowledge graph
- **Context-aware responses** using global/local search
- **Example queries**:
  - "What are the credit risk requirements in EU regulations?"
  - "Compare liquidity risk requirements between EU and Finnish regulations"
  - "What are the capital requirements under CRR?"

### 4. Knowledge Graph Visualization
- **Interactive HTML visualization** of the GraphRAG knowledge graph
- **Entity and relationship explorer**
- **Community clusters** with hierarchical structure

---

## 📈 Performance Characteristics

| Stage | Processing | Speed | Cost | Model |
|-------|-----------|-------|------|-------|
| **Stage 1: Extraction** | Parallel (200 workers) | Fast | Low | Gemini 2.5 Flash Lite |
| **Stage 2: Relationships** | Sequential per category | Medium | Medium | Gemini 2.5 Pro |
| **Stage 3: Export** | Database queries | Fast | None | N/A |
| **GraphRAG Indexing** | Batch processing | Slow (one-time) | High | Custom embeddings |
| **Frontend Serving** | REST API | Real-time | None | N/A |

**Dataset Scale**:
- **Documents**: 98 regulatory documents
- **Paragraphs**: 125,172 text chunks
- **Risk Categories**: 21 distinct types
- **Jurisdictions**: EU, Finnish, International
- **Regulatory Sources**: 13 different authorities

**Top Risk Categories**:
- Credit Risk: 2,084 mentions
- Market Risk: 790 mentions
- Liquidity Risk: 563 mentions

---

## 🎯 Key Innovation Points

1. **Large Context Windows**: 100-page chunks capture full regulatory context
2. **Massive Parallelization**: 200 concurrent API calls for 200x speedup
3. **Grouped Analysis**: LLM analyzes ALL requirements together (not pairwise O(n²))
4. **Multi-document Relationships**: Can link 3+ documents (A↔B↔C)
5. **Synthesis Phase**: Deduplicates and consolidates per document per risk
6. **Specific Page Lists**: Exact page numbers, not misleading ranges
7. **Idempotent Pipeline**: Safe to re-run, skips already-processed documents
8. **Higher Dimensional GraphRAG**: 6+ dimensional embeddings for advanced queries
9. **Dual Interface**: Both structured graph view and natural language chat

---

## 📁 Project Structure

```
junction1/
├── backend_api.py                    # Flask REST API server
├── legal_documents.db                # SQLite database (documents, pages, paragraphs, requirements)
├── frontend/
│   ├── index.html                    # Main web interface
│   ├── app.js                        # Frontend logic (graph, chat, tabs)
│   ├── styles.css                    # Styling
│   ├── requirements_analysis.json    # Pipeline output (used by frontend)
│   └── graphrag_knowledge_graph.html # GraphRAG visualization
├── risk-graphrag-project/            # GraphRAG implementation
│   ├── config/risk_categories.yaml   # 21 risk categories with synonyms
│   ├── scripts/
│   │   ├── extract_corpus.py         # Extract 125k+ paragraphs
│   │   ├── dimension_mapper.py       # Higher dimensional embeddings
│   │   ├── risk_visualizer.py        # Visualization system
│   │   ├── query_engine.py           # Multi-dimensional queries
│   │   └── interactive_dashboard.py  # Streamlit dashboard
│   ├── output/                       # GraphRAG knowledge graph files
│   └── visualizations/               # Generated charts and graphs
├── risk_requirement_pipeline/
│   ├── requirement_extraction_pipeline.py      # Stage 1
│   ├── requirement_relationship_pipeline.py    # Stage 2
│   └── export_for_frontend.py                  # Stage 3
├── run_full_pipeline.py              # Orchestrator script
├── REQUIREMENT_PIPELINE_WORKFLOW.md  # Detailed pipeline documentation
└── requirements.txt                  # Python dependencies
```

---

## 🔧 API Endpoints

### Documents
- `GET /api/documents` - List all documents
- `GET /api/document/<id>` - Get document with all paragraphs
- `GET /api/document/metadata/<filename>` - Get document metadata and PDF path
- `GET /api/document/by-filename/<filename>` - Get document content by filename
- `POST /api/document/<filename>/paragraphs` - Get specific paragraphs by page/index

### Statistics
- `GET /api/stats` - Database statistics (document counts, categories)

### GraphRAG
- `POST /api/graphrag/chat` - Natural language query interface
- `GET /api/graphrag/visualization` - Serve knowledge graph HTML

### Static Files
- `GET /` - Main frontend interface
- `GET /<path>` - Frontend assets (JS, CSS)
- `GET /bronze_data/<path>` - Serve PDF files

---

## 📝 Configuration

### Risk Categories (`risk_categories.yaml`)

21 risk categories with:
- Primary terms and synonyms
- Regulatory context and definitions
- Examples of related requirements

Categories include:
- Credit Risk, Market Risk, Liquidity Risk
- Operational Risk, Compliance Risk, Strategic Risk
- Reputational Risk, Legal Risk, Model Risk
- Concentration Risk, Country Risk, Interest Rate Risk
- And 10+ more specialized categories

### Pipeline Parameters

Adjustable in pipeline scripts:
- **Chunk size**: 100 pages (Stage 1)
- **Concurrency**: 200 workers (Stage 1)
- **Temperature**: 0.4 (Stages 1 & 2)
- **Models**:
  - Extraction: `gemini-2.5-flash-lite`
  - Relationships: `gemini-2.5-pro`

---

## 🎨 Use Cases

### Regulatory Compliance
- **Gap Identification**: Find missing requirements across jurisdictions
- **Conflict Resolution**: Identify contradictory regulations
- **Requirement Mapping**: Complete coverage analysis for each risk type

### Risk Management
- **Cross-Jurisdictional Analysis**: Compare EU vs Finnish requirements
- **Implementation Planning**: Map requirements to remediation actions
- **Compliance Monitoring**: Track regulatory changes and updates

### Research & Analysis
- **Regulatory Evolution**: Temporal analysis of requirement changes
- **Comparative Studies**: Jurisdiction-specific risk treatment
- **Policy Development**: Evidence-based regulatory gap analysis

---

## 🔮 Future Enhancements

- **Real-time Updates**: Automatic regulatory document monitoring
- **Machine Learning**: Predictive conflict detection
- **API Integration**: RESTful endpoints for external systems
- **Advanced Analytics**: Temporal trend analysis and forecasting
- **Multi-language Support**: Extend beyond English translations
- **Export Capabilities**: PDF reports, Excel summaries

---

## 📞 Support & Documentation

For detailed pipeline workflow, see [REQUIREMENT_PIPELINE_WORKFLOW.md](REQUIREMENT_PIPELINE_WORKFLOW.md)

For GraphRAG specifics, see [risk-graphrag-project/PROJECT_SUMMARY.md](risk-graphrag-project/PROJECT_SUMMARY.md)
