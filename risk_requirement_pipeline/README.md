# Risk Requirement Pipeline

Custom requirement extraction and relationship analysis pipeline for financial regulation documents.

## Overview

This pipeline extracts regulatory requirements from financial documents and identifies **document-level** overlaps and contradictions, focusing on key risk categories.

### Key Concept: Document-Level Relationships

- **Nodes = Documents** (e.g., Basel Framework, CRR, EBA Guidelines)
- **Edges = Relationships** between documents (overlaps or contradictions)
- **One edge can connect 2, 3, 4, or more documents** (multi-way relationships)
- **Requirements replace paragraphs** as the evidence for each relationship
  - Old: Documents connected via `relevant_paragraphs` with page/paragraph references
  - New: Documents connected via `requirements` with full requirement text and page ranges

### Architecture Flow

```
1. EXTRACT REQUIREMENTS (100-page chunks + concurrent processing + synthesis)
   Pages 1-100 → Extract reqs → [Raw A1, A2, A3, A4, A5]
   Pages 101-200 → Extract reqs → [Raw A6, A7, A8]     ← Concurrent (up to 10)
   Pages 201-300 → Extract reqs → [Raw A9, A10]        ← Concurrent (up to 10)

   SYNTHESIS BY RISK: Group by risk category, then consolidate each group
   Credit Risk: [A1, A6, A9] → Gemini → [Final Credit Req 1, Final Credit Req 2]
   Market Risk: [A2, A7] → Gemini → [Final Market Req 1]
   Operational Risk: [A3, A8, A10] → Gemini → [Final Ops Req 1, Final Ops Req 2]

   Document A → [Credit Req 1, Market Req 1, Ops Req 1, Ops Req 2] (synthesized by risk)
   Document B → [Req B1, Req B2] (synthesized by risk)
   Document C → [Req C1, Req C2, Req C3] (synthesized by risk)

2. GROUPED RELATIONSHIP ANALYSIS (Gemini analyzes all requirements together)
   ALL requirements for risk category → Gemini finds groups of overlaps/contradictions

   Example for "Credit Risk":
   Requirements: [A1, A2, A3, B1, B2, C1, C2, C3]

   Gemini identifies:
   - Group 1: A1, B1, C1 (OVERLAP - all define capital ratios)
   - Group 2: A2, B2 (CONTRADICTION - conflicting provisioning rules)
   - Group 3: A3, C3 (OVERLAP - loan classification standards)

3. CONVERT TO DOCUMENT-LEVEL RELATIONSHIPS
   Group 1: A1+B1+C1 → Documents A↔B↔C (capital ratios overlap)
   Group 2: A2+B2 → Documents A↔B (provisioning contradiction)
   Group 3: A3+C3 → Documents A↔C (classification overlap)

   Result: Direct multi-document relationships without pairwise-then-merge complexity

4. EXPORT & VISUALIZE
   JSON format matching analysis_results.json
   Graph visualization with multi-document edges
```

## Components

### 1. `requirement_extraction_pipeline.py`
Extracts requirements from documents using 100-page chunks with concurrent processing:
- **Non-Overlapping Chunks**: 100 pages at a time (pages 1-100, 101-200, 201-300, etc.)
- **Concurrent Processing**: Up to 10 chunks processed simultaneously for speed
- **No Previous Context**: Each chunk processed independently (no requirement context from other chunks)
- **Strong Risk Filtering**: Only extracts requirements strongly related to risk categories
- **One Risk Per Requirement**: Picks the MOST SIGNIFICANT risk category for each requirement
- **Document Context**: Tells Gemini which document and which pages it's analyzing
- **Precise Page Ranges**: Gemini specifies exact page numbers as arrays for each requirement based on page markers
- **Synthesis Step**: After all chunks, Gemini consolidates requirements by risk category (removes redundancies, merges related ones)
- **LLM**: Uses gemini-2.5-flash-lite-preview-09-2025

### 2. `requirement_relationship_pipeline.py`
Finds **document-level** relationships via grouped requirement analysis:
- **Grouped Analysis (Gemini)**: Analyzes ALL requirements for a risk category together in one API call
- **Direct Group Identification**: Gemini identifies GROUPS of overlapping/contradicting requirements from DIFFERENT documents
- **Relationship Types**: OVERLAP (similar requirements) or CONTRADICTION (conflicting requirements)
- **Cross-Document Only**: Groups must contain requirements from different documents (never same document)
- **Document-Level Output**: Each relationship connects 2+ documents directly from Gemini groups
- **Result**: Multi-way relationships like A↔B↔C when Gemini identifies shared requirement topics
- **Separated by Risk**: Each risk category processed independently
- **No Rate Limiting**: Designed for higher API quotas
- **Simplified Architecture**: No complex pairwise-then-merge logic

### 3. `synthesize_requirements.py`
Consolidates requirements per document:
- Removes duplicates and near-duplicates
- Merges requirements describing same obligation
- Keeps all distinct significant requirements
- Called automatically at end of document extraction

### 4. `export_for_frontend.py`
Exports analysis results in frontend-compatible JSON format:
- Groups relationships by risk category
- Each relationship connects 2+ documents (can be multi-way like A↔B↔C)
- For each document: includes all relevant requirements involved
- Output structure matches analysis_results.json format (requirements replace relevant_paragraphs)

### 5. `run_full_pipeline.py`
Master script that orchestrates the complete pipeline in sequence

```bash
# Run full pipeline (incremental by default)
python run_full_pipeline.py

# Force reprocess everything
python run_full_pipeline.py --force

# Test with limited documents
python run_full_pipeline.py 3
```

## Database Schema

### Tables Created
- `requirements`: Stores extracted requirements with risk category and specific page lists
- `requirement_relationships`: Stores overlap/contradiction relationships
- `relationship_requirements`: Junction table linking requirements to relationships

### Incremental Processing
The pipeline automatically skips documents that have already been processed (have requirements in the database). Use `--force` flag to reprocess all documents.

## Usage

### Step 1: Extract Requirements
```bash
# Extract from all documents (incremental - skips already processed)
python risk_requirement_pipeline/requirement_extraction_pipeline.py

# Force reprocess all documents (clears and re-does everything)
python risk_requirement_pipeline/requirement_extraction_pipeline.py --force

# Test with limited documents
python risk_requirement_pipeline/requirement_extraction_pipeline.py 5

# Combined: reprocess first 3 documents
python risk_requirement_pipeline/requirement_extraction_pipeline.py 3 --force
```

### Step 2: Find Relationships
```bash
python risk_requirement_pipeline/requirement_relationship_pipeline.py
```

### Step 3: Export for Frontend
```bash
python risk_requirement_pipeline/export_for_frontend.py
```

## Configuration

Risk categories are defined in:
`risk-graphrag-project/config/risk_categories.yaml`

## Environment Variables

Requires `GEMINI_API_KEY` in `.env` file.

## Output Structure

Matches `analysis_results.json` format:

```json
{
  "risk_categories": [
    {
      "risk_name": "Credit Risk",
      "overlaps": [
        {
          "id": "credit-overlap-1",
          "documents": [
            {
              "filename": "DocA.json",
              "requirements": [
                {
                  "text": "Banks must maintain minimum capital ratios...",
                  "pages": [10, 12, 15],
                  "risk_category": "CREDIT_RISK"
                }
              ]
            },
            {
              "filename": "DocB.json",
              "requirements": [
                {
                  "text": "Capital adequacy requirements for credit institutions...",
                  "pages": [45, 46],
                  "risk_category": "CREDIT_RISK"
                }
              ]
            },
            {
              "filename": "DocC.json",
              "requirements": [...]
            }
          ],
          "reason": "All three documents define capital adequacy requirements for credit risk..."
        }
      ],
      "contradictions": [...]
    }
  ]
}
```

**Key differences from paragraph-based approach:**

| Aspect | Paragraph-Based (OLD) | Requirements-Based (NEW) |
|--------|----------------------|--------------------------|
| Evidence | `relevant_paragraphs: [{page: 45, paragraph_index: 3}]` | `requirements: [{text: "Full text...", pages: [45, 46, 47]}]` |
| Extraction | Text chunking | LLM extracts structured requirements |
| Context | Limited to chunk | 100-page chunks + concurrent processing |
| Finding Relationships | Embedding similarity | LLM grouped analysis (all reqs together) |
| Merging | N/A | Direct group identification |
| Multi-document edges | Manual | Automatic via Gemini groups |
| Risk Focus | Post-processing | Strong filtering + one significant risk per req |
| Page Ranges | Chunk ranges | Gemini specifies exact page lists (no ranges, just specific pages) |
| Duplicates | Possible | Independent chunk processing |
| Performance | Sequential | 10 concurrent API calls |

## Key Benefits

✅ **Full requirement text** - Better evidence than paragraph indices
✅ **Multi-document relationships** - One edge can connect 3+ documents
✅ **Synthesis step** - Gemini consolidates requirements per document
✅ **Grouped relationship analysis** - Gemini analyzes all requirements together (more efficient than pairwise)
✅ **Concurrent processing** - Up to 10 simultaneous API calls for 10x faster extraction
✅ **100-page chunks** - More context per API call, better LLM understanding
✅ **Strong risk filtering** - Only extracts highly relevant requirements
✅ **Specific page lists** - Gemini specifies exact page arrays for each requirement (no misleading ranges)
✅ **No previous context dependency** - Each chunk processed independently
✅ **Simplified architecture** - Direct group identification instead of pairwise-then-merge
✅ **No embeddings** - Pure LLM-based analysis
✅ **No rate limiting delays** - Fast processing for higher API quotas
✅ **Incremental processing** - Automatically skips already processed documents

