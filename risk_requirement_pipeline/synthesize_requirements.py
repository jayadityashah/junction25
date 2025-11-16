"""
Synthesize and deduplicate requirements for a document using Gemini
"""

import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


def synthesize_document_requirements(
    requirements: List[Dict[str, Any]],
    document_name: str,
    llm: ChatGoogleGenerativeAI
) -> List[Dict[str, Any]]:
    """
    Use Gemini to synthesize/consolidate requirements for a document.
    Removes redundancies, merges related requirements, keeps only significant ones.
    """
    
    if len(requirements) <= 5:
        # Not enough to meaningfully synthesize
        return requirements
    
    # Format requirements for Gemini
    req_list = ""
    for idx, req in enumerate(requirements, 1):
        req_list += f"\n{idx}. [{req['risk_category']}] {req['requirement_text']}\n   Pages: {req['start_page']}-{req['end_page']}\n"
    
    prompt = f"""You are synthesizing regulatory requirements extracted from: {document_name}

These {len(requirements)} requirements were extracted from different pages. Your task:
1. REMOVE duplicates or near-duplicates
2. MERGE requirements that describe the same obligation but from different pages
3. KEEP all DISTINCT significant requirements
4. Ensure each final requirement is complete and self-contained

REQUIREMENTS TO SYNTHESIZE:
{req_list}

OUTPUT FORMAT - Return ONLY valid JSON array:
[
  {{
    "requirement": "Final synthesized requirement text (merge related ones, keep distinct ones)",
    "risk_category": "EXACT_RISK_CATEGORY",
    "relevant_pages": [all_page_numbers_from_merged_requirements]
  }}
]

Guidelines:
- If 2-3 requirements describe same obligation → MERGE into one comprehensive requirement
- If requirements are truly distinct → KEEP as separate
- If requirement is redundant/trivial → REMOVE
- Preserve the most complete version of each requirement
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        
        # Clean markdown
        if content.startswith('```'):
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
        
        # Parse JSON
        synthesized = json.loads(content)
        
        # Validate
        validated = []
        for req in synthesized:
            if 'requirement' in req and 'risk_category' in req:
                # Handle page ranges - prefer relevant_pages array, fallback to start/end, then overall range
                if 'relevant_pages' in req and req['relevant_pages']:
                    # Convert page array to min/max range
                    pages = [p for p in req['relevant_pages'] if isinstance(p, int)]
                    if pages:
                        start_page = min(pages)
                        end_page = max(pages)
                    else:
                        start_page = requirements[0]['start_page']
                        end_page = requirements[-1]['end_page']
                elif 'start_page' in req and 'end_page' in req:
                    start_page = req['start_page']
                    end_page = req['end_page']
                else:
                    # Fallback to overall range of all requirements
                    start_page = requirements[0]['start_page']
                    end_page = requirements[-1]['end_page']

                # Store specific pages in extraction_window if available
                extraction_window = None
                if 'relevant_pages' in req and req['relevant_pages']:
                    extraction_window = json.dumps(sorted(req['relevant_pages']))

                validated.append({
                    'requirement_text': req['requirement'],
                    'risk_category': req['risk_category'],
                    'start_page': start_page,
                    'end_page': end_page,
                    'extraction_window': extraction_window
                })
        
        return validated
        
    except Exception as e:
        print(f"  ⚠️  Error synthesizing requirements: {e}")
        print(f"      Keeping original {len(requirements)} requirements")
        return requirements

