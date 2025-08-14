import os
import json
import logging
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, field_validator, model_validator
from typing import ClassVar

logger = logging.getLogger(__name__)


class Segment(BaseModel):
    type: str
    data: str
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ['text', 'formula']:
            raise ValueError(f"Invalid segment type: {v}. Must be 'text' or 'formula'")
        return v


class SegmentMapping(BaseModel):
    segments: list[Segment]
    _expected_segments: ClassVar[list[dict[str, str]]] = []
    
    @classmethod
    def set_expected_structure(cls, gt_segments: list[dict[str, str]]) -> None:
        cls._expected_segments = gt_segments
    
    @model_validator(mode='after')
    def validate_structure(self):
        if not self._expected_segments:
            return self
            
        expected_count = len(self._expected_segments)
        actual_count = len(self.segments)
        
        if actual_count != expected_count:
            raise ValueError(f"Segment count mismatch: expected {expected_count}, got {actual_count}")
            
        for i, (expected, actual) in enumerate(zip(self._expected_segments, self.segments)):
            if expected['type'] != actual.type:
                raise ValueError(f"Type mismatch at index {i}: expected '{expected['type']}', got '{actual.type}'")
        
        return self

def create_llm_mapping_prompt(gt_segments: list[dict[str, str]], markdown_content: str) -> str:
    """Create a detailed prompt for the LLM to map segments."""
    
    # Create a structured representation of ground truth
    gt_structure = []
    for i, segment in enumerate(gt_segments):
        if segment["type"] == "text":
            gt_structure.append(f"[{i}] TEXT: {segment['data']}")
        else:
            gt_structure.append(f"[{i}] FORMULA: {segment['data']}")
    
    prompt = f"""You are an expert document analyst tasked with precisely mapping parsed markdown content back to its original ground truth structure.

GROUND TRUTH STRUCTURE (sequential order):
{"\n".join(gt_structure)}

PARSED MARKDOWN CONTENT:
```markdown
{markdown_content}
```

CRITICAL MAPPING REQUIREMENTS:
Your task is to create a 1:1 mapping between the parsed markdown and the ground truth segments. Follow these rules strictly:

1. OUTPUT FORMAT: Return exactly a JSON array with {len(gt_segments)} elements in the same order as ground truth
2. STRUCTURE: Each element must have "type" (text/formula) and "data" fields
3. COMPLETENESS: Every character, symbol, word, and sentence from the parsed markdown MUST be assigned to exactly one ground truth segment. Include any leading/trailing symbols.
4. BEST FIT MAPPING: Even heavily deformed or incomplete content  must be assigned to the most similar ground truth segment 
5. MISSING CONTENT: If a ground truth segment has no corresponding content in the markdown, use empty string "" for "data"
6. FORMULA HANDLING: For formula segments, extract the complete LaTeX formula with all delimiters (e.g., $...$ or $$...$$)
7. NO ORPHANED CONTENT: Do not leave any content from the parsed markdown unmapped
8. LATEX BACKSLASH PRESERVATION: Copy LaTeX backslashes EXACTLY as they appear in the markdown. Do NOT add extra backslashes for JSON escaping.

MAPPING STRATEGY:
- Read through the parsed markdown sequentially
- For each piece of content, determine which ground truth segment it best corresponds to
- When in doubt, choose the chronologically closest ground truth segment

Return ONLY the JSON array, no explanations or additional text."""

    return prompt

def extract_segments_using_llm(
    gt_segments: list[dict[str, str]],
    markdown_content: str,
    model: str
) -> list[dict[str, str]]:
    """Use LLM to extract and map segments from markdown to ground truth structure."""

    if os.getenv("LLM_PROXY_URL"):
        client = OpenAI(base_url=os.getenv("LLM_PROXY_URL"),
                        api_key=os.getenv("LLM_PROXY_API_KEY"))
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Set expected structure in the Pydantic model
    SegmentMapping.set_expected_structure(gt_segments)
    
    prompt = create_llm_mapping_prompt(gt_segments, markdown_content)

    try:
        response = client.responses.parse(
            model=model,
            input=prompt,
            text_format=SegmentMapping
        )
        
        # With structured output, access the parsed object directly
        return [{"type": seg.type, "data": seg.data} for seg in response.output_parsed.segments]
            
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise

def verify_content_completeness(result_segments: list[dict[str, str]], 
                               markdown_content: str) -> bool:
    """
    Verify that all content from markdown is captured in result segments.
    
    Args:
        result_segments: Mapped result segments  
        markdown_content: Original markdown content
        
    Returns:
        True if content completeness check passes, False otherwise
    """
    # Collect all text from result segments
    result_text = "".join(seg["data"] for seg in result_segments if seg["data"])
    
    # Remove whitespace and newlines for comparison
    md_clean = "".join(markdown_content.split())
    result_clean = "".join(result_text.split())
    
    if len(md_clean) != len(result_clean):
        diff = len(md_clean) - len(result_clean)
        logger.warning(f"Character count mismatch: Markdown={len(md_clean)}, Result={len(result_clean)}, Diff={diff}")
        return False

    return True

def extract_segments_from_md_llm(
    gt_json_path: Path,
    input_md_path: Path,
    output_json_path: Path,
    model: str = "gpt-5-mini"
) -> None:
    """
    Extract segments from markdown using LLM to map to ground truth structure.
    
    Args:
        gt_json_path: Path to ground truth JSON file
        input_md_path: Path to input markdown file
        output_json_path: Path to output JSON file
        model: OpenAI model to use
    """

    # Load ground truth data
    with open(gt_json_path, 'r', encoding='utf-8') as f:
        gt_segments = json.load(f)

    # Load markdown content
    with open(input_md_path, 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    # Extract segments using LLM
    result_segments = extract_segments_using_llm(
        gt_segments,
        markdown_content,
        model
    )

    # Verify content completeness
    verification_passed = verify_content_completeness(result_segments, markdown_content)

    # Save result
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_segments, f, indent=2, ensure_ascii=False)
    
    # Log completion status
    if verification_passed:
        logger.info(f"Successfully extracted and saved to: {output_json_path}")
    else:
        logger.warning(f"Extracted {len(result_segments)} segments with verification issues, saved to: {output_json_path}")
