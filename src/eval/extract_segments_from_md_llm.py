import os
import json
import logging
from pathlib import Path
from openai import OpenAI

logger = logging.getLogger(__name__)

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

    prompt = create_llm_mapping_prompt(gt_segments, markdown_content)

    try:
        response = client.responses.create(
            model=model,
            input=prompt,
            text={
                "verbosity": "low"
            },
            reasoning={
                "effort": "minimal"
            }

        )
        
        result_text = response.text.strip()
        
        # Try to parse JSON from the response
        try:
            # Remove potential markdown code blocks
            if result_text.startswith("```json"):
                logger.warning("LLM response contained '```json' markdown formatting, stripping it")
                result_text = result_text[7:]
            if result_text.endswith("```"):
                logger.warning("LLM response contained closing '```' markdown formatting, stripping it")
                result_text = result_text[:-3]
            if result_text.startswith("```"):
                logger.warning("LLM response contained opening '```' markdown formatting, stripping it")
                result_text = result_text[3:]
            result_text = result_text.strip()
            
            # Try direct parsing first
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError as e:
                # If parsing fails due to escape sequences, try fixing common LaTeX escapes
                logger.info("Direct JSON parsing failed, attempting to fix LaTeX escapes")
                
                # Replace problematic single backslashes with double backslashes in LaTeX contexts
                import re
                
                # Fix backslashes within formula data fields
                def fix_latex_escapes(match):
                    content = match.group(1)
                    # Replace single backslashes with double backslashes, but avoid already escaped ones
                    content = re.sub(r'(?<!\\)\\(?!\\)', r'\\\\', content)
                    return f'"data": "{content}"'
                
                fixed_text = re.sub(r'"data":\s*"([^"]*(?:\\.[^"]*)*)"', fix_latex_escapes, result_text)
                result = json.loads(fixed_text)
            
            # Validate result structure
            if not isinstance(result, list) or len(result) != len(gt_segments):
                raise ValueError(f"Result length {len(result)} doesn't match ground truth length {len(gt_segments)}")
            
            # Ensure all elements have required fields and correct types
            for i, item in enumerate(result):
                if not isinstance(item, dict) or "type" not in item or "data" not in item:
                    raise ValueError(f"Invalid structure at index {i}")
                    
                # Ensure type matches ground truth
                if item["type"] != gt_segments[i]["type"]:
                    logger.warning(f"Type mismatch at index {i}: expected {gt_segments[i]['type']}, got {item['type']}")
                    item["type"] = gt_segments[i]["type"]  # Force correct type
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"LLM response: {result_text}")
            raise ValueError("LLM returned invalid JSON")
            
    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        raise

def verify_mapping_completeness(gt_segments: list[dict[str, str]], 
                               result_segments: list[dict[str, str]], 
                               markdown_content: str) -> bool:
    """
    Verify that the mapping between ground truth and result segments is complete.
    
    Args:
        gt_segments: Original ground truth segments
        result_segments: Mapped result segments  
        markdown_content: Original markdown content
        
    Returns:
        True if verification passes, False otherwise
    """
    issues = []
    
    # Check segment count
    if len(gt_segments) != len(result_segments):
        issues.append(f"Segment count mismatch: GT={len(gt_segments)}, Result={len(result_segments)}")
        
    # Check types match
    for i, (gt_seg, res_seg) in enumerate(zip(gt_segments, result_segments)):
        if gt_seg["type"] != res_seg["type"]:
            issues.append(f"Type mismatch at index {i}: GT={gt_seg['type']}, Result={res_seg['type']}")
    
    # Collect all text from result segments
    result_text = "".join(seg["data"] for seg in result_segments if seg["data"])
    
    # Remove whitespace and newlines for comparison
    md_clean = "".join(markdown_content.split())
    result_clean = "".join(result_text.split())
    
    if len(md_clean) != len(result_clean):
        diff = len(md_clean) - len(result_clean)
        issues.append(f"Character count mismatch: Markdown={len(md_clean)}, Result={len(result_clean)}, Diff={diff}")
        

    if issues:
        logger.warning("Verification issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
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

    # Verify mapping completeness
    verification_passed = verify_mapping_completeness(gt_segments, result_segments, markdown_content)

    # Save result
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result_segments, f, indent=2, ensure_ascii=False)
    
    # Log completion status
    if verification_passed:
        logger.info(f"Successfully extracted and saved to: {output_json_path}")
    else:
        logger.warning(f"Extracted {len(result_segments)} segments with verification issues, saved to: {output_json_path}")
