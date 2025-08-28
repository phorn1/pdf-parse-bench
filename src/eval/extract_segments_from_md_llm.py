import os
import json
import logging
from pathlib import Path
from openai import OpenAI
from pydantic import BaseModel, field_validator, model_validator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

logger = logging.getLogger(__name__)





@dataclass
class SegmentExtractionJob:
    gt_json_path: Path
    input_md_path: Path
    output_json_path: Path
    model: str = "gpt-5-mini"


class ParallelSegmentExtractor:
    """Parallel segment extraction processor similar to ParallelLaTeXPDFGenerator."""
    
    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        
    def _process_single_job(self, job: SegmentExtractionJob) -> tuple[bool, str]:
        """Process a single segment extraction job."""
        try:
            # Load ground truth data
            with open(job.gt_json_path, 'r', encoding='utf-8') as f:
                gt_segments = json.load(f)

            # Load markdown content
            with open(job.input_md_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            # Extract segments using LLM
            result_segments = extract_segments_using_llm(
                gt_segments,
                markdown_content,
                job.model
            )

            # Verify content completeness
            verification_result = verify_content_completeness(result_segments, markdown_content)

            # Save result
            with open(job.output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result_segments, f, indent=2, ensure_ascii=False)
            
            job_name = f"{job.input_md_path.parent.name}/{job.input_md_path.parent.parent.name}"
            if verification_result["passed"]:
                return True, f"✅ {job_name}"
            else:
                return True, f"⚠️ {job_name}: Character count mismatch (original: {verification_result['original_count']}, extracted: {verification_result['extracted_count']}, diff: {verification_result['diff']})"
            
        except Exception as e:
            error_msg = f"✗ {job.input_md_path.parent.name}/{job.input_md_path.parent.parent.name}: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}"
            return False, error_msg
    
    def extract_segments_parallel(self, jobs: list[SegmentExtractionJob]):
        """Extract segments in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_job = {executor.submit(self._process_single_job, job): job for job in jobs}
            
            for future in as_completed(future_to_job):
                success, message = future.result()
                yield success, message


def create_llm_mapping_prompt(gt_segments: list[dict[str, str]], markdown_content: str) -> str:
    """Create a detailed prompt for the LLM to map segments."""
    
    # Create a structured representation of ground truth
    gt_structure = []
    for i, segment in enumerate(gt_segments):
        if segment["type"] == "text":
            gt_structure.append(f"[{i}] TEXT: {segment['data']}")
        elif segment["type"] == "inline-formula":
            gt_structure.append(f"[{i}] INLINE-FORMULA: {segment['data']}")
        elif segment["type"] == "display-formula":
            gt_structure.append(f"[{i}] DISPLAY-FORMULA: {segment['data']}")
    
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
2. STRUCTURE: Each element must have "type" (text/inline-formula/display-formula) and "data" fields
3. COMPLETENESS: Every character, symbol, word, and sentence from the parsed markdown MUST be assigned to exactly one ground truth segment. Include any leading/trailing symbols.
4. BEST FIT MAPPING: Even heavily deformed or incomplete content  must be assigned to the most similar ground truth segment 
5. MISSING CONTENT: If a ground truth segment has no corresponding content in the markdown, use empty string "" for "data"
6. FORMULA HANDLING: For formula segments, extract the complete LaTeX formula with all delimiters (e.g., $...$ or $$...$$)
7. FORMULA SPLITTING: When multiple individual formulas in ground truth are combined into one formula block in the parsed markdown, split them by identifying logical formula boundaries (typically separated by \\\\ or line breaks). Each formula should be mapped to its corresponding ground truth segment with appropriate delimiters.
8. NO ORPHANED CONTENT: Do not leave any content from the parsed markdown unmapped
9. LATEX BACKSLASH PRESERVATION: Copy LaTeX backslashes EXACTLY as they appear in the markdown. Do NOT add extra backslashes for JSON escaping.

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
    
    # ========== CREATE THREAD-SAFE VALIDATION CLASS ==========

    class Segment(BaseModel):
        type: str
        data: str

        @field_validator('type')
        @classmethod
        def validate_type(cls, v: str) -> str:
            valid_types = ['text', 'inline-formula', 'display-formula']
            if v not in valid_types:
                raise ValueError(f"Invalid segment type: {v}. Must be one of {valid_types}")
            return v

    class SegmentMapping(BaseModel):
        """Thread-safe segment mapping with validation for this specific job."""
        segments: list[Segment]
        
        @model_validator(mode='after')
        def validate_structure(self):
            expected_count = len(gt_segments)
            actual_count = len(self.segments)
            
            if actual_count != expected_count:
                raise ValueError(f"Segment count mismatch: expected {expected_count}, got {actual_count}")
                
            for i, (expected, actual) in enumerate(zip(gt_segments, self.segments)):
                if expected['type'] != actual.type:
                    raise ValueError(f"Type mismatch at index {i}: expected '{expected['type']}', got '{actual.type}'")
            
            return self
    
    # ========== SETUP LLM CLIENT ==========
    
    if os.getenv("LLM_PROXY_URL"):
        client = OpenAI(base_url=os.getenv("LLM_PROXY_URL"),
                        api_key=os.getenv("LLM_PROXY_API_KEY"))
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ========== EXTRACT AND VALIDATE SEGMENTS ==========
    
    prompt = create_llm_mapping_prompt(gt_segments, markdown_content)
    
    try:
        response = client.responses.parse(
            model=model,
            input=prompt,
            text_format=SegmentMapping
        )

        return [{"type": seg.type, "data": seg.data} for seg in response.output_parsed.segments]

    except Exception as e:
        raise

def verify_content_completeness(result_segments: list[dict[str, str]], 
                               markdown_content: str) -> dict[str, int | bool]:
    """
    Verify that all content from markdown is captured in result segments.
    
    Args:
        result_segments: Mapped result segments  
        markdown_content: Original markdown content
        
    Returns:
        Dictionary containing verification results with keys:
        - passed: True if content completeness check passes, False otherwise
        - original_count: Character count of original markdown (excluding whitespace)
        - extracted_count: Character count of extracted segments (excluding whitespace)
        - diff: Difference between original and extracted counts
    """
    # Collect all text from result segments
    result_text = "".join(seg["data"] for seg in result_segments if seg["data"])
    
    # Remove whitespace and newlines for comparison
    md_clean = "".join(markdown_content.split())
    result_clean = "".join(result_text.split())
    
    original_count = len(md_clean)
    extracted_count = len(result_clean)
    diff = original_count - extracted_count
    
    return {
        "passed": original_count == extracted_count,
        "original_count": original_count,
        "extracted_count": extracted_count,
        "diff": diff
    }

