import os
import json
from pathlib import Path
from openai import OpenAI
import instructor
from instructor.hooks import Hooks
from instructor.core import InstructorRetryException
from typing import Literal
from pydantic import BaseModel, model_validator, Field
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .formula_renderer import FormulaRenderer
from .formula_normalizer import normalize_segments_formulas


# ========== SEGMENT EXTRACTION ==========

@dataclass
class SegmentExtractionJob:
    gt_json_path: Path
    input_md_path: Path
    output_json_path: Path
    rendered_formulas_dir: Path | None = None
    model: str = "gpt-5-mini"


class ParallelSegmentExtractor:
    """Parallel segment extraction processor with integrated progress tracking."""

    def __init__(self, max_workers: int):
        self.max_workers = max_workers

    def _process_single_job(self, job: SegmentExtractionJob, console) -> bool:
        """Process a single segment extraction job.

        Returns:
            True if successful, False if error occurred
        """
        job_name = f"{job.input_md_path.parent.name}/{job.input_md_path.parent.parent.name}"

        try:
            # Load ground truth data
            with open(job.gt_json_path, 'r', encoding='utf-8') as f:
                gt_segments = json.load(f)

            # Load markdown content
            with open(job.input_md_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            # Extract segments using LLM with console for retry logging
            result_segments = extract_segments_using_llm(
                gt_segments,
                markdown_content,
                job.model,
                console=console,
                job_name=job_name
            )

            # Normalize formulas in extracted segments
            result_segments = normalize_segments_formulas(result_segments)

            # Render formulas if path is provided
            if job.rendered_formulas_dir is not None:
                renderer = FormulaRenderer()
                renderer.render_formulas_in_segments(result_segments, job.rendered_formulas_dir)

            # Verify content completeness
            verification_result = verify_content_completeness(result_segments, markdown_content)

            # Save result
            with open(job.output_json_path, 'w', encoding='utf-8') as f:
                json.dump(result_segments, f, indent=2, ensure_ascii=False)

            # Log result
            if verification_result["passed"]:
                console.print(f"   ✅ {job_name}")
            else:
                console.print(f"   ⚠️ {job_name}: Character count mismatch (original: {verification_result['original_count']}, extracted: {verification_result['extracted_count']}, diff: {verification_result['diff']})")

            return True

        except InstructorRetryException as e:
            # Final error message (details already logged via hooks during retries)
            console.print(f"   ❌ {job_name}: Failed after {e.n_attempts} attempts")
            return False
        except Exception as e:
            console.print(f"   ❌ {job_name}: {str(e)}")
            return False

    def extract_segments_parallel(self, jobs: list[SegmentExtractionJob]):
        """Extract segments in parallel using ThreadPoolExecutor with progress tracking."""
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Extracting segments...", total=len(jobs))

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_job = {executor.submit(self._process_single_job, job, progress.console): job for job in jobs}

                for future in as_completed(future_to_job):
                    future.result()  # Wait for completion, logging happens inside _process_single_job
                    progress.update(task, advance=1)


def create_llm_mapping_prompt(gt_segments: list[dict[str, str]], markdown_content: str) -> str:
    """Create a detailed prompt for the LLM to map segments."""

    # Calculate expected character count for verification
    md_clean = "".join(markdown_content.split())
    expected_char_count = len(md_clean)

    # Create structured representation of ground truth
    gt_structure = []
    for i, segment in enumerate(gt_segments):
        segment_type = segment["type"].upper()
        gt_structure.append(f"[{i}] {segment_type}: {segment['data']}")

    prompt = f"""You are an expert document analyst. Your task: map parsed markdown content back to the original ground truth segment structure.

CRITICAL CONSTRAINT:
The parsed markdown contains EXACTLY {expected_char_count} characters (excluding whitespace).
Your output MUST contain EXACTLY {expected_char_count} characters across all segments (excluding whitespace).

GROUND TRUTH STRUCTURE ({len(gt_segments)} segments in sequential order):
{"\n".join(gt_structure)}

PARSED MARKDOWN CONTENT:
```markdown
{markdown_content}
```

INSTRUCTIONS:

1. EXTRACT, DON'T TRANSFORM: Copy content from the parsed markdown exactly as it appears. Do not modify, normalize, or reformat anything.

2. COMPLETENESS: Every character from the parsed markdown must be assigned to exactly one segment. Nothing should be skipped.

3. STRUCTURE MAPPING: The ground truth shows the expected sequence and types of segments. Map the parsed content to match this structure.

4. FORMULA HANDLING: Formulas may be split or merged differently between ground truth and parsed markdown. Adjust the mapping as needed to maintain completeness while respecting the ground truth structure.

5. MISSING CONTENT: If a ground truth segment has no corresponding content in the parsed markdown, use empty string "".

6. OUTPUT: Return a JSON array with {len(gt_segments)} elements, each with "type" and "data" fields.

VERIFICATION BEFORE RETURNING:
- Total non-whitespace characters across all segments = {expected_char_count}
- All {len(gt_segments)} segments present
- All content from parsed markdown accounted for
"""

    return prompt

def extract_segments_using_llm(
    gt_segments: list[dict[str, str]],
    markdown_content: str,
    model: str,
    console=None,
    job_name: str = ""
) -> list[dict[str, str]]:
    """Use LLM to extract and map segments from markdown to ground truth structure.

    Uses OpenAI models with structured outputs via Instructor.
    """

    # ========== CREATE THREAD-SAFE VALIDATION CLASS ==========

    class Segment(BaseModel):
        type: Literal['text', 'inline-formula', 'display-formula'] = Field(
            description="Segment type from the ground truth structure - use the exact type specified in the corresponding ground truth segment"
        )
        data: str = Field(
            description="Exact content from the parsed markdown - copy verbatim without any modifications, normalization, or formatting changes"
        )

    class SegmentMapping(BaseModel):
        """Thread-safe segment mapping with validation for this specific job."""
        segments: list[Segment] = Field(
            min_length=len(gt_segments),
            max_length=len(gt_segments),
            description=f"Must contain exactly {len(gt_segments)} segments in order"
        )

        @model_validator(mode='after')
        def validate_structure(self):
            for i, (expected, actual) in enumerate(zip(gt_segments, self.segments)):
                if expected['type'] != actual.type:
                    raise ValueError(f"Type mismatch at segment {i}: expected '{expected['type']}', got '{actual.type}'")
            return self

    # ========== SETUP OPENAI CLIENT WITH INSTRUCTOR ==========

    prompt = create_llm_mapping_prompt(gt_segments, markdown_content)

    if os.getenv("LLM_PROXY_URL"):
        base_client = OpenAI(
            base_url=os.getenv("LLM_PROXY_URL"),
            api_key=os.getenv("LLM_PROXY_API_KEY")
        )
    else:
        base_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    client = instructor.from_openai(base_client)

    # ========== SETUP RETRY LOGGING HOOKS ==========

    def extract_error_message(error: Exception) -> str:
        """Extract readable error message from validation exception."""
        if hasattr(error, 'errors'):
            errors = error.errors()
            if errors:
                return errors[0].get('msg', str(error))
        return str(error)

    # Create hooks for logging retry attempts
    retry_hooks = None
    if console and job_name:
        retry_counter = {"count": 0}

        def log_validation_error(e: Exception):
            retry_counter["count"] += 1
            console.print(f"      🔄 {job_name} - Retry {retry_counter['count']}/2: {extract_error_message(e)}")

        retry_hooks = Hooks()
        retry_hooks.on("parse:error", log_validation_error)

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        response_model=SegmentMapping,
        max_retries=2,
        hooks=retry_hooks
    )

    return [{"type": seg.type, "data": seg.data} for seg in response.segments]

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

