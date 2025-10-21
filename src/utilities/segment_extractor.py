import os
import json
import re
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from Levenshtein import distance as levenshtein_distance

from openai import OpenAI
from pydantic import BaseModel, Field
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

from .formula_renderer import FormulaRenderer


@dataclass
class SegmentExtractionJob:
    gt_json_path: Path
    input_md_path: Path
    output_json_path: Path
    stripped_parsed_text_path: Path
    rendered_formulas_dir: Path | None = None


class ParallelSegmentExtractor:
    """Parallel segment extraction processor with integrated progress tracking."""

    def __init__(self, max_workers: int, model: str = "gpt-5-mini"):
        self.max_workers = max_workers
        self.model = model

    def _process_single_job(self, job: SegmentExtractionJob, console) -> bool:
        """Process a single segment extraction job.

        Returns:
            True if successful, False if error occurred
        """
        job_name = f"{job.input_md_path.parent.name}/{job.input_md_path.parent.parent.name}"

        try:
            # Load ground truth formulas
            with open(job.gt_json_path, 'r', encoding='utf-8') as f:
                gt_segments = json.load(f)
            gt_formulas = [
                {"gt_data": segment["data"]}
                for segment in gt_segments
                if segment["type"] in ["inline-formula", "display-formula"]
            ]

            # Load parsed markdown content
            with open(job.input_md_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            # Extract formulas using LLM
            formula_extraction_result, remaining_text = extract_formulas_using_llm(
                gt_formulas,
                markdown_content,
                self.model,
                console=console,
                job_name=job_name
            )

            # Normalize parsed formulas in-place
            for formula_pair in formula_extraction_result:
                if formula_pair["parsed_formula"] is not None:
                    formula_pair["parsed_formula"] = normalize_gathered_display_formula(
                        formula_pair["parsed_formula"]
                    )

            # Render formulas if path is provided
            if job.rendered_formulas_dir is not None:
                renderer = FormulaRenderer()
                for i, formula_pair in enumerate(formula_extraction_result):
                    if formula_pair["parsed_formula"]:  # Only render non-empty formulas
                        formula_pair["rendered_png"] = renderer.render_formula(
                            formula_pair["parsed_formula"],
                            job.rendered_formulas_dir,
                            f"formula_{i:03d}"
                        )

            # Check for failed extractions
            failed_extractions = [
                (i, pair["gt_formula"])
                for i, pair in enumerate(formula_extraction_result)
                if pair["parsed_formula"] is None
            ]

            # Save result
            with open(job.output_json_path, 'w', encoding='utf-8') as f:
                json.dump(formula_extraction_result, f, indent=2, ensure_ascii=False)
            with open(job.stripped_parsed_text_path, 'w', encoding='utf-8') as f:
                f.write(remaining_text)

            # Log result with detailed failure information
            if failed_extractions:
                console.print(f"   ⚠️  {job_name} - {len(failed_extractions)} formula(s) not extracted:")
                for idx, gt_formula in failed_extractions:
                    console.print(f"   ❌ {idx} GT Formula: {gt_formula}")
            else:
                console.print(f"   ✅ {job_name}")

            return True

        except Exception as e:
            console.print(f"   ❌ {job_name}: {str(e)}")
            return False

    def extract_segments_parallel(self, jobs: list[SegmentExtractionJob]):
        """Extract segments in parallel using ThreadPoolExecutor with progress tracking."""

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Extracting segments...", total=len(jobs))

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_job = {executor.submit(self._process_single_job, job, progress.console): job for job in jobs}

                for future in as_completed(future_to_job):
                    future.result()  # Wait for completion, logging happens inside _process_single_job
                    progress.update(task, advance=1)


# ========== LLM FORMULA EXTRACTION ==========

def create_formula_extraction_prompt(gt_formulas_segments: list[dict[str, str]], markdown_content: str) -> str:
    """Create a focused prompt for extracting formula segments."""

    # Build the ground truth formula structure (with 0-based sequential indices)
    gt_formula_structure = [
        f"[{idx}] {formula['gt_data']}"
        for idx, formula in enumerate(gt_formulas_segments)
    ]

    prompt = f"""You are a mathematical formula extraction specialist.

SCENARIO:
You are given two inputs:
1. A reference list of {len(gt_formula_structure)} LaTeX formulas (GROUND TRUTH)
2. A markdown document containing text with embedded formulas (PARSED MARKDOWN)

The formulas from the ground truth list should theoretically appear in the markdown document, embedded between text snippets. Note that every formula in the markdown definitively originates from the ground truth list. 

CHALLENGES:
- Formulas in the markdown may be slightly or significantly modified compared to the ground truth
- Some formulas from the ground truth may be missing in the markdown
- Formula order is often preserved, but in some cases may differ from the ground truth order

YOUR TASK:
Extract the formulas from the markdown document and return them as a JSON list. The output list must follow the same order and structure as the ground truth list. For each formula in the ground truth list, find and extract the corresponding formula from the markdown.

GROUND TRUTH FORMULAS ({len(gt_formula_structure)} total):
{"\n".join(gt_formula_structure)}

PARSED MARKDOWN CONTENT:
```markdown
{markdown_content}
```

INSTRUCTIONS:

1. For each ground truth formula, find its match in the markdown
2. EXTRACT EXACTLY, DON'T TRANSFORM: Copy formulas character-by-character as they appear in markdown
   - Extract the COMPLETE formula from the very beginning to the very end, including ALL delimiter characters (e.g., $, $$, \\[, \\], \\(, \\))
   - Preserve ALL whitespace using actual newline and tab characters in JSON (not escaped \\n or \\t sequences)
   - Do NOT add, remove, or normalize anything
3. HANDLE VARIATIONS: Formulas in markdown may be split or merged differently than ground truth
   - A single ground truth formula might be split into multiple parts in the markdown
   - Multiple ground truth formulas might be merged into one in the markdown
   - Adjust your mapping accordingly while maintaining the ground truth structure
4. If a formula is not found, use empty string ""

OUTPUT:
JSON list with {len(gt_formula_structure)} objects:
- index: Sequential index from 0 to {len(gt_formula_structure)-1}
- data: Extracted formula from markdown, or "" if missing
"""

    return prompt


def extract_formulas_using_llm(
    gt_formulas: list[dict[str, str]],
    markdown_content: str,
    model: str,
    console=None,
    job_name: str = "",
    max_retries: int = 1
) -> tuple[list[dict[str, str]], str]:
    """Extract formula segments using LLM with structured output and post-validation.

    Returns:
        Tuple of:
        - List of dicts with format: [{"gt_formula": "...", "parsed_formula": "..."}, ...]
        - Remaining text with extracted formulas removed
    """

    # ========== OPENAI CLIENT SETUP ==========

    if os.getenv("LLM_PROXY_URL"):
        client = OpenAI(
            base_url=os.getenv("LLM_PROXY_URL"),
            api_key=os.getenv("LLM_PROXY_API_KEY")
        )
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ========== EXTRACTION STATE ==========

    current_text = markdown_content
    formulas_dict = {
        i: {"gt_data": gt["gt_data"], "parsed_formula": None}
        for i, gt in enumerate(gt_formulas)
    }

    for attempt in range(max_retries + 1):
        # Get formulas that still need extraction (parsed_formula is None)
        to_extract = {idx: data for idx, data in formulas_dict.items() if data["parsed_formula"] is None}

        if not to_extract:
            break  # All formulas extracted

        # Build list for prompt (LLM expects sequential 0-based indices)
        formulas_for_prompt = [{"gt_data": data["gt_data"]} for data in to_extract.values()]

        # Define Pydantic models dynamically based on current extraction batch size
        class FormulaExtraction(BaseModel):
            index: int = Field(description=f"Sequential index (0 to {len(formulas_for_prompt)-1})")
            data: str = Field(description="Exact formula from markdown, verbatim. Empty string if not found.")

        class ExtractedFormulas(BaseModel):
            formulas: list[FormulaExtraction] = Field(
                min_length=len(formulas_for_prompt),
                max_length=len(formulas_for_prompt)
            )

        # ========== LLM CALL ==========

        prompt = create_formula_extraction_prompt(formulas_for_prompt, current_text)

        response = client.responses.parse(
            model=model,
            input=[{"role": "user", "content": prompt}],
            text_format=ExtractedFormulas
        )

        extraction_batch = response.output_parsed

        # ========== VALIDATE INDICES ==========

        expected_indices = list(range(len(formulas_for_prompt)))
        actual_indices = [f.index for f in extraction_batch.formulas]

        if expected_indices != actual_indices:
            is_last_attempt = attempt == max_retries
            retry_status = f" (giving up after {max_retries + 1} attempts)" if is_last_attempt else " (retrying...)"

            if console:
                console.print(f"   ⚠️  {job_name}: Index mismatch - expected {expected_indices}, got {actual_indices}{retry_status}")

            # Retry ENTIRE extraction if not last attempt
            if not is_last_attempt:
                continue

        # ========== POST-VALIDATION WITH WARNINGS ==========

        original_indices = list(to_extract.keys())

        for local_idx, formula in enumerate(extraction_batch.formulas):
            original_idx = original_indices[local_idx]

            # Skip empty formulas (intentionally not found)
            if not formula.data:
                formulas_dict[original_idx]["parsed_formula"] = ""
                continue

            # Try exact match first (fast path)
            if formula.data in current_text:
                formulas_dict[original_idx]["parsed_formula"] = formula.data
                current_text = current_text.replace(formula.data, "", 1)
                continue

            # Try fuzzy matching
            matched_formula = find_original_formula_in_markdown(
                llm_formula=formula.data,
                markdown_content=current_text
            )

            if matched_formula:
                if matched_formula not in current_text:
                    raise Exception(f"Unexpected: matched formula not in text: {matched_formula!r}")
                formulas_dict[original_idx]["parsed_formula"] = matched_formula
                current_text = current_text.replace(matched_formula, "", 1)
                if console:
                    console.print(f"   🔧 {job_name}: Matched formula [{formula.index}] via normalization:\n"
                                  f"LLM formula:\n"
                                  f"{formula.data}\n"
                                  f"Parsed formula:\n"
                                  f"{matched_formula}")
            else:
                # Keep as None to retry in next iteration
                if console:
                    console.print(f"   ⚠️  {job_name}: Formula not found in markdown - Index:{formula.index}: {formula.data}")

        # ========== CHECK IF RETRY NEEDED ==========

        has_failed_formulas = any(data["parsed_formula"] is None for data in formulas_dict.values())
        if not has_failed_formulas or attempt == max_retries:
            break
        if console:
            console.print(f"   🔄 {job_name}: Retrying failed formulas in cleaned text...")

    # ========== COMBINE RESULTS ==========

    result = [
        {
            "gt_formula": formulas_dict[i]["gt_data"],
            "parsed_formula": formulas_dict[i]["parsed_formula"]
        }
        for i in range(len(gt_formulas))
    ]

    return result, current_text


# ========== FORMULA NORMALIZATION ==========

def find_original_formula_in_markdown(
    llm_formula: str,
    markdown_content: str,
    edit_distance_ratio: float = 0.15,
    search_radius: int = 10
) -> str | None:
    """
    Find the original formula in markdown using normalized sliding window matching.

    Strategy:
    1. Normalize both strings (remove whitespace AND backslashes)
    2. Use sliding window with Levenshtein distance to find best position
    3. Map normalized position back to original text
    4. Refine by testing small boundary variations around that position

    Args:
        llm_formula: Formula extracted by LLM (may have whitespace/backslash differences or errors)
        markdown_content: Original markdown content to search in
        edit_distance_ratio: Max allowed edit distance as ratio of formula length
        search_radius: Characters to expand/shrink boundaries during refinement

    Returns:
        Original formula string from markdown, or None if no match within threshold
    """
    # Unescape string-escaped newlines and tabs in LLM output
    # (only when they're NOT part of LaTeX commands like \theta or \text)
    llm_formula = re.sub(r'\\n(?![a-zA-Z])', '\n', llm_formula)
    llm_formula = re.sub(r'\\t(?![a-zA-Z])', '\t', llm_formula)

    # Normalize both strings (remove whitespace AND backslashes)
    normalized_llm = re.sub(r'[\s\\]+', '', llm_formula)
    normalized_markdown = re.sub(r'[\s\\]+', '', markdown_content)

    # Early return: Can't match if formula is empty or longer than content
    if not normalized_llm or len(normalized_llm) > len(normalized_markdown):
        return None

    threshold = int(len(normalized_llm) * edit_distance_ratio)

    # Find best position in normalized markdown using sliding window
    best_pos, best_dist = 0, float('inf')
    for i in range(len(normalized_markdown) - len(normalized_llm) + 1):
        window = normalized_markdown[i:i + len(normalized_llm)]
        dist = levenshtein_distance(normalized_llm, window)
        if dist < best_dist:
            best_pos, best_dist = i, dist

    # Build mapping from normalized indices to original indices
    norm_to_orig = {
        norm_idx: orig_idx
        for norm_idx, (orig_idx, char) in enumerate(
            (i, c) for i, c in enumerate(markdown_content) if not c.isspace() and c != '\\'
        )
    }

    # Map normalized window to original text boundaries
    orig_start = norm_to_orig[best_pos]
    orig_end = norm_to_orig[best_pos + len(normalized_llm) - 1] + 1

    # Refine by testing boundary variations
    best_match, best_final_dist = None, float('inf')
    best_score = float('inf')

    def calculate_delimiter_bonus(text: str) -> float:
        """Calculate bonus for matching formula delimiters in case they were missing in the llm extraction. """
        bonus = 0.0

        # Award bonus for start delimiters
        if text.startswith('$$'):
            bonus += 2.5
        elif text.startswith(('$', r'\[', r'\(')):
            bonus += 1.5

        # Award bonus for end delimiters
        if text.endswith('$$'):
            bonus += 2.5
        elif text.endswith(('$', r'\]', r'\)')):
            bonus += 1.5

        return bonus

    for start_delta in range(-search_radius, search_radius + 1):
        for end_delta in range(-search_radius, search_radius + 1):
            s = max(0, orig_start + start_delta)
            e = min(len(markdown_content), orig_end + end_delta)

            if s >= e:
                continue

            candidate = markdown_content[s:e]
            candidate_norm = re.sub(r'[\s\\]+', '', candidate)
            dist = levenshtein_distance(normalized_llm, candidate_norm)

            score = dist - calculate_delimiter_bonus(candidate)

            if score < best_score:
                best_match, best_final_dist, best_score = candidate, dist, score

    return best_match if best_final_dist <= threshold else None


def normalize_gathered_display_formula(formula_data: str) -> str:
    """
    Normalize display formula by removing unmatched gathered environments
    and ensuring proper $$ delimiters.

    WHY THIS IS NECESSARY:
    Some parsers merge consecutive formulas from the PDF into a single formula using
    `gathered` environments. During LLM extraction, these merged formulas get split back
    into individual formulas, breaking the gathered environment and leaving unmatched
    `\\begin{gathered}` or `\\end{gathered}` fragments that cause rendering errors.

    Args:
        formula_data: Raw LaTeX formula string

    Returns:
        Normalized LaTeX formula string with proper delimiters
    """
    # Remove unmatched gathered environments
    begin_matches = list(re.finditer(r'\\begin\{gathered\}', formula_data))
    end_matches = list(re.finditer(r'\\end\{gathered\}', formula_data))

    if len(begin_matches) != len(end_matches):
        formula_data = re.sub(r'\\(?:begin|end)\{gathered\}', '', formula_data)
        print(f"Removed unmatched gathered environment: {len(begin_matches)} begin + {len(end_matches)} end")
    else:
        return formula_data

    # Ensure proper $$ delimiters for the repaired formula
    stripped = formula_data.strip()

    if not stripped.startswith('$$'):
        stripped = f"$${stripped}"
    if not stripped.endswith('$$'):
        stripped = f"{stripped}$$"

    return stripped
