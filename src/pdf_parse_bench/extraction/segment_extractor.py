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

from ..utilities import FormulaRenderer


@dataclass
class SegmentExtractionJob:
    gt_json_path: Path
    input_md_path: Path
    output_json_path: Path
    stripped_parsed_text_path: Path
    rendered_formulas_dir: Path | None = None


class ParallelSegmentExtractor:
    """Parallel segment extraction processor with integrated progress tracking."""

    def __init__(self, max_workers: int, model: str = "gpt-5-mini", verbose: bool = False):
        self.max_workers = max_workers
        self.model = model
        self.verbose = verbose

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
                job_name=job_name,
                verbose=self.verbose
            )

            # Split grouped formulas
            process_grouped_formulas(
                formula_extraction_result,
                self.model,
                console=console,
                job_name=job_name
            )

            # Remove is_grouped field after processing
            for formula_pair in formula_extraction_result:
                formula_pair.pop("is_grouped", None)

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

            # Convert None to empty string for failed extractions
            for formula_pair in formula_extraction_result:
                if formula_pair["parsed_formula"] is None:
                    formula_pair["parsed_formula"] = ""

            # Save result
            with open(job.output_json_path, 'w', encoding='utf-8') as f:
                json.dump(formula_extraction_result, f, indent=2, ensure_ascii=False)
            with open(job.stripped_parsed_text_path, 'w', encoding='utf-8') as f:
                f.write(remaining_text)

            # Log result with detailed failure information
            if failed_extractions:
                console.print(f"   ⚠️  {job_name} - {len(failed_extractions)} formula(s) not extracted:")
                for idx, gt_formula in failed_extractions:
                    console.print(f"   ⚠️  {idx} GT Formula: {gt_formula}")
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
   - Extract the COMPLETE formula including ALL delimiters (e.g., $, $$, \\[, \\], \\(, \\))
   - If the parser split a formula incorrectly, include adjacent text that belongs to it
       Example: Markdown has "$x = 5$ meters" but GT is "$x = 5 \\text{{meters}}$" → extract "$x = 5$ meters" (include "meters")
   - Preserve ALL whitespace using actual newline and tab characters in JSON (not escaped \\n or \\t sequences)
   - Do NOT add, remove, or normalize anything
3. GROUPED FORMULAS: If multiple ground truth formulas appear merged together (within the same delimiters or in environments like aligned/gathered/array):
   - Extract the COMPLETE grouped content and assign it to the FIRST formula
   - For subsequent formulas that are part of this group: set data="" and is_grouped=true
4. If a formula is genuinely missing from the markdown, use empty string "" (is_grouped defaults to false)

OUTPUT:
JSON list with {len(gt_formula_structure)} objects:
- index: Sequential index from 0 to {len(gt_formula_structure)-1}
- data: Extracted formula from markdown, or "" if missing/grouped
- is_grouped: true if this formula is part of a previous formula's group
"""

    return prompt


def extract_formulas_using_llm(
    gt_formulas: list[dict[str, str]],
    markdown_content: str,
    model: str,
    console=None,
    job_name: str = "",
    max_retries: int = 1,
    verbose: bool = False
) -> tuple[list[dict[str, str]], str]:
    """Extract formula segments using LLM with structured output and post-validation.

    Returns:
        Tuple of:
        - List of dicts with format: [{"gt_formula": "...", "parsed_formula": "..."}, ...]
        - Remaining text with extracted formulas removed
    """

    # ========== OPENAI CLIENT SETUP ==========

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ========== EXTRACTION STATE ==========

    current_text = markdown_content
    formulas_dict = {
        i: {"gt_data": gt["gt_data"], "parsed_formula": None, "is_grouped": False}
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
            is_grouped: bool = Field(
                default=False,
                description=(
                    "Set to true ONLY if this formula is part of a grouped environment "
                    "in a PREVIOUS formula (e.g., aligned/gathered/array). "
                    "Set to false if the formula is genuinely missing from the markdown."
                )
            )

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

        # ========== VALIDATE IS_GROUPED CONSISTENCY ==========

        for local_idx, formula in enumerate(extraction_batch.formulas):
            if formula.is_grouped:
                if formula.data != "":
                    raise ValueError(
                        f"{job_name}: Formula [{formula.index}] has is_grouped=true but data is not empty: {formula.data!r}"
                    )
                if local_idx == 0:
                    raise ValueError(
                        f"{job_name}: Formula [{formula.index}] has is_grouped=true but is the first formula"
                    )

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

            # Skip empty formulas (intentionally not found or grouped)
            if not formula.data:
                formulas_dict[original_idx]["parsed_formula"] = ""
                formulas_dict[original_idx]["is_grouped"] = formula.is_grouped
                continue

            # Try exact match first (fast path) - only if proper delimiters are present
            has_delimiters = any(
                formula.data.startswith(start) and formula.data.endswith(end)
                for start, end in [('$$', '$$'), ('$', '$'), (r'\[', r'\]'), (r'\(', r'\)')]
            )
            if has_delimiters and formula.data in current_text:
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
                if console and verbose:
                    console.print(f"   🔧 {job_name}: Matched formula [{formula.index}] via normalization:\n"
                                  f"LLM formula:\n"
                                  f"{formula.data}\n"
                                  f"Parsed formula:\n"
                                  f"{matched_formula}")
            # else: Keep as None to retry in next iteration

        # ========== CHECK IF RETRY NEEDED ==========

        failed_indices = [idx for idx, data in formulas_dict.items() if data["parsed_formula"] is None]
        if not failed_indices or attempt == max_retries:
            break
        if console:
            failed_indices_str = ", ".join(f"[{idx}]" for idx in failed_indices)
            console.print(f"   🔄 {job_name}: Retrying {len(failed_indices)} failed formula(s) in cleaned text: {failed_indices_str}")

    # ========== COMBINE RESULTS ==========

    result = [
        {
            "gt_formula": formulas_dict[i]["gt_data"],
            "parsed_formula": formulas_dict[i]["parsed_formula"],
            "is_grouped": formulas_dict[i]["is_grouped"]
        }
        for i in range(len(gt_formulas))
    ]

    return result, current_text


# ========== GROUPED FORMULA SPLITTING ==========

def process_grouped_formulas(
    formula_results: list[dict[str, str | bool]],
    model: str,
    console=None,
    job_name: str = ""
) -> None:
    """
    Process and split grouped formulas in-place.

    Args:
        formula_results: List of formula extraction results (modified in-place)
        model: LLM model to use for splitting
        console: Console for logging
        job_name: Job name for logging
    """
    i = 0
    while i < len(formula_results):
        # Check if next formula(s) are marked as grouped
        if i + 1 < len(formula_results) and formula_results[i + 1].get('is_grouped', False):
            # Found start of group - collect all grouped members
            group_members = []
            j = i + 1

            while j < len(formula_results) and formula_results[j].get('is_grouped', False):
                group_members.append(j)
                j += 1

            # Split the grouped formula using LLM
            grouped_formula = formula_results[i]["parsed_formula"]
            gt_formulas_for_split = [formula_results[i]["gt_formula"]] + [
                formula_results[idx]["gt_formula"] for idx in group_members
            ]

            try:
                split_formulas = split_grouped_formula(
                    grouped_formula,
                    gt_formulas_for_split,
                    model,
                    console=console,
                    job_name=job_name
                )

                # Assign split formulas back to results
                formula_results[i]["parsed_formula"] = split_formulas[0]
                formula_results[i]["is_grouped"] = False
                for idx, member_idx in enumerate(group_members):
                    formula_results[member_idx]["parsed_formula"] = split_formulas[idx + 1]
                    formula_results[member_idx]["is_grouped"] = False

            except Exception as e:
                if console:
                    console.print(f"   ⚠️  {job_name}: Failed to split grouped formula at [{i}]: {str(e)}")

            i = j
        else:
            i += 1


def split_grouped_formula(
    grouped_formula: str,
    gt_formulas: list[str],
    model: str,
    console=None,
    job_name: str = ""
) -> list[str]:
    """
    Split a grouped formula into individual formulas using LLM.

    Args:
        grouped_formula: The complete grouped formula with environment
        gt_formulas: List of ground truth formulas to match against
        model: LLM model to use
        console: Console for logging
        job_name: Job name for logging

    Returns:
        List of individual formulas (same length as gt_formulas)
    """

    # ========== EXTRACT DELIMITERS ==========

    delimiter_start = ""
    delimiter_end = ""
    grouped_formula = grouped_formula.strip()

    # Check for delimiters and extract them
    for start, end in [('$$', '$$'), ('$', '$'), (r'\[', r'\]'), (r'\(', r'\)')]:
        if grouped_formula.startswith(start) and grouped_formula.endswith(end):
            delimiter_start = start
            delimiter_end = end
            grouped_formula = grouped_formula[len(start):-len(end)]
            break

    # ========== OPENAI CLIENT SETUP ==========

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # ========== CREATE PROMPT ==========

    gt_formula_list = "\n".join([f"[{i}] {formula}" for i, formula in enumerate(gt_formulas)])

    prompt = f"""You are a LaTeX formula splitting specialist.

TASK:
You are given a grouped formula that contains {len(gt_formulas)} individual formulas merged together.
Your task is to split this grouped formula into {len(gt_formulas)} separate formulas.

GROUPED FORMULA CONTENT:
```
{grouped_formula}
```

GROUND TRUTH FORMULAS (reference - may differ significantly from actual content):
{gt_formula_list}

INSTRUCTIONS:
1. Remove grouping environment wrappers (e.g., \\begin{{aligned}}...\\end{{aligned}})
2. Use line breaks (\\\\) as split indicators where applicable - REMOVE these separators from output
3. Match parts to ground truth formulas by content similarity
4. EXTRACT EXACTLY - preserve ALL content character-by-character, no transformations/normalization, NO content loss
5. If a formula cannot be extracted, return empty string ""

OUTPUT:
JSON list with {len(gt_formulas)} strings, one for each ground truth formula in order.
"""

    # ========== PYDANTIC MODEL ==========

    class SplitFormulas(BaseModel):
        formulas: list[str] = Field(
            min_length=len(gt_formulas),
            max_length=len(gt_formulas)
        )

    # ========== LLM CALL ==========

    response = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=SplitFormulas
    )

    raw_formulas = response.output_parsed.formulas

    # ========== VALIDATE AND ADD DELIMITERS ==========

    result_formulas = []
    for i, raw_formula in enumerate(raw_formulas):
        if not raw_formula:
            result_formulas.append("")
            continue

        # Validate that the formula content exists in grouped_formula
        if raw_formula not in grouped_formula:
            # Try fuzzy matching
            matched = find_original_formula_in_markdown(
                llm_formula=raw_formula,
                markdown_content=grouped_formula
            )
            if matched:
                raw_formula = matched

        # Add delimiters back if they were detected
        if delimiter_start:
            final_formula = f"{delimiter_start}{raw_formula}{delimiter_end}"
        else:
            final_formula = raw_formula

        result_formulas.append(final_formula)

    return result_formulas


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
