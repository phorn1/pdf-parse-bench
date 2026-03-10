import os
import json
import re
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from Levenshtein import distance as levenshtein_distance

from openai import OpenAI
from pydantic import BaseModel, Field
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn

from ..eval.results import FormulaResult, TableResult, save_results
from ..utilities import FormulaRenderer


@dataclass
class SegmentExtractionJob:
    gt_json_path: Path
    input_md_path: Path
    output_formulas_json_path: Path
    output_tables_json_path: Path
    stripped_parsed_text_path: Path
    rendered_formulas_dir: Path | None = None


class ParallelSegmentExtractor:
    """Parallel segment extraction processor with integrated progress tracking."""

    def __init__(self, max_workers: int, model: str = "google/gemini-3-flash-preview", verbose: bool = False):
        self.max_workers = max_workers
        self.model = model
        self.verbose = verbose
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def _process_single_job(self, job: SegmentExtractionJob, console) -> bool:
        """Process a single segment extraction job.

        Returns:
            True if successful, False if error occurred
        """
        job_name = f"{job.input_md_path.parent.name}/{job.input_md_path.parent.parent.name}"

        try:
            # Load ground truth segments
            gt_segments = json.loads(job.gt_json_path.read_text())

            gt_tables = [
                {"gt_data": segment["data"], "complexity": segment["complexity"]}
                for segment in gt_segments
                if segment["type"] == "table"
            ]

            gt_formulas = [
                {"gt_data": segment["data"], "formula_type": segment["type"]}
                for segment in gt_segments
                if segment["type"] in ["inline-formula", "display-formula"]
            ]

            # Load parsed markdown content
            markdown_content = job.input_md_path.read_text()

            remaining_text = markdown_content

            # ========== TABLE EXTRACTION ==========

            table_extraction_result, remaining_text = extract_tables_using_llm(
                gt_tables,
                remaining_text,
                self.model,
                self.client,
                console=console,
                job_name=job_name,
                verbose=self.verbose
            )

            failed_table_extractions = []
            for i, pair in enumerate(table_extraction_result):
                if pair["extracted_table"] is None:
                    failed_table_extractions.append((i, pair["gt_table"]))
                    pair["extracted_table"] = ""

            if table_extraction_result:
                table_results = [
                    TableResult(
                        index=i,
                        gt_table=pair["gt_table"],
                        extracted_table=pair["extracted_table"],
                        complexity=pair["complexity"],
                    )
                    for i, pair in enumerate(table_extraction_result)
                ]
                save_results(job.output_tables_json_path, table_results)

            # ========== FORMULA EXTRACTION ==========

            formula_extraction_result, remaining_text = extract_formulas_using_llm(
                gt_formulas,
                remaining_text,
                self.model,
                self.client,
                console=console,
                job_name=job_name,
                verbose=self.verbose
            )

            # Split grouped formulas
            process_grouped_formulas(
                formula_extraction_result,
                self.model,
                self.client,
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
                    if formula_pair["extracted_formula"]:  # Only render non-empty formulas
                        formula_pair["rendered_png"] = renderer.render_formula(
                            formula_pair["extracted_formula"],
                            job.rendered_formulas_dir,
                            f"formula_{i:03d}"
                        )

            failed_formula_extractions = []
            for i, pair in enumerate(formula_extraction_result):
                if pair["extracted_formula"] is None:
                    failed_formula_extractions.append((i, pair["gt_formula"]))
                    pair["extracted_formula"] = ""

            if formula_extraction_result:
                formula_results = [
                    FormulaResult(
                        index=i,
                        gt_formula=pair["gt_formula"],
                        extracted_formula=pair["extracted_formula"],
                        formula_type=pair["formula_type"],
                    )
                    for i, pair in enumerate(formula_extraction_result)
                ]
                save_results(job.output_formulas_json_path, formula_results)

            job.stripped_parsed_text_path.write_text(remaining_text)

            # ========== LOG RESULTS ==========

            has_failures = failed_formula_extractions or failed_table_extractions
            if has_failures:
                if failed_table_extractions:
                    console.print(f"   ⚠️  {job_name} - {len(failed_table_extractions)} table(s) not extracted:")
                    for idx, gt_table in failed_table_extractions:
                        console.print(f"       [{idx}] GT Table: {gt_table}")
                if failed_formula_extractions:
                    console.print(f"   ⚠️  {job_name} - {len(failed_formula_extractions)} formula(s) not extracted:")
                    for idx, gt_formula in failed_formula_extractions:
                        console.print(f"       [{idx}] GT Formula: {gt_formula}")
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
    client: OpenAI,
    console=None,
    job_name: str = "",
    max_retries: int = 1,
    verbose: bool = False
) -> tuple[list[dict[str, str]], str]:
    """Extract formula segments using LLM with structured output and post-validation.

    Returns:
        Tuple of:
        - List of dicts with format: [{"gt_formula": "...", "extracted_formula": "..."}, ...]
        - Remaining text with extracted formulas removed
    """

    if not gt_formulas:
        return [], markdown_content

    # ========== EXTRACTION STATE ==========

    current_text = markdown_content
    formulas_dict = {
        i: {"gt_data": gt["gt_data"], "formula_type": gt["formula_type"], "extracted_formula": None, "is_grouped": False}
        for i, gt in enumerate(gt_formulas)
    }

    for attempt in range(max_retries + 1):
        # Get formulas that still need extraction (extracted_formula is None)
        to_extract = {idx: data for idx, data in formulas_dict.items() if data["extracted_formula"] is None}

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

        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=ExtractedFormulas,
                max_tokens=32000,
            )
            extraction_batch = response.choices[0].message.parsed
        except Exception as e:
            if console:
                console.print(f"   ⚠️  {job_name}: Formula LLM call failed: {e} (retrying...)")
            continue

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
                formulas_dict[original_idx]["extracted_formula"] = ""
                formulas_dict[original_idx]["is_grouped"] = formula.is_grouped
                continue

            # Try exact match first (fast path) - only if proper delimiters are present
            has_delimiters = any(
                formula.data.startswith(start) and formula.data.endswith(end)
                for start, end in [('$$', '$$'), ('$', '$'), (r'\[', r'\]'), (r'\(', r'\)')]
            )
            if has_delimiters and formula.data in current_text:
                formulas_dict[original_idx]["extracted_formula"] = formula.data
                current_text = current_text.replace(formula.data, "", 1)
                continue

            # Try fuzzy matching
            matched_formula = find_original_segment_in_markdown(
                llm_segment=formula.data,
                markdown_content=current_text,
                bonus_fn=calculate_formula_delimiter_bonus
            )

            if matched_formula:
                if matched_formula not in current_text:
                    raise Exception(f"Unexpected: matched formula not in text: {matched_formula!r}")
                formulas_dict[original_idx]["extracted_formula"] = matched_formula
                current_text = current_text.replace(matched_formula, "", 1)
                if console and verbose:
                    console.print(f"   🔧 {job_name}: Matched formula [{formula.index}] via normalization:\n"
                                  f"LLM formula:\n"
                                  f"{formula.data}\n"
                                  f"Parsed formula:\n"
                                  f"{matched_formula}")
            # else: Keep as None to retry in next iteration

        # ========== CHECK IF RETRY NEEDED ==========

        failed_indices = [idx for idx, data in formulas_dict.items() if data["extracted_formula"] is None]
        if not failed_indices or attempt == max_retries:
            break
        if console:
            failed_indices_str = ", ".join(f"[{idx}]" for idx in failed_indices)
            console.print(f"   🔄 {job_name}: Retrying {len(failed_indices)} failed formula(s) in cleaned text: {failed_indices_str}")

    # ========== COMBINE RESULTS ==========

    result = [
        {
            "gt_formula": formulas_dict[i]["gt_data"],
            "extracted_formula": formulas_dict[i]["extracted_formula"],
            "formula_type": formulas_dict[i]["formula_type"],
            "is_grouped": formulas_dict[i]["is_grouped"],
        }
        for i in range(len(gt_formulas))
    ]

    return result, current_text


# ========== GROUPED FORMULA SPLITTING ==========

def process_grouped_formulas(
    formula_results: list[dict[str, str | bool]],
    model: str,
    client: OpenAI,
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
            grouped_formula = formula_results[i]["extracted_formula"]
            gt_formulas_for_split = [formula_results[i]["gt_formula"]] + [
                formula_results[idx]["gt_formula"] for idx in group_members
            ]

            try:
                split_formulas = split_grouped_formula(
                    grouped_formula,
                    gt_formulas_for_split,
                    model,
                    client,
                    console=console,
                    job_name=job_name
                )

                # Assign split formulas back to results
                formula_results[i]["extracted_formula"] = split_formulas[0]
                formula_results[i]["is_grouped"] = False
                for idx, member_idx in enumerate(group_members):
                    formula_results[member_idx]["extracted_formula"] = split_formulas[idx + 1]
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
    client: OpenAI,
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

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=SplitFormulas,
        max_tokens=16000,
    )

    raw_formulas = response.choices[0].message.parsed.formulas

    # ========== VALIDATE AND ADD DELIMITERS ==========

    result_formulas = []
    for i, raw_formula in enumerate(raw_formulas):
        if not raw_formula:
            result_formulas.append("")
            continue

        # Validate that the formula content exists in grouped_formula
        if raw_formula not in grouped_formula:
            # Try fuzzy matching
            matched = find_original_segment_in_markdown(
                llm_segment=raw_formula,
                markdown_content=grouped_formula,
                bonus_fn=calculate_formula_delimiter_bonus
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


# ========== SEGMENT MATCHING ==========

def calculate_formula_delimiter_bonus(text: str) -> float:
    """Calculate bonus for matching formula delimiters in case they were missing in the LLM extraction."""
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


def find_original_segment_in_markdown(
    llm_segment: str,
    markdown_content: str,
    edit_distance_ratio: float = 0.15,
    search_radius: int = 10,
    bonus_fn: Callable[[str], float] | None = None
) -> str | None:
    """
    Find the original segment in markdown using normalized sliding window matching.

    Strategy:
    1. Normalize both strings (remove whitespace AND backslashes)
    2. Use sliding window with Levenshtein distance to find best position
    3. Map normalized position back to original text
    4. Refine by testing small boundary variations around that position

    Args:
        llm_segment: Segment extracted by LLM (may have whitespace/backslash differences or errors)
        markdown_content: Original markdown content to search in
        edit_distance_ratio: Max allowed edit distance as ratio of segment length
        search_radius: Characters to expand/shrink boundaries during refinement
        bonus_fn: Optional function to calculate bonus score for candidates (e.g., delimiter matching)

    Returns:
        Original segment string from markdown, or None if no match within threshold
    """
    # Unescape string-escaped newlines and tabs in LLM output
    # (only when they're NOT part of LaTeX commands like \theta or \text)
    llm_segment = re.sub(r'\\n(?![a-zA-Z])', '\n', llm_segment)
    llm_segment = re.sub(r'\\t(?![a-zA-Z])', '\t', llm_segment)

    # Normalize both strings (remove whitespace AND backslashes)
    normalized_llm = re.sub(r'[\s\\]+', '', llm_segment)
    normalized_markdown = re.sub(r'[\s\\]+', '', markdown_content)

    # Early return: Can't match if segment is empty or longer than content
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
    norm_to_orig = {}
    norm_idx = 0
    for orig_idx, c in enumerate(markdown_content):
        if not c.isspace() and c != '\\':
            norm_to_orig[norm_idx] = orig_idx
            norm_idx += 1

    # Map normalized window to original text boundaries
    orig_start = norm_to_orig[best_pos]
    orig_end = norm_to_orig[best_pos + len(normalized_llm) - 1] + 1

    # Refine by testing boundary variations
    best_match, best_final_dist = None, float('inf')
    best_score = float('inf')

    for start_delta in range(-search_radius, search_radius + 1):
        for end_delta in range(-search_radius, search_radius + 1):
            s = max(0, orig_start + start_delta)
            e = min(len(markdown_content), orig_end + end_delta)

            if s >= e:
                continue

            candidate = markdown_content[s:e]
            candidate_norm = re.sub(r'[\s\\]+', '', candidate)
            dist = levenshtein_distance(normalized_llm, candidate_norm)

            bonus = bonus_fn(candidate) if bonus_fn else 0.0
            score = dist - bonus

            if score < best_score:
                best_match, best_final_dist, best_score = candidate, dist, score

    return best_match if best_final_dist <= threshold else None


# ========== LLM TABLE EXTRACTION ==========

def create_table_extraction_prompt(gt_tables_segments: list[dict[str, str]], markdown_content: str) -> str:
    """Create a focused prompt for extracting table segments."""

    gt_table_structure = [
        f"[{idx}] {table['gt_data']}"
        for idx, table in enumerate(gt_tables_segments)
    ]

    prompt = f"""You are a table extraction specialist.

SCENARIO:
You are given two inputs:
1. A reference list of {len(gt_table_structure)} LaTeX tables (GROUND TRUTH)
2. A markdown document containing text with embedded tables (PARSED MARKDOWN)

The tables from the ground truth list appear in the markdown document, but may be represented in various formats (e.g. markdown, HTML, LaTeX, lists, or other representations) and are likely to be significantly modified in content and structure.

YOUR TASK:
Extract the tables from the markdown document and return them as a JSON list. The output list must follow the same order and structure as the ground truth list. For each table in the ground truth list, find and extract the corresponding table from the markdown.

GROUND TRUTH TABLES ({len(gt_table_structure)} total):
{"\n".join(gt_table_structure)}

PARSED MARKDOWN CONTENT:
```markdown
{markdown_content}
```

INSTRUCTIONS:

1. For each ground truth table, find its match in the markdown
2. EXTRACT EXACTLY, DON'T TRANSFORM: Copy tables character-by-character as they appear in markdown
   - Extract the COMPLETE table including ALL formatting
   - Preserve ALL whitespace using actual newline and tab characters in JSON
   - Do NOT add, remove, or normalize anything
4. Only use empty string "" if the table content is truly absent from the document (very rare)

OUTPUT:
JSON list with {len(gt_table_structure)} objects:
- index: Sequential index from 0 to {len(gt_table_structure)-1}
- data: Extracted table from markdown, or "" if missing
"""

    return prompt


def extract_tables_using_llm(
    gt_tables: list[dict[str, str]],
    markdown_content: str,
    model: str,
    client: OpenAI,
    console=None,
    job_name: str = "",
    max_retries: int = 1,
    verbose: bool = False
) -> tuple[list[dict[str, str]], str]:
    """Extract table segments using LLM with structured output and post-validation.

    Returns:
        Tuple of:
        - List of dicts with format: [{"gt_table": "...", "extracted_table": "..."}, ...]
        - Remaining text with extracted tables removed
    """

    # Early return if no tables to extract
    if not gt_tables:
        return [], markdown_content

    # ========== EXTRACTION STATE ==========

    current_text = markdown_content
    tables_dict = {
        i: {"gt_data": gt["gt_data"], "complexity": gt["complexity"], "extracted_table": None}
        for i, gt in enumerate(gt_tables)
    }

    for attempt in range(max_retries + 1):
        to_extract = {idx: data for idx, data in tables_dict.items() if data["extracted_table"] is None}

        if not to_extract:
            break

        tables_for_prompt = [{"gt_data": data["gt_data"]} for data in to_extract.values()]

        # Define Pydantic models dynamically
        class TableExtraction(BaseModel):
            index: int = Field(description=f"Sequential index (0 to {len(tables_for_prompt)-1})")
            data: str = Field(description="Exact table from markdown, verbatim. Empty string if not found.")

        class ExtractedTables(BaseModel):
            tables: list[TableExtraction] = Field(
                min_length=len(tables_for_prompt),
                max_length=len(tables_for_prompt)
            )

        # ========== LLM CALL ==========

        prompt = create_table_extraction_prompt(tables_for_prompt, current_text)

        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=ExtractedTables,
                max_tokens=32000,
            )
            extraction_batch = response.choices[0].message.parsed
        except Exception as e:
            if console:
                console.print(f"   ⚠️  {job_name}: Table LLM call failed: {e} (retrying...)")
            continue

        # ========== VALIDATE INDICES ==========

        expected_indices = list(range(len(tables_for_prompt)))
        actual_indices = [t.index for t in extraction_batch.tables]

        if expected_indices != actual_indices:
            is_last_attempt = attempt == max_retries
            retry_status = f" (giving up after {max_retries + 1} attempts)" if is_last_attempt else " (retrying...)"

            if console:
                console.print(f"   ⚠️  {job_name}: Table index mismatch - expected {expected_indices}, got {actual_indices}{retry_status}")

            if not is_last_attempt:
                continue

        # ========== POST-VALIDATION ==========

        original_indices = list(to_extract.keys())

        for local_idx, table in enumerate(extraction_batch.tables):
            original_idx = original_indices[local_idx]

            if not table.data:
                tables_dict[original_idx]["extracted_table"] = ""
                continue

            # Try exact match first
            if table.data in current_text:
                tables_dict[original_idx]["extracted_table"] = table.data
                current_text = current_text.replace(table.data, "", 1)
                continue

            # Try fuzzy matching for tables
            matched_table = find_original_segment_in_markdown(
                llm_segment=table.data,
                markdown_content=current_text
            )

            if matched_table:
                if matched_table not in current_text:
                    raise Exception(f"Unexpected: matched table not in text: {matched_table[:100]!r}...")
                tables_dict[original_idx]["extracted_table"] = matched_table
                current_text = current_text.replace(matched_table, "", 1)
                if console and verbose:
                    console.print(f"   🔧 {job_name}: Matched table [{table.index}] via normalization")

        # ========== CHECK IF RETRY NEEDED ==========

        failed_indices = [idx for idx, data in tables_dict.items() if data["extracted_table"] is None]
        if not failed_indices or attempt == max_retries:
            break
        if console:
            failed_indices_str = ", ".join(f"[{idx}]" for idx in failed_indices)
            console.print(f"   🔄 {job_name}: Retrying {len(failed_indices)} failed table(s): {failed_indices_str}")

    # ========== COMBINE RESULTS ==========

    result = [
        {
            "gt_table": tables_dict[i]["gt_data"],
            "extracted_table": tables_dict[i]["extracted_table"],
            "complexity": tables_dict[i]["complexity"],
        }
        for i in range(len(gt_tables))
    ]

    return result, current_text

