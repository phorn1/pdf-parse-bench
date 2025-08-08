import json
import difflib
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from Levenshtein import distance as levenshtein_distance

logger = logging.getLogger(__name__)


class TextMatchingError(Exception):
    """Exception raised when text cannot be matched in markdown content with sufficient similarity."""
    pass


def _calculate_levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Calculate the similarity ratio using Levenshtein distance.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity ratio between 0 and 1
    """
    if not s1 or not s2:
        return 0.0

    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0

    lev_dist = levenshtein_distance(s1, s2)
    return 1 - (lev_dist / max_len)


def _find_fuzzy_prefix_matches(
        corpus: str,
        prefix: str,
        similarity_threshold: float = 0.6,
        max_candidates: int = 5
) -> List[Tuple[int, float]]:
    """
    Find potential matches for a prefix using Levenshtein distance.

    Args:
        corpus: Text to search in
        prefix: Prefix to search for
        similarity_threshold: Minimum similarity to consider a match
        max_candidates: Maximum number of candidates to return

    Returns:
        List of tuples (start_index, similarity_score)
    """
    candidates = []
    prefix_len = len(prefix)

    # Normalize prefix for better matching
    normalized_prefix = ' '.join(prefix.split()).lower()

    # Sliding window through corpus
    for i in range(len(corpus) - prefix_len + 1):
        window = corpus[i:i + prefix_len]
        normalized_window = ' '.join(window.split()).lower()

        # Calculate similarity
        similarity = _calculate_levenshtein_ratio(normalized_prefix, normalized_window)

        if similarity >= similarity_threshold:
            candidates.append((i, similarity))

    # Sort by similarity and return top candidates
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:max_candidates]


def _find_text_in_markdown(
        target_text: str,
        markdown_content: str,
        search_start_offset: int,
        max_search_distance: int,
        similarity_threshold: float = 0.5,
) -> Tuple[int, int]:
    """
    Find the boundaries of target_text within markdown_content.

    Args:
        target_text: The text to find.
        markdown_content: The markdown content to search in.
        search_start_offset: The offset to start searching from.
        max_search_distance: Maximum distance to search for the text.
        similarity_threshold: The minimum similarity ratio required for a match.

    Returns:
        tuple: (absolute_start_idx, absolute_end_idx) of the matched text.

    Raises:
        TextMatchingError: If the text cannot be matched with sufficient similarity.
    """
    corpus_to_search = markdown_content[search_start_offset:]

    if not corpus_to_search:
        raise TextMatchingError("No remaining markdown content to search in.")

    best_match = {
        "start": -1,
        "end": -1,
        "ratio": -1.0
    }

    # Calculate search parameters
    target_len = len(target_text)
    min_len = int(target_len * 0.9)  # Min length of MD segment to check
    max_len = int(target_len * 1.1)  # Max length of MD segment to check

    # Get effective prefix for initial matching
    effective_prefix = ' '.join(target_text.split())[:12]

    # Find potential starting points in the corpus with prev_formula_length
    try:
        potential_start_indices = _find_potential_start_indices(
            corpus_to_search, effective_prefix, max_search_distance
        )
    except Exception as e:
        logger.error(f"Failed to find potential start indices: {e}")
        logger.error(f"Context - Target text prefix: '{target_text[:50]}...'")
        logger.error(f"Context - Corpus preview: '{corpus_to_search[:200]}...'")
        raise

    # Test each potential starting point with different lengths
    for start_idx in potential_start_indices:
        best_match = _find_best_match_at_position(
            corpus_to_search,
            target_text,
            start_idx,
            min_len,
            max_len,
            best_match
        )

    # Check if we found a good enough match
    if best_match["ratio"] >= similarity_threshold:
        return (
            search_start_offset + best_match["start"],
            search_start_offset + best_match["end"]
        )
    else:
        # If close to threshold, log the best match for debugging
        if best_match["ratio"] > 0.5:
            logger.warning(
                f"Best match below threshold: {best_match['ratio']:.2f} "
                f"(threshold: {similarity_threshold})"
            )

        raise TextMatchingError(
            f"Could not find text with sufficient similarity (threshold: {similarity_threshold}). "
            f"Best match had similarity: {best_match['ratio']:.2f}. "
            f"Text to find (truncated): '{target_text[:50]}...'"
        )


def _find_potential_start_indices(corpus: str, prefix: str, max_search_distance: int = 0) -> List[int]:
    """
    Find potential starting indices for the target text in the corpus.

    Args:
        corpus: The text to search in.
        prefix: The prefix to use for initial matching.
        max_search_distance: Maximum distance to search within.

    Returns:
        List of potential starting indices in the corpus.
    """
    indices = []

    # Limit corpus to search within max_search_distance
    limited_corpus = corpus[:max_search_distance]

    current_idx = 0
    while current_idx < len(limited_corpus):
        try:
            relative_idx = limited_corpus[current_idx:].find(prefix)
            if relative_idx == -1:
                break
            actual_idx = current_idx + relative_idx
            indices.append(actual_idx)
            current_idx = actual_idx + len(prefix)  # Move past this found prefix
        except Exception as e:
            logger.warning(f"Error during prefix search: {e}")
            break

    if not indices and prefix:
        logger.info("No exact matches found, trying fuzzy matching...")

        # Try with full prefix first
        fuzzy_candidates = _find_fuzzy_prefix_matches(
            limited_corpus,
            prefix,
            similarity_threshold=0.9
        )

        if fuzzy_candidates:
            logger.info(f"Found {len(fuzzy_candidates)} fuzzy matches for prefix")
        else:
            logger.warning(f"No fuzzy matches found for prefix: '{prefix}'")

        # Convert fuzzy candidates to indices
        indices.extend([idx for idx, _ in fuzzy_candidates])

    if not indices:
        logger.error(f"No potential matches found for prefix: '{prefix}'")
        raise Exception("No potential match found.")

    return indices


def _find_best_match_at_position(
        corpus: str,
        target: str,
        start_idx: int,
        min_len: int,
        max_len: int,
        current_best: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Find the best match for the target text starting at the given position in the corpus.

    Args:
        corpus: The text to search in.
        target: The target text to match.
        start_idx: The starting index in the corpus.
        min_len: The minimum length to check.
        max_len: The maximum length to check.
        current_best: The current best match information.

    Returns:
        Updated best match information.
    """
    for length in range(min_len, max_len + 1):
        if start_idx + length > len(corpus):
            break  # Segment would go out of bounds

        segment = corpus[start_idx:start_idx + length]

        # Calculate similarity ratio
        current_ratio = difflib.SequenceMatcher(
            None, target, segment, autojunk=False
        ).ratio()

        if current_ratio > current_best["ratio"]:
            current_best = {
                "start": start_idx,
                "end": start_idx + length,
                "ratio": current_ratio
            }

    return current_best


def _create_result_json(gt_data: List[Dict[str, str]], markdown_string: str) -> str:
    """
    Process ground truth JSON and Markdown to produce a new JSON.
    Uses text blocks from ground truth as anchors to extract formulas from Markdown.

    Args:
        gt_data: The ground truth data as a list of dictionaries.
        markdown_string: A string containing the Markdown content.

    Returns:
        A JSON string for result.json, structured like gt.json.
    """
    result_list = []
    search_position = 0
    prev_formula_length = 0  # Track previous formula length

    # Pre-compute text anchors with their positions for later use
    text_anchors = []
    for i, gt_item in enumerate(gt_data):
        if gt_item.get("type") == 'text':
            text_anchors.append({
                "index": i,
                "data": gt_item.get("data", ""),
                "start": -1,  # Will be filled when we find the text
                "end": -1
            })

    current_anchor_idx = 0

    for i, gt_item in enumerate(gt_data):
        item_type = gt_item.get("type")
        item_data = gt_item.get("data", "")

        if item_type == 'text':
            try:
                # Find and extract text from markdown with previous formula length
                text_start, text_end = _find_text_in_markdown(
                    item_data, markdown_string, search_position,
                    max_search_distance=prev_formula_length + len(item_data)
                )

                extracted_text = markdown_string[text_start:text_end]
                result_list.append({"type": "text", "data": extracted_text})
                search_position = text_end
                prev_formula_length = 0  # Reset after finding text

                # Store found positions for later use
                if current_anchor_idx < len(text_anchors) and text_anchors[current_anchor_idx]["index"] == i:
                    text_anchors[current_anchor_idx]["start"] = text_start
                    text_anchors[current_anchor_idx]["end"] = text_end
                    current_anchor_idx += 1

            except TextMatchingError as e:
                logger.error(f"Text matching error: {e}")
                logger.info("Using ground truth text as fallback and continuing.")
                result_list.append({"type": "text", "data": item_data})

                # Advance the search position to avoid getting stuck
                advance_amount = len(item_data) if item_data else 50
                search_position = min(search_position + advance_amount, len(markdown_string))

        elif item_type == 'formula':
            formula_start = search_position
            formula_end = len(markdown_string)

            # Find the next text anchor to determine where this formula ends
            # Instead of calling find_text_in_markdown again, use the pre-computed anchors
            next_anchor = None
            for anchor in text_anchors:
                if anchor["index"] > i and anchor["start"] != -1:
                    next_anchor = anchor
                    break

            if next_anchor:
                formula_end = next_anchor["start"]
            else:
                # If no next anchor found, check if there's one that needs computing
                next_text_anchor = _find_next_text_anchor(gt_data, i)
                if next_text_anchor:
                    try:
                        next_text_start, _ = _find_text_in_markdown(
                            next_text_anchor,
                            markdown_string,
                            formula_start,
                            max_search_distance=int(len(item_data) * 5) + len(next_text_anchor)
                        )
                        formula_end = next_text_start
                    except TextMatchingError as e:
                        logger.warning(f"Warning when finding formula boundaries: {e}")
                        logger.info("Formula may extend to the end of Markdown.")

            extracted_formula = markdown_string[formula_start:formula_end].strip()
            prev_formula_length = len(extracted_formula)  # Store formula length for next iteration
            result_list.append({"type": "formula", "data": extracted_formula})
            search_position = formula_end

    return json.dumps(result_list, indent=4)


def _find_next_text_anchor(gt_data: List[Dict[str, Any]], current_index: int) -> Optional[str]:
    """
    Find the next text item in the ground truth data after current_index.

    Args:
        gt_data: The ground truth data list.
        current_index: The current item index.

    Returns:
        The next text anchor or None if not found.
    """
    for j in range(current_index + 1, len(gt_data)):
        if gt_data[j].get("type") == 'text':
            return gt_data[j].get("data", "")
    return None


def extract_segments_from_md(
        gt_json_path: Path,
        input_md_path: Path,
        output_json_path: Path,
) -> None:
    try:
        # Load and parse input files
        with open(gt_json_path, 'r', encoding='utf-8') as f_gt:
            gt_data = json.load(f_gt)

        with open(input_md_path, 'r', encoding='utf-8') as f_md:
            markdown_str = f_md.read()

        # Generate and save result
        result = _create_result_json(gt_data, markdown_str)

        with open(output_json_path, 'w', encoding='utf-8') as f_res:
            f_res.write(result)

        logger.info(f"Result JSON has been generated at {output_json_path}")

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        raise