import json
import re
import gradio as gr
import subprocess
import tempfile
import os
import logging
from pathlib import Path
from pylatexenc.latexencode import UnicodeToLatexEncoder
from ..pipeline import PipelinePaths, BenchmarkRunConfig


# ========== CONSTANTS ==========

CUSTOM_CSS = """
.summary-stats {
    text-align: center;
    margin-bottom: 20px;
}
.summary-value {
    font-size: 20px;
    font-weight: bold;
}
.formula-container {
    margin: 15px 0;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 8px;
    background-color: #f9f9f9;
}
.formula-display-container {
    display: flex;
    min-height: 150px;
    gap: 10px;
    align-items: stretch;
}
.formula-box {
    flex: 1;
    padding: 15px;
    border: 1px solid #ddd;
    border-radius: 8px;
    background-color: #ffffff;
    display: flex;
    flex-direction: column;
}
.comparison-box {
    width: 120px;
    min-width: 120px;
    max-width: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 0;
    flex-shrink: 0;
}
.navigation-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 15px 0;
}
.file-selection {
    margin-bottom: 20px;
}
"""

STANDARD_DISPLAY_OUTPUTS = [
    "formula_progress", "ground_truth_formula", "extracted_formula",
    "combined_status", "explanation", "errors", "formula_index", "formula_number_input"
]

FULL_RELOAD_OUTPUTS = [
    "summary_html_component", "formula_progress", "ground_truth_formula", "extracted_formula",
    "combined_status", "explanation", "errors", "formula_index", "formula_number_input",
    "current_data", "current_stats_data", "formula_number_input", "judge_model_dropdown"
]


# ========== EXCEPTIONS ==========

class LatexRenderError(Exception):
    """Exception raised when LaTeX rendering fails."""
    pass


# ========== FORMULA RENDERING ==========

class FormulaRenderer:
    """Handles LaTeX to PNG conversion."""
    
    LATEX_TEMPLATE = """
\\documentclass[preview, border=5pt]{{standalone}}
\\usepackage{{amsmath,amssymb,amsfonts}}
\\usepackage[version=4]{{mhchem}}
\\usepackage{{varwidth}}
\\begin{{document}}
\\begin{{varwidth}}{{25cm}}
{formula}
\\end{{varwidth}}
\\end{{document}}
"""
    
    def preprocess_unicode(self, text: str) -> str:
        """Convert Unicode mathematical symbols to LaTeX commands using pylatexenc."""
        try:
            has_dollars = '$' in text
            rules = 'unicode-xml' if has_dollars else 'defaults'
            
            encoder = UnicodeToLatexEncoder(
                conversion_rules=[rules],
                non_ascii_only=True,
                replacement_latex_protection='braces',
            )
            
            converted = encoder.unicode_to_latex(text)
            return converted
        except Exception as e:
            raise LatexRenderError(f"Unicode conversion failed: {e}")

    def render(self, latex_formula: str) -> str:
        """Convert LaTeX formula to PNG using pdflatex + magick."""
        preprocessed_formula = self.preprocess_unicode(latex_formula)
        latex_doc = self.LATEX_TEMPLATE.format(formula=preprocessed_formula)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
            output_path = tmp_png.name
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_file = os.path.join(tmpdir, 'formula.tex')
            pdf_file = os.path.join(tmpdir, 'formula.pdf')
            
            with open(tex_file, 'w') as f:
                f.write(latex_doc)
            
            result = subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', '-output-directory', tmpdir, tex_file], 
                capture_output=True, text=True, encoding='utf-8', errors='replace'
            )
            
            if result.returncode != 0:
                logging.debug(f"LaTeX Warning: {result.stdout}\n{result.stderr}")

            if not os.path.exists(pdf_file):
                raise LatexRenderError(f"PDF not generated for formula: {latex_formula}")
            
            try:
                subprocess.run(
                    ['magick', '-density', '800', pdf_file, '-quality', '100', output_path],
                    check=True, capture_output=True
                )
            except subprocess.CalledProcessError as e:
                raise LatexRenderError(f"ImageMagick conversion failed: {e}")
            
            return output_path
    
    def create_error_image(self, error_msg: str) -> str:
        """Create a simple error image with text."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_png:
            error_png_path = tmp_png.name
        
        short_error = error_msg[:50] + "..." if len(error_msg) > 50 else error_msg

        subprocess.run([
            'magick', '-size', '600x300', 'xc:white',
            '-pointsize', '14', '-fill', 'red',
            '-gravity', 'center', '-annotate', '+0-20', 'LaTeX Render Error',
            '-pointsize', '10', '-annotate', '+0+20', short_error,
            error_png_path
        ], check=True, capture_output=True)
        return error_png_path


# ========== DATA LOADING ==========

def load_formula_data(formula_results_path: Path) -> dict[int, dict[str, dict]] | None:
    """Load formula evaluation results from JSON file and group by formula_number and judge_model."""
    if not formula_results_path.exists():
        return None
    
    with open(formula_results_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Group evaluations by formula_number, then by judge_model
    grouped_data = {}
    for entry in raw_data:
        formula_num = entry['formula_number']
        ground_truth = entry['ground_truth_formula']
        extracted = entry['extracted_formula']
        
        if formula_num not in grouped_data:
            grouped_data[formula_num] = {}
        
        # Process each evaluation in the llm_evals array
        for eval_entry in entry['llm_evals']:
            judge_model = eval_entry['judge_model']
            
            # Create a combined entry with formula data and evaluation results
            combined_entry = {
                'formula_number': formula_num,
                'ground_truth_formula': ground_truth,
                'extracted_formula': extracted,
                'judge_model': judge_model,
                'explanation': eval_entry['explanation'],
                'is_correct': eval_entry['is_correct'],
                'score': eval_entry['score'],
                'errors': eval_entry['errors']
            }
            
            grouped_data[formula_num][judge_model] = combined_entry
    
    return grouped_data


def load_stats_data(stats_path: Path) -> dict | None:
    """Load evaluation statistics from JSON file."""
    if not stats_path.exists():
        return None
    
    with open(stats_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ========== HTML GENERATION ==========

def create_summary_html(stats: dict, judge_model: str = None) -> str:
    """Create HTML for summary statistics for a specific judge model."""
    formula_stats_all = stats.get('formula_statistics', {})
    text_stats = stats.get('text_statistics', {})
    
    # Handle the llm_judge array format
    llm_judge_array = formula_stats_all.get('llm_judge', [])
    
    if llm_judge_array:
        # Find the specific judge model or use the first one
        if judge_model:
            formula_stats = next((stats for stats in llm_judge_array if stats.get('judge_model') == judge_model), {})
            if not formula_stats:
                # Fallback to first if specified judge not found
                formula_stats = llm_judge_array[0]
                judge_model = formula_stats.get('judge_model', 'N/A')
        else:
            formula_stats = llm_judge_array[0]
            judge_model = formula_stats.get('judge_model', 'N/A')
    else:
        formula_stats = {}
        judge_model = 'N/A'
    
    # Get total formulas from the main formula_statistics level
    total_formulas = formula_stats_all.get('total_formulas', 'N/A')
    
    # Format accuracy percentage
    accuracy = formula_stats.get('accuracy_percentage')
    accuracy_str = f"{accuracy:.1f}" if accuracy and accuracy != int(accuracy) else str(int(accuracy)) if accuracy else "N/A"
    
    # Format average score
    avg_score = formula_stats.get('average_score')
    score_str = f"{avg_score:.1f}" if avg_score and avg_score != int(avg_score) else str(int(avg_score)) if avg_score else "N/A"
    
    # Format levenshtein similarity
    lev_sim = text_stats.get('average_levenshtein_similarity')
    lev_sim_str = f"{lev_sim:.3f}" if lev_sim else "N/A"
    
    return f"""
    <div class="summary-stats">
        <div style="display: inline-block; margin: 0 15px;">
            <div>Judge Model</div>
            <div class="summary-value">{judge_model}</div>
        </div>
        <div style="display: inline-block; margin: 0 15px;">
            <div>Total Formulas</div>
            <div class="summary-value">{total_formulas}</div>
        </div>
        <div style="display: inline-block; margin: 0 15px;">
            <div>Correct Formulas</div>
            <div class="summary-value">{formula_stats.get('correct_formulas', 'N/A')}</div>
        </div>
        <div style="display: inline-block; margin: 0 15px;">
            <div>Accuracy</div>
            <div class="summary-value">{accuracy_str}%</div>
        </div>
        <div style="display: inline-block; margin: 0 15px;">
            <div>Average Score</div>
            <div class="summary-value">{score_str}</div>
        </div>
        <div style="display: inline-block; margin: 0 15px;">
            <div>Avg Levenshtein Sim</div>
            <div class="summary-value">{lev_sim_str}</div>
        </div>
    </div>
    """


def create_display_html(formula_number: int, total_formulas: int, is_correct: bool, score: float) -> tuple[str, str]:
    """Create HTML for progress and result display."""
    # Progress HTML
    progress_percentage = (formula_number / total_formulas) * 100 if total_formulas > 0 else 0
    progress_html = f"""
    <div style="margin: 10px 0; display: flex; flex-direction: column;">
        <div style="font-weight: bold; text-align: center; margin-bottom: 5px;">
            Formula {formula_number} of {total_formulas}
        </div>
        <div style="width: 100%; background-color: #eee; border-radius: 10px; height: 10px;">
            <div style="background-color: #3498db; width: {progress_percentage}%; height: 100%; border-radius: 10px;"></div>
        </div>
    </div>
    """
    
    # Result HTML
    bg_color = 'rgba(46, 204, 113, 0.2)' if is_correct else 'rgba(231, 76, 60, 0.2)'
    border_color = '#2ecc71' if is_correct else '#e74c3c'
    status_text = "Correct" if is_correct else "Incorrect"
    status_icon = "✓" if is_correct else "✗"
    
    result_html = f"""
    <div style="padding: 10px; border-radius: 5px; background-color: {bg_color}; 
                border: 1px solid {border_color}; display: flex; flex-direction: column; 
                align-items: center; justify-content: center; margin: 0; width: 100%;">
        <span style="font-size: 24px; margin-bottom: 5px;">{status_icon}</span>
        <span style="font-weight: bold; margin-bottom: 10px;">{status_text}</span>
        <div style="display: flex; align-items: center;">
            <span style="font-weight: bold; margin-right: 5px;">Score:</span>
            <span>{score}</span>
        </div>
    </div>
    """
    
    return progress_html, result_html


def create_errors_html(errors: list[str]) -> str:
    """Create HTML for errors list."""
    if not errors:
        return "No errors found."

    return "<ul>" + "".join([f"<li>{error}</li>" for error in errors]) + "</ul>"


# ========== DATA UTILITIES ==========

def get_available_runs_and_parsers(paths: PipelinePaths) -> dict[str, dict[str, list[str]]]:
    """Get all available run/parser combinations organized by timestamp -> pdf -> parsers."""
    data = {}
    
    if not paths.runs_dir.exists():
        return data
    
    for timestamp_dir in paths.runs_dir.iterdir():
        if not timestamp_dir.is_dir():
            continue
            
        timestamp_data = {}
        
        for pdf_dir in timestamp_dir.iterdir():
            if not pdf_dir.is_dir():
                continue
                
            parsers = []
            
            for parser_dir in pdf_dir.iterdir():
                if not parser_dir.is_dir():
                    continue
                    
                # Check if evaluation results exist
                eval_formula_path = parser_dir / "eval_formula_results.json"
                eval_stats_path = parser_dir / "eval_stats.json"
                
                if eval_formula_path.exists() and eval_stats_path.exists():
                    parsers.append(parser_dir.name)
            
            if parsers:
                timestamp_data[pdf_dir.name] = sorted(parsers)
        
        if timestamp_data:
            data[timestamp_dir.name] = timestamp_data
    
    return data


def get_timestamps_list(data: dict) -> list[str]:
    """Get sorted list of available timestamps."""
    return sorted(data.keys(), reverse=True)  # Most recent first


def get_pdfs_for_timestamp(data: dict, timestamp: str) -> list[str]:
    """Get sorted list of PDFs for a given timestamp."""
    if timestamp not in data:
        return []
    return sorted(data[timestamp].keys())


def get_parsers_for_pdf(data: dict, timestamp: str, pdf_name: str) -> list[str]:
    """Get sorted list of parsers for a given timestamp and PDF."""
    if timestamp not in data or pdf_name not in data[timestamp]:
        return []
    return sorted(data[timestamp][pdf_name])




def get_judge_models_for_combination(formula_data: dict[int, dict[str, dict]] | None) -> list[str]:
    """Get sorted list of available judge models from the formula data."""
    if not formula_data:
        return []
    
    judge_models = set()
    for formula_evaluations in formula_data.values():
        judge_models.update(formula_evaluations.keys())
    
    return sorted(judge_models)


# ========== EVENT HANDLERS ==========

def render_formula_to_image(renderer: FormulaRenderer, formula_text: str) -> str | None:
    """Render formula to PNG image, return path or error image if failed."""
    try:
        return renderer.render(formula_text)
    except LatexRenderError as e:
        logging.warning(f"Formula rendering failed: {e}")
        return renderer.create_error_image(str(e))


def update_display(idx: int, judge_model: str, formula_data: dict[int, dict[str, dict]] | None, renderer: FormulaRenderer):
    """Update display based on index and judge model."""
    if not formula_data or not judge_model or idx < 0:
        return ("", None, None, "", "", "", "", "", idx, 1)

    formula_numbers = sorted(formula_data.keys())
    if idx >= len(formula_numbers):
        return ("", None, None, "", "", "", "", "", idx, 1)

    formula_num = formula_numbers[idx]
    judge_evaluations = formula_data[formula_num]
    
    if judge_model not in judge_evaluations:
        # If selected judge model doesn't exist for this formula, use first available
        available_judges = list(judge_evaluations.keys())
        if not available_judges:
            return ("", None, None, "", "", "", "", "", idx, 1)
        judge_model = available_judges[0]

    formula = judge_evaluations[judge_model]
    formula_number = idx + 1
    total_formulas = len(formula_numbers)

    # Create HTML components
    progress_html, result_html = create_display_html(
        formula_number, total_formulas, 
        formula.get('is_correct'),
        formula.get('score')
    )
    errors_html = create_errors_html(formula.get('errors'))
    
    # Render formulas to images
    ground_truth_img = render_formula_to_image(renderer, formula.get('ground_truth_formula'))
    extracted_img = render_formula_to_image(renderer, formula.get('extracted_formula'))
    
    # Get raw LaTeX text
    ground_truth_text = formula.get('ground_truth_formula', '')
    extracted_text = formula.get('extracted_formula', '')

    return (
        progress_html,
        ground_truth_img,
        extracted_img,
        result_html,
        formula.get('explanation'),
        errors_html,
        ground_truth_text,
        extracted_text,
        idx,
        formula_number
    )


def update_pdf_choices(timestamp: str, data_dict: dict):
    """Update PDF choices based on timestamp."""
    pdfs = get_pdfs_for_timestamp(data_dict, timestamp)
    first_pdf = pdfs[0] if pdfs else None
    parsers = get_parsers_for_pdf(data_dict, timestamp, first_pdf) if first_pdf else []
    first_parser = parsers[0] if parsers else None
    return (
        gr.update(choices=pdfs, value=first_pdf),
        gr.update(choices=parsers, value=first_parser)
    )


def update_parser_choices(timestamp: str, pdf_name: str, data_dict: dict, current_parser: str = None):
    """Update parser choices based on timestamp and PDF."""
    parsers = get_parsers_for_pdf(data_dict, timestamp, pdf_name)
    
    # Try to keep current parser if it's available for the new PDF
    if current_parser and current_parser in parsers:
        selected_parser = current_parser
    else:
        selected_parser = parsers[0] if parsers else None
        
    return gr.update(choices=parsers, value=selected_parser)


def load_combination(timestamp: str, pdf_name: str, parser: str, current_idx: int, paths: PipelinePaths, renderer: FormulaRenderer):
    """Load a combination and update the interface."""
    if not all([timestamp, pdf_name, parser]):
        return ("", "", "", "", "", "", "", "", "", 0, 1, None, None, 1, gr.update(choices=[], value=None))

    try:
        # Create BenchmarkRunConfig to get file paths
        run_config = BenchmarkRunConfig(
            name=pdf_name,
            timestamp=timestamp,
            paths=paths
        )

        # Load the evaluation data
        formula_data = load_formula_data(run_config.eval_formula_results_path(parser))
        stats_data = load_stats_data(run_config.eval_stats_path(parser))

        if not formula_data or not stats_data:
            return (
                "No evaluation data found",
                "", "", "", "", "", "", "", "",
                0, 1, None, None, 1, gr.update(choices=[], value=None)
            )

        # Get available judge models and select first one
        judge_models = get_judge_models_for_combination(formula_data)
        first_judge = judge_models[0] if judge_models else None

        if not first_judge:
            return (
                "No judge models found",
                "", "", "", "", "", "", "", "",
                0, 1, None, None, 1, gr.update(choices=[], value=None)
            )

        # Validate index
        formula_numbers = sorted(formula_data.keys())
        if current_idx >= len(formula_numbers):
            current_idx = len(formula_numbers) - 1
        if current_idx < 0:
            current_idx = 0
        
        # Create summary HTML and display
        summary_html = create_summary_html(stats_data, first_judge)
        display_result = update_display(current_idx, first_judge, formula_data, renderer)

        # Return everything needed to update the UI
        return (
            summary_html,
            display_result[0],  # progress_html
            display_result[1],  # ground_truth image
            display_result[2],  # extracted image
            display_result[3],  # result_html
            display_result[4],  # explanation
            display_result[5],  # errors_html
            display_result[6],  # ground_truth_text
            display_result[7],  # extracted_text
            current_idx,  # formula_index
            display_result[9],  # formula_number for input
            formula_data,  # current_data
            stats_data,  # current_stats_data
            display_result[9],  # formula_number for input (duplicate for consistency)
            gr.update(choices=judge_models, value=first_judge)  # judge model dropdown
        )
        
    except Exception as e:
        print(f"Error loading combination {timestamp}/{pdf_name}/{parser}: {e}")
        return (
            "Error loading data",
            "", "", "", "", "", "", "", "",
            0, 1, None, None, 1, gr.update(choices=[], value=None)
        )


def go_to_prev(current_idx: int, judge_model: str, formula_data: dict[int, dict[str, dict]] | None, renderer: FormulaRenderer):
    """Navigate to previous formula."""
    if formula_data and current_idx > 0:
        current_idx -= 1
    return update_display(current_idx, judge_model, formula_data, renderer)


def go_to_next(current_idx: int, judge_model: str, formula_data: dict[int, dict[str, dict]] | None, renderer: FormulaRenderer):
    """Navigate to next formula."""
    if formula_data:
        formula_numbers = sorted(formula_data.keys())
        if current_idx < len(formula_numbers) - 1:
            current_idx += 1
    return update_display(current_idx, judge_model, formula_data, renderer)


def go_to_number(formula_num: int, judge_model: str, formula_data: dict[int, dict[str, dict]] | None, renderer: FormulaRenderer):
    """Navigate to specific formula number."""
    # Convert 1-based formula number to 0-based index
    idx = formula_num - 1
    if formula_data:
        formula_numbers = sorted(formula_data.keys())
        if idx < 0:
            idx = 0
        if idx >= len(formula_numbers):
            idx = len(formula_numbers) - 1
    else:
        idx = 0
    return update_display(idx, judge_model, formula_data, renderer)


def judge_model_changed(new_judge_model: str, current_idx: int, formula_data: dict[int, dict[str, dict]] | None, renderer: FormulaRenderer):
    """Update display when judge model changes."""
    return update_display(current_idx, new_judge_model, formula_data, renderer)


# ========== GRADIO INTERFACE ==========

def create_formula_viewer():
    """Create the Gradio interface for formula comparison."""
    paths = PipelinePaths()
    

    # Get all available combinations
    data = get_available_runs_and_parsers(paths)

    if not data:
        print(f"Error: No evaluation results found in {paths.runs_dir}")
        return None

    # Get initial choices
    timestamps = get_timestamps_list(data)
    initial_timestamp = timestamps[0] if timestamps else None
    initial_pdfs = get_pdfs_for_timestamp(data, initial_timestamp) if initial_timestamp else []
    initial_pdf = initial_pdfs[0] if initial_pdfs else None
    initial_parsers = get_parsers_for_pdf(data, initial_timestamp, initial_pdf) if initial_timestamp and initial_pdf else []
    initial_parser = initial_parsers[0] if initial_parsers else None

    # Create the interface
    with gr.Blocks(title="Formula Comparison Viewer", theme=gr.themes.Soft(), css=CUSTOM_CSS) as interface:
        gr.Markdown("# Formula Comparison Viewer")

        # Hierarchical file selection
        with gr.Row(elem_classes="file-selection"):
            with gr.Column():
                timestamp_dropdown = gr.Dropdown(
                    choices=timestamps,
                    label="Select Timestamp",
                    value=initial_timestamp
                )
            with gr.Column():
                pdf_dropdown = gr.Dropdown(
                    choices=initial_pdfs,
                    label="Select PDF",
                    value=initial_pdf
                )
            with gr.Column():
                parser_dropdown = gr.Dropdown(
                    choices=initial_parsers,
                    label="Select Parser",
                    value=initial_parser
                )
            with gr.Column():
                judge_model_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Judge Model",
                    value=None
                )

        # Summary statistics
        summary_html_component = gr.HTML()

        # Formula progress indicator
        formula_progress = gr.HTML()

        # Direct navigation controls
        with gr.Row(elem_classes="navigation-row"):
            prev_btn = gr.Button("← Previous")
            formula_number_input = gr.Number(
                label="Formula Number",
                value=1,
                minimum=1,
                step=1
            )
            next_btn = gr.Button("Next →")

        gr.Markdown("## Formula Comparison")

        # Side by side display with comparison in middle
        with gr.Row(elem_classes="formula-display-container", elem_id="formula-comparison-row"):
            with gr.Column(elem_classes="formula-box"):
                gr.Markdown("### Ground Truth Formula")
                ground_truth_formula = gr.Image(show_label=False, show_download_button=False, container=True)
                ground_truth_text = gr.Textbox(label="Raw LaTeX", lines=3, interactive=False)

            # Comparison box with fixed width
            with gr.Column(elem_classes="comparison-box", elem_id="status-comparison-box", scale=0):
                combined_status = gr.HTML()

            with gr.Column(elem_classes="formula-box"):
                gr.Markdown("### Extracted Formula")
                extracted_formula = gr.Image(show_label=False, show_download_button=False, container=True)
                extracted_text = gr.Textbox(label="Raw LaTeX", lines=3, interactive=False)

        # Explanation and Errors
        gr.Markdown("### Explanation")
        explanation = gr.Textbox(lines=3)

        gr.Markdown("### Errors")
        errors = gr.HTML()

        # State variables to track current data
        current_data = gr.State(None)  # Will hold the current formula data
        current_stats_data = gr.State(None)  # Store the current stats data
        formula_index = gr.State(0)  # 0-based index for internal use
        current_judge_model = gr.State(None)  # Current selected judge model
        hierarchical_data = gr.State(data)  # Store the hierarchical data

        # Initialize formula renderer
        renderer = FormulaRenderer()
        
        # Connect event handlers for dropdown changes
        timestamp_dropdown.change(
            update_pdf_choices,
            inputs=[timestamp_dropdown, hierarchical_data],
            outputs=[pdf_dropdown, parser_dropdown]
        )

        pdf_dropdown.change(
            update_parser_choices,
            inputs=[timestamp_dropdown, pdf_dropdown, hierarchical_data, parser_dropdown],
            outputs=[parser_dropdown]
        )

        # Connect event handlers for loading data
        def reload_data(timestamp, pdf_name, parser, current_idx):
            return load_combination(timestamp, pdf_name, parser, current_idx, paths, renderer)

        # Full reload outputs for dropdown changes
        full_reload_outputs = [
            summary_html_component, formula_progress, ground_truth_formula, extracted_formula,
            combined_status, explanation, errors, ground_truth_text, extracted_text,
            formula_index, formula_number_input, current_data, current_stats_data, 
            formula_number_input, judge_model_dropdown
        ]
        
        for dropdown in [timestamp_dropdown, pdf_dropdown, parser_dropdown]:
            dropdown.change(
                reload_data,
                inputs=[timestamp_dropdown, pdf_dropdown, parser_dropdown, formula_index],
                outputs=full_reload_outputs
            )

        # Standard navigation events with same output pattern
        navigation_outputs = [
            formula_progress, ground_truth_formula, extracted_formula,
            combined_status, explanation, errors, ground_truth_text, extracted_text,
            formula_index, formula_number_input
        ]
        
        judge_model_dropdown.change(
            lambda judge_model, idx, data: judge_model_changed(judge_model, idx, data, renderer),
            inputs=[judge_model_dropdown, formula_index, current_data],
            outputs=navigation_outputs
        )

        prev_btn.click(
            lambda idx, judge_model, data: go_to_prev(idx, judge_model, data, renderer),
            inputs=[formula_index, judge_model_dropdown, current_data],
            outputs=navigation_outputs
        )

        next_btn.click(
            lambda idx, judge_model, data: go_to_next(idx, judge_model, data, renderer),
            inputs=[formula_index, judge_model_dropdown, current_data],
            outputs=navigation_outputs
        )

        formula_number_input.change(
            lambda formula_num, judge_model, data: go_to_number(formula_num, judge_model, data, renderer),
            inputs=[formula_number_input, judge_model_dropdown, current_data],
            outputs=navigation_outputs
        )

        # Initialize the interface
        interface.load(
            reload_data,
            inputs=[timestamp_dropdown, pdf_dropdown, parser_dropdown, formula_index],
            outputs=full_reload_outputs
        )

    return interface


# ========== MAIN ENTRY POINT ==========

def main():
    # Create and launch the Gradio interface
    app = create_formula_viewer()
    if app:
        app.launch()
    else:
        print("Failed to create Gradio interface.")


if __name__ == "__main__":
    main()