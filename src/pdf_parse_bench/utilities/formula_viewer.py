import json
import logging
from pathlib import Path
import gradio as gr
from pydantic import BaseModel

from ..pipeline import PipelinePaths, BenchmarkRunConfig
from ..eval import LLMJudgeEval, LLMJudgeStatistics, SummaryStatistics, FormulaEvaluationSummary


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



# ========== PYDANTIC MODELS ==========


class ViewerState(BaseModel):
    """Current state of the formula viewer."""
    run_config: BenchmarkRunConfig
    parser: str
    formula_evaluations: dict[int, FormulaEvaluationSummary]  # formula_number -> evaluation
    stats: SummaryStatistics
    current_formula_index: int = 0
    current_judge_model: str | None = None
    
    @property
    def available_judge_models(self) -> list[str]:
        """Get sorted list of available judge models."""
        judge_models = set()
        for evaluation in self.formula_evaluations.values():
            judge_models.update(evaluation.llm_evals_by_model.keys())
        return sorted(judge_models)
    
    @property
    def total_formulas(self) -> int:
        """Get total number of formulas."""
        return len(self.formula_evaluations)
    
    @property
    def current_formula(self) -> FormulaEvaluationSummary | None:
        """Get current formula evaluation."""
        formula_numbers = sorted(self.formula_evaluations.keys())
        if 0 <= self.current_formula_index < len(formula_numbers):
            formula_num = formula_numbers[self.current_formula_index]
            return self.formula_evaluations[formula_num]
        return None
    
    @property
    def current_evaluation(self) -> LLMJudgeEval | None:
        """Get current LLM evaluation."""
        formula = self.current_formula
        if formula and self.current_judge_model and self.current_judge_model in formula.llm_evals_by_model:
            return formula.llm_evals_by_model[self.current_judge_model]
        return None

    class Config:
        arbitrary_types_allowed = True  # Allow BenchmarkRunConfig


# ========== DATA LOADING ==========

def load_formula_evaluations(formula_results_path: Path) -> dict[int, FormulaEvaluationSummary] | None:
    """Load formula evaluation results from JSON file."""
    if not formula_results_path.exists():
        return None
    
    with open(formula_results_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    evaluations = {}
    for raw_entry in raw_data:
        formula_eval = FormulaEvaluationSummary.model_validate(raw_entry)
        evaluations[formula_eval.formula_number] = formula_eval
    
    return evaluations


def load_evaluation_stats(stats_path: Path) -> SummaryStatistics | None:
    """Load evaluation statistics from JSON file."""
    if not stats_path.exists():
        return None
    
    with open(stats_path, 'r', encoding='utf-8') as f:
        raw_stats = json.load(f)
        
    return SummaryStatistics.model_validate(raw_stats)


def create_viewer_state(run_config: BenchmarkRunConfig, parser: str) -> ViewerState | None:
    """Create a complete viewer state from run config and parser."""
    formula_evaluations = load_formula_evaluations(run_config.eval_formula_results_path(parser))
    stats = load_evaluation_stats(run_config.eval_stats_path(parser))
    
    if not formula_evaluations or not stats:
        return None
    
    # Get first available judge model
    first_judge = None
    if formula_evaluations:
        for evaluation in formula_evaluations.values():
            if evaluation.llm_evals:
                first_judge = next(iter(evaluation.llm_evals_by_model.keys()))
                break
    
    return ViewerState(
        run_config=run_config,
        parser=parser,
        formula_evaluations=formula_evaluations,
        stats=stats,
        current_formula_index=0,
        current_judge_model=first_judge
    )


# ========== HTML GENERATION ==========

def create_summary_html(stats: SummaryStatistics, judge_model: str | None = None) -> str:
    """Create HTML for summary statistics for a specific judge model."""
    judge_stats_dict = stats.llm_judge_stats_by_model
    if not judge_model or judge_model not in judge_stats_dict:
        # Use first available judge model
        if judge_stats_dict:
            judge_model = next(iter(judge_stats_dict.keys()))
        else:
            judge_model = 'N/A'
    
    if judge_model == 'N/A' or judge_model not in judge_stats_dict:
        formula_stats = LLMJudgeStatistics(
            judge_model='N/A',
            correct_formulas=0,
            accuracy_percentage=0.0,
            average_score=0.0,
            average_inline_score=0.0,
            average_display_score=0.0
        )
    else:
        formula_stats = judge_stats_dict[judge_model]
    
    # Format numbers
    accuracy_str = f"{formula_stats.accuracy_percentage:.1f}" if formula_stats.accuracy_percentage != int(formula_stats.accuracy_percentage) else str(int(formula_stats.accuracy_percentage))
    score_str = f"{formula_stats.average_score:.1f}" if formula_stats.average_score != int(formula_stats.average_score) else str(int(formula_stats.average_score))
    lev_sim = stats.text_statistics.average_levenshtein_similarity
    lev_sim_str = f"{lev_sim:.3f}" if lev_sim else "N/A"
    
    return f"""
    <div class="summary-stats">
        <div style="display: inline-block; margin: 0 15px;">
            <div>Judge Model</div>
            <div class="summary-value">{formula_stats.judge_model}</div>
        </div>
        <div style="display: inline-block; margin: 0 15px;">
            <div>Total Formulas</div>
            <div class="summary-value">{stats.formula_statistics.total_formulas}</div>
        </div>
        <div style="display: inline-block; margin: 0 15px;">
            <div>Correct Formulas</div>
            <div class="summary-value">{formula_stats.correct_formulas}</div>
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
                
            run_config = BenchmarkRunConfig(
                name=pdf_dir.name,
                timestamp=timestamp_dir.name,
                paths=paths
            )
            
            parsers = []
            
            for parser_dir in pdf_dir.iterdir():
                if not parser_dir.is_dir():
                    continue
                    
                eval_formula_path = run_config.eval_formula_results_path(parser_dir.name)
                eval_stats_path = run_config.eval_stats_path(parser_dir.name)
                
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






# ========== EVENT HANDLERS ==========

def get_formula_image_path(run_config: BenchmarkRunConfig, parser: str | None, formula_number: int, is_ground_truth: bool = False) -> str | None:
    """Get path to pre-rendered formula PNG file."""
    try:
        if is_ground_truth:
            # Ground truth formulas are in run_directory/rendered_formulas/
            rendered_dir = run_config.rendered_formulas_dir()
        else:
            # Parser formulas are in run_directory/parser_name/rendered_formulas/
            rendered_dir = run_config.rendered_formulas_dir(parser)
        
        # Formula files are named formula_000.png, formula_001.png, etc.
        formula_filename = f"formula_{formula_number-1:03d}.png"  # Convert to 0-based index
        formula_path = rendered_dir / formula_filename
        
        if formula_path.exists() and formula_path.is_file():
            return str(formula_path)
        else:
            if formula_path.exists():
                logging.warning(f"Formula path exists but is not a file: {formula_path}")
            else:
                logging.warning(f"Formula PNG not found: {formula_path}")
            return None
    except Exception as e:
        logging.error(f"Error getting formula image path: {e}")
        return None


def update_display_from_state(state: ViewerState) -> tuple:
    """Update display based on viewer state."""
    if not state.current_formula or not state.current_evaluation:
        return ("", None, None, "", "", "", "", "", state.current_formula_index, 1)

    formula = state.current_formula
    evaluation = state.current_evaluation
    formula_number = state.current_formula_index + 1
    total_formulas = state.total_formulas

    # Create HTML components
    progress_html, result_html = create_display_html(
        formula_number, total_formulas, 
        evaluation.is_correct,
        evaluation.score
    )
    errors_html = create_errors_html(evaluation.errors)
    
    # Load pre-rendered formula images
    ground_truth_img = get_formula_image_path(state.run_config, None, formula_number, is_ground_truth=True)
    extracted_img = get_formula_image_path(state.run_config, state.parser, formula_number, is_ground_truth=False)
    
    return (
        progress_html,
        ground_truth_img,
        extracted_img,
        result_html,
        evaluation.explanation,
        errors_html,
        formula.ground_truth_formula,
        formula.extracted_formula,
        state.current_formula_index,
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


def update_parser_choices(timestamp: str, pdf_name: str, data_dict: dict, current_parser: str | None = None):
    """Update parser choices based on timestamp and PDF."""
    parsers = get_parsers_for_pdf(data_dict, timestamp, pdf_name)
    
    # Try to keep current parser if it's available for the new PDF
    if current_parser and current_parser in parsers:
        selected_parser = current_parser
    else:
        selected_parser = parsers[0] if parsers else None
        
    return gr.update(choices=parsers, value=selected_parser)


def load_combination(timestamp: str, pdf_name: str, parser: str, current_idx: int, paths: PipelinePaths):
    """Load a combination and update the interface."""
    if not all([timestamp, pdf_name, parser]):
        return ("", "", "", "", "", "", "", "", "", 0, 1, None, 1, gr.update(choices=[], value=None))

    try:
        # Create BenchmarkRunConfig
        run_config = BenchmarkRunConfig(
            name=pdf_name,
            timestamp=timestamp,
            paths=paths
        )

        # Create viewer state
        state = create_viewer_state(run_config, parser)
        if not state:
            return (
                "No evaluation data found",
                "", "", "", "", "", "", "", "",
                0, 1, None, 1, gr.update(choices=[], value=None)
            )

        # Validate and set current index
        if current_idx >= state.total_formulas:
            current_idx = state.total_formulas - 1
        if current_idx < 0:
            current_idx = 0
        state.current_formula_index = current_idx
        
        # Create summary HTML and display
        summary_html = create_summary_html(state.stats, state.current_judge_model)
        display_result = update_display_from_state(state)
        judge_models = state.available_judge_models

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
            state,  # current_state
            display_result[9],  # formula_number for input (duplicate for consistency)
            gr.update(choices=judge_models, value=state.current_judge_model)  # judge model dropdown
        )
        
    except Exception as e:
        print(f"Error loading combination {timestamp}/{pdf_name}/{parser}: {e}")
        return (
            "Error loading data",
            "", "", "", "", "", "", "", "",
            0, 1, None, 1, gr.update(choices=[], value=None)
        )


def navigate_previous(state: ViewerState) -> tuple:
    """Navigate to previous formula."""
    if state.current_formula_index > 0:
        state.current_formula_index -= 1
    return update_display_from_state(state)


def navigate_next(state: ViewerState) -> tuple:
    """Navigate to next formula."""
    if state.current_formula_index < state.total_formulas - 1:
        state.current_formula_index += 1
    return update_display_from_state(state)


def navigate_to_number(formula_num: int, state: ViewerState) -> tuple:
    """Navigate to specific formula number."""
    # Convert 1-based formula number to 0-based index
    idx = formula_num - 1
    if idx < 0:
        idx = 0
    if idx >= state.total_formulas:
        idx = state.total_formulas - 1
    state.current_formula_index = idx
    return update_display_from_state(state)


def change_judge_model(new_judge_model: str, state: ViewerState) -> tuple:
    """Update display when judge model changes."""
    state.current_judge_model = new_judge_model
    return update_display_from_state(state)


# ========== UI COMPONENT BUILDERS ==========

def create_file_selection_row(data: dict) -> tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown]:
    """Create hierarchical file selection dropdowns."""
    timestamps = get_timestamps_list(data)
    initial_timestamp = timestamps[0] if timestamps else None
    initial_pdfs = get_pdfs_for_timestamp(data, initial_timestamp) if initial_timestamp else []
    initial_pdf = initial_pdfs[0] if initial_pdfs else None
    initial_parsers = get_parsers_for_pdf(data, initial_timestamp, initial_pdf) if initial_timestamp and initial_pdf else []
    initial_parser = initial_parsers[0] if initial_parsers else None
    
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
    
    return timestamp_dropdown, pdf_dropdown, parser_dropdown, judge_model_dropdown


def create_navigation_controls() -> tuple[gr.Button, gr.Number, gr.Button]:
    """Create navigation controls."""
    with gr.Row(elem_classes="navigation-row"):
        prev_btn = gr.Button("← Previous")
        formula_number_input = gr.Number(
            label="Formula Number",
            value=1,
            minimum=1,
            step=1
        )
        next_btn = gr.Button("Next →")
    
    return prev_btn, formula_number_input, next_btn


def create_formula_comparison_section() -> tuple[gr.Image, gr.Textbox, gr.HTML, gr.Image, gr.Textbox]:
    """Create the main formula comparison section."""
    gr.Markdown("## Formula Comparison")
    
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
    
    return ground_truth_formula, ground_truth_text, combined_status, extracted_formula, extracted_text


def create_explanation_section() -> tuple[gr.Textbox, gr.HTML]:
    """Create explanation and errors section."""
    gr.Markdown("### Explanation")
    explanation = gr.Textbox(lines=3)

    gr.Markdown("### Errors")
    errors = gr.HTML()
    
    return explanation, errors


# ========== GRADIO INTERFACE ==========

def create_formula_viewer():
    """Create the Gradio interface for formula comparison."""
    paths = PipelinePaths()
    
    # Get all available combinations
    data = get_available_runs_and_parsers(paths)

    if not data:
        print(f"Error: No evaluation results found in {paths.runs_dir}")
        return None

    # Create the interface
    with gr.Blocks(title="Formula Comparison Viewer", theme=gr.themes.Soft(), css=CUSTOM_CSS) as interface:
        gr.Markdown("# Formula Comparison Viewer")

        # Create UI components using extracted functions
        timestamp_dropdown, pdf_dropdown, parser_dropdown, judge_model_dropdown = create_file_selection_row(data)
        
        # Summary statistics
        summary_html_component = gr.HTML()
        
        # Formula progress indicator
        formula_progress = gr.HTML()
        
        # Navigation controls
        prev_btn, formula_number_input, next_btn = create_navigation_controls()
        
        # Formula comparison section
        ground_truth_formula, ground_truth_text, combined_status, extracted_formula, extracted_text = create_formula_comparison_section()
        
        # Explanation section
        explanation, errors = create_explanation_section()

        # State variables to track current data
        current_data = gr.State(None)  # Will hold the ViewerState
        formula_index = gr.State(0)  # 0-based index for internal use
        hierarchical_data = gr.State(data)  # Store the hierarchical data
        
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
            return load_combination(timestamp, pdf_name, parser, current_idx, paths)

        # Full reload outputs for dropdown changes
        full_reload_outputs = [
            summary_html_component, formula_progress, ground_truth_formula, extracted_formula,
            combined_status, explanation, errors, ground_truth_text, extracted_text,
            formula_index, formula_number_input, current_data, 
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
            lambda judge_model, state: change_judge_model(judge_model, state),
            inputs=[judge_model_dropdown, current_data],
            outputs=navigation_outputs
        )

        prev_btn.click(
            lambda state: navigate_previous(state),
            inputs=[current_data],
            outputs=navigation_outputs
        )

        next_btn.click(
            lambda state: navigate_next(state),
            inputs=[current_data],
            outputs=navigation_outputs
        )

        formula_number_input.change(
            lambda formula_num, state: navigate_to_number(formula_num, state),
            inputs=[formula_number_input, current_data],
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