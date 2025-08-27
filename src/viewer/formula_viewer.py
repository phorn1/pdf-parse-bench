import json
import re
import gradio as gr
from pathlib import Path
from ..pipeline import PipelinePaths, BenchmarkRunConfig


def load_formula_data(formula_results_path: Path) -> list[dict] | None:
    """Load formula evaluation results from JSON file."""
    if not formula_results_path.exists():
        return None
    
    with open(formula_results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process formulas to ensure consistent MathJax formatting
    for entry in data:
        entry['ground_truth_formula'] = standardize_formula_notation(entry['ground_truth_formula'])
        entry['extracted_formula'] = standardize_formula_notation(entry['extracted_formula'])
    
    return data


def load_stats_data(stats_path: Path) -> dict | None:
    """Load evaluation statistics from JSON file."""
    if not stats_path.exists():
        return None
    
    with open(stats_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def standardize_formula_notation(formula_text: str) -> str:
    """Standardize formula notation for MathJax."""
    if not formula_text:
        return formula_text

    # If formula already has $$ delimiters, return as is
    if formula_text.strip().startswith('$$') and formula_text.strip().endswith('$$'):
        return formula_text

    # Temporarily replace existing $$ to avoid processing them
    formula_text = formula_text.replace('$$', '@@DOUBLE_DOLLAR@@')

    # Replace single $ formulas with double $$ for display formulas
    formula_text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', r'$$\1$$', formula_text)

    # Restore original $$ notation
    formula_text = formula_text.replace('@@DOUBLE_DOLLAR@@', '$$')

    return formula_text


def create_summary_html(stats: dict) -> str:
    """Create HTML for summary statistics."""
    formula_stats = stats.get('formula_statistics', {})
    text_stats = stats.get('text_statistics', {})
    
    return f"""
    <div class="summary-stats">
        <div style="display: inline-block; margin: 0 15px;">
            <div>Total Formulas</div>
            <div class="summary-value">{formula_stats.get('total_formulas', 0)}</div>
        </div>
        <div style="display: inline-block; margin: 0 15px;">
            <div>Correct Formulas</div>
            <div class="summary-value">{formula_stats.get('correct_formulas', 0)}</div>
        </div>
        <div style="display: inline-block; margin: 0 15px;">
            <div>Accuracy</div>
            <div class="summary-value">{formula_stats.get('accuracy_percentage', 0)}%</div>
        </div>
        <div style="display: inline-block; margin: 0 15px;">
            <div>Average Score</div>
            <div class="summary-value">{formula_stats.get('average_score', 0)}</div>
        </div>
        <div style="display: inline-block; margin: 0 15px;">
            <div>Average Levenshtein Similarity</div>
            <div class="summary-value">{text_stats.get('average_levenshtein_similarity', 0)}</div>
        </div>
    </div>
    """


def create_progress_html(formula_number: int, total_formulas: int) -> str:
    """Create HTML for progress bar."""
    progress_percentage = (formula_number / total_formulas) * 100 if total_formulas > 0 else 0

    return f"""
    <div style="margin: 10px 0; display: flex; flex-direction: column;">
        <div style="font-weight: bold; text-align: center; margin-bottom: 5px;">
            Formula {formula_number} of {total_formulas}
        </div>
        <div style="width: 100%; background-color: #eee; border-radius: 10px; height: 10px;">
            <div style="background-color: #3498db; width: {progress_percentage}%; height: 100%; border-radius: 10px;"></div>
        </div>
    </div>
    """


def create_result_html(is_correct: bool, score: float) -> str:
    """Create HTML for formula result display."""
    bg_color = 'rgba(46, 204, 113, 0.2)' if is_correct else 'rgba(231, 76, 60, 0.2)'
    border_color = '#2ecc71' if is_correct else '#e74c3c'
    status_text = "Correct" if is_correct else "Incorrect"
    status_icon = "✓" if is_correct else "✗"

    return f"""
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


def create_errors_html(errors: list[str]) -> str:
    """Create HTML for errors list."""
    if not errors:
        return "No errors found."

    return "<ul>" + "".join([f"<li>{error}</li>" for error in errors]) + "</ul>"


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


def create_formula_viewer():
    """Create the Gradio interface for formula comparison."""
    paths = PipelinePaths()
    
    # MathJax configuration for proper formula rendering
    mathjax_js = """
function() {
    if (typeof window.MathJax === 'undefined') {
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$']],
                displayMath: [['$$', '$$']],
                packages: {'[+]': ['mhchem']}
            },
            loader: {load: ['[tex]/mhchem']}
        };
        
        const script = document.createElement('script');
        script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
        script.async = true;
        document.head.appendChild(script);
    }
    
    setInterval(() => {
        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise();
        }
    }, 500);
}
"""
    
    # Custom CSS for better styling
    custom_css = """
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
    with gr.Blocks(title="Formula Comparison Viewer", theme=gr.themes.Soft(), css=custom_css, js=mathjax_js) as interface:
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
                ground_truth_formula = gr.Markdown()

            # Comparison box with fixed width
            with gr.Column(elem_classes="comparison-box", elem_id="status-comparison-box", scale=0):
                combined_status = gr.HTML()

            with gr.Column(elem_classes="formula-box"):
                gr.Markdown("### Extracted Formula")
                extracted_formula = gr.Markdown()

        # Explanation and Errors
        gr.Markdown("### Explanation")
        explanation = gr.Textbox(lines=3)

        gr.Markdown("### Errors")
        errors = gr.HTML()

        # State variables to track current index and current data
        formula_index = gr.State(0)  # 0-based index for internal use
        current_data = gr.State(None)  # Will hold the current formula data
        hierarchical_data = gr.State(data)  # Store the hierarchical data

        # Function to update display based on index
        def update_display(idx: int, formula_data: list[dict] | None):
            if not formula_data or idx < 0 or idx >= len(formula_data):
                # Handle edge cases
                return ("", "", "", "", "", "", idx, 1)

            formula = formula_data[idx]

            # Extract formula details
            ground_truth = formula.get('ground_truth_formula', 'No ground truth formula available')
            extracted = formula.get('extracted_formula', 'No extracted formula available')
            is_correct = formula.get('is_correct', False)
            score = formula.get('score', 0)
            explanation_text = formula.get('explanation', 'No explanation available')
            errors_list = formula.get('errors', [])

            # Calculate formula number (1-based) and total
            formula_number = idx + 1
            total_formulas = len(formula_data)

            # Create HTML components
            progress_html = create_progress_html(formula_number, total_formulas)
            result_html = create_result_html(is_correct, score)
            errors_html = create_errors_html(errors_list)

            return (
                progress_html,
                ground_truth,
                extracted,
                result_html,
                explanation_text,
                errors_html,
                idx,
                formula_number
            )

        # Function to update PDF choices based on timestamp
        def update_pdf_choices(timestamp: str, data_dict: dict):
            pdfs = get_pdfs_for_timestamp(data_dict, timestamp)
            first_pdf = pdfs[0] if pdfs else None
            parsers = get_parsers_for_pdf(data_dict, timestamp, first_pdf) if first_pdf else []
            first_parser = parsers[0] if parsers else None
            return (
                gr.update(choices=pdfs, value=first_pdf),
                gr.update(choices=parsers, value=first_parser)
            )

        # Function to update parser choices based on timestamp and PDF
        def update_parser_choices(timestamp: str, pdf_name: str, data_dict: dict):
            parsers = get_parsers_for_pdf(data_dict, timestamp, pdf_name)
            first_parser = parsers[0] if parsers else None
            return gr.update(choices=parsers, value=first_parser)

        # Function to load a combination and update the interface
        def load_combination(timestamp: str, pdf_name: str, parser: str, current_idx: int):
            if not all([timestamp, pdf_name, parser]):
                return None, "", "", "", "", "", "", 0, 1, None, 1

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
                        "", "", "", "", "", "",
                        0, 1, None, 1
                    )

                # Create summary HTML
                summary_html = create_summary_html(stats_data)

                # Validate index
                if current_idx >= len(formula_data):
                    current_idx = len(formula_data) - 1
                if current_idx < 0:
                    current_idx = 0

                # Update the display
                display_result = update_display(current_idx, formula_data)

                # Return everything needed to update the UI
                return (
                    summary_html,
                    display_result[0],  # progress_html
                    display_result[1],  # ground_truth
                    display_result[2],  # extracted
                    display_result[3],  # result_html
                    display_result[4],  # explanation
                    display_result[5],  # errors_html
                    current_idx,  # index state
                    display_result[7],  # formula_number (1-based)
                    formula_data,  # formula_data state
                    display_result[7]  # formula_number for input
                )
                
            except Exception as e:
                print(f"Error loading combination {timestamp}/{pdf_name}/{parser}: {e}")
                return (
                    "Error loading data",
                    "", "", "", "", "", "",
                    0, 1, None, 1
                )

        # Navigation functions
        def go_to_prev(current_idx: int, formula_data: list[dict] | None):
            if formula_data and current_idx > 0:
                current_idx -= 1
            return update_display(current_idx, formula_data)

        def go_to_next(current_idx: int, formula_data: list[dict] | None):
            if formula_data and current_idx < len(formula_data) - 1:
                current_idx += 1
            return update_display(current_idx, formula_data)

        def go_to_number(formula_num: int, formula_data: list[dict] | None):
            # Convert 1-based formula number to 0-based index
            idx = formula_num - 1
            if formula_data:
                if idx < 0:
                    idx = 0
                if idx >= len(formula_data):
                    idx = len(formula_data) - 1
            else:
                idx = 0
            return update_display(idx, formula_data)

        # Connect event handlers for dropdown changes
        timestamp_dropdown.change(
            update_pdf_choices,
            inputs=[timestamp_dropdown, hierarchical_data],
            outputs=[pdf_dropdown, parser_dropdown]
        )

        pdf_dropdown.change(
            update_parser_choices,
            inputs=[timestamp_dropdown, pdf_dropdown, hierarchical_data],
            outputs=[parser_dropdown]
        )

        # Connect event handlers for loading data
        def reload_data(timestamp, pdf_name, parser, current_idx):
            return load_combination(timestamp, pdf_name, parser, current_idx)

        for dropdown in [timestamp_dropdown, pdf_dropdown, parser_dropdown]:
            dropdown.change(
                reload_data,
                inputs=[timestamp_dropdown, pdf_dropdown, parser_dropdown, formula_index],
                outputs=[
                    summary_html_component,
                    formula_progress,
                    ground_truth_formula,
                    extracted_formula,
                    combined_status,
                    explanation,
                    errors,
                    formula_index,
                    formula_number_input,
                    current_data,
                    formula_number_input
                ]
            )

        prev_btn.click(
            go_to_prev,
            inputs=[formula_index, current_data],
            outputs=[
                formula_progress,
                ground_truth_formula,
                extracted_formula,
                combined_status,
                explanation,
                errors,
                formula_index,
                formula_number_input
            ]
        )

        next_btn.click(
            go_to_next,
            inputs=[formula_index, current_data],
            outputs=[
                formula_progress,
                ground_truth_formula,
                extracted_formula,
                combined_status,
                explanation,
                errors,
                formula_index,
                formula_number_input
            ]
        )

        formula_number_input.change(
            go_to_number,
            inputs=[formula_number_input, current_data],
            outputs=[
                formula_progress,
                ground_truth_formula,
                extracted_formula,
                combined_status,
                explanation,
                errors,
                formula_index,
                formula_number_input
            ]
        )

        # Initialize the interface
        interface.load(
            reload_data,
            inputs=[timestamp_dropdown, pdf_dropdown, parser_dropdown, formula_index],
            outputs=[
                summary_html_component,
                formula_progress,
                ground_truth_formula,
                extracted_formula,
                combined_status,
                explanation,
                errors,
                formula_index,
                formula_number_input,
                current_data,
                formula_number_input
            ]
        )

    return interface


def main():
    # Create and launch the Gradio interface
    app = create_formula_viewer()
    if app:
        app.launch()
    else:
        print("Failed to create Gradio interface.")


if __name__ == "__main__":
    main()