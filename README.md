# PDF Parse Bench

This benchmark evaluates how effectively different PDF parsing solutions extract mathematical formulas from documents. We generate synthetic PDFs with diverse formatting scenarios, parse them with different parsers, and assess the quality of the parsed output through a two-stage evaluation pipeline: identifying formulas in the parsed text, then scoring them based on semantic similarity to the ground truth.

## 🏆 Leaderboard - Latest Results (2025-q4)

| Rank | Parser | Overall | Inline | Display | CDM |
|------|--------|---------|--------|---------|-----|
| 1 | Qwen3-VL-235B-A22B-Instruct | 9.76 | 9.75 | 9.65 | 0.99 |
| 2 | Gemini 3 Pro | 9.75 | 9.72 | 9.71 | 0.99 |
| 3 | PaddleOCR-VL | 9.65 | 9.64 | 9.60 | 0.98 |
| 4 | Mathpix | 9.64 | 9.60 | 9.62 | 0.97 |
| 5 | dots.ocr | 9.43 | 9.31 | 9.55 | 0.94 |
| 6 | PP-StructureV3 | 9.34 | 9.26 | 9.47 | 0.98 |
| 7 | Nanonets-OCR-s | 9.31 | 9.30 | 9.26 | 0.96 |
| 8 | Gemini 2.5 Pro | 9.28 | 9.16 | 9.39 | 0.97 |
| 9 | MonkeyOCR-pro-3B | 9.25 | 9.26 | 9.20 | 0.98 |
| 10 | MinerU2.5 | 9.17 | 9.16 | 9.17 | 0.96 |
| 11 | olmOCR-2-7B-1025-FP8 | 8.94 | 8.88 | 8.94 | 0.92 |
| 12 | Gemini 2.5 Flash | 8.78 | 8.63 | 8.98 | 0.96 |
| 13 | Mistral OCR | 8.66 | 8.46 | 9.01 | 0.95 |
| 14 | DeepSeek-OCR | 8.55 | 8.65 | 8.30 | 0.95 |
| 15 | LlamaParse | 8.14 | 8.06 | 8.20 | 0.85 |
| 16 | GPT-5 nano | 7.79 | 7.77 | 7.67 | 0.75 |
| 17 | PyPDF | 7.69 | 7.75 | 7.52 | 0.75 |
| 18 | GOT-OCR2.0 | 7.38 | 7.07 | 7.95 | 0.91 |
| 19 | PyMuPDF4LLM | 6.67 | 6.70 | 6.41 | 0.58 |
| 20 | GPT-5 mini | 6.61 | 6.55 | 6.56 | 0.75 |
| 21 | GROBID | 5.70 | 5.97 | 5.07 | 0.71 |

**Legend:**
- **Overall**: Average LLM-as-a-Judge score (0-10 scale) across all formulas (1411 inline + 641 display formulas from 100 PDFs)
- **Inline**: Average score for inline formulas only (1411 formulas)
- **Display**: Average score for display-mode formulas only (641 formulas)
- **CDM**: Character Detection Metric (0-1 scale) - measures character-level accuracy using visual rendering comparison


## Benchmark Dataset

PDFs are generated synthetically using LaTeX with randomized parameters:

- **PDF Generation:** Each document contains randomly selected formulas embedded in text passages, displayed as inline or display-mode equations. Parameters include formats, styling, languages, and content structure. Layout and structure vary to test parser robustness across different scenarios.

- **Formula Dataset:** Mathematical formulas are randomly sampled from our dataset of 319,000 formulas extracted from Wikipedia, ensuring diversity in formula complexity and real-world relevance. Dataset: [piushorn/wikipedia-latex-formulas-319k](https://huggingface.co/datasets/piushorn/wikipedia-latex-formulas-319k)

- **Ground Truth:** Since PDFs are generated from LaTeX source, we automatically obtain exact ground truth for each formula as a byproduct of the generation process.

## Evaluation Pipeline

Parser outputs are assessed using a two-step pipeline:

### Step 1: Formula Extraction

Given a parser's output (the extracted text from a PDF), an LLM establishes initial formula-to-ground-truth correspondences, then fuzzy search reliably extracts exact formula strings from the parsed output. This achieves robust alignment even when parser outputs differ significantly from ground truth.

### Step 2: Scoring with LLM-as-a-Judge

The primary metric is the **LLM-as-a-Judge score** (0-10 scale, default: GPT-5-mini). For each formula pair, the LLM judge evaluates three key criteria: (1) **Correctness** - whether mathematical symbols, variables, and operations are accurately preserved, (2) **Completeness** - whether all parts are present without omissions, and (3) **Semantic equivalence** - whether the extracted formula conveys the same mathematical meaning as the ground truth. Our research demonstrates that using LLMs as judges provides a robust and meaningful metric for comparing ground truth LaTeX formulas against parsed output, focusing on semantic equivalence and mathematical correctness rather than relying solely on text similarity metrics or visual rendering comparison. Scores are computed separately for inline and display formulas.

## Quick Start

**Benchmark Datasets:** New benchmark datasets are released quarterly (e.g., 2025-q4), each containing 100 PDFs with diverse mathematical content.

There are two ways to use this benchmark, depending on your needs:

### Option 1: Evaluate Your Existing Parser (pip install)

**Use this if:** You quickly want to evaluate your PDF Parsing tool against the benchmark.

**Advantage:** Simple pip install, no need to integrate with the repository structure.

#### Installation

```bash
pip install pdf-parse-bench
```

**Note:** Set `OPENAI_API_KEY` environment variable for evaluation.

#### Step 1: Parse the Benchmark PDFs

Get the benchmark PDFs and parse them with your parser:

```python
from pdf_parse_bench import get_benchmark_pdfs_dir
from pathlib import Path

# Get benchmark PDFs (included in the package)
pdfs_dir = get_benchmark_pdfs_dir()

# Parse each PDF with your parser
output_dir = Path("results/my_parser")
for pdf_path in pdfs_dir.glob("*.pdf"):
    # Parse PDF with your parser
    parsed_text = your_parser.parse(pdf_path)

    # Save to expected format: {output_dir}/{pdf_name}/parsed.md
    (output_dir / pdf_path.stem / "parsed.md").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / pdf_path.stem / "parsed.md").write_text(parsed_text)
```

**Required output structure:**
```
results/my_parser/
├── 000/
│   └── parsed.md
├── 001/
│   └── parsed.md
├── 002/
│   └── parsed.md
...
```

#### Step 2: Run Evaluation

Run the benchmark evaluation on your parsed results:

```python
from pdf_parse_bench import run_benchmark, get_benchmark_ground_truth_dir

# Run evaluation on your parsed results
run_benchmark(
    parser_output_dir="results/my_parser",
    ground_truth_dir=get_benchmark_ground_truth_dir()
)
```

---

### Option 2: Add Parser to Repository (for reproducibility)

**Use this if:** You want to contribute your parser to the benchmark, reproduce published results, or ensure full reproducibility of your evaluation setup.

**Advantage:** Full automation with CLI, parser configuration is versioned and reproducible, easy to share exact setup with others.

#### Clone Repository

```bash
git clone https://github.com/phorn1/pdf-parse-bench.git
cd pdf-parse-bench

# Install with uv
uv sync

# Configure environment (copy and edit .env.example)
cp .env.example .env
```

#### Add Your Parser Implementation

Create a new parser module in the `parsers/` directory:

```python
# parsers/my_parser/__main__.py
from pathlib import Path
from pdf_parse_bench.utilities import PDFParser
from pdf_parse_bench.pipeline import run_cli

class MyParser(PDFParser):
    @classmethod
    def display_name(cls) -> str:
        return "My Parser"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        # Your parsing logic here
        markdown = "# Parsed content"
        self._write_output(markdown, output_path)
        return markdown

if __name__ == "__main__":
    run_cli(MyParser())
```

#### Run Your Parser

```bash
uv run -m parsers.my_parser
```

The benchmark infrastructure handles everything automatically:
- Loading test PDFs from `data/2025-q4/pdfs/`
- Parsing each PDF with your parser
- Extracting formulas from parsed output
- Running evaluation against ground truth
- Saving results in standardized format

## CLI Options

The benchmark CLI provides several options to customize execution:

```bash
# Run only specific steps
uv run -m parsers.my_parser --only parse
uv run -m parsers.my_parser --only extract
uv run -m parsers.my_parser --only evaluate

# Skip specific steps
uv run -m parsers.my_parser --skip-parse
uv run -m parsers.my_parser --skip-extract

# Reprocess existing results
uv run -m parsers.my_parser --reprocess all
uv run -m parsers.my_parser --reprocess parse --reprocess extract

# Use different LLM judges for evaluation
uv run -m parsers.my_parser --llm-judge-models "gpt-5-mini,gemini-2.5-flash"

# Enable Character Detection Metrics (CDM)
# Note: Requires CDM service installation (https://github.com/opendatalab/UniMERNet/tree/main/cdm)
# and CDM_SERVICE_URL environment variable
uv run -m parsers.my_parser --enable-cdm

# Custom input/output directories
uv run -m parsers.my_parser -i data/2025-q4 -o results/custom
```

## Project Structure

```
pdf-parse-bench/
├── src/pdf_parse_bench/       # Core benchmark infrastructure
│   ├── pipeline/              # Benchmark execution pipeline
│   ├── eval/                  # Evaluation metrics and judges
│   ├── extraction/            # Formula extraction from parsed text
│   ├── utilities/             # Base classes and helpers
│   └── synth_pdf/             # Synthetic PDF generation (optional)
├── parsers/                   # Parser implementations
│   ├── pymupdf4llm/
│   ├── llamaparse/
│   ├── mathpix/
│   └── ...                    # Add your own!
├── data/                      # Benchmark datasets
│   └── 2025-q4/              # Current benchmark version
│       ├── pdfs/             # Test PDFs
│       └── ground_truth/     # LaTeX formulas
```

## Contributing

Contributions are welcome!

**Adding a parser implementation:** See [Option 2](#option-2-add-parser-to-repository-for-reproducibility) above for instructions on adding your parser to the repository.

**Bug reports and feature requests:** Please open an issue on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this benchmark in your research or project, please cite:

```bibtex
@software{pdf_parse_bench_2025,
  author = {Horn, Pius; Keuper, Janis},
  title = {Benchmarking PDF-to-Text Parsers on Mathematical Formula Extraction},
  year = {2025},
  url = {https://github.com/phorn1/pdf-parse-bench}
}
```

## Acknowledgments
This work has been supported by the German Federal Ministry of Research, Technology and Space (BMFTR) in the program "Forschung an Fachhochschulen in Kooperation mit Unternehmen (FH-Kooperativ)" within the joint project **LLMpraxis** under grant 13FH622KX2.

<p align="center">
  <img src="https://raw.githubusercontent.com/phorn1/pdf-parse-bench/main/assets/BMFTR_logo.png" alt="BMFTR_logo" width="150" />
  <img src="https://raw.githubusercontent.com/phorn1/pdf-parse-bench/main/assets/HAW_logo.png" alt="HAW_logo" width="150" />
</p>
