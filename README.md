# PDF Parse Bench

A comprehensive benchmark for evaluating PDF parser quality on mathematical formula extraction.

## Overview

This benchmark evaluates how effectively different PDF parsing solutions extract mathematical formulas from documents. Our approach uses synthetic PDF generation with LaTeX to create a robust and scalable evaluation framework.

### Methodology

**Synthetic Dataset Generation:** We generate PDFs synthetically using LaTeX with randomized parameters including formats, styling, languages, and content. Each PDF contains numerous randomly selected formulas embedded in random text passages, displayed either as inline formulas or display-mode equations. The layout and structure vary across documents to test parser robustness across different formatting scenarios.

**Formula Dataset:** The mathematical formulas used in our benchmark are randomly sampled from our dataset of 319,000 formulas that we extracted from Wikipedia. This ensures diversity in formula complexity and real-world relevance. Dataset: [piushorn/wikipedia-latex-formulas-319k](https://huggingface.co/datasets/piushorn/wikipedia-latex-formulas-319k)

**Ground Truth:** Since PDFs are generated synthetically from LaTeX source, we automatically obtain exact ground truth as a byproduct of the generation process. Each formula in the benchmark has a corresponding LaTeX ground truth.

### Evaluation Approach

Our benchmark uses a two-step evaluation pipeline:

**1. Extraction Step:** An LLM establishes initial formula-to-ground-truth correspondences, then fuzzy search reliably extracts exact formula strings from the parsed output. This achieves robust alignment even when parser outputs differ significantly from ground truth.

**2. Evaluation Step:** The primary metric is the **LLM-as-a-Judge score** (0-10 scale, default: GPT-5-mini). For each formula pair, the LLM judge evaluates three key criteria: (1) **Correctness** - whether mathematical symbols, variables, and operations are accurately preserved, (2) **Completeness** - whether all parts are present without omissions, and (3) **Semantic equivalence** - whether the extracted formula conveys the same mathematical meaning as the ground truth. Our research demonstrates that using LLMs as judges provides a robust and meaningful metric for comparing ground truth LaTeX formulas against parsed output, focusing on semantic equivalence and mathematical correctness rather than relying solely on text similarity metrics or visual rendering comparison. Scores are computed separately for inline and display formulas.

## Quick Start

There are two ways to use this benchmark, depending on your needs:

### Option 1: Evaluate Your Existing Parser (pip install)

**Use this if:** You already have a PDF parser and want to quickly evaluate it against the benchmark.

**Advantage:** Simple pip install, no need to integrate with the repository structure.

#### Installation

```bash
pip install pdf-parse-bench
```

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
```

#### Add Your Parser Implementation

Create a new parser module in the `parsers/` directory:

```python
# parsers/my_parser/__main__.py
from pdf_parse_bench.utilities.base_parser import PDFParser
from pdf_parse_bench.pipeline.cli import run_cli

class MyParser(PDFParser):
    @staticmethod
    def parser_name() -> str:
        return "my_parser"

    def parse_pdf(self, pdf_path: str) -> str:
        # Your parsing logic here
        return parsed_text

if __name__ == "__main__":
    run_cli(MyParser())
```

#### Run Your Parser

```bash
uv run -m parsers.my_parser
```

The benchmark infrastructure handles everything automatically:
- Loading test PDFs from `data/2025-10-v1/pdfs/`
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
uv run -m parsers.my_parser -i data/2025-10-v1 -o results/custom
```

## Dataset Details

- **Size**: 50 PDFs with diverse mathematical content
- **Metric**: LLM-as-a-Judge score (0-10 scale, default: GPT-5-mini)
- **Extensibility**: Can be regenerated or expanded with different parameters

## 🏆 Latest Results (2025-10-v1)

| Rank | Parser | Avg Score | Accuracy (%) | Inline Score | Display Score | Samples |
|------|--------|-----------|--------------|--------------|---------------|---------|
| 1 | PaddleOCR-VL | 9.58 | 91.31% | 9.55 | 9.62 | 50 |
| 2 | dots.ocr | 9.17 | 85.23% | 9.22 | 9.09 | 50 |
| 3 | Nanonets-OCR-s | 9.17 | 85.26% | 9.13 | 9.24 | 50 |
| 4 | Gemini 2.5 Pro | 9.15 | 84.08% | 9.07 | 9.28 | 49 |
| 5 | MonkeyOCR-pro-3B | 9.11 | 83.61% | 9.32 | 8.74 | 50 |
| 6 | MinerU2.5 | 9.08 | 84.99% | 9.02 | 9.24 | 50 |
| 7 | PP-StructureV3 | 9.03 | 82.16% | 8.87 | 9.26 | 50 |
| 8 | olmOCR-2-7B-1025-FP8 | 8.94 | 84.7% | 9.0 | 8.9 | 50 |
| 9 | Gemini 2.5 Flash | 8.75 | 77.65% | 8.69 | 8.88 | 48 |
| 10 | Mathpix | 8.48 | 78.63% | 9.52 | 6.81 | 50 |
| 11 | DeepSeek-OCR | 8.16 | 72.85% | 8.27 | 8.08 | 50 |
| 12 | LlamaParse | 7.99 | 74.03% | 7.95 | 8.08 | 47 |
| 13 | Mistral OCR | 7.63 | 67.78% | 8.51 | 6.17 | 50 |
| 14 | GPT-5 nano | 7.18 | 53.7% | 7.42 | 6.65 | 50 |
| 15 | GOT-OCR2.0 | 6.31 | 50.09% | 6.71 | 6.02 | 50 |
| 16 | PyMuPDF4LLM | 6.27 | 42.32% | 0.0 | 6.27 | 50 |
| 17 | GPT-5 mini | 5.94 | 38.06% | 6.35 | 5.28 | 49 |

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
│   └── 2025-10-v1/           # Current benchmark version
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
  <img src="assets/BMFTR_logo.png" alt="BMFTR_logo" width="150" />
  <img src="assets/HAW_logo.png" alt="HAW_logo" width="150" />
</p>
