# PDF Parse Bench

This benchmark evaluates how effectively different PDF parsing solutions extract mathematical formulas and tables from documents. We generate synthetic PDFs with diverse formatting scenarios, parse them with different parsers, and assess the quality of the parsed output through a two-stage evaluation pipeline: identifying formulas and tables in the parsed text, then scoring them with an LLM-as-a-Judge based on semantic similarity to the ground truth.

![Workflow Overview](assets/workflow.png)

## 🏆 Leaderboard (2026-q1)

Results are based on two separate benchmark datasets, each containing 100 synthetic PDFs:
- **`2026-q1-tables-only`** — PDFs with tables of varying complexity (simple, moderate, complex)
- **`2026-q1-formulas-only`** — PDFs with inline and display-mode mathematical formulas

| Parser | Tables | Formulas | Cost/Time | Inference |
|--------|--------|----------|-----------|-----------|
| Gemini 3 Flash | 9.50 | 9.79 | $0.57 | API |
| LightOnOCR-2-1B | 9.08 | 9.57 | 30 min | GPU |
| Mistral OCR | 8.89 | 9.48 | $0.20 | API |
| dots.ocr | 8.73 | 9.55 | 20 min | GPU |
| Mathpix | 8.53 | 9.66 | $0.35–0.50 | API |
| Chandra | 8.43 | 9.45 | 4 h | GPU |
| Qwen3-VL-235B | 8.43 | 9.84 | $0.20 | API/GPU |
| MonkeyOCR-pro-3B | 8.39 | 9.50 | 20 min | GPU |
| GLM-4.5V | 7.98 | 9.37 | $0.60 | API |
| GPT-5 mini | 7.14 | 5.57 | $1.00 | API |
| Claude Sonnet 4.6 | 7.02 | 8.50 | $3.00 | API |
| Nanonets-OCR-s | 6.92 | 9.21 | 50 min | GPU |
| Gemini 2.5 Flash | 6.85 | — | $0.40 | API |
| MinerU2.5 | 6.49 | 9.32 | — | API/GPU |
| GPT-5 nano | 6.48 | 4.78 | $0.35 | API |
| DeepSeek-OCR | 5.75 | 8.97 | 4 min | GPU |
| PyMuPDF4LLM | 5.25 | — | 30 s | CPU |
| GOT-OCR2.0 | 5.13 | 8.01 | 20 min | GPU |
| olmOCR-2-7B | 4.05 | 9.35 | 25 min | GPU |
| GROBID | 2.10 | 7.01 | 2 min | CPU |
| PaddleOCR-VL | — | 8.47 | — | GPU |
| Gemini 2.5 Pro | — | 7.56 | — | API |

**Legend:**
- **Tables**: Average LLM-as-a-Judge score (0-10) across 451 tables from `2026-q1-tables-only`
- **Formulas**: Average LLM-as-a-Judge score (0-10) across 1413 inline + 657 display formulas from `2026-q1-formulas-only`
- **Cost/Time**: API cost (USD) for 100 pages, or GPU wall-clock time on a single NVIDIA RTX 4090
- **Inference**: Deployment options (CPU, GPU, API)

<details>
<summary>📊 Detailed table scores (Simple/Moderate/Complex/TEDS)</summary>

| Rank | Parser | Overall | Simple | Moderate | Complex | TEDS |
|------|--------|---------|--------|----------|---------|------|
| 1 | Gemini 3 Flash | 9.50 | 9.53 | 9.38 | 9.61 | 0.85 |
| 2 | LightOnOCR-2-1B | 9.08 | 9.41 | 8.90 | 8.91 | 0.84 |
| 3 | Mistral OCR | 8.89 | 8.92 | 8.69 | 9.07 | 0.88 |
| 4 | dots.ocr | 8.73 | 9.01 | 8.43 | 8.76 | 0.81 |
| 5 | Mathpix | 8.53 | 9.32 | 8.40 | 7.77 | 0.74 |
| 6 | Chandra | 8.43 | 8.96 | 8.14 | 8.15 | 0.77 |
| 7 | Qwen3-VL-235B | 8.43 | 9.23 | 8.27 | 7.67 | 0.78 |
| 8 | MonkeyOCR-pro-3B | 8.39 | 8.60 | 8.10 | 8.47 | 0.80 |
| 9 | GLM-4.5V | 7.98 | 9.19 | 7.59 | 7.00 | 0.78 |
| 10 | GPT-5 mini | 7.14 | 8.03 | 6.82 | 6.48 | 0.68 |
| 11 | Claude Sonnet 4.6 | 7.02 | 6.94 | 7.10 | 7.01 | 0.52 |
| 12 | Nanonets-OCR-s | 6.92 | 8.27 | 6.51 | 5.82 | 0.69 |
| 13 | Gemini 2.5 Flash | 6.85 | 7.93 | 6.52 | 5.94 | 0.72 |
| 14 | MinerU2.5 | 6.49 | 7.07 | 6.03 | 6.35 | 0.78 |
| 15 | GPT-5 nano | 6.48 | 7.63 | 6.18 | 5.47 | 0.32 |
| 16 | DeepSeek-OCR | 5.75 | 7.45 | 5.34 | 4.20 | 0.66 |
| 17 | PyMuPDF4LLM | 5.25 | 6.78 | 4.86 | 3.91 | 0.11 |
| 18 | GOT-OCR2.0 | 5.13 | 5.89 | 4.95 | 4.45 | 0.58 |
| 19 | olmOCR-2-7B | 4.05 | 4.64 | 3.78 | 3.68 | 0.35 |
| 20 | GROBID | 2.10 | 2.27 | 1.94 | 2.09 | 0.00 |

- **Overall/Simple/Moderate/Complex**: LLM-as-a-Judge score (0-10 scale) across 451 tables by complexity level
- **TEDS**: Tree-Edit-Distance-based Similarity (0-1 scale) - structural accuracy metric

</details>

<details>
<summary>📊 Detailed formula scores (Inline/Display/CDM)</summary>

| Rank | Parser | Inline | Display | CDM |
|------|--------|--------|---------|-----|
| 1 | Qwen3-VL-235B | 9.82 | 9.86 | — |
| 2 | Gemini 3 Flash | 9.77 | 9.82 | — |
| 3 | Mathpix | 9.64 | 9.72 | — |
| 4 | LightOnOCR-2-1B | 9.51 | 9.70 | — |
| 5 | dots.ocr | 9.44 | 9.77 | — |
| 6 | MonkeyOCR-pro-3B | 9.54 | 9.42 | — |
| 7 | Mistral OCR | 9.39 | 9.68 | — |
| 8 | Chandra | 9.43 | 9.50 | — |
| 9 | GLM-4.5V | 9.33 | 9.46 | — |
| 10 | olmOCR-2-7B | 9.34 | 9.37 | — |
| 11 | MinerU2.5 | 9.36 | 9.25 | — |
| 12 | Nanonets-OCR-s | 9.18 | 9.26 | — |
| 13 | DeepSeek-OCR | 8.95 | 9.02 | — |
| 14 | Claude Sonnet 4.6 | 8.42 | 8.67 | — |
| 15 | PaddleOCR-VL | 8.50 | 8.42 | — |
| 16 | GOT-OCR2.0 | 7.77 | 8.53 | — |
| 17 | Gemini 2.5 Pro | 7.42 | 7.87 | — |
| 18 | GROBID | 7.33 | 6.33 | — |
| 19 | GPT-5 mini | 5.87 | 4.94 | — |
| 20 | GPT-5 nano | 4.88 | 4.57 | — |

- **Inline**: LLM-as-a-Judge score for inline formulas (1413 formulas)
- **Display**: LLM-as-a-Judge score for display-mode formulas (657 formulas)
- **CDM**: Character Detection Metric (0-1 scale) - character-level accuracy via visual rendering comparison (TODO)

</details>


## Benchmark Datasets

PDFs are generated synthetically using LaTeX with randomized parameters. Layout, styling, languages, and content structure vary to test parser robustness across different scenarios. Since PDFs are generated from LaTeX source, we automatically obtain exact ground truth as a byproduct of the generation process.

- **Formula Dataset:** Each PDF contains randomly selected formulas embedded in text passages, displayed as inline or display-mode equations. Formulas are sampled from our dataset of 319,000 formulas extracted from Wikipedia, ensuring diversity in complexity and real-world relevance. Dataset: [piushorn/wikipedia-latex-formulas-319k](https://huggingface.co/datasets/piushorn/wikipedia-latex-formulas-319k)

- **Table Dataset:** Each PDF contains tables of varying complexity (simple, moderate, complex) with diverse content types, column layouts, and formatting. Dataset coming soon on Hugging Face.

## Evaluation Pipeline

Parser outputs are assessed using a two-step pipeline:

### Step 1: Formula Extraction

Given a parser's output (the extracted text from a PDF), an LLM establishes initial formula-to-ground-truth correspondences, then fuzzy search reliably extracts exact formula strings from the parsed output. This achieves robust alignment even when parser outputs differ significantly from ground truth.

### Step 2: Scoring with LLM-as-a-Judge

The primary metric is the **LLM-as-a-Judge score** (0-10 scale, default: Gemini 3 Flash via OpenRouter). For each formula pair, the LLM judge evaluates three key criteria: (1) **Correctness** - whether mathematical symbols, variables, and operations are accurately preserved, (2) **Completeness** - whether all parts are present without omissions, and (3) **Semantic equivalence** - whether the extracted formula conveys the same mathematical meaning as the ground truth. Our research demonstrates that using LLMs as judges provides a robust and meaningful metric for comparing ground truth LaTeX formulas against parsed output, focusing on semantic equivalence and mathematical correctness rather than relying solely on text similarity metrics or visual rendering comparison. For detailed metric comparison studies, see [formula-metric-study](https://github.com/phorn1/formula-metric-study) and [table-metric-study](https://github.com/phorn1/table-metric-study). Scores are computed separately for inline and display formulas.

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

**Note:** Set `OPENROUTER_API_KEY` environment variable for evaluation (used for LLM-as-a-Judge scoring via [OpenRouter](https://openrouter.ai/)).

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
from pathlib import Path
from pdf_parse_bench import Benchmark, PDFParser, get_benchmark_ground_truth_dir

class MyParser(PDFParser):
    @classmethod
    def display_name(cls) -> str:
        return "My Parser"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        raise NotImplementedError  # Already parsed in Step 1

bench = Benchmark(
    parser_output_dir=Path("results/my_parser"),
    ground_truth_dir=get_benchmark_ground_truth_dir(),
    llm_judge_models=["google/gemini-3-flash-preview"],
    parser=MyParser(),
)
bench.extract()
bench.evaluate()
bench.save_benchmark_summary()
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
uv run -m parsers.my_parser --step parse
uv run -m parsers.my_parser --step extract --step evaluate

# Reprocess existing results
uv run -m parsers.my_parser --reprocess all
uv run -m parsers.my_parser --reprocess parse --reprocess extract

# Use a different LLM judge for evaluation (OpenRouter model format)
uv run -m parsers.my_parser --llm-judge-model openai/gpt-5-mini

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

If you use this benchmark in your research or project, please cite our paper:

```bibtex
@misc{horn2025benchmarking,
    title = {Benchmarking Document Parsers on Mathematical Formula Extraction from PDFs},
    author = {Horn, Pius and Keuper, Janis},
    year = {2025},
    eprint={2511.10390},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url = {https://arxiv.org/abs/2512.09874}
}
```

📄 **Paper:** [arXiv:2512.09874](https://arxiv.org/abs/2512.09874)

## Acknowledgments
This work has been supported by the German Federal Ministry of Research, Technology and Space (BMFTR) in the program "Forschung an Fachhochschulen in Kooperation mit Unternehmen (FH-Kooperativ)" within the joint project **LLMpraxis** under grant 13FH622KX2.

<p align="center">
  <img src="https://raw.githubusercontent.com/phorn1/pdf-parse-bench/main/assets/BMFTR_logo.png" alt="BMFTR_logo" width="150" />
  <img src="https://raw.githubusercontent.com/phorn1/pdf-parse-bench/main/assets/HAW_logo.png" alt="HAW_logo" width="150" />
</p>
