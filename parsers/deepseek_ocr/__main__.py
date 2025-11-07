"""CLI entry point for DeepSeek-OCR parser benchmark."""

import tempfile
from pathlib import Path

from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser


class DeepSeekOCRParser(PDFParser):
    """PDF parser using DeepSeek-OCR with vLLM."""

    def __init__(self):
        """Initialize DeepSeek-OCR parser."""
        super().__init__()
        self.llm = None

    def _load_model(self):
        """Load DeepSeek-OCR model using vLLM (official configuration)."""
        if self.llm is not None:
            return

        from vllm import LLM
        from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

        self.llm = LLM(
            model="deepseek-ai/DeepSeek-OCR",
            enable_prefix_caching=False,
            mm_processor_cache_gb=0,
            logits_processors=[NGramPerReqLogitsProcessor],
            mm_processor_kwargs={
                "base_size": 1024,
                "image_size": 640,
                "crop_mode": True,
            },
        )

    @classmethod
    def display_name(cls) -> str:
        return "DeepSeek-OCR"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """Parse single-page PDF to markdown using DeepSeek-OCR."""
        self._load_model()

        # ========== PDF TO IMAGE ==========
        import fitz
        from vllm import SamplingParams
        from PIL import Image

        doc = fitz.open(pdf_path)
        page = doc[0]  # Single page only
        pix = page.get_pixmap(dpi=300)

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = Path(temp_file.name)
        pix.save(temp_path)
        doc.close()

        # ========== RUN OCR WITH VLLM ==========
        try:
            image = Image.open(temp_path).convert("RGB")
            prompt = "<image>\nConvert the document to markdown."

            model_input = [{"prompt": prompt, "multi_modal_data": {"image": image}}]

            sampling_param = SamplingParams(
                temperature=0.0,
                max_tokens=8192,
                extra_args=dict(
                    ngram_size=30,
                    window_size=90,
                    whitelist_token_ids={128821, 128822},
                ),
                skip_special_tokens=False,
            )

            model_outputs = self.llm.generate(model_input, sampling_param)
            markdown = model_outputs[0].outputs[0].text

        finally:
            temp_path.unlink()

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=DeepSeekOCRParser())