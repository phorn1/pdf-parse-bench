"""CLI entry point for LightOnOCR-2-1B parser benchmark."""

import tempfile
from pathlib import Path

from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser


class LightOnOCRParser(PDFParser):
    """PDF parser using LightOnOCR-2-1B with transformers."""

    def __init__(self):
        """Initialize LightOnOCR-2-1B parser."""
        super().__init__()
        self.model = None
        self.processor = None
        self._device = None
        self._dtype = None

    def _load_model(self):
        """Load LightOnOCR-2-1B model and processor."""
        if self.model is not None and self.processor is not None:
            return

        import torch
        from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if self._device == "cuda" else torch.float32

        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            "lightonai/LightOnOCR-2-1B",
            torch_dtype=self._dtype,
        ).to(self._device)
        self.model.eval()

        self.processor = LightOnOcrProcessor.from_pretrained("lightonai/LightOnOCR-2-1B")

    @classmethod
    def display_name(cls) -> str:
        return "LightOnOCR-2-1B"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """Parse single-page PDF to markdown using LightOnOCR-2-1B."""
        self._load_model()

        # ========== PDF TO IMAGE ==========
        import fitz
        from PIL import Image

        doc = fitz.open(pdf_path)
        page = doc[0]  # Single page only
        pix = page.get_pixmap(dpi=200)  # Model card recommends 200 DPI

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_path = Path(temp_file.name)
        pix.save(temp_path)
        doc.close()

        # ========== RUN OCR ==========
        try:
            image = Image.open(temp_path)

            conversation = [
                {"role": "user", "content": [{"type": "image", "image": image}]}
            ]

            inputs = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {
                k: v.to(device=self._device, dtype=self._dtype) if v.is_floating_point() else v.to(self._device)
                for k, v in inputs.items()
            }

            output_ids = self.model.generate(**inputs, max_new_tokens=8000, do_sample=False)
            generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            markdown = self.processor.decode(generated_ids, skip_special_tokens=True)

        finally:
            temp_path.unlink()

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=LightOnOCRParser())
