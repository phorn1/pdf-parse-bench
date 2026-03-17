import tempfile
from pathlib import Path

from pdf_parse_bench.pipeline import run_cli
from pdf_parse_bench.utilities import PDFParser


class PaddleOCRVLParser(PDFParser):
    """PDF parser using PaddleOCR-VL-1.5 via HuggingFace Transformers."""

    def __init__(self):
        super().__init__()
        self.model = None
        self.processor = None

    def _load_model(self):
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_path = "PaddlePaddle/PaddleOCR-VL-1.5"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda").eval()

    @classmethod
    def display_name(cls) -> str:
        return "PaddleOCR-VL"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """Parse single-page PDF to markdown using PaddleOCR-VL-1.5."""
        self._load_model()

        import fitz
        from PIL import Image

        doc = fitz.open(pdf_path)
        pix = doc[0].get_pixmap(dpi=300)
        doc.close()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        pix.save(tmp_path)

        try:
            image = Image.open(tmp_path).convert("RGB")
        finally:
            tmp_path.unlink()

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR:"},
            ],
        }]

        max_pixels = 1280 * 28 * 28
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            images_kwargs={
                "size": {
                    "shortest_edge": self.processor.image_processor.min_pixels,
                    "longest_edge": max_pixels,
                }
            },
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=4096)
        markdown = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:-1])

        self._write_output(markdown, output_path)
        return markdown


if __name__ == "__main__":
    run_cli(parser=PaddleOCRVLParser())