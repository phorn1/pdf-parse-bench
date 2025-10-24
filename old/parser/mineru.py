import os
import time
import uuid
import zipfile
import tempfile
from pathlib import Path
import requests
from dotenv import load_dotenv

from .core import PDFParser, parser_registry

load_dotenv()


@parser_registry()
class MineRUParser(PDFParser):
    """PDF parser using MineRU API."""
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("MINERU_API_KEY")
        self.base_url = "https://mineru.net/api/v4"
        
        if not self.api_key:
            raise ValueError("MINERU_API_KEY environment variable is required")
    
    @classmethod
    def parser_name(cls) -> str:
        return "mineru"

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF to markdown using MineRU API.
        
        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file
            
        Returns:
            str: Generated markdown content
        """
        headers = {'Authorization': f'Bearer {self.api_key}'}
        
        # Request upload URL
        data = {
            "language": "auto",
            "files": [{"name": pdf_path.name, "is_ocr": True, "data_id": str(uuid.uuid4())}]
        }
        
        response = requests.post(f"{self.base_url}/file-urls/batch", 
                               headers={**headers, 'Content-Type': 'application/json'}, 
                               json=data)
        
        if response.status_code != 200 or response.json().get("code") != 0:
            raise Exception(f"Upload URL request failed: {response.text}")
            
        result = response.json()
        batch_id = result["data"]["batch_id"]
        upload_url = result["data"]["file_urls"][0]
        
        # Upload file
        with open(pdf_path, 'rb') as f:
            response = requests.put(upload_url, data=f)
            
        if response.status_code != 200:
            raise Exception(f"File upload failed: {response.status_code}")
        
        # Wait for processing and download result
        for _ in range(60):
            response = requests.get(f"{self.base_url}/extract-results/batch/{batch_id}", headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 0:
                    extract_results = result["data"].get("extract_result", [])
                    if extract_results:
                        state = extract_results[0].get("state")
                        if state == "done":
                            zip_url = extract_results[0].get("full_zip_url")
                            if zip_url:
                                markdown_content = self._extract_from_zip(zip_url)
                                self._write_output(markdown_content, output_path)
                                return markdown_content
                        elif state == "failed":
                            raise Exception(f"Processing failed: {extract_results[0].get('err_msg')}")
            
            time.sleep(5)
            
        raise Exception("Processing timeout")

    def _extract_from_zip(self, zip_url: str) -> str:
        """Download zip and extract markdown content."""
        response = requests.get(zip_url)
        if response.status_code != 200:
            raise Exception(f"Failed to download results: {response.status_code}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "results.zip"
            zip_path.write_bytes(response.content)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
                
                for file_path in Path(temp_dir).rglob("*.md"):
                    content = file_path.read_text(encoding='utf-8')
                    if content.strip():
                        return content
                
                for file_path in Path(temp_dir).rglob("*.txt"):
                    content = file_path.read_text(encoding='utf-8')
                    if content.strip():
                        return content
                        
                raise Exception("No content found in zip file")