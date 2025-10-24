import logging
import os
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv

from .core import PDFParser, parser_registry

# Load environment variables
load_dotenv()


@parser_registry()
class AdobeParser(PDFParser):
    """Adobe PDF Services parser implementation."""
    
    def __init__(self):
        super().__init__()
        self.client_id = os.getenv("ADOBE_CLIENT_ID")
        self.client_secret = os.getenv("ADOBE_CLIENT_SECRET")
        
        if not self.client_id or not self.client_secret:
            raise ValueError("ADOBE_CLIENT_ID and ADOBE_CLIENT_SECRET environment variables are required")
    
    @classmethod
    def parser_name(cls) -> str:
        return "adobe"
    

    def parse(self, pdf_path: Path, output_path: Path) -> str:
        """
        Parse PDF using Adobe PDF Services.
        
        Args:
            pdf_path: Path to input PDF file
            output_path: Path for output markdown file
            config: Adobe configuration with client_id and client_secret
            
        Returns:
            str: Generated markdown content
        """
        from adobe.pdfservices.operation.auth.service_principal_credentials import ServicePrincipalCredentials
        from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, \
            SdkException
        from adobe.pdfservices.operation.io.cloud_asset import CloudAsset
        from adobe.pdfservices.operation.io.stream_asset import StreamAsset
        from adobe.pdfservices.operation.pdf_services import PDFServices
        from adobe.pdfservices.operation.pdf_services_media_type import PDFServicesMediaType
        from adobe.pdfservices.operation.pdfjobs.jobs.extract_pdf_job import ExtractPDFJob
        from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_element_type import ExtractElementType
        from adobe.pdfservices.operation.pdfjobs.params.extract_pdf.extract_pdf_params import ExtractPDFParams
        from adobe.pdfservices.operation.pdfjobs.result.extract_pdf_result import ExtractPDFResult
        import zipfile

        temp_dir = "temp_adobe_extract"  # Define temp_dir outside try block for finally
        try:
            with open(pdf_path, 'rb') as file:
                input_stream = file.read()

            # Initial setup, create credentials instance
            credentials = ServicePrincipalCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )

            # Creates a PDF Services instance
            pdf_services = PDFServices(credentials=credentials)

            # Creates an asset from source file and upload
            input_asset = pdf_services.upload(input_stream=input_stream, mime_type=PDFServicesMediaType.PDF)

            # Create parameters for the job to extract text and styling information
            # Removed unsupported parameters: get_element_to_extract_bounding_box and get_images_as_base64
            extract_pdf_params = ExtractPDFParams(
                elements_to_extract=[ExtractElementType.TEXT, ExtractElementType.TABLES]
                # Include tables for more complete markdown
            )

            # Creates a new job instance
            extract_pdf_job = ExtractPDFJob(input_asset=input_asset, extract_pdf_params=extract_pdf_params)

            # Submit the job and gets the job result
            location = pdf_services.submit(extract_pdf_job)
            pdf_services_response = pdf_services.get_job_result(location, ExtractPDFResult)

            # Get content from the resulting asset
            result_asset: CloudAsset = pdf_services_response.get_result().get_resource()
            stream_asset: StreamAsset = pdf_services.get_content(result_asset)

            os.makedirs(temp_dir, exist_ok=True)
            zip_file_path = os.path.join(temp_dir, "extract_result.zip")

            # Write the zip file content to a temporary file
            with open(zip_file_path, "wb") as file:
                file.write(stream_asset.get_input_stream())

            markdown_content = []
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                if 'structuredData.json' in zip_ref.namelist():
                    with zip_ref.open('structuredData.json') as json_file:
                        structured_data = json.load(json_file)

                        if 'elements' in structured_data:
                            for element in structured_data['elements']:
                                text_content = element.get('Text', '')
                                # Get the last part of the path, e.g., 'P', 'H1'
                                path = element.get('Path', '').split('/')[-1]

                                if text_content:  # Only process if there's actual text
                                    if path.startswith('H'):  # Simple heuristic for headings
                                        # Use the number after H to determine heading level, e.g., H1 -> #, H2 -> ##
                                        try:
                                            level = int(path[1:])
                                            markdown_content.append("#" * min(level, 6) + " " + text_content.strip() + "\n")
                                        except ValueError:
                                            # If H is not followed by a number, treat as a paragraph
                                            markdown_content.append(text_content.strip() + "\n")
                                    elif path == 'P':  # Paragraph
                                        markdown_content.append(text_content.strip() + "\n")
                                    elif path == 'L':  # List item (simple bullet)
                                        markdown_content.append(f"- {text_content.strip()}\n")
                                    elif path == 'Table':  # Table content - might need more complex parsing for actual markdown tables
                                        # For simplicity, we'll just append text from table cells,
                                        # a more robust solution would build a markdown table.
                                        markdown_content.append(
                                            f"\n{text_content.strip()}\n")  # Add newlines around table text
                                    elif path == 'Figure' and 'Text' in element:  # Text within a figure (e.g., captions or labels)
                                        markdown_content.append(f"*{text_content.strip()}*\n")  # Italic for figure text
                                    else:  # Default to plain text with a newline
                                        markdown_content.append(text_content.strip() + "\n")

                                # Add an extra newline between blocks for better readability in Markdown
                                if path in ['P', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'L', 'Table', 'Figure']:
                                    markdown_content.append("\n")  # Add a blank line after common block elements

                else:
                    logging.warning(
                        "structuredData.json not found in the extracted zip. Cannot generate structured Markdown.")
                    # Fallback: if structuredData.json is not there, extract any other text file found
                    for file_name in zip_ref.namelist():
                        if file_name.endswith('.txt'):
                            with zip_ref.open(file_name) as extracted_file:
                                markdown_content.append(extracted_file.read().decode('utf-8'))
                                break  # Just take the first text file if JSON is absent

            # Write the aggregated markdown content to the output path
            markdown_text = "".join(markdown_content)
            self._write_output(markdown_text, output_path)

            return markdown_text

        except ServiceApiException as service_api_exception:
            logging.error(
                f"ServiceApiException: {service_api_exception.message} (Status Code: {service_api_exception.status_code})")
            raise
        except ServiceUsageException as service_usage_exception:
            logging.error(
                f"ServiceUsageException: {service_usage_exception.message} (Status Code: {service_usage_exception.status_code})")
            raise
        except SdkException as sdk_exception:
            logging.error(f"SdkException: {sdk_exception.message}")
            raise
        except FileNotFoundError:
            logging.error(f"Error: Input PDF file not found at {pdf_path}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from structuredData.json: {e}. The JSON might be malformed or incomplete.")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise
        finally:
            # Clean up temporary directory and zip file
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


