from PIL import Image
from autogameplayer.core.interfaces import IOCRBackend


class TesseractOCRBackend(IOCRBackend):
    """Default OCR Backend using pytesseract."""

    def extract_text(self, image: Image.Image) -> str:
        try:
            import pytesseract

            # Run text extraction
            text = pytesseract.image_to_string(image).strip()
            return text
        except ImportError:
            return ""
        except Exception as e:
            import logging

            logging.debug(f"OCR Extraction failed: {e}")
            return ""
