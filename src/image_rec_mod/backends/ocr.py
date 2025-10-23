import re
from pathlib import Path
from typing import Any, Dict

import pytesseract
from PIL import Image

from image_rec_mod.extractor import Extractor


class TesseractExtractor(Extractor):
    """
    Extracts numerical data from an image using Tesseract OCR.
    """

    def extract(self, image_path: str) -> Dict[str, Any] | None:
        """
        Extracts the first numerical value from an image.

        Args:
            image_path: The path to the image file.

        Returns:
            A dictionary containing the extracted information, or None if no
            number is found.
        """
        try:
            print(f"--- OCR Debug ---")
            print(f"Processing image: {image_path}")
            image = Image.open(image_path)
            print(f"Image format: {image.format}, size: {image.size}, mode: {image.mode}")

            # Preprocessing: convert to grayscale
            gray_image = image.convert("L")
            print("Converted to grayscale.")

            # Preprocessing: apply thresholding
            import numpy as np
            threshold = 128
            np_img = np.array(gray_image)
            binary_img = np.where(np_img > threshold, 255, 0).astype(np.uint8)
            from PIL import Image as PILImage
            processed_image = PILImage.fromarray(binary_img)
            print("Applied thresholding.")

            # OCR on processed image
            text = pytesseract.image_to_string(processed_image)
            print(f"Raw Tesseract output after preprocessing:\n---\n{text}\n---")
            
            # Find the first number in the extracted text
            match = re.search(r'\d+', text)
            print(f"Regex match for numbers: {match}")
            number = int(match.group(0)) if match else None
            print(f"Extracted number: {number}")
            print(f"-------end----------")

            return {
                "image_name": Path(image_path).name,
                "number": number,
                "certainty": 0.95,
            }
        except FileNotFoundError:
            # Handle cases where the image path is invalid
            print(f"Error: Image file not found at {image_path}")
            return None
        except Exception as e:
            # Handle other potential errors during OCR
            print(f"An unexpected error occurred: {e}")
            return None