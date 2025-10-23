import os
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from PIL import Image

from image_rec_mod.extractor import Extractor

# Configure the Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)


class LLMExtractor(Extractor):
    """
    Extractor that uses a Large Language Model to extract the bib number.
    """

    def __init__(self, model: str = None):
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        self.model = genai.GenerativeModel(self.model_name)

    def extract(self, file: Path) -> Optional[int]:
        """
        Extract the bib number from an image file.

        Args:
            file: Path to the image file.

        Returns:
            The extracted bib number, or None if no number could be found.
        """
        prompt = """
You are analyzing an image of a runner wearing a bib number on their chest.

Task: Extract ONLY the number visible on the runner's bib/chest.

Rules:
1. Return only the number itself (digits only)
2. If multiple runners, extract the most prominent/centered one
3. If no clear bib number visible, return "null"
4. Ignore any other text or numbers not on the bib

Format: Return just the number, nothing else.
"""
        try:
            image = Image.open(file)
            response = self.model.generate_content([prompt, image])
            text = response.text.strip().lower()

            if text == "null":
                return None

            return int(text)
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
