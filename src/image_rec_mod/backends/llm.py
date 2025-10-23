import os
import json
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from PIL import Image

from image_rec_mod.extractor import Extractor

# Configure the Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)


class RemoteLLMExtractor(Extractor):
    """
    Extractor that uses a remote Large Language Model to extract the bib number.
    """

    def __init__(self, model: str):
        self.model_name = model
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set.")
        self.model = genai.GenerativeModel(self.model_name)

    def extract(self, file: Path) -> dict:
        """
        Extract the bib number from an image file.

        Args:
            file: Path to the image file.

        Returns:
            A dictionary containing the extracted bib number or an error.
        """

        prompt = """You are a specialized image analyzer for extracting runner bib numbers.

TASK: Identify and extract the bib number worn on a runner's chest/torso.

INSTRUCTIONS:
1. Look for numbers printed on fabric bibs attached to the runner's chest or torso
2. If multiple runners are visible, focus on the most prominent/centered runner
3. Extract only the bib number itself - ignore timing chips, advertising, or other text
4. Common bib locations: center chest, upper torso, sometimes on shorts/belt
5. If no bib number is visible, bib might be the name of the runner. extract as an error in that case.

RESPONSE FORMAT: Return ONLY a valid JSON object with no additional text or markdown formatting.

Success case:
{"bib": 256}

Failure cases:
{"error": "No runner visible in image"}
{"error": "No bib number visible on runner"}
{"error": "Image too blurry to read number"}
{"error": "Multiple runners, cannot determine primary subject"}
{"error": "Name of the Runner: John Doe"}

CRITICAL: 
- Return ONLY the JSON object, no markdown code blocks, no explanations
- The "bib" value must be an integer (digits only)
- Use "error" key when extraction is not possible. Give as much context as possible.
"""
        try:
            image = Image.open(file)
            response = self.model.generate_content([prompt, image])
            text = response.text.strip()

            # The model might wrap the JSON in markdown, so we strip it
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            data = json.loads(text)
            if "bib" in data:
                return data
            elif "error" in data:
                return data
            else:
                return {"error": f"LLM returned unknown JSON format: {text}"}

        except json.JSONDecodeError:
            # It's possible the model just returns a number.
            if text.isdigit():
                return {"bib": int(text)}
            return {"error": f"Failed to decode JSON from LLM response: {text}"}
        except Exception as e:
            return {"error": f"An error occurred: {e}"}
