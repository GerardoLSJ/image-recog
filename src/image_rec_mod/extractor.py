from abc import ABC, abstractmethod
from typing import Dict, Any

class Extractor(ABC):
    """
    An abstract base class for data extractors from images.

    This interface defines the contract for all data extraction providers.
    Each provider must implement the `extract` method, which takes an image
    path and returns the extracted information. This can include methods like
    OCR, VLM, or other LLM-based approaches.
    """

    @abstractmethod
    def extract(self, image_path: str) -> Dict[str, Any]:
        """
        Extracts data from an image using a specific implementation.

        Args:
            image_path: The path to the image file.

        Returns:
            A dictionary containing the extracted information. The structure of
            the dictionary depends on the specific extractor implementation.
            Example:
            {
                "image_name": "img.jpg",
                "number": 123,
                "certainty": 0.95
            }
        """
        pass