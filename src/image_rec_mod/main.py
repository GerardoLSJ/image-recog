from pathlib import Path
from typing import Optional

from image_rec_mod.backends.llm import LLMExtractor
from image_rec_mod.backends.ocr import TesseractExtractor as OCRExtractor
from image_rec_mod.backends.vlm import (
    VLMExtractor,
    Qwen2VL2BInstructExtractor,
)
from image_rec_mod.extractor import Extractor


def get_extractor(name: str) -> Extractor:
    """
    Get an extractor instance by name.

    Args:
        name: The name of the extractor.

    Returns:
        An instance of the specified extractor.
    """
    if name == "ocr":
        return OCRExtractor()
    if name == "vlm":
        return VLMExtractor()
    if name == "llm":
        return LLMExtractor()
    if name == "qwen2-vl-2b-instruct":
        return Qwen2VL2BInstructExtractor()
    raise ValueError(f"Unknown extractor: {name}")


def extract_bib_number(file: Path, extractor_name: str) -> Optional[int]:
    """
    Extract the bib number from an image file using the specified extractor.

    Args:
        file: Path to the image file.
        extractor_name: The name of the extractor to use.

    Returns:
        The extracted bib number, or None if no number could be found.
    """
    extractor = get_extractor(extractor_name)
    return extractor.extract(file)
