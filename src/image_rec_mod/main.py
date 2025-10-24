from pathlib import Path
from typing import Optional

from image_rec_mod.backends.llm import RemoteLLMExtractor
from image_rec_mod.backends.ocr import TesseractExtractor as OCRExtractor
from image_rec_mod.backends.vlm import LocalVLMExtractor, RemoteVLLMExtractor
from image_rec_mod.extractor import Extractor


_EXTRACTOR_CACHE = {}


def get_extractor(extractor_name: str) -> Extractor:
    """
    Get an extractor instance by name. This function caches extractor instances to avoid
    reloading models on every call.

    Args:
        extractor_name: The name of the extractor.

    Returns:
        An instance of the specified extractor.
    """
    # Check cache first for supported models
    if extractor_name in _EXTRACTOR_CACHE:
        return _EXTRACTOR_CACHE[extractor_name]

    # Handle cached local VLMs
    if extractor_name == "qwen-inline-cpu":
        extractor = LocalVLMExtractor("Qwen/Qwen2-VL-2B-Instruct", device="cpu")
        _EXTRACTOR_CACHE[extractor_name] = extractor
        return extractor
    if extractor_name == "qwen-inline-gpu":
        extractor = LocalVLMExtractor("Qwen/Qwen2-VL-2B-Instruct", device="cuda")
        _EXTRACTOR_CACHE[extractor_name] = extractor
        return extractor

    # Handle non-cached extractors
    if extractor_name == "ocr":
        return OCRExtractor()
    if extractor_name == "gemini-flash":
        return RemoteLLMExtractor("gemini-2.0-flash-exp")
    if extractor_name == "qwen-vllm":
        return RemoteVLLMExtractor("Qwen/Qwen2-VL-2B-Instruct")
    if extractor_name == "smol-vllm":
        return RemoteVLLMExtractor("HuggingFaceTB/SmolVLM-2.2B-Instruct")

    raise ValueError(f"Unknown extractor: {extractor_name}")


def extract_bib_number(image_path: str, extractor_name: str) -> dict:
    """
    Extract the bib number from an image file using the specified extractor.

    Args:
        image_path: Path to the image file.
        extractor_name: The name of the extractor to use.

    Returns:
        The extracted bib number, or None if no number could be found.
    """
    extractor = get_extractor(extractor_name)
    return extractor.extract(image_path)
