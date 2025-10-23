from pathlib import Path
from typing import Optional

from image_rec_mod.backends.llm import RemoteLLMExtractor
from image_rec_mod.backends.ocr import TesseractExtractor as OCRExtractor
from image_rec_mod.backends.vlm import LocalVLMExtractor, RemoteVLLMExtractor
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
    if name == "gemini-flash":
        return RemoteLLMExtractor("gemini-2.0-flash-exp")
    if name == "qwen-inline-cpu":
        return LocalVLMExtractor("Qwen/Qwen2-VL-2B-Instruct", device="cpu")
    if name == "qwen-inline-gpu":
        return LocalVLMExtractor("Qwen/Qwen2-VL-2B-Instruct", device="cuda")
    if name == "qwen-vllm":
        return RemoteVLLMExtractor("Qwen/Qwen2-VL-2B-Instruct")
    if name == "smol-vllm":
        return RemoteVLLMExtractor("HuggingFaceTB/SmolVLM-2.2B-Instruct")
    raise ValueError(f"Unknown extractor: {name}")


def extract_bib_number(file: Path, extractor_name: str) -> dict:
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
