from enum import Enum
from pathlib import Path

import typer

from image_rec_mod.main import extract_bib_number

app = typer.Typer()


class ExtractorType(str, Enum):
    OCR = "ocr"
    GEMINI_FLASH = "gemini-flash"
    QWEN_INLINE_CPU = "qwen-inline-cpu"
    QWEN_INLINE_GPU = "qwen-inline-gpu"
    QWEN_VLLM = "qwen-vllm"
    SMOL_VLLM = "smol-vllm"


@app.command()
def run(
    image_path: Path = typer.Argument(..., help="Path to the image file."),
    extractor: ExtractorType = typer.Option(
        ExtractorType.OCR, help="Extractor to use."
    ),
    scale_width: int = typer.Option(
        None, help="Maximum width to scale the image."
    ),
    scale_height: int = typer.Option(
        None, help="Maximum height to scale the image."
    ),
):
    """
    Extracts information from an image using the specified extractor.
    """
    result = extract_bib_number(
        image_path, extractor.value, scale_width, scale_height
    )
    if "bib" in result:
        print(f"Bib number found: {result['bib']}")
    elif "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Unknown response format: {result}")


if __name__ == "__main__":
    app()