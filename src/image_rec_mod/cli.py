from enum import Enum
from pathlib import Path

import typer

from image_rec_mod.main import extract_bib_number

app = typer.Typer()


class ExtractorType(str, Enum):
    OCR = "ocr"
    VLM = "vlm"
    LLM = "llm"


@app.command()
def run(
    image_path: Path = typer.Argument(..., help="Path to the image file."),
    extractor: ExtractorType = typer.Option(
        ExtractorType.OCR, help="Extractor to use."
    ),
):
    """
    Extracts information from an image using the specified extractor.
    """
    result = extract_bib_number(image_path, extractor.value)
    if result is not None:
        print(f"Bib number found: {result}")
    else:
        print("No bib number found.")


if __name__ == "__main__":
    app()