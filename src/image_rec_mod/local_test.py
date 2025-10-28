import sys
from pathlib import Path
import time
from datetime import datetime
import argparse

# Add the 'src' directory to the Python path to allow running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_rec_mod.main import get_extractor # Import the new function

def test_image_folder(extractor_name="qwen-inline-gpu", scale_width=500, scale_height=500):
    """
    Iterates through images in the test folder, extracts bib numbers efficiently,
    and prints them.
    
    Args:
        extractor_name: Name of the extractor to use. Defaults to "qwen-inline-gpu".
        scale_width: Maximum width to scale images. Defaults to 500.
        scale_height: Maximum height to scale images. Defaults to 500.
    """
    start_time = time.time()
    project_root = Path(__file__).resolve().parent.parent.parent
    folder_to_test = project_root / "tests" / "test_images"

    print(f"Testing images in: {folder_to_test}")
    print(f"Using image scaling: {scale_width}x{scale_height}")

    image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    image_paths = [
        str(f) for f in sorted(folder_to_test.iterdir())
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_paths:
        print("No images found to test.")
        return

    try:
        # 1. Get the extractor instance (model loads here, ONCE)
        print("Loading model...")
        extractor = get_extractor(extractor_name)
        print("Model loaded. Extracting bib numbers...")

        # 2. Process all images with the new method
        results = extractor.extract_multiple_bib_numbers(
            image_paths, scale_width, scale_height
        )

        # 3. Print results
        for item in results:
            image_name = Path(item['image']).name
            bib_number = item['result'].get('bib')
            print(f"Processing {image_name}...")
            if bib_number is not None:
                print(f"  Bib number: {bib_number}")
            else:
                print("  Bib number not found.")
                print(f"  Full result: {item['result']}")

    except Exception as e:
        print(f"  An error occurred: {e}")
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        
        summary = (
            f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Extractor Used: {extractor_name}\n"
            f"Image Folder: {folder_to_test}\n"
            f"Number of Images: {len(image_paths)}\n"
            f"Image Scaling: {scale_width}x{scale_height}\n"
            f"Total Execution Time: {total_time:.2f} seconds\n"
            f"--------------------------------------------------\n"
        )
        
        summary_file = project_root / "logs_summaries.txt"
        with open(summary_file, "a") as f:
            f.write(summary)
        
        print(f"\nSummary written to {summary_file}")


def main():
    """Entry point for the local-test script."""
    parser = argparse.ArgumentParser(
        description="Test bib number extraction on images in the test folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available extractors:
  qwen-inline-cpu       Qwen2-VL-2B on CPU
  qwen-inline-gpu       Qwen2-VL-2B on GPU (default)
  smol-inline-cpu       SmolVLM-2.2B on CPU
  smol-inline-gpu       SmolVLM-2.2B on GPU
  qwen-vllm             Qwen via vLLM server
  smol-vllm             SmolVLM via vLLM server
  gemini-flash          Google Gemini Flash (remote)
  ocr                   Tesseract OCR

Examples:
  poetry run local-test
  poetry run local-test --extractor smol-inline-gpu
  poetry run local-test --extractor qwen-inline-cpu --width 800 --height 600
        """
    )

    parser.add_argument(
        "--extractor", "-e",
        type=str,
        default="qwen-inline-gpu",
        help="Name of the extractor to use (default: qwen-inline-gpu)"
    )

    parser.add_argument(
        "--width", "-w",
        type=int,
        default=500,
        help="Maximum width to scale images (default: 500)"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=500,
        help="Maximum height to scale images (default: 500)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"Bib Number Extraction Test")
    print("=" * 60)
    print(f"Extractor: {args.extractor}")
    print("=" * 60)
    print()

    test_image_folder(
        extractor_name=args.extractor,
        scale_width=args.width,
        scale_height=args.height
    )


if __name__ == "__main__":
    main()
