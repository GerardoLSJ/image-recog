import sys
from pathlib import Path

# Add the 'src' directory to the Python path to allow running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from image_rec_mod.main import get_extractor # Import the new function

def test_image_folder():
    """
    Iterates through images in the test folder, extracts bib numbers efficiently,
    and prints them.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    folder_to_test = project_root / "tests" / "test_images"

    print(f"Testing images in: {folder_to_test}")

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
        extractor = get_extractor("qwen-inline-cpu")
        print("Model loaded. Extracting bib numbers...")

        # 2. Process all images with the new method
        results = extractor.extract_multiple_bib_numbers(image_paths)

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


if __name__ == "__main__":
    test_image_folder()