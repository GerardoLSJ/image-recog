# Image Recognition CLI

This project is a command-line interface (CLI) application to extract numerical data from images using a modular architecture that supports various extraction backends like OCR, VLM, and LLM.

## Installation

This project uses Poetry for dependency management. To install the necessary dependencies and the application itself, follow these steps:

1.  **Install Tesseract OCR**:
    *   **macOS (using Homebrew)**:
        ```bash
        brew install tesseract
        ```
    *   **Other Systems**: Please refer to the official [Tesseract documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html) for installation instructions.

2.  **Install Project Dependencies**:
    Navigate to the project's root directory and run:
    ```bash
    poetry install
    ```

## Usage

Once the installation is complete, you can use the `image-recog` command to extract numbers from images.

### Basic Command

```bash
poetry run image-recog <path-to-image> --extractor <extractor-type>
```

### Arguments

*   `<path-to-image>`: The path to the image file you want to process.
*   `--extractor`: The type of extractor to use. The available options are:
    *   `ocr`: Uses the Tesseract OCR engine.
    *   `gemini-flash`: Uses the Google Gemini 2.0 Flash Experimental model. Requires a `GOOGLE_API_KEY` environment variable.
    *   `qwen-inline-cpu`: Runs the `Qwen/Qwen2-VL-2B-Instruct` model locally on the CPU.
    *   `qwen-vllm`: Connects to a `vllm` server running the `Qwen/Qwen2-VL-2B-Instruct` model.
    *   `smol-vllm`: Connects to a `vllm` server running the `HuggingFaceTB/SmolVLM-2.2B-Instruct` model.

### Example

Here is an example of how to run the application with the test images provided in the project:

```bash
# This image contains no numbers and will return null
poetry run image-recog src/image_rec_mod/test1.jpg --extractor ocr

# This image contains the number 15
poetry run image-recog src/image_rec_mod/test2.webp --extractor ocr

# To use a remote VLM model (make sure the vLLM server is running first)
poetry run image-recog src/image_rec_mod/test2.webp --extractor qwen-vllm

# Example with SmolVLM
poetry run image-recog src/image_rec_mod/test2.webp --extractor smol-vllm
```

## VLM Server Setup (for vLLM models)

To use the `qwen-vllm` or `smol-vllm` extractors, you need to have a `vllm` server running with the desired model.

1.  **Install `vllm`**:
    ```bash
    pip install vllm
    ```

2.  **Start the server**:
    Due to a validation issue with the default settings in vLLM for these models, you need to specify the `--max-model-len` argument to prevent errors. A value of `2048` is recommended.

    *   For Qwen:
        ```bash
        vllm serve Qwen/Qwen2-VL-2B-Instruct --max-model-len 2048
        ```
    *   For SmolVLM:
        ```bash
        vllm serve HuggingFaceTB/SmolVLM-2.2B-Instruct --max-model-len 2048
        ```

    The server will start on `http://localhost:8000` by default.

## File Structure

```
.
├── .gitignore
├── .roomodes
├── plan.md
├── poetry.lock
├── pyproject.toml
├── README.md
├── src
│   ├── image_rec_mod
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── extractor.py
│   │   ├── main.py
│   │   ├── test1.jpg
│   │   ├── test2.webp
│   │   ├── test3.webp
│   │   ├── utils.py
│   │   └── backends
│   │       ├── __init__.py
│   │       ├── llm.py
│   │       ├── ocr.py
│   │       └── vlm.py
│   └── webapp
│       ├── __init__.py
│       ├── main.py
│       ├── static
│       │   └── style.css
│       └── templates
│           └── index.html
└── tests
    ├── test_image_rec_mod.py
    └── test_webapp.py
