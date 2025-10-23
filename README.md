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
    *   `ocr` (default): Uses the Tesseract OCR engine.
    *   `vlm`: Uses a Vision Language Model. The following VLM models are supported:
        *   `qwen2-vl-2b-instruct`: Uses the [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) model directly.
        *   `qwen2-vl-2b-instruct-vllm`: Uses the [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) model served with `vllm`. See VLM Server Setup for more details.
    *   `llm`: (Placeholder) Will use a Large Language Model for refinement.

### Example

Here is an example of how to run the application with the test images provided in the project:

```bash
# This image contains no numbers and will return null
poetry run image-recog src/image_rec_mod/test1.jpg --extractor ocr

# This image contains the number 15
poetry run image-recog src/image_rec_mod/test2.webp --extractor ocr
```


## VLM Server Setup (for `qwen2-vl-2b-instruct-vllm`)

To use the `qwen2-vl-2b-instruct-vllm` extractor, you need to have a `vllm` server running with the `Qwen/Qwen2-VL-2B-Instruct` model.

1.  **Install `vllm`**:
    ```bash
    pip install vllm
    ```

2.  **Start the server**:
    ```bash
    vllm serve "Qwen/Qwen2-VL-2B-Instruct"
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
```
