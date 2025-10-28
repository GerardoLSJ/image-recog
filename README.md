# Image Recognition CLI

This project is a command-line interface (CLI) application to extract numerical data from images using a modular architecture that supports various extraction backends like OCR, VLM, and LLM.

## Installation
      ```bash
poetry run pip uninstall torch torchvision -y
poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

## Run
      ```bash

poetry run image-recog C:\Users\gerar\Projects\image-recog\tests\test_images\test6.webp --extractor smol-inline-gpu




```
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

3.  **Install PyTorch with CUDA Support** (for GPU-accelerated models):

    After installing dependencies, you need to install PyTorch with CUDA support for GPU acceleration:

    **For systems with NVIDIA GPU (CUDA 12.1 compatible):**
    ```bash
    poetry run pip uninstall -y torch torchvision torchaudio
    poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

    **For CPU-only systems or if you don't have an NVIDIA GPU:**
    ```bash
    poetry run pip uninstall -y torch torchvision torchaudio
    poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

    **Verify your installation:**
    ```bash
    poetry run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
    ```

    **Note:** GPU systems should show `CUDA available: True`. The CUDA 12.1 build is compatible with NVIDIA drivers supporting CUDA 12.1 and later (including CUDA 13.0).

## Usage

Once the installation is complete, you can use the `image-recog` command to extract numbers from images.

### Basic Command

```bash
poetry run image-recog <path-to-image> --extractor <extractor-type>
```

$env:VARIABLE_NAME="value"

### Arguments

*   `<path-to-image>`: The path to the image file you want to process.
*   `--extractor`: The type of extractor to use. The available options are:
    *   `ocr`: Uses the Tesseract OCR engine.
    *   `gemini-flash`: Uses the Google Gemini 2.0 Flash Experimental model. Requires a `GOOGLE_API_KEY` environment variable.
    *   `qwen-inline-cpu`: Runs the `Qwen/Qwen2-VL-2B-Instruct` model locally on the CPU.
    *   `qwen-inline-gpu`: Runs the `Qwen/Qwen2-VL-2B-Instruct` model locally on the GPU.
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

# Example with Qwen inline GPU
poetry run image-recog src/image_rec_mod/test2.webp --extractor qwen-inline-gpu

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




gerry-ub@Midgard:/mnt/c/Users/gerar/Projects/ubuntu-py$ source venv/bin/activate

wsl.exe -d Ubuntu

/home/gerry-ub/projects/ubuntu-py

C:\Users\gerar\Projects\image-recog\src\test6.webp

poetry run image-recog C:\Users\gerar\Projects\image-recog\src\test6.webp --extractor qwen-inline-gpu