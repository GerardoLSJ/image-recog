import os
import re
import json
import torch
import time
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from image_rec_mod.extractor import Extractor
import requests
import base64
from image_rec_mod.utils import ImageScaler
from typing import Optional
from io import BytesIO


class VLMExtractor(Extractor):
    """
    A placeholder for a Vision Language Model extractor.
    This class is meant to be subclassed by specific VLM implementations.
    """

    def extract(self, image_path: str) -> dict:
        raise NotImplementedError("VLM extractor is not yet implemented")


class LocalVLMExtractor(VLMExtractor):
    """
    Extractor for Vision Language Models running locally.
    """

    def __init__(self, model_name: str, device: str = "cpu", scaler: Optional[ImageScaler] = None):
        """
        Initializes the extractor by loading the model and processor.
        Adds debug logging for CUDA and model loading.
        """
        self.scaler = scaler
        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("vlm_debug")
        self.logger.info(f"Initializing Qwen2VLForConditionalGeneration with model_name={model_name}, device={device}")
        
        start_time = time.time()

        if torch.cuda.is_available():
            self.logger.info(f"PyTorch version: {torch.__version__}")
            self.logger.info(f"CUDA is available. Device count: {torch.cuda.device_count()}")
            self.logger.info(f"CUDA version: {torch.version.cuda}")
            self.logger.info(f"cuDNN version: {torch.backends.cudnn.version()}")
            self.logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
            self.logger.info(f"CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            self.logger.info("CUDA is NOT available. Using CPU.")
        if device != "cpu" and not torch.cuda.is_available():
            self.logger.warning(f"Requested device '{device}' but CUDA is not available. Falling back to CPU.")
            device = "cpu"
        
        try:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                device_map=device,
            )
            self.logger.info(f"Model loaded successfully on device: {self.model.device}")
        except RuntimeError as e:
            self.logger.error(f"Failed to load model on device '{device}': {e}")
            if "CUDA out of memory" in str(e):
                raise RuntimeError("CUDA out of memory. Try a smaller model or free up GPU memory.") from e
            raise RuntimeError(f"Failed to load model: {e}") from e
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during model loading: {e}")
            raise RuntimeError(f"An unexpected error occurred during model loading: {e}") from e

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.logger.info("Processor loaded successfully.")
        
        end_time = time.time()
        self.logger.info(f"Model and processor loaded in {end_time - start_time:.2f} seconds.")

    def extract(self, image_path: str) -> dict:
        """
        Extracts a number from the given image using the specified model.
        """
        start_time = time.time()
        self.logger.info(f"Extracting bib number from: {image_path}")

        image_to_process = image_path
        if self.scaler:
            self.logger.info(f"Scaling image: {image_path}")
            scaled_image = self.scaler.scale(image_path)
            # Create a temporary path for the scaled image
            _, ext = os.path.splitext(image_path)
            temp_image_path = str(image_path).replace(ext, f"_scaled{ext}")
            scaled_image.save(temp_image_path)
            image_to_process = temp_image_path
            self.logger.info(f"Using scaled image: {image_to_process}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(image_to_process)}"},
                    {"type": "text", "text": """You are a specialized image analyzer for extracting runner bib numbers.

TASK: Identify and extract the bib number worn on a runner's chest/torso.

RESPONSE FORMAT: Return ONLY a valid JSON object with no additional text or markdown formatting.

Success case:
{"bib": 256}

Failure cases:
{"error": "No bib number visible on runner"}
"""},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        text = output_text[0].strip()

        # The model might wrap the JSON in markdown, so we strip it
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            # Use a more lenient regex to find the JSON object
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_text = match.group(0)
                # The model sometimes returns numbers with leading zeros, which is invalid in JSON
                # We can clean this up before parsing
                json_text = re.sub(r':\s*0+(\d+)', r': \1', json_text)
                data = json.loads(json_text)
                end_time = time.time()
                self.logger.info(f"Extraction for {os.path.basename(image_path)} completed in {end_time - start_time:.2f} seconds.")
                return data
            else:
                self.logger.error(f"No JSON object found in VLM response: {text}")
                return {"error": f"No JSON object found in VLM response: {text}"}
        except json.JSONDecodeError:
            self.logger.error(f"Failed to decode JSON from VLM response: {text}")
            return {"error": f"Failed to decode JSON from VLM response: {text}"}
        except Exception as e:
            self.logger.error(f"An error occurred during extraction: {e}")
            return {"error": f"An error occurred: {e}"}

    def extract_multiple_bib_numbers(
        self, 
        image_paths: list[str], 
        scale_width: Optional[int] = None, 
        scale_height: Optional[int] = None
    ) -> list[dict]:
        """
        Extracts bib numbers from a list of images.
        
        Args:
            image_paths: List of paths to image files.
            scale_width: Optional maximum width to scale images.
            scale_height: Optional maximum height to scale images.
        
        Returns:
            List of dictionaries containing image names and extraction results.
        """
        total_start_time = time.time()
        self.logger.info(f"Starting batch extraction for {len(image_paths)} images.")
        
        # Create temporary scaler if scale parameters are provided
        original_scaler = self.scaler
        if scale_width and scale_height:
            self.logger.info(f"Using temporary scaler with dimensions: {scale_width}x{scale_height}")
            self.scaler = ImageScaler(scale_width, scale_height)
        
        results = []
        try:
            for i, image_path in enumerate(image_paths):
                self.logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                result = self.extract(image_path)
                results.append({"image": os.path.basename(image_path), "result": result})
        finally:
            # Restore original scaler
            self.scaler = original_scaler
        
        total_end_time = time.time()
        self.logger.info(f"Batch extraction for {len(image_paths)} images completed in {total_end_time - total_start_time:.2f} seconds.")
        return results


class RemoteVLLMExtractor(VLMExtractor):
    """
    Extractor for Vision Language Models served with vLLM.
    """

    def __init__(self, model_name: str, url="http://localhost:8000/v1/chat/completions", scaler: Optional[ImageScaler] = None):
        """
        Initializes the extractor with the vLLM server URL and model name.
        Adds debug logging for HTTP requests and responses.
        """
        self.scaler = scaler
        import logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("remote_vllm_debug")
        self.model_name = model_name
        self.url = url
        self.logger.info(f"RemoteVLLMExtractor initialized with model_name={model_name}, url={url}")

    def extract(self, image_path: str) -> dict:
        """
        Extracts a number from the given image by sending a request to the vLLM server.
        Adds debug logging for request payload, response, and errors.
        """
        import base64, json, requests, os, time
        start_time = time.time()
        self.logger.info(f"Preparing to extract bib number from image: {image_path}")

        image_bytes = b""
        if self.scaler:
            self.logger.info(f"Scaling image: {image_path}")
            scaled_image = self.scaler.scale(image_path)
            buffer = BytesIO()
            scaled_image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
        else:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()

        encoded_string = base64.b64encode(image_bytes).decode("utf-8")
        self.logger.info(f"Base64 encoded image length: {len(encoded_string)} characters")
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_string}"
                            },
                        },
                        {"type": "text", "text": """You are a specialized image analyzer for extracting runner bib numbers.

            TASK: Identify and extract the bib number worn on a runner's chest/torso.

            RESPONSE FORMAT: Return ONLY a valid JSON object with no additional text or markdown formatting.

            Success case:
            {"bib": 256}

            Failure cases:
            {"error": "No bib number visible on runner"}
            """},
                    ],
                }
            ],
            "max_tokens": 128,
        }
        self.logger.info(f"Sending POST request to {self.url} with payload: {json.dumps(data)[:500]}...")
        try:
            response = requests.post(self.url, headers=headers, json=data)
            self.logger.info(f"HTTP status: {response.status_code}")
            response.raise_for_status()
            self.logger.info(f"Raw response: {response.text[:500]}")
            text = response.json()["choices"][0]["message"]["content"].strip()
            # The model might wrap the JSON in markdown, so we strip it
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            self.logger.info(f"Parsed model output: {text}")
            result = json.loads(text)
            end_time = time.time()
            self.logger.info(f"Remote extraction for {os.path.basename(image_path)} completed in {end_time - start_time:.2f} seconds.")
            return result
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP request failed: {e}")
            return {"error": f"HTTP request failed: {e}"}
        except (json.JSONDecodeError, KeyError):
            self.logger.error(f"Failed to parse response from vLLM server: {response.text}")
            return {"error": f"Failed to parse response from vLLM server: {response.text}"}
