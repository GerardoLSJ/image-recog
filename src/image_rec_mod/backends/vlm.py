import os
import re
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from image_rec_mod.extractor import Extractor
import requests
import base64


class VLMExtractor(Extractor):
    """
    A placeholder for a Vision Language Model extractor.
    This class is meant to be subclassed by specific VLM implementations.
    """

    def extract(self, image_path: str) -> dict:
        raise NotImplementedError("VLM extractor is not yet implemented")


class Qwen2VL2BInstructExtractor(VLMExtractor):
    """
    Extractor using the Qwen2-VL-2B-Instruct model.

    This extractor uses the Qwen2-VL-2B-Instruct model from Hugging Face to
    extract numerical data from images. It requires the `transformers` and
    `qwen-vl-utils` packages to be installed.
    """

    def __init__(self):
        """
        Initializes the extractor by loading the model and processor.
        """
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            device_map="cpu",
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    def extract(self, image_path: str) -> dict:
        """
        Extracts a number from the given image using the Qwen2-VL-2B-Instruct model.

        Args:
            image_path: The path to the image file.

        Returns:
            A dictionary containing the extracted number, image name, and certainty.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                    {"type": "text", "text": "What is the number in this image?"},
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

        # Parse the number from the output string
        match = re.search(r'\d+', output_text[0])
        extracted_number = match.group(0) if match else None

        return {
            "image_name": os.path.basename(image_path),
            "number": extracted_number,
            "certainty": 0.95,  # Placeholder certainty
        }

class Qwen2VL2BInstructVLLMExtractor(VLMExtractor):
    """
    Extractor using the Qwen2-VL-2B-Instruct model served with vLLM.

    This extractor sends requests to a vLLM server that is serving the
    Qwen2-VL-2B-Instruct model. It requires the `requests` package to be
    installed.
    """

    def __init__(self, url="http://localhost:8000/v1/chat/completions"):
        """
        Initializes the extractor with the vLLM server URL.

        Args:
            url: The URL of the vLLM server's chat completions endpoint.
        """
        self.url = url

    def extract(self, image_path: str) -> dict:
        """
        Extracts a number from the given image by sending a request to the vLLM server.

        Args:
            image_path: The path to the image file.

        Returns:
            A dictionary containing the extracted number, image name, and certainty.
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        headers = {"Content-Type": "application/json"}
        data = {
            "model": "Qwen/Qwen2-VL-2B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"data:image;base64,{encoded_string}",
                        },
                        {"type": "text", "text": "What is the number in this image?"},
                    ],
                }
            ],
            "max_tokens": 128,
        }

        response = requests.post(self.url, headers=headers, json=data)
        response.raise_for_status()
        
        extracted_number = response.json()["choices"][0]["message"]["content"].strip()

        return {
            "image_name": os.path.basename(image_path),
            "number": extracted_number,
            "certainty": 0.95,  # Placeholder certainty
        }