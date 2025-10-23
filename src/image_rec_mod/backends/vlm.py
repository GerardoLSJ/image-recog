import os
import re
import json
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


class LocalVLMExtractor(VLMExtractor):
    """
    Extractor for Vision Language Models running locally.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initializes the extractor by loading the model and processor.
        """
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def extract(self, image_path: str) -> dict:
        """
        Extracts a number from the given image using the specified model.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
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
                return data
            else:
                return {"error": f"No JSON object found in VLM response: {text}"}
        except json.JSONDecodeError:
            return {"error": f"Failed to decode JSON from VLM response: {text}"}
        except Exception as e:
            return {"error": f"An error occurred: {e}"}


class RemoteVLLMExtractor(VLMExtractor):
    """
    Extractor for Vision Language Models served with vLLM.
    """

    def __init__(self, model_name: str, url="http://localhost:8000/v1/chat/completions"):
        """
        Initializes the extractor with the vLLM server URL and model name.
        """
        self.model_name = model_name
        self.url = url

    def extract(self, image_path: str) -> dict:
        """
        Extracts a number from the given image by sending a request to the vLLM server.
        """
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

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

        try:
            response = requests.post(self.url, headers=headers, json=data)
            response.raise_for_status()
            
            text = response.json()["choices"][0]["message"]["content"].strip()
            
            # The model might wrap the JSON in markdown, so we strip it
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            return json.loads(text)
        except requests.exceptions.RequestException as e:
            return {"error": f"HTTP request failed: {e}"}
        except (json.JSONDecodeError, KeyError):
            return {"error": f"Failed to parse response from vLLM server: {response.text}"}