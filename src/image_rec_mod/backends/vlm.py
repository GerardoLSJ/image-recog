import os

from image_rec_mod.extractor import Extractor


class VLMExtractor(Extractor):
    def extract(self, image_path: str) -> dict:
        print("VLM extractor is not yet implemented")
        return {
            "image_name": os.path.basename(image_path),
            "number": "vlm_placeholder",
            "certainty": 0.99,
        }