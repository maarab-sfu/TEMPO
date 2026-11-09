import torch
import numpy as np

from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEncoder:

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        device=None
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading CLIP on {self.device}")

        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode_images(self, image_paths, batch_size=16):

        embeddings = []

        for start in range(0, len(image_paths), batch_size):

            batch_paths = image_paths[start:start + batch_size]

            images = []

            for path in batch_paths:
                with Image.open(path) as img:
                    images.append(img.convert("RGB"))

            inputs = self.processor(
                images=images,
                return_tensors="pt",
                padding=True
            )

            inputs = {
                k: v.to(self.device)
                for k, v in inputs.items()
            }

            features = self.model.visual_projection(
                self.model.vision_model(
                    pixel_values=inputs["pixel_values"]
                ).pooler_output
            )

            # Normalize embeddings
            features = torch.nn.functional.normalize(
                features,
                p=2,
                dim=-1
            )

            embeddings.append(
                features.cpu().numpy()
            )

        return np.concatenate(embeddings, axis=0)

    @torch.inference_mode()
    def encode_text(self, texts):

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        features = self.model.text_projection(
            self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            ).pooler_output
        )

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=-1
        )

        return features.cpu().numpy()