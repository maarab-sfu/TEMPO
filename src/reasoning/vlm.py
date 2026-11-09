import torch

from pathlib import Path
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)


class VisionLanguageModel:

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
    ):

        self.device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading VLM: {model_name}")
        print(f"Available device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            attn_implementation="sdpa",
        )

        self.model.eval()
        if torch.cuda.is_available():
            print(
                "GPU memory allocated:",
                round(torch.cuda.memory_allocated() / 1024**3, 2),
                "GB"
            )

        print("VLM loaded successfully")

    @torch.inference_mode()
    def answer_image(
        self,
        image_path,
        question,
        max_new_tokens=128,
    ):

        image_path = Path(image_path)

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a visual question-answering assistant. "
                    "Always respond in English. "
                    "Answer using only the visual evidence provided. "
                    "Do not assume that an event occurred merely because it seems likely. "
                    "If the evidence is insufficient, explicitly say so. "
                    "Keep answers concise."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": question,
                    },
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )

        # Inputs go to the model's input embedding device.
        input_device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(input_device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        # Remove prompt tokens from the output.
        generated_ids = generated_ids[
            :,
            inputs["input_ids"].shape[1]:,
        ]

        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return answer.strip()


    @torch.inference_mode()
    def answer_video_frames(
        self,
        frames,
        question,
        max_new_tokens=256,
    ):
        """
        Answer a question using multiple timestamped video frames.

        Args:
            frames: List of dictionaries containing:
                - image_path
                - timestamp
            question: Natural-language question.
            max_new_tokens: Maximum generated tokens.

        Returns:
            Generated answer.
        """

        if not frames:
            raise ValueError("At least one frame is required.")

        # Ensure chronological ordering.
        frames = sorted(
            frames,
            key=lambda x: x["timestamp"]
        )

        images = []
        content = []

        content.append({
            "type": "text",
            "text": (
                "The following images are sampled frames from "
                "the same video, ordered chronologically. "
                "Their timestamps are provided in seconds. "
                "Use only the observed frames as evidence. "
                "Do not assume what happened between frames.\n"
            )
        })

        for i, frame in enumerate(frames):

            with Image.open(frame["image_path"]) as img:
                images.append(img.convert("RGB"))

            content.append({
                "type": "text",
                "text": (
                    f"Frame {i + 1}, "
                    f"timestamp {frame['timestamp']:.2f} seconds:"
                )
            })

            content.append({
                "type": "image"
            })

        content.append({
            "type": "text",
            "text": f"Question: {question}"
        })

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a video question-answering assistant. "
                    "Always answer in English. "
                    "Reason about the chronological sequence "
                    "of the provided frames. "
                    "Distinguish observed events from assumptions. "
                    "If the frames do not establish an answer, "
                    "state that the evidence is insufficient. "
                    "When possible, reference relevant timestamps."
                )
            },
            {
                "role": "user",
                "content": content
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt"
        )

        input_device = (
            self.model.get_input_embeddings().weight.device
        )

        inputs = inputs.to(input_device)

        print("Input token shape:", inputs["input_ids"].shape)
        print("Pixel values shape:", inputs["pixel_values"].shape)
        print("Model dtype:", self.model.dtype)
        print(
            "GPU memory allocated:",
            round(torch.cuda.memory_allocated() / 1024**3, 2),
            "GB"
        )

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            use_cache=True,
        )

        generated_ids = generated_ids[
            :,
            inputs["input_ids"].shape[1]:
        ]

        answer = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return answer.strip()