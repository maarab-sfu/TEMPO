import json
import re
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
        structured=False,
        attribution_answer=None,
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
        if attribution_answer is not None:
            content.append({
                "type": "text",
                "text": (
                    "Attribution task only. The answer below has ALREADY been generated. "
                    "Do not answer the question again or revise the answer. "
                    f"Existing answer: {attribution_answer}\n"
                    f"Choose supporting frame numbers ONLY from {list(range(1, len(frames) + 1))}. "
                    "Return ONLY JSON with the single key cited_frames, e.g. "
                    "{\"cited_frames\": [1, 2]}. "
                    "Select frames only if they visibly support the existing answer; "
                    "otherwise return {\"cited_frames\": []}. "
                    "Do not use timestamps or indexed frame IDs as frame numbers."
                )
            })
        elif structured:
            content.append({
                "type": "text",
                "text": (
                    "Return ONLY a JSON object with keys answer and cited_frames. "
                    "cited_frames must be a list of frame numbers from the provided "
                    "Frame 1, Frame 2, etc. labels that directly support the answer. "
                    "Use [] if evidence is insufficient. Do not invent frame numbers. "
                    "For insufficient evidence, say so in answer."
                )
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
    def answer_video_frames_grounded(self, frames, question):
        """Generate the answer with the original prompt, then attribute separately.

        Citations are model-selected, not independently verified support.
        """
        ordered_frames = sorted(frames, key=lambda f: f["timestamp"])
        # Stage 1 must remain identical to the M3.2 answer-generation call.
        answer = self.answer_video_frames(ordered_frames, question)
        # Stage 2 may fail, but it must never overwrite Stage 1's answer.
        raw = self.answer_video_frames(
            ordered_frames, question, max_new_tokens=128,
            attribution_answer=answer,
        )
        result = parse_attribution_response(raw, ordered_frames, answer)
        print("[M3.3] Raw attribution:", repr(raw), flush=True)
        print("[M3.3] Parsed attribution:", repr(result), flush=True)
        return result


def parse_grounded_response(raw, frames):
    """Parse model output without allowing fabricated or out-of-range citations."""
    answer = raw.strip()
    cited = []
    parsed = False
    try:
        # Allow a fenced JSON response, but do not guess citations from prose.
        candidate = answer
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate).strip()
        payload = json.loads(candidate)
        if isinstance(payload, dict) and isinstance(payload.get("answer"), str):
            answer = payload["answer"].strip()
            numbers = payload.get("cited_frames", [])
            if isinstance(numbers, list):
                parsed = True
                for number in numbers:
                    if type(number) is int and 1 <= number <= len(frames):
                        frame_id = frames[number - 1]["frame_id"]
                        if frame_id not in cited:
                            cited.append(frame_id)
    except (ValueError, TypeError, KeyError):
        pass
    return {
        "answer": answer,
        "cited_frame_ids": cited,
        "grounding_status": "cited" if cited else "unverified",
        "citation_parse_ok": parsed,
    }


def parse_attribution_response(raw, frames, answer):
    """Accept only a JSON list of in-range, 1-based supplied-frame numbers."""
    cited = []
    parsed = False
    try:
        candidate = raw.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate).strip()
        payload = json.loads(candidate)
        if isinstance(payload, dict) and isinstance(payload.get("cited_frames"), list):
            parsed = True
            for number in payload["cited_frames"]:
                if type(number) is int and 1 <= number <= len(frames):
                    frame_id = frames[number - 1]["frame_id"]
                    if frame_id not in cited:
                        cited.append(frame_id)
    except (ValueError, TypeError, KeyError):
        pass
    return {
        "answer": answer,
        "cited_frame_ids": cited,
        "grounding_status": "cited" if cited else "unverified",
        "citation_parse_ok": parsed,
    }
