import os, torch
from PIL import Image
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from loguru import logger

CAPTION_DIR = "outputs/captions"
os.makedirs(CAPTION_DIR, exist_ok=True)

class FrameCaptioner:
    def __init__(self, device: int = 0):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.pipe = pipeline(
            "image-text-to-text",
            model="google/gemma-3n-e2b-it",
            device=device if torch.cuda.is_available() else -1,
            torch_dtype=torch.bfloat16,
        )

    def caption_single(self, frame_path: str) -> str:
        try:
            out = self.pipe(frame_path, text="<image_soft_token> Describe what is shown.")
            caption = out[0]["generated_text"]
            return caption.split("</s>")[-1].strip()
        except Exception as e:
            logger.error(f"Caption pipeline failed for {frame_path}: {e}")
            return "Error: could not generate caption"

    def caption_frames(self, frame_paths: list[str]) -> dict:
        cached = {}
        if os.path.exists("captions_cache.json"):
            with open("captions_cache.json", "r") as f:
                cached = json.load(f)

        results = {}
        for path in frame_paths:
            name = os.path.basename(path)
            if name in cached:
                results[name] = cached[name]
                logger.info(f"Skipping cached: {name}")
            else:
                results[name] = self.caption_single(path)
                with open(os.path.join(CAPTION_DIR, name.replace(".jpg", ".txt")), "w") as f:
                    f.write(results[name])

        with open("captions_cache.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

