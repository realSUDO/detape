"""
Caption Frames Module
Generates captions for each extracted frame using Gemma 3n Vision
"""
import os
import torch
from PIL import Image
from transformers import pipeline
from typing import List
from loguru import logger

from .extract_frames import extract_frames_from_video

# Load configuration - Using official Gemma 3n multimodal model
# PRIMARY: Gemma 3n E2B (2B params, ~2GB RAM, optimized for speed)
MODEL_NAME = "google/gemma-3n-e2b-it"  # Official Gemma 3n instruction-tuned model
# FALLBACK: Gemma 3n E4B for better quality if needed
FALLBACK_MODEL = "google/gemma-3n-e4b-it"  # Larger Gemma 3n variant
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTION_OUTPUT_DIR = "outputs/captions"

logger.info(f"Using device: {DEVICE}")
logger.info(f"Using Gemma 3n E2B model: {MODEL_NAME}")

# Ensure output directory exists
os.makedirs(CAPTION_OUTPUT_DIR, exist_ok=True)

class FrameCaptioner:
    def __init__(self, model_name: str = MODEL_NAME):
        self.pipeline = None
        self.model_name = model_name
        logger.info(f"Initializing FrameCaptioner with Gemma 3n model: {model_name}")

    def load_model(self):
        logger.info("Loading Gemma 3n Vision model using pipeline...")
        try:
            # Setup pipeline
            self.pipeline = pipeline(
                task="image-text-to-text",
                model=self.model_name,
                device=0 if DEVICE == "cuda" else -1,
                torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
            )
            logger.success("Gemma 3n E2B pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise e

    def caption_frames(self, video_path: str) -> List[str]:
        if not self.pipeline:
            self.load_model()

        # Extract frames
        frame_paths = extract_frames_from_video(video_path)
        captions = []

        for i, frame_path in enumerate(frame_paths):
            try:
                image = Image.open(frame_path).convert("RGB")
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Describe what you see in this image. Focus on any vehicles, people, activities, or incidents that might be occurring."}
                        ]
                    }
                ]

                logger.info(f"Captioning frame {i+1}/{len(frame_paths)}: {os.path.basename(frame_path)}")
                
                # Run the pipeline
                output = self.pipeline(
                    text=messages,
                    max_new_tokens=100,
                    return_full_text=False
                )
                
                caption = output[0]["generated_text"].strip()

                captions.append(caption)
                
                # Save caption alongside the frame
                frame_name = os.path.basename(frame_path)
                self.save_caption(caption, frame_name)

                logger.debug(f"Caption for {frame_name}: {caption[:100]}...")
                
            except Exception as e:
                logger.error(f"Failed to caption frame {frame_path}: {e}")
                captions.append(f"Error: Could not process {os.path.basename(frame_path)}")

        return captions

    def save_caption(self, caption: str, frame_name: str):
        # Save caption to a text file with the same name as the frame
        caption_filename = os.path.join(CAPTION_OUTPUT_DIR, frame_name.replace('.jpg', '.txt'))
        with open(caption_filename, "w") as f:
            f.write(caption)
        logger.info(f"Saved caption to {caption_filename}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python caption_frames.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    captioner = FrameCaptioner()
    all_captions = captioner.caption_frames(video_path)

    print(f"Generated {len(all_captions)} captions")
