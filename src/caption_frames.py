"""
Caption Frames Module
Generates captions for each extracted frame using Gemma 3n Vision
"""
import os
import torch
from PIL import Image
from transformers import Gemma3nForConditionalGeneration, AutoProcessor
from typing import List
from loguru import logger

from .extract_frames import extract_frames_from_video

# Load configuration - Using official Gemma 3n multimodal model
# PRIMARY: Gemma 3n (requires HF access)
MODEL_NAME = "google/gemma-3n-e2b-it"  # Gemma 3n instruction-tuned model
# FALLBACK: Open alternative for testing
FALLBACK_MODEL = "Salesforce/blip-image-captioning-base"  # Open alternative
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTION_OUTPUT_DIR = "outputs/captions"

logger.info(f"Using device: {DEVICE}")
logger.info(f"Using Gemma 3n model: {MODEL_NAME}")

# Ensure output directory exists
os.makedirs(CAPTION_OUTPUT_DIR, exist_ok=True)

class FrameCaptioner:
    def __init__(self, model_name: str = MODEL_NAME):
        self.processor = None
        self.model = None
        self.model_name = model_name
        logger.info(f"Initializing FrameCaptioner with model: {model_name}")

    def load_model(self):
        logger.info("Loading Gemma 3n multimodal model...")
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map="auto"
            ).eval()
            logger.success("Gemma 3n model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Trying fallback to CPU...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = Gemma3nForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            ).to(DEVICE)
            logger.success("Model loaded with fallback settings")

    def caption_frames(self, video_path: str) -> List[str]:
        if not self.model or not self.processor:
            self.load_model()

        # Extract frames
        frame_paths = extract_frames_from_video(video_path)
        captions = []

        # Prompt for incident analysis
        prompt = "Describe what you see in this image. Focus on any vehicles, people, activities, or incidents that might be occurring."

        for i, frame_path in enumerate(frame_paths):
            try:
                image = Image.open(frame_path).convert("RGB")
                
                # Process inputs with both image and text prompt
                inputs = self.processor(
                    images=image, 
                    text=prompt, 
                    return_tensors="pt"
                )
                
                # Move inputs to device if not using device_map="auto"
                if DEVICE != "auto":
                    inputs = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in inputs.items()}

                logger.info(f"Captioning frame {i+1}/{len(frame_paths)}: {os.path.basename(frame_path)}")
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        do_sample=False,
                        temperature=0.1
                    )
                
                # Decode the response
                response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                
                # Clean up response (remove prompt if it's included)
                if prompt in response:
                    caption = response.replace(prompt, "").strip()
                else:
                    caption = response.strip()

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
