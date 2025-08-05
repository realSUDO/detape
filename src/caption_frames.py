"""
Caption Frames Module
Generates captions for each extracted frame using Gemma 3n Vision
"""
import os
import torch
from PIL import Image
from transformers import pipeline
from typing import List, Dict
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

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

    def caption_single_frame(self, image_path: str) -> str:
        """Caption a single frame with optimizations"""
        try:
            # Optimize: Resize image to reduce processing time
            image = Image.open(image_path).convert("RGB")
            # Resize to 640x360 for faster processing while maintaining quality
            image = image.resize((640, 360), Image.Resampling.LANCZOS)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": "Describe what you see in this image. Focus on any vehicles, people, activities, or incidents that might be occurring."}
                    ]
                }
            ]
            
            # Run the pipeline
            output = self.pipeline(
                text=messages,
                max_new_tokens=100,
                return_full_text=False
            )
            
            return output[0]["generated_text"].strip()
            
        except Exception as e:
            logger.error(f"Failed to caption frame {image_path}: {e}")
            return f"Error: Could not process {os.path.basename(image_path)}"

    def caption_frame_safe(self, image_path: str) -> tuple:
        """Thread-safe wrapper for frame captioning"""
        frame_name = os.path.basename(image_path)
        caption = self.caption_single_frame(image_path)
        return frame_name, caption

    def load_existing_captions(self) -> Dict[str, str]:
        """Load existing captions from cache file"""
        cache_file = os.path.join(CAPTION_OUTPUT_DIR, "captions_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_captions = json.load(f)
                logger.info(f"Loaded {len(cached_captions)} cached captions")
                return cached_captions
            except Exception as e:
                logger.warning(f"Failed to load caption cache: {e}")
        return {}

    def save_captions_cache(self, captions: Dict[str, str]):
        """Save captions to cache file"""
        cache_file = os.path.join(CAPTION_OUTPUT_DIR, "captions_cache.json")
        try:
            with open(cache_file, 'w') as f:
                json.dump(captions, f, indent=2)
            logger.info(f"Saved {len(captions)} captions to cache")
        except Exception as e:
            logger.error(f"Failed to save caption cache: {e}")

    def caption_frames(self, video_path: str) -> List[str]:
        if not self.pipeline:
            self.load_model()

        # Extract frames
        frame_paths = extract_frames_from_video(video_path)
        
        # Load existing captions (resume functionality)
        cached_captions = self.load_existing_captions()
        
        # Filter out already processed frames
        frames_to_process = []
        captions_dict = {}
        
        for frame_path in frame_paths:
            frame_name = os.path.basename(frame_path)
            if frame_name in cached_captions:
                captions_dict[frame_name] = cached_captions[frame_name]
                logger.info(f"📋 Skipping cached frame: {frame_name}")
            else:
                frames_to_process.append(frame_path)
        
        if not frames_to_process:
            logger.info("✅ All frames already processed! Using cached captions.")
        else:
            logger.info(f"🚀 Processing {len(frames_to_process)} new frames with {min(4, len(frames_to_process))} threads...")
            
            # Parallel processing with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(4, len(frames_to_process))) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(self.caption_frame_safe, frame_path): frame_path
                    for frame_path in frames_to_process
                }
                
                # Process completed tasks
                for i, future in enumerate(as_completed(future_to_path), 1):
                    frame_path = future_to_path[future]
                    frame_name, caption = future.result()
                    
                    captions_dict[frame_name] = caption
                    logger.info(f"✅ [{i}/{len(frames_to_process)}] {frame_name}: {caption[:60]}...")
                    
                    # Save individual caption file
                    self.save_caption(caption, frame_name)
        
        # Save updated cache
        self.save_captions_cache(captions_dict)
        
        # Return captions in original frame order
        captions = []
        for frame_path in frame_paths:
            frame_name = os.path.basename(frame_path)
            captions.append(captions_dict.get(frame_name, "Error: Caption not found"))
        
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
