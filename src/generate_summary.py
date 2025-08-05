"""
Generate Summary Module
Converts frame captions into a coherent incident summary using Gemma 3n Text
"""
import os
import torch
import json
from transformers import Gemma3nForCausalLM, AutoTokenizer
from typing import List
from loguru import logger

# Load configuration - Using official Gemma 3n text model
MODEL_NAME = "google/gemma-3n-e2b-it"  # Text generation model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUMMARY_OUTPUT_FILE = "outputs/summary.txt"

logger.info(f"Using device: {DEVICE}")

class SummaryGenerator:
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = None
        self.model = None
        self.model_name = model_name
        logger.info(f"Initializing SummaryGenerator with model: {model_name}")

    def load_model(self):
        logger.info("Loading Gemma 3n Text model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = Gemma3nForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                device_map={"":0}  # Force all to GPU 0
            ).eval()

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.success("Gemma 3n text model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load text model: {e}")
            logger.info("Trying fallback to CPU...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = Gemma3nForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            ).to(DEVICE)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.success("Text model loaded with fallback settings")

    def create_summary_prompt(self, captions: List[str]) -> str:
        """Create a prompt for summarizing video descriptions"""
        caption_text = ""
        for i, caption in enumerate(captions):
            timestamp = i  # 1 FPS assumption
            caption_text += f"Time {timestamp}s: {caption}\n"

        prompt = f"""
You are a professional incident analyst trained to summarize video sequences.
Below is a list of short image descriptions captured once per second over time.
Your task is to construct a smooth, time-aware narrative of the observed scene.

Visual Observations (Chronologically ordered):
{caption_text}

Instructions:
- Use the above time-stamped descriptions to build a **coherent paragraph** summarizing what happens in the video.
- Focus on **event progression**, identifying **any anomalies, movements, or incidents**.
- Maintain **chronological flow**, as if narrating the video.
- If nothing unusual happens, describe it as **routine/normal activity**.
- Be **factual and objective**, avoiding exaggeration or fictional assumptions.
- Output must be **50 to 150 words**, and read as a fluid summary.

Summary:
"""
        return prompt

    def generate_summary_from_captions(self, captions: List[str]) -> str:
        if not self.model or not self.tokenizer:
            self.load_model()

        try:
            logger.info(f"Generating summary from {len(captions)} captions")
            prompt = self.create_summary_prompt(captions)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(DEVICE)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = full_response.split("Summary:")[-1].strip()
            summary = self._clean_summary(summary)

            logger.success("Summary generated successfully")
            logger.debug(f"Generated summary: {summary}")

            return summary

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Unable to generate summary from the provided captions."

    def _clean_summary(self, summary: str) -> str:
        sentences = summary.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            summary = '.'.join(sentences[:-1]) + '.'

        summary = ' '.join(summary.split())
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'

        return summary

    def load_captions_from_files(self, captions_dir: str = "outputs/captions") -> List[str]:
        captions = []

        if not os.path.exists(captions_dir):
            logger.error(f"Captions directory not found: {captions_dir}")
            return captions

        caption_files = [f for f in os.listdir(captions_dir) if f.endswith('.txt')]
        caption_files.sort()

        logger.info(f"Loading {len(caption_files)} caption files")

        for caption_file in caption_files:
            filepath = os.path.join(captions_dir, caption_file)
            try:
                with open(filepath, 'r') as f:
                    caption = f.read().strip()
                    captions.append(caption)
            except Exception as e:
                logger.error(f"Error reading caption file {caption_file}: {e}")

        return captions

    def load_captions_from_cache(self, cache_file: str = "outputs/captions/captions_cache.json") -> List[str]:
        """Load captions from cache JSON file"""
        captions = []
        
        if not os.path.exists(cache_file):
            logger.error(f"Cache file not found: {cache_file}")
            return captions
            
        try:
            with open(cache_file, 'r') as f:
                captions_dict = json.load(f)
                
            # Sort by frame order (assumes frame_000.jpg, frame_001.jpg, etc.)
            sorted_frames = sorted(captions_dict.keys())
            captions = [captions_dict[frame] for frame in sorted_frames]
            logger.info(f"Loaded {len(captions)} captions from cache")
            
        except Exception as e:
            logger.error(f"Error reading cache file: {e}")
            
        return captions

    def save_summary(self, summary: str, output_file: str = SUMMARY_OUTPUT_FILE):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(summary)
        logger.success(f"Summary saved to: {output_file}")

    def generate_summary_from_video(self, video_path: str) -> str:
        from .caption_frames import FrameCaptioner
        captioner = FrameCaptioner()
        captions = captioner.caption_frames(video_path)
        summary = self.generate_summary_from_captions(captions)
        self.save_summary(summary)
        return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generate_summary.py <video_path_or_option>")
        print("Options:")
        print("  - Provide video path to process entire pipeline")
        print("  - 'from_captions' to use existing caption .txt files")
        print("  - 'from_cache' to use captions_cache.json")
        sys.exit(1)

    generator = SummaryGenerator()

    if sys.argv[1] == "from_captions":
        captions = generator.load_captions_from_files()
        if captions:
            summary = generator.generate_summary_from_captions(captions)
            generator.save_summary(summary)
            print(f"Generated summary: {summary}")
        else:
            print("No caption files found!")
    elif sys.argv[1] == "from_cache":
        captions = generator.load_captions_from_cache()
        if captions:
            summary = generator.generate_summary_from_captions(captions)
            generator.save_summary(summary)
            print(f"Generated summary: {summary}")
        else:
            print("No cache file found!") 
    else:
        video_path = sys.argv[1]
        summary = generator.generate_summary_from_video(video_path)
        print(f"Generated summary: {summary}")

