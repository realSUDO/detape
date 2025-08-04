"""
Generate Summary Module
Converts frame captions into a coherent incident summary using Gemma 3n Text
"""
import os
import torch
from transformers import Gemma3nForCausalLM, AutoTokenizer
from typing import List
from loguru import logger

# Load configuration - Using official Gemma 3n text model
MODEL_NAME = "google/gemma-3n-e2b-it"  # Same model, used for text generation
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
                device_map="auto"
            ).eval()
            
            # Set pad token if not present
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
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.success("Text model loaded with fallback settings")

    def create_summary_prompt(self, captions: List[str]) -> str:
        """Create a prompt for summarizing frame captions"""
        
        # Format captions into a structured input
        caption_text = ""
        for i, caption in enumerate(captions):
            timestamp = i  # Since we extract at 1 FPS, frame number ≈ seconds
            caption_text += f"Time {timestamp}s: {caption}\n"
        
        prompt = f"""You are an expert incident report analyst. Based on the following chronological frame descriptions from a video, create a clear and concise incident summary.

Video Frame Descriptions:
{caption_text}

Instructions:
- Write a coherent paragraph summarizing what happened in the video
- Focus on key events, actions, and any incidents
- Maintain chronological order when describing events
- Be objective and factual
- If no significant incident occurred, describe it as routine/normal activity
- Keep the summary between 50-150 words

Incident Summary:"""

        return prompt

    def generate_summary_from_captions(self, captions: List[str]) -> str:
        """Generate a coherent summary from frame captions"""
        if not self.model or not self.tokenizer:
            self.load_model()
        
        try:
            logger.info(f"Generating summary from {len(captions)} captions")
            
            # Create the prompt
            prompt = self.create_summary_prompt(captions)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(DEVICE)
            
            # Generate summary
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the summary part (after the prompt)
            summary = full_response.split("Incident Summary:")[-1].strip()
            
            # Clean up the summary
            summary = self._clean_summary(summary)
            
            logger.success("Summary generated successfully")
            logger.debug(f"Generated summary: {summary}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Unable to generate summary from the provided captions."

    def _clean_summary(self, summary: str) -> str:
        """Clean and format the generated summary"""
        # Remove any trailing incomplete sentences
        sentences = summary.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            summary = '.'.join(sentences[:-1]) + '.'
        
        # Remove extra whitespace
        summary = ' '.join(summary.split())
        
        # Ensure it ends with proper punctuation
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary

    def load_captions_from_files(self, captions_dir: str = "outputs/captions") -> List[str]:
        """Load captions from saved text files"""
        captions = []
        
        if not os.path.exists(captions_dir):
            logger.error(f"Captions directory not found: {captions_dir}")
            return captions
        
        # Get all caption files and sort them
        caption_files = [f for f in os.listdir(captions_dir) if f.endswith('.txt')]
        caption_files.sort()  # Ensure chronological order
        
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

    def save_summary(self, summary: str, output_file: str = SUMMARY_OUTPUT_FILE):
        """Save the generated summary to a file"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(summary)
        
        logger.success(f"Summary saved to: {output_file}")

    def generate_summary_from_video(self, video_path: str) -> str:
        """Complete pipeline: extract frames, caption them, and generate summary"""
        from .caption_frames import FrameCaptioner
        
        # Caption the frames
        captioner = FrameCaptioner()
        captions = captioner.caption_frames(video_path)
        
        # Generate summary
        summary = self.generate_summary_from_captions(captions)
        
        # Save summary
        self.save_summary(summary)
        
        return summary


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_summary.py <video_path_or_captions_dir>")
        print("  - Provide video path to process entire pipeline")
        print("  - Provide 'from_captions' to use existing caption files")
        sys.exit(1)
    
    generator = SummaryGenerator()
    
    if sys.argv[1] == "from_captions":
        # Generate summary from existing caption files
        captions = generator.load_captions_from_files()
        if captions:
            summary = generator.generate_summary_from_captions(captions)
            generator.save_summary(summary)
            print(f"Generated summary: {summary}")
        else:
            print("No captions found!")
    else:
        # Process entire video pipeline
        video_path = sys.argv[1]
        summary = generator.generate_summary_from_video(video_path)
        print(f"Generated summary: {summary}")

