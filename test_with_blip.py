#!/usr/bin/env python3
"""
Test Detape with BLIP model (open alternative)
"""
import os
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path

def test_captioning_with_blip():
    """Test frame captioning with BLIP model (open alternative)"""
    
    # Setup
    model_name = "Salesforce/blip-image-captioning-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🧠 Testing image captioning with BLIP...")
    print(f"📱 Device: {device}")
    print(f"🔧 Model: {model_name}")
    
    try:
        # Load model
        print("⏳ Loading BLIP model...")
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        print("✅ Model loaded successfully!")
        
        # Test with first few frames
        frames_dir = "data/frames"
        if not os.path.exists(frames_dir):
            print("❌ No frames directory found!")
            return False
            
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])[:3]  # Test with first 3 frames
        
        print(f"\n🖼️ Testing with {len(frame_files)} frames...")
        
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_file)
            
            # Load and process image
            image = Image.open(frame_path).convert("RGB")
            
            # Generate caption
            inputs = processor(image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_length=50)
            
            caption = processor.decode(output_ids[0], skip_special_tokens=True)
            
            print(f"Frame {i+1}: {frame_file}")
            print(f"Caption: {caption}")
            print("-" * 50)
            
        print("🎉 BLIP captioning test successful!")
        print("\n💡 This shows the pipeline works!")
        print("🔑 Now you just need Gemma 3n access for the full experience.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during BLIP test: {e}")
        return False

if __name__ == "__main__":
    test_captioning_with_blip()
