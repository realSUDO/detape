#!/usr/bin/env python3
"""
Basic test script to verify Detape functionality
"""
import sys
import os
from pathlib import Path

def test_video_file():
    """Test if video file exists"""
    video_path = "data/test_2.mp4"
    if os.path.exists(video_path):
        print(f"✅ Video file found: {video_path}")
        return True
    else:
        print(f"❌ Video file not found: {video_path}")
        return False

def test_frames():
    """Test if frames exist"""
    frames_dir = "data/frames"
    if os.path.exists(frames_dir):
        frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
        print(f"✅ Found {len(frame_files)} frame files in {frames_dir}")
        return True
    else:
        print(f"❌ Frames directory not found: {frames_dir}")
        return False

def test_dependencies():
    """Test if key dependencies are available"""
    deps = ['cv2', 'PIL', 'torch']
    available = []
    missing = []
    
    for dep in deps:
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'PIL':
                from PIL import Image
            elif dep == 'torch':
                import torch
            available.append(dep)
            print(f"✅ {dep} available")
        except ImportError:
            missing.append(dep)
            print(f"❌ {dep} missing")
    
    return len(missing) == 0

def main():
    print("🔍 DETAPE BASIC TEST")
    print("=" * 30)
    
    # Test basic requirements
    video_ok = test_video_file()
    frames_ok = test_frames()
    deps_ok = test_dependencies()
    
    print("\n" + "=" * 30)
    if video_ok and frames_ok and deps_ok:
        print("🎉 All basic tests passed!")
        print("🚀 Ready to run: python main.py data/test_2.mp4")
    else:
        print("⚠️  Some issues found. Address them before running main.py")
        
        if not deps_ok:
            print("\n💡 To fix dependencies:")
            print("pip install opencv-python pillow torch loguru transformers")

if __name__ == "__main__":
    main()
