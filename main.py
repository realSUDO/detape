"""
Detape v1.0 - Main Runner
Master script for the video-to-report pipeline with enhanced Gemma 3n support

Usage:
    python main.py <video_path>
    
Features:
- Enhanced frame captioning with community workarounds for Gemma 3n audio issues
- Bulletproof error handling and validation
- Comprehensive logging and debugging
- Memory optimization for GPU processing
"""
import sys
import os
import time
from pathlib import Path
from loguru import logger
import torch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.extract_frames import extract_frames_from_video, get_video_info
from src.caption_frames import FrameCaptioner
try:
    from src.generate_summary import SummaryGenerator
except ImportError:
    logger.warning("SummaryGenerator not available - summary generation will be skipped")
    SummaryGenerator = None

def setup_logging():
    """Setup logging configuration"""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    logger.add(
        "detape.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}",
        level="DEBUG"
    )

def print_banner():
    """Print Detape banner with system information"""
    banner = """
    ╔═══════════════════════════════════════╗
    ║           DETAPE v1.0 Beta            ║
    ║     AI Video Incident Report Tool     ║
    ║        Enhanced Gemma 3n Support      ║
    ╚═══════════════════════════════════════╝
    """
    print(banner)
    
    # System information
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"    🖥️  Processing Device: {device} ({gpu_name})")
        print(f"    💾 GPU Memory: {gpu_memory:.1f} GB")
    else:
        print(f"    🖥️  Processing Device: {device}")
    
    print(f"    🧠 AI Models: Gemma 3n E2B (Vision + Text)")
    print(f"    🔧 Community Workarounds: Enabled")
    print()
    
    logger.info(f"Starting Detape v1.0 Beta on {device}")

def validate_video_file(video_path: str) -> bool:
    """Validate input video file"""
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return False
    
    # Check file extension
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    file_ext = Path(video_path).suffix.lower()
    
    if file_ext not in valid_extensions:
        logger.error(f"Unsupported video format: {file_ext}")
        logger.info(f"Supported formats: {', '.join(valid_extensions)}")
        return False
    
    return True

def process_video(video_path: str) -> dict:
    """
    Complete video processing pipeline for v0.1 Alpha
    
    Steps:
    1. Extract frames at 1 FPS
    2. Generate captions using Gemma 3n E2B Vision (lightweight)
    3. Save raw captions
    
    Returns:
        Dictionary with processing results
    """
    results = {
        "video_path": video_path,
        "frames_extracted": 0,
        "captions_generated": 0,
        "processing_time": 0,
        "success": False
    }
    
    start_time = time.time()
    
    try:
        # Step 1: Get video information
        logger.info("📹 Analyzing video file...")
        video_info = get_video_info(video_path)
        logger.info(f"Video: {video_info['resolution']}, {video_info['duration']:.1f}s, {video_info['fps']:.1f} FPS")
        
        # Step 2: Extract frames
        logger.info("🎞️ Extracting frames at 1 FPS...")
        frame_paths = extract_frames_from_video(video_path, fps=1.0)
        results["frames_extracted"] = len(frame_paths)
        logger.success(f"Extracted {len(frame_paths)} frames")
        
        # Step 3: Generate captions
        logger.info("🧠 Generating captions with Gemma 3n Vision...")
        captioner = FrameCaptioner()
        captions = captioner.caption_frames(video_path)
        results["captions_generated"] = len(captions)
        logger.success(f"Generated {len(captions)} captions")
        
        # Step 4: Save processing summary
        processing_time = time.time() - start_time
        results["processing_time"] = processing_time
        results["success"] = True
        
        # Save results summary
        save_processing_summary(results, captions)
        
        logger.success(f"✅ Processing complete in {processing_time:.1f}s")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        results["error"] = str(e)
    
    return results

def save_processing_summary(results: dict, captions: list):
    """Save processing summary to outputs directory"""
    summary_file = "outputs/processing_summary.txt"
    os.makedirs("outputs", exist_ok=True)
    
    with open(summary_file, 'w') as f:
        f.write("DETAPE v0.1 Alpha - Processing Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Video: {results['video_path']}\n")
        f.write(f"Frames extracted: {results['frames_extracted']}\n")
        f.write(f"Captions generated: {results['captions_generated']}\n")
        f.write(f"Processing time: {results['processing_time']:.1f}s\n")
        f.write(f"Status: {'SUCCESS' if results['success'] else 'FAILED'}\n\n")
        
        if captions:
            f.write("Generated Captions:\n")
            f.write("-" * 20 + "\n")
            for i, caption in enumerate(captions):
                f.write(f"Frame {i:02d}: {caption}\n")
    
    logger.info(f"Processing summary saved to: {summary_file}")

def main():
    """Main entry point"""
    setup_logging()
    print_banner()
    
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python main.py <video_path>")
        print("\nExample:")
        print("  python main.py data/test_1.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Validate input
    if not validate_video_file(video_path):
        sys.exit(1)
    
    # Process video
    logger.info(f"Processing video: {video_path}")
    results = process_video(video_path)
    
    # Print results
    print("\n" + "="*50)
    if results["success"]:
        print("🎉 SUCCESS - Video processing completed!")
        print(f"📊 Results:")
        print(f"   • Frames extracted: {results['frames_extracted']}")
        print(f"   • Captions generated: {results['captions_generated']}")
        print(f"   • Processing time: {results['processing_time']:.1f}s")
        print(f"📁 Outputs saved to: outputs/")
        print(f"   • Raw captions: outputs/captions/")
        print(f"   • Summary: outputs/processing_summary.txt")
        
        print(f"\n🔍 Next Steps for v0.5 Beta:")
        print(f"   • Run summary generation")
        print(f"   • Add event classification")
        print(f"   • Implement CLI interface")
        
    else:
        print("❌ FAILED - Video processing encountered errors")
        if "error" in results:
            print(f"Error: {results['error']}")
    
    print("="*50)

if __name__ == "__main__":
    main()
