"""
Frame Extraction Module
Extracts frames from video files at 2 FPS for analysis
"""
import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple
from loguru import logger


def extract_frames_from_video(video_path: str, output_dir: str = None, fps: float = 2.0, max_frames: int = 60) -> List[str]:
    """
    Extract frames from video at specified FPS
    
    Args:
        video_path: Path putideo file
        output_dir: Directory to save extracted frames (optional)
        fps: Frames per second to extract (default: 2.0)
        max_frames: Maximum number of frames to extract (default: 60)
    
    Returns:
        List of paths to extracted frame images
    """
    logger.info(f"Extracting frames from: {video_path}")
    
    # Validate input
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join("data", "frames")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    logger.info(f"Video properties - FPS: {video_fps:.2f}, Duration: {duration:.2f}s, Total frames: {total_frames}")
    
    # Calculate frame interval
    frame_interval = int(video_fps / fps)
    frame_paths = []
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified interval
        if frame_count % frame_interval == 0:
            # Generate filename
            timestamp = frame_count / video_fps
            frame_filename = f"frame_{extracted_count:04d}_t{timestamp:.1f}s.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            
            # Save frame
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            extracted_count += 1
            
            logger.debug(f"Extracted frame {extracted_count} at {timestamp:.1f}s")
            
            # Check max frames limit
            if extracted_count >= max_frames:
                logger.warning(f"Reached maximum frame limit ({max_frames})")
                break
        
        frame_count += 1
    
    cap.release()
    
    logger.success(f"Extracted {len(frame_paths)} frames to {output_dir}")
    return frame_paths


def get_video_info(video_path: str) -> dict:
    """
    Get basic information about a video file
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps
    
    cap.release()
    
    return {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration,
        "resolution": f"{width}x{height}"
    }


def cleanup_frames(frames_dir: str):
    """Clean up extracted frame files"""
    if os.path.exists(frames_dir):
        for file in os.listdir(frames_dir):
            if file.endswith(('.jpg', '.png')):
                os.remove(os.path.join(frames_dir, file))
        logger.info(f"Cleaned up frames in {frames_dir}")


if __name__ == "__main__":
    # Test the frame extraction
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_frames.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    try:
        # Get video info
        info = get_video_info(video_path)
        print(f"Video Info: {info}")
        
        # Extract frames
        frames = extract_frames_from_video(video_path)
        print(f"Extracted {len(frames)} frames")
        
        # Print first few frame paths
        for i, frame_path in enumerate(frames[:5]):
            print(f"Frame {i+1}: {frame_path}")
            
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        sys.exit(1)
