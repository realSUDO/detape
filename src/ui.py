"""
Detape v1.0 - Streamlit UI
Interactive web interface for video-to-report pipeline
"""
import streamlit as st
import os
import tempfile
import time
from pathlib import Path
import json

# Import our modules
import sys
sys.path.append('.')
sys.path.append('..')

try:
    from extract_frames import extract_frames_from_video, get_video_info
    from caption_frames import FrameCaptioner
    from generate_summary import SummaryGenerator
except ImportError:
    try:
        from src.extract_frames import extract_frames_from_video, get_video_info
        from src.caption_frames import FrameCaptioner
        from src.generate_summary import SummaryGenerator
    except ImportError:
        st.error("❌ Cannot import modules. Make sure you're running from the correct directory.")
        st.stop()

# Page configuration
st.set_page_config(
    page_title="Detape - AI Video Incident Analyzer",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
    color: white;
    margin-bottom: 2rem;
}
.status-box {
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.success-box {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
}
.warning-box {
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
    color: #856404;
}
.error-box {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎞️ DETAPE v1.0</h1>
        <h3>AI-Powered Video Incident Analysis</h3>
        <p>Transform real-world videos into structured incident reports using offline AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("📋 Process Steps")
        st.write("1. **Upload Video** - .mp4, .avi, .mov")
        st.write("2. **Extract Frames** - 1 FPS sampling")
        st.write("3. **Generate Captions** - Gemma 3n Vision")
        st.write("4. **Create Summary** - Gemma 3n Text")
        st.write("5. **View Results** - Structured report")
        
        st.header("⚙️ Settings")
        fps_setting = st.slider("Frames per second", 0.5, 2.0, 1.0, 0.5)
        max_duration = st.slider("Max video duration (seconds)", 10, 120, 60, 10)
        
        # Processing time estimate
        estimated_frames = int(max_duration * fps_setting)
        est_time = estimated_frames * 0.5 + 120  # ~0.5s per frame + 2min model loading
        
        st.warning(f"⏱️ **Processing Time Estimate**: ~{est_time/60:.1f} minutes for {estimated_frames} frames")
        st.info("🛋️ **Please be patient!** AI models take time to load. Don't refresh the page.")
        
        st.header("📊 Model Info")
        st.write("**Vision Model**: Gemma 3n E2B")
        st.write("**Text Model**: Gemma 3n E2B")
        st.write("**Processing**: Offline")

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file (max 60 seconds recommended)"
        )
        
        if uploaded_file is not None:
            # Display video info
            st.video(uploaded_file)
            
            # Process button
            if st.button("🚀 Process Video", type="primary"):
                process_video(uploaded_file, fps_setting, max_duration)
    
    with col2:
        st.header("📋 Processing Status")
        
        # Status container
        status_container = st.empty()
        
        # Results container
        results_container = st.empty()
        
        # Load existing results if available
        load_existing_results(results_container)

def process_video(uploaded_file, fps_setting, max_duration):
    """Process the uploaded video through the complete pipeline"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Step 1: Video Analysis
        status_text.text("📹 Analyzing video... (This will take a moment)")
        progress_bar.progress(10)
        
        video_info = get_video_info(video_path)
        
        if video_info['duration'] > max_duration:
            st.error(f"Video too long ({video_info['duration']:.1f}s). Maximum allowed: {max_duration}s")
            return
            
        st.success(f"✅ Video: {video_info['resolution']}, {video_info['duration']:.1f}s, {video_info['fps']:.1f} FPS")
        
        # Step 2: Extract Frames
        status_text.text("🎞️ Extracting frames... (Quick step)")
        progress_bar.progress(25)
        
        frame_paths = extract_frames_from_video(video_path, fps=fps_setting)
        st.success(f"✅ Extracted {len(frame_paths)} frames")
        
        # Step 3: Generate Captions
        status_text.text("🧠 Loading Gemma 3n Vision model... (This might take 1-2 minutes, please be patient)")
        progress_bar.progress(35)
        
        captioner = FrameCaptioner()
        
        status_text.text(f"🧠 Generating captions for {len(frame_paths)} frames... (Processing each frame, hold tight!)")
        progress_bar.progress(40)
        
        captions = captioner.caption_frames(video_path)
        st.success(f"✅ Generated {len(captions)} captions")
        
        # Memory cleanup - Free vision model before loading text model
        status_text.text("🧹 Cleaning up GPU memory... (Preparing for text model)")
        if hasattr(captioner, 'model') and captioner.model is not None:
            del captioner.model
            del captioner.processor
            import torch
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        progress_bar.progress(70)
        
        # Step 4: Generate Summary
        status_text.text("📝 Loading text model for summary generation... (Almost there, hang in there!)")
        
        generator = SummaryGenerator() 
        summary = generator.generate_summary_from_captions(captions)
        generator.save_summary(summary)
        
        progress_bar.progress(90)
        st.success("✅ Summary generated successfully")
        
        # Step 5: Display Results
        status_text.text("📊 Displaying results...")
        progress_bar.progress(100)
        
        display_results(captions, summary, video_info)
        
        # Cleanup
        os.unlink(video_path)
        
        status_text.text("🎉 Processing complete!")
        
    except Exception as e:
        st.error(f"❌ Processing failed: {str(e)}")
        progress_bar.progress(0)

def display_results(captions, summary, video_info):
    """Display the processing results"""
    
    st.header("📊 Analysis Results")
    
    # Summary section
    st.subheader("📝 Incident Summary")
    st.markdown(f"""
    <div class="status-box success-box">
        <h4>Generated Report</h4>
        <p>{summary}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Video info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Duration", f"{video_info['duration']:.1f}s")
    with col2:
        st.metric("Resolution", video_info['resolution'])
    with col3:
        st.metric("Frames Analyzed", len(captions))
    
    # Caption details (expandable)
    with st.expander("🔍 View Frame-by-Frame Analysis"):
        for i, caption in enumerate(captions):
            st.write(f"**Frame {i:02d}** ({i}s): {caption}")
    
    # Download section
    st.subheader("📥 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download summary
        st.download_button(
            label="📄 Download Summary",
            data=summary,
            file_name="incident_summary.txt",
            mime="text/plain"
        )
    
    with col2:
        # Download full report
        full_report = create_full_report(captions, summary, video_info)
        st.download_button(
            label="📋 Download Full Report",
            data=full_report,
            file_name="full_incident_report.txt",
            mime="text/plain"
        )

def load_existing_results(container):
    """Load and display existing results if available"""
    
    summary_file = "outputs/summary.txt"
    cache_file = "outputs/captions/captions_cache.json"
    
    if os.path.exists(summary_file) and os.path.exists(cache_file):
        with container.container():
            st.info("📁 Found existing analysis results")
            
            if st.button("📊 Load Previous Results"):
                try:
                    # Load summary
                    with open(summary_file, 'r') as f:
                        summary = f.read()
                    
                    # Load captions
                    with open(cache_file, 'r') as f:
                        captions_dict = json.load(f)
                    
                    sorted_frames = sorted(captions_dict.keys()) 
                    captions = [captions_dict[frame] for frame in sorted_frames]
                    
                    # Mock video info for display
                    video_info = {
                        'duration': len(captions),
                        'resolution': 'Unknown',
                        'fps': 1.0
                    }
                    
                    display_results(captions, summary, video_info)
                    
                except Exception as e:
                    st.error(f"Failed to load results: {e}")

def create_full_report(captions, summary, video_info):
    """Create a comprehensive report"""
    
    report = f"""
DETAPE v1.0 - INCIDENT ANALYSIS REPORT
{'='*50}

SUMMARY
{'-'*20}
{summary}

VIDEO INFORMATION
{'-'*20}
Duration: {video_info['duration']:.1f} seconds
Resolution: {video_info['resolution']}
Frames Analyzed: {len(captions)}
Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

FRAME-BY-FRAME ANALYSIS
{'-'*30}
"""
    
    for i, caption in enumerate(captions):
        report += f"Time {i:02d}s: {caption}\n"
    
    report += f"""

TECHNICAL DETAILS
{'-'*20}
AI Models Used:
- Vision: Gemma 3n E2B (Image Captioning)
- Text: Gemma 3n E2B (Summary Generation)
- Processing: Offline/Local

Generated by Detape v1.0
AI Video Incident Analysis Tool
"""
    
    return report

if __name__ == "__main__":
    main()
