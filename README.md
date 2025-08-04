# Detape v0.1 Alpha 🎬🤖

**AI Video Incident Report Generator**

Detape transforms short real-world videos (10-60 seconds) into structured incident reports using AI. Perfect for analyzing dashcam footage, CCTV clips, or phone recordings of accidents, incidents, and unusual activities.

## 🎯 Current Status: v0.1 Alpha

**✅ IMPLEMENTED:**
- 🎞️ Frame extraction from videos at 1 FPS using OpenCV
- 🧠 Gemma 3n Vision model for frame captioning
- 📄 Raw caption saving and processing
- 🏗️ Basic pipeline architecture

**🔄 COMING IN v0.5 Beta:**
- 📝 Text summarization using Gemma 3n Text
- 🏷️ Event classification and tagging
- 💻 Command-line interface

**🚀 PLANNED FOR v1.0:**
- 🎛️ Streamlit web interface
- 🔊 Audio analysis with librosa
- 🗣️ Text-to-speech narration
- 📊 Enhanced reporting

## 🏗️ Project Structure

```
detape/
├── README.md
├── requirements.txt
├── main.py                 # Master runner script
├── data/
│   ├── test_1.mp4         # Sample video files
│   ├── test_2.mp4
│   └── frames/            # Extracted frames
├── models/                # Gemma 3n models cache
├── src/
│   ├── extract_frames.py  # Frame extraction with OpenCV
│   ├── caption_frames.py  # Gemma 3n Vision captioning
│   └── generate_summary.py # Gemma 3n Text summarization
└── outputs/
    ├── captions/          # Individual frame captions
    ├── summary.txt        # Generated summary
    └── processing_summary.txt
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone/setup the project (if not already done)
cd detape

# Install dependencies
pip install -r requirements.txt
```

### 2. Run v0.1 Alpha

Process a video to extract frames and generate captions:

```bash
python main.py data/test_1.mp4
```

### 3. Check Results

After processing, check the outputs:

```bash
# View processing summary
cat outputs/processing_summary.txt

# Check individual captions
ls outputs/captions/

# View extracted frames
ls data/frames/
```

## 🧠 AI Agents Architecture

| Agent | Model | Purpose | Status |
|-------|-------|---------|--------|
| **Vision Scout** | Gemma 3n Vision | Frame captioning | ✅ Active |
| **Text Crafter** | Gemma 3n Text | Summary generation | 🔄 v0.5 |
| **Tag Oracle** | Rule-based + AI | Event classification | 🔄 v0.5 |
| **Sound Sentry** | librosa + ffmpeg | Audio analysis | 🚀 v1.0 |
| **Voice Relay** | pyttsx3 | Text-to-speech | 🚀 v1.0 |
| **Frontline UI** | Streamlit | Web interface | 🚀 v1.0 |

## 💻 Usage Examples

### Basic Processing
```bash
# Process a dashcam video
python main.py data/dashcam_incident.mp4

# Process a security camera clip
python main.py data/security_footage.mp4
```

### Individual Components (Advanced)
```bash
# Extract frames only
cd src
python extract_frames.py ../data/test_1.mp4

# Generate captions only (requires frames)
python caption_frames.py ../data/test_1.mp4
```

## 🔧 Configuration

Key settings can be found in the source files:

- **Frame extraction**: 1 FPS, max 60 frames
- **Models**: Gemma 3n Vision and Text
- **Supported formats**: .mp4, .avi, .mov, .mkv
- **Device**: Auto-detects CUDA/CPU

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- Transformers (Hugging Face)
- CUDA-capable GPU (recommended)

## 🐛 Troubleshooting

### Common Issues

**Model Loading Errors**:
```bash
# Check if models are accessible
python -c "from transformers import AutoProcessor; print('OK')"
```

**CUDA Issues**:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Video Format Issues**:
- Ensure video file is in supported format (.mp4, .avi, .mov, .mkv)
- Check video file isn't corrupted

## 🗺️ Roadmap

### v0.5 Beta (Next)
- [ ] Complete text summarization pipeline
- [ ] Add event classification (Accident, Normal, Fire, etc.)
- [ ] Implement full CLI with options
- [ ] Add batch processing capability

### v1.0 Final
- [ ] Streamlit web interface
- [ ] Audio analysis integration
- [ ] TTS summary narration
- [ ] Enhanced report formatting
- [ ] Demo video creation

## 🤝 Contributing

This is a hackathon project! Current focus:
1. Getting v0.1 Alpha stable
2. Implementing v0.5 Beta features
3. Preparing for demo

## 📄 License

Hackathon project - MIT License

---

**Detape** - Making sense of video incidents with AI 🎬🤖
