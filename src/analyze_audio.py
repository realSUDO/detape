"""
Audio Analysis Module
Analyzes audio for crash-like sounds, volume spikes, and other audio indicators
"""
import os
import numpy as np
import librosa
import librosa.display
from scipy import signal
from typing import Dict, List, Tuple
from loguru import logger
import json

# Configuration
SAMPLE_RATE = 22050  # Standard sample rate for analysis
AUDIO_OUTPUT_DIR = "outputs/audio"

class AudioAnalyzer:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
        logger.info("AudioAnalyzer initialized")

    def extract_audio_from_video(self, video_path: str) -> Tuple[np.ndarray, int]:
        """Extract audio from video file"""
        try:
            logger.info(f"Extracting audio from: {video_path}")
            
            # Load audio using librosa
            audio, sr = librosa.load(video_path, sr=self.sample_rate)
            
            logger.success(f"Extracted audio: {len(audio)/sr:.2f}s at {sr}Hz")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            return np.array([]), 0

    def analyze_volume_profile(self, audio: np.ndarray, sr: int) -> Dict:
        """Analyze volume changes and detect spikes"""
        if len(audio) == 0:
            return {"error": "No audio data"}
        
        # Calculate RMS energy in windows
        hop_length = sr // 4  # 0.25 second windows
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Convert to dB
        rms_db = librosa.amplitude_to_db(rms)
        
        # Time stamps for each window
        times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
        
        # Detect volume spikes
        threshold = np.mean(rms_db) + 2 * np.std(rms_db)
        spikes = []
        
        for i, (time, db) in enumerate(zip(times, rms_db)):
            if db > threshold:
                spikes.append({
                    "time": float(time),
                    "volume_db": float(db),
                    "intensity": "high" if db > threshold + 5 else "medium"
                })
        
        return {
            "duration": float(len(audio) / sr),
            "max_volume_db": float(np.max(rms_db)),
            "mean_volume_db": float(np.mean(rms_db)),
            "volume_spikes": spikes,
            "spike_count": len(spikes)
        }

    def detect_crash_indicators(self, audio: np.ndarray, sr: int) -> Dict:
        """Detect audio patterns that might indicate crashes or impacts"""
        if len(audio) == 0:
            return {"error": "No audio data"}
        
        # Calculate spectral features
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Detect sudden amplitude changes (potential impacts)
        onset_frames = librosa.onset.onset_detect(
            y=audio, sr=sr, units='time', hop_length=512
        )
        
        # Analyze frequency content for crash-like sounds
        # High frequency content often indicates crashes, breaking glass, etc.
        high_freq_energy = np.mean(magnitude[magnitude.shape[0]//2:, :], axis=0)
        high_freq_threshold = np.mean(high_freq_energy) + np.std(high_freq_energy)
        
        # Detect zero crossing rate changes (indicates texture changes in audio)
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        zcr_mean = np.mean(zcr)
        zcr_spikes = np.where(zcr > zcr_mean + 2 * np.std(zcr))[0]
        
        # Identify potential crash events
        crash_indicators = []
        for onset_time in onset_frames:
            # Check if onset coincides with high frequency activity
            frame_idx = librosa.time_to_frames(onset_time, sr=sr, hop_length=512)
            if frame_idx < len(high_freq_energy) and high_freq_energy[frame_idx] > high_freq_threshold:
                crash_indicators.append({
                    "time": float(onset_time),
                    "type": "impact",
                    "confidence": "high" if high_freq_energy[frame_idx] > high_freq_threshold * 1.5 else "medium",
                    "frequency_content": "high"
                })
        
        return {
            "total_onsets": len(onset_frames),
            "crash_indicators": crash_indicators,
            "high_frequency_events": int(np.sum(high_freq_energy > high_freq_threshold)),
            "texture_changes": len(zcr_spikes),
            "audio_classification": self._classify_audio_content(crash_indicators, len(onset_frames))
        }

    def _classify_audio_content(self, crash_indicators: List[Dict], total_onsets: int) -> str:
        """Classify the overall audio content"""
        if len(crash_indicators) > 2:
            return "high_activity"
        elif len(crash_indicators) > 0:
            return "moderate_activity"
        elif total_onsets > 10:
            return "busy_environment"
        else:
            return "quiet_environment"

    def analyze_audio_timeline(self, audio: np.ndarray, sr: int, frame_count: int) -> List[Dict]:
        """Create per-second audio analysis to match video frames"""
        if len(audio) == 0:
            return []
        
        duration = len(audio) / sr
        seconds = min(int(duration), frame_count)  # Match frame count
        
        timeline = []
        
        for i in range(seconds):
            start_sample = int(i * sr)
            end_sample = int((i + 1) * sr)
            
            if end_sample > len(audio):
                segment = audio[start_sample:]
            else:
                segment = audio[start_sample:end_sample]
            
            if len(segment) > 0:
                # Calculate features for this second
                rms = librosa.feature.rms(y=segment)[0]
                rms_db = float(librosa.amplitude_to_db(rms).mean())
                
                zcr = float(librosa.feature.zero_crossing_rate(segment)[0].mean())
                
                # Detect onsets in this segment
                onsets = librosa.onset.onset_detect(y=segment, sr=sr)
                
                timeline.append({
                    "time": i,
                    "volume_db": rms_db,
                    "texture": "rough" if zcr > 0.1 else "smooth",
                    "activity": "high" if len(onsets) > 2 else "low",
                    "onset_count": len(onsets)
                })
            else:
                timeline.append({
                    "time": i,
                    "volume_db": -60.0,
                    "texture": "silence",
                    "activity": "none",
                    "onset_count": 0
                })
        
        return timeline

    def analyze_video_audio(self, video_path: str, frame_count: int = None) -> Dict:
        """Complete audio analysis for a video file"""
        logger.info(f"Starting audio analysis for: {video_path}")
        
        # Extract audio
        audio, sr = self.extract_audio_from_video(video_path)
        
        if len(audio) == 0:
            return {"error": "Could not extract audio from video"}
        
        # Perform all analyses
        volume_analysis = self.analyze_volume_profile(audio, sr)
        crash_analysis = self.detect_crash_indicators(audio, sr)
        
        if frame_count:
            timeline = self.analyze_audio_timeline(audio, sr, frame_count)
        else:
            # Estimate frame count from duration (assuming 1 FPS)
            estimated_frames = int(volume_analysis.get("duration", 0))
            timeline = self.analyze_audio_timeline(audio, sr, estimated_frames)
        
        # Combine results
        results = {
            "video_path": video_path,
            "audio_duration": volume_analysis.get("duration", 0),
            "volume_analysis": volume_analysis,
            "crash_analysis": crash_analysis,
            "timeline": timeline,
            "summary": self._create_audio_summary(volume_analysis, crash_analysis)
        }
        
        # Save results
        self.save_audio_analysis(results)
        
        logger.success("Audio analysis completed")
        return results

    def _create_audio_summary(self, volume_analysis: Dict, crash_analysis: Dict) -> str:
        """Create a text summary of audio analysis"""
        summary_parts = []
        
        # Volume analysis
        max_vol = volume_analysis.get("max_volume_db", -60)
        spike_count = volume_analysis.get("spike_count", 0)
        
        if max_vol > -10:
            summary_parts.append("Loud audio detected")
        elif max_vol > -30:
            summary_parts.append("Moderate audio levels")
        else:
            summary_parts.append("Quiet audio environment")
        
        if spike_count > 0:
            summary_parts.append(f"{spike_count} volume spike(s) detected")
        
        # Crash analysis
        crash_indicators = crash_analysis.get("crash_indicators", [])
        classification = crash_analysis.get("audio_classification", "unknown")
        
        if len(crash_indicators) > 0:
            summary_parts.append(f"{len(crash_indicators)} potential impact sound(s)")
        
        summary_parts.append(f"Audio environment: {classification}")
        
        return ". ".join(summary_parts) + "."

    def save_audio_analysis(self, results: Dict, filename: str = "audio_analysis.json"):
        """Save audio analysis results to file"""
        output_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.success(f"Audio analysis saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save audio analysis: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_audio.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    analyzer = AudioAnalyzer()
    results = analyzer.analyze_video_audio(video_path)
    
    print("\n📊 Audio Analysis Results:")
    print(f"Duration: {results.get('audio_duration', 0):.1f}s")
    print(f"Summary: {results.get('summary', 'No summary available')}")
    
    volume_analysis = results.get('volume_analysis', {})
    print(f"Volume spikes: {volume_analysis.get('spike_count', 0)}")
    
    crash_analysis = results.get('crash_analysis', {})
    crash_indicators = crash_analysis.get('crash_indicators', [])
    print(f"Potential crashes: {len(crash_indicators)}")
    
    if crash_indicators:
        for indicator in crash_indicators:
            print(f"  - {indicator['time']:.1f}s: {indicator['type']} ({indicator['confidence']} confidence)")
