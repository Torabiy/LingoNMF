# data_loader.py
# This module handles loading and preprocessing of audio data.

import librosa

def load_audio_data(file1, file2):
    """Loads and preprocesses two audio files."""
    y1, fs1 = librosa.load(file1, sr=None)
    y2, fs2 = librosa.load(file2, sr=None)
    
    # Ensure the sample rates match
    if fs1 != fs2:
        y2 = librosa.resample(y2, orig_sr=fs2, target_sr=fs1)
    
    return y1, fs1, y2, fs2
