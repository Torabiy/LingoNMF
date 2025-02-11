import librosa
import numpy as np

def load_audio_files():
    """
    Loads heart and lung sound audio files.
    """
    y1, fs1 = librosa.load('M_AF_LC.wav', sr=None)
    y2, fs2 = librosa.load('M_W_RLA.wav', sr=None)

    if fs1 != fs2:
        y2 = librosa.resample(y2, orig_sr=fs2, target_sr=fs1)

    min_length = min(len(y1), len(y2))
    h = y1[:min_length]
    l = y2[:min_length]

    return np.vstack([h, l]), fs1
