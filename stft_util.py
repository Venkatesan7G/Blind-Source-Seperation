import numpy as np
import librosa

SR = 16000
N_FFT = 512
HOP = 128
WIN_LENGTH = 512
CENTER = True

FULL_F = N_FFT // 2 + 1  # 257
N_BINS = 256             # train on first 256 bins (drop Nyquist)


def stft_1d(x: np.ndarray) -> np.ndarray:
    return librosa.stft(
        x.astype(np.float32),
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN_LENGTH,
        center=CENTER,
    )


def stft_stereo(x: np.ndarray) -> np.ndarray:
    """
    Accepts stereo waveform:
      - (N,2)  from soundfile
      - (2,N)
    Returns:
      - (2,F,T) complex
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D stereo array, got {x.shape}")

    if x.shape[1] == 2:         # (N,2)
        ch1 = x[:, 0]
        ch2 = x[:, 1]
    elif x.shape[0] == 2:       # (2,N)
        ch1 = x[0, :]
        ch2 = x[1, :]
    else:
        raise ValueError(f"Not stereo: {x.shape}")

    X1 = stft_1d(ch1)
    X2 = stft_1d(ch2)
    return np.stack([X1, X2], axis=0)


def crop_F(arr: np.ndarray, target_F: int = N_BINS) -> np.ndarray:
    # supports (F,T) or (C,F,T)
    if arr.ndim == 2:
        return arr[:target_F, :]
    if arr.ndim == 3:
        return arr[:, :target_F, :]
    raise ValueError(f"Unsupported shape for crop_F: {arr.shape}")


def pad_or_trim_T(arr: np.ndarray, T: int) -> np.ndarray:
    curT = arr.shape[-1]
    if curT == T:
        return arr
    if curT > T:
        return arr[..., :T]
    pad = [(0, 0)] * arr.ndim
    pad[-1] = (0, T - curT)
    return np.pad(arr, pad, mode="constant")


def istft_1d(X: np.ndarray, length: int | None = None) -> np.ndarray:
    """
    If X is cropped to 256 bins, pad back to 257 bins before ISTFT.
    """
    if X.shape[0] == N_BINS:
        X = np.pad(X, ((0, 1), (0, 0)), mode="constant")

    y = librosa.istft(
        X,
        hop_length=HOP,
        win_length=WIN_LENGTH,
        center=CENTER,
        length=length,
    )
    return y.astype(np.float32)
