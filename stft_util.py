import numpy as np
import librosa

# Keep these consistent with feature_extraction.py
N_FFT = 512
HOP = 128
WIN_LENGTH = 512
CENTER = True

FULL_F = N_FFT // 2 + 1  # 257


def stft_1d(x: np.ndarray) -> np.ndarray:
    """
    x: (N,)
    returns complex STFT: (F,T) where F=257 for n_fft=512
    """
    return librosa.stft(
        x.astype(np.float32),
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN_LENGTH,
        center=CENTER,
    )


def istft_1d(X: np.ndarray, length: int | None = None) -> np.ndarray:
    """
    X: complex STFT (F,T).
    If F==256 (cropped), pad back to 257 for librosa.istft with n_fft=512.
    """
    if X.shape[0] == FULL_F - 1:  # 256
        X = np.pad(X, ((0, 1), (0, 0)), mode="constant")  # -> (257,T)

    return librosa.istft(
        X,
        hop_length=HOP,
        win_length=WIN_LENGTH,
        center=CENTER,
        length=length,
    ).astype(np.float32)


def pad_or_trim_T(arr: np.ndarray, T: int) -> np.ndarray:
    """
    Pads or truncates along the LAST axis to exactly T frames.
    Works for (..., T).
    """
    curT = arr.shape[-1]
    if curT == T:
        return arr
    if curT > T:
        return arr[..., :T]
    pad = [(0, 0)] * arr.ndim
    pad[-1] = (0, T - curT)
    return np.pad(arr, pad, mode="constant")


def crop_F(arr: np.ndarray, target_F: int) -> np.ndarray:
    """
    Crop frequency axis to target_F (keep low frequencies).
    Supports arrays shaped:
      (F,T) or (C,F,T) or (F,T,C)
    """
    if arr.ndim == 2:
        return arr[:target_F, :]
    if arr.ndim == 3:
        # Could be (C,F,T) or (F,T,C)
        if arr.shape[1] == FULL_F or arr.shape[1] > target_F:
            # assume (C,F,T)
            return arr[:, :target_F, :]
        else:
            # assume (F,T,C)
            return arr[:target_F, :, :]
    # fallback: just slice first axis
    slicer = [slice(None)] * arr.ndim
    slicer[0] = slice(0, target_F)
    return arr[tuple(slicer)]
