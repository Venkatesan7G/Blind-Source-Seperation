import numpy as np
import matplotlib.pyplot as plt

def add_dither_to_signal(signal, bit_depth, dither_type="uniform"):
    """
    Add dither noise to a signal before quantization.
    
    Parameters:
    - signal: The input signal (e.g., a numpy array).
    - bit_depth: The bit depth to quantize the signal to (e.g., 8, 16, 24 bits).
    - dither_type: Type of dither ('uniform' or 'gaussian').
    
    Returns:
    - quantized_signal: The quantized signal with added dither.
    """
    
    # Calculate the maximum value based on the bit depth (assuming signal is in range [-1, 1])
    max_value = 2 ** (bit_depth - 1) - 1

    # Generate dither signal based on chosen dither type
    if dither_type == "uniform":
        # Uniform dither in range [-1/2, 1/2]
        dither = np.random.uniform(-0.5, 0.5, size=signal.shape)
    elif dither_type == "gaussian":
        # Gaussian dither (zero mean, small standard deviation)
        dither = np.random.normal(0, 1/255, size=signal.shape)
    else:
        raise ValueError("Unsupported dither type. Use 'uniform' or 'gaussian'.")
    
    # Add dither to the signal
    signal_with_dither = signal + dither
    
    # Quantize the signal: map the signal to the range of [-max_value, max_value]
    quantized_signal = np.round(signal_with_dither * max_value).astype(np.int32)
    
    # Clip the values to ensure they are within the valid range for the bit depth
    quantized_signal = np.clip(quantized_signal, -max_value, max_value)
    
    # Normalize back to the original range [-1, 1]
    quantized_signal = quantized_signal / max_value
    
    return quantized_signal

# Example Usage
if __name__ == "__main__":
    # Create a sample sine wave signal (for example purposes)
    fs = 44100  # Sampling rate
    t = np.linspace(0, 1, fs)  # 1-second duration
    frequency = 1000  # Frequency of sine wave in Hz
    signal = np.sin(2 * np.pi * frequency * t)  # Sine wave signal
    
    # Apply dithering to the signal
    bit_depth = 8
    signal_with_dither = add_dither_to_signal(signal, bit_depth, dither_type="uniform")
    
    # Plot the original vs. the dithered signal
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t[:2000], signal[:2000], label="Original Signal")
    plt.title("Original Signal")
    plt.subplot(2, 1, 2)
    plt.plot(t[:2000], signal_with_dither[:2000], label="Signal with Dither", color='orange')
    plt.title("Signal with Dither")
    plt.tight_layout()
    plt.show()
