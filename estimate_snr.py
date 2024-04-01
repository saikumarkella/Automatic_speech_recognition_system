import numpy as np
from scipy.fft import fft
import soundfile as sf

def calculate_snr(signal, samplerate=24000, resolution=10):
    # Step 1: Compute FFT
    fft_length = int(2 ** np.ceil(np.log2(samplerate / resolution)))
    fft_length = 1024
    
    fft_result = fft(signal, n=fft_length)
    print(fft_result.shape)
    # Step 2: Identify bins for 300-3000 Hz
    k_300 = int(300 * fft_length / samplerate)
    k_3000 = int(3000 * fft_length / samplerate)


    print(k_300, k_3000)
    # Step 3: Calculate energy in each bin
    energy = np.abs(fft_result[k_300:k_3000+1]) ** 2
    

    # Step 4: Calculate noise floor (N)
    N = np.min(energy) * (k_3000 - k_300 + 1)

    # Step 5: Calculate sum of energies (S)
    S = np.sum(energy)

    # Step 6: Calculate SNR
    SNR = (S - N) / N

    # Step 7: Calculate SNR level in dB
    SNR_dB = 10 * np.log10(SNR)

    return SNR_dB

# Example usage:
# Assuming you have your audio signal stored in a variable named 'audio_signal'
# SNR_level = calculate_snr(audio_signal)

if __name__ == "__main__":
    path = "/home/sai/Downloads/intro.mp3"
    audio,sr = sf.read(path)
    snr = calculate_snr(audio)
    print(snr)