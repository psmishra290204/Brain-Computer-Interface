import numpy as np
import os

# Set the correct path
project_path = r'C:\Users\KIIT\OneDrive\Desktop\Emotion_Detection_Project\data'
file_path = os.path.join(project_path, 's01.dat')  # Adjust this to access the specific file

# Function to load .dat file
def load_dat_file(file_path):
    data = np.fromfile(file_path, dtype=np.uint8)
    print("Data size (including header):", data.size)
    
    # Extract the EEG data portion by ignoring the initial headers
    expected_size = 40 * 40 * 8064
    header_size = data.size - expected_size
    print("Header size:", header_size)
    
    # Skip the header portion and reshape the remaining data
    eeg_data = data[header_size:]
    if eeg_data.size != expected_size:
        print(f"Warning: Data size mismatch after header adjustment. Expected {expected_size}, but got {eeg_data.size}")
    
    eeg_data = eeg_data.reshape((40, 40, 8064))
    eeg_data = eeg_data.astype(np.float32) / 255.0
    return eeg_data

# Load the data
eeg_data = load_dat_file(file_path)

print(eeg_data.shape)  # Expected output: (40, 40, 8064)

from scipy.signal import butter, lfilter

# Function to apply bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = lfilter(b, a, data, axis=-1)
    return filtered_data

fs = 128  # Sampling frequency
lowcut = 1.0
highcut = 50.0

# Apply the bandpass filter to each channel of the EEG data
filtered_eeg_data = np.array([bandpass_filter(trial, lowcut, highcut, fs) for trial in eeg_data])

print(filtered_eeg_data.shape)  # Same shape as original data

from scipy.signal import butter, lfilter

# Function to apply bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = lfilter(b, a, data, axis=-1)
    return filtered_data

fs = 128  # Sampling frequency
lowcut = 1.0
highcut = 50.0

# Apply the bandpass filter to each channel of the EEG data
filtered_eeg_data = np.array([bandpass_filter(trial, lowcut, highcut, fs) for trial in eeg_data])

print(filtered_eeg_data.shape)  # Same shape as original data