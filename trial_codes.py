import numpy as np
import os
from scipy.signal import butter, lfilter, coherence
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load .dat file
def load_dat_file(file_path):
    data = np.fromfile(file_path, dtype=np.uint8)
    expected_size = 40 * 40 * 8064
    header_size = data.size - expected_size
    eeg_data = data[header_size:]
    eeg_data = eeg_data.reshape((40, 40, 8064))
    eeg_data = eeg_data.astype(np.float32) / 255.0
    return eeg_data

# Function to apply bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = lfilter(b, a, data, axis=-1)
    return filtered_data

# Function to calculate coherence between two channels
def calculate_coherence(data, fs):
    n_participants, n_trials, n_channels, n_samples = data.shape
    coherence_matrix = np.zeros((n_participants, n_trials, n_channels, n_channels))

    for participant in range(n_participants):
        for trial in range(n_trials):
            for i in range(n_channels):
                for j in range(n_channels):
                    if i != j:
                        f, Cxy = coherence(data[participant, trial, i, :], data[participant, trial, j, :], fs=fs)
                        coherence_matrix[participant, trial, i, j] = np.mean(Cxy)
    
    return coherence_matrix

# Parameters
data_folder = r'C:\Users\KIIT\OneDrive\Desktop\Emotion_Detection_Project\data'
fs = 128  # Sampling frequency
lowcut = 1.0  # Lower bound of the frequency range
highcut = 50.0  # Upper bound of the frequency range

# Load and preprocess data
all_filtered_eeg_data = []
for file_name in os.listdir(data_folder):
    if file_name.endswith('.dat'):
        file_path = os.path.join(data_folder, file_name)
        eeg_data = load_dat_file(file_path)
        filtered_eeg_data = np.array([bandpass_filter(trial, lowcut, highcut, fs) for trial in eeg_data])
        all_filtered_eeg_data.append(filtered_eeg_data)

all_filtered_eeg_data = np.array(all_filtered_eeg_data)

# Generate synthetic multi-class labels (e.g., 3 classes)
labels = np.random.randint(0, 3, size=(32, 40))  # Shape: (32 participants, 40 trials)

# Calculate coherence
coherence_matrix = calculate_coherence(all_filtered_eeg_data, fs)

# Reshape coherence matrix for model input
X = coherence_matrix.reshape(32 * 40, -1)  # Shape: (1280, 1600)
y = labels.reshape(32 * 40)  # Shape: (1280,) - Original labels (not one-hot encoded)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)  # Use original labels here

# Evaluate the model
y_pred = gb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)  # Use original labels here
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)  # Use original labels here
print("Confusion Matrix:\n", cm)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Multi-Class Classification')
plt.show()

# ROC Curve for each class
y_score = gb_model.predict_proba(X_test)  # Get predicted probabilities for each class
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])  # Binarize y_test for ROC curve

fpr = {}
tpr = {}
roc_auc = {}
colors = ['blue', 'red', 'green']

plt.figure()
for i in range(3):  # Assuming 3 classes
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multi-Class')
plt.legend(loc="lower right")
plt.show()