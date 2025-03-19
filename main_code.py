import numpy as np
import os
from scipy.signal import butter, lfilter, coherence
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving and loading models
from tqdm import tqdm  # For progress tracking

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

# Function to calculate coherence between two channels and create connectivity matrix
def calculate_coherence(data, fs):
    n_participants, n_trials, n_channels, n_samples = data.shape
    coherence_matrix = np.zeros((n_participants, n_trials, n_channels, n_channels))

    for participant in range(n_participants):
        for trial in tqdm(range(n_trials), desc='Calculating Coherence'):
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

# Load and preprocess data with progress tracking
all_filtered_eeg_data = []
for file_name in tqdm(os.listdir(data_folder), desc='Loading Data'):
    if file_name.endswith('.dat'):
        file_path = os.path.join(data_folder, file_name)
        eeg_data = load_dat_file(file_path)
        filtered_eeg_data = np.array([bandpass_filter(trial, lowcut, highcut, fs) for trial in eeg_data])
        all_filtered_eeg_data.append(filtered_eeg_data)

all_filtered_eeg_data = np.array(all_filtered_eeg_data)

# Generate synthetic multi-class labels (e.g., 3 classes)
labels = np.random.randint(0, 3, size=(32, 40))  # Shape: (32 participants, 40 trials)

# Calculate coherence and create connectivity matrix with progress tracking
coherence_matrix = calculate_coherence(all_filtered_eeg_data, fs)

# Reshape coherence matrix for model input
X = coherence_matrix.reshape(32 * 40, -1)  # Shape: (1280, 1600)
y = labels.reshape(32 * 40)  # Shape: (1280,) - Original labels (not one-hot encoded)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning with Grid Search and progress tracking
param_grid = {
    'estimator__n_estimators': [100, 200],
    'estimator__learning_rate': [0.01, 0.1],
    'estimator__max_depth': [3, 5, 7],
    'estimator__subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(OneVsRestClassifier(GradientBoostingClassifier(random_state=42)), param_grid, cv=5, verbose=2, n_jobs=-1)
with tqdm(total=len(param_grid['estimator__n_estimators']) * len(param_grid['estimator__learning_rate']) * len(param_grid['estimator__max_depth']) * len(param_grid['estimator__subsample']), desc='Grid Search') as pbar:
    grid_search.fit(X_train, y_train)
    pbar.update(1)

# Print best parameters
print(f"Best parameters: {grid_search.best_params_}")

# Use the best estimator to predict
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with hyperparameter tuning: {accuracy * 100:.2f}%")

# Detailed Metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Hyperparameter Tuning')
plt.show()

# Model Stacking
base_learners = [
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
]

meta_learner = LogisticRegression()
stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=meta_learner, cv=5)
with tqdm(total=5, desc='Model Stacking') as pbar:
    stacking_clf.fit(X_train, y_train)
    pbar.update(1)

# Evaluate the stacked model
y_pred = stacking_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with stacking: {accuracy * 100:.2f}%")

# Detailed Metrics for Stacked Model
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f"Precision with stacking: {precision:.2f}")
print(f"Recall with stacking: {recall:.2f}")
print(f"F1 Score with stacking: {f1:.2f}")

# Confusion Matrix for Stacked Model
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1', 'Class 2'], yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix with Stacking')
plt.show()

# Cross-validation for stacking classifier with progress tracking
with tqdm(total=5, desc='Cross-validation') as pbar:
    cv_scores = cross_val_score(stacking_clf, X, y, cv=5)
    pbar.update(1)
print(f"Cross-validated accuracy: {np.mean(cv_scores) * 100:.2f}%")

# ROC Curve for each class
y_score = stacking_clf.predict_proba(X_test)
y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])

fpr = {}
tpr = {}
roc_auc = {}
colors = ['blue', 'red', 'green']

plt.figure()
for i in range(len(np.unique(y))):  # Use the number of unique classes
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

# Save the trained[_{{{CITATION{{{_1{](https://github.com/MDrance/Filbi-project/tree/0729748706b08c82aee0b80224153288f5553576/knn.py)[_{{{CITATION{{{_2{](https://github.com/eugeneOlkhovik/isic-2019/tree/cfb3492d96cbe265e94431470c24206cac6a8979/vis_utils.py)[_{{{CITATION{{{_3{](https://github.com/m4ni5h/UdacityMLND2/tree/636395e783e8420cfcd0636a7e62ffc8a3e680d4/Dermatologist-AI%2Fget_results.py)[_{{{CITATION{{{_4{](https://github.com/JRLi/untitled/tree/f940ab166ba57f06288a64ae0b6978c11450a806/QuickService%2FlogPBMC2.py)
