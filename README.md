EEG-Based Emotion Detection

**Overview**

This project implements an EEG-based emotion detection system using machine learning techniques. The system processes EEG signals, applies a bandpass filter, extracts connectivity features using coherence, and employs machine learning models, including Gradient Boosting, Stacking Classifiers, and Support Vector Machines, for classification.

**Features**

Loads EEG data from .dat files

Applies a bandpass filter to preprocess EEG signals

Computes coherence-based connectivity matrices

Performs hyperparameter tuning using GridSearchCV

Implements Gradient Boosting, Stacking Classifiers, and Random Forest for emotion classification

Visualizes results using confusion matrices and ROC curves

Saves trained models for future use

**Installation**

**Prerequisites**

Ensure you have Python installed (>=3.7) along with the required dependencies:

pip install numpy scipy scikit-learn matplotlib seaborn joblib tqdm

Usage

**1. Load EEG Data**

Modify data_folder in the script to point to your EEG dataset directory.

data_folder = r'C:\Users\KIIT\OneDrive\Desktop\Emotion_Detection_Project\data'

**2. Preprocess Data**

Loads .dat files and extracts EEG signals.

Applies a bandpass filter (1-50 Hz) to remove noise.

Computes coherence-based connectivity matrices.

**3. Train & Evaluate Models**

Splits the dataset into training and test sets.

Performs hyperparameter tuning with GridSearchCV.

Evaluates models using accuracy, precision, recall, and F1-score.

Uses Stacking Classifier to improve classification performance.

**4. Visualization**
Generates confusion matrices.

Plots ROC curves for multi-class classification.

**5. Save and Load Models**

Save the trained model for reuse:

joblib.dump(best_model, 'emotion_detection_model.pkl')

Load the trained model:

model = joblib.load('emotion_detection_model.pkl')

**Results**

Best hyperparameters are automatically selected using GridSearchCV.

Final accuracy and performance metrics are printed.

Model stacking improves classification accuracy.

**License**

This project is open-source and available under the MIT License.
