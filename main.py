# Install required libraries
!pip install wfdb matplotlib numpy pandas tensorflow scikit-learn

# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from wfdb import rdsamp
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, GRU, Dense, Flatten, Dropout, Bidirectional

# Define constants
sampling_rate = 500  # Hz
gain = 100  # Analog gain
n_splits = 5  # K-Fold Cross Validation splits

# Load ECG data
def load_ecg_data(folder_path):
    ecg_data = []
    labels = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".dat"):
                file_path = os.path.join(root, file[:-4])
                activity = file[-5]  # Extract activity from filename
                try:
                    signal, fields = rdsamp(file_path)
                    ecg_data.append(signal)
                    labels.append(activity)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
    return ecg_data, labels

# Dataset path
data_path = "./macecgdb/physionet.org/files/macecgdb/1.0.0/"

# Load data and preprocess
ecg_data, labels = load_ecg_data(data_path)
ecg_data = np.array(ecg_data)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
onehot_encoder = OneHotEncoder(sparse_output=False)
labels_onehot = onehot_encoder.fit_transform(integer_encoded.reshape(-1, 1))

# Normalize and reshape ECG signals
ecg_data = ecg_data / np.max(np.abs(ecg_data))
ecg_data = ecg_data.reshape(ecg_data.shape[0], ecg_data.shape[1], -1)

# CRNN Model Design
def build_crnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(GRU(64, return_sequences=True))(x)
    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# K-Fold Cross Validation
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold = 1
fold_accuracies = []

for train_index, val_index in kf.split(ecg_data):
    print(f"\nTraining on Fold {fold}/{n_splits}")
    
    # Train-validation split
    X_train, X_val = ecg_data[train_index], ecg_data[val_index]
    y_train, y_val = labels_onehot[train_index], labels_onehot[val_index]
    
    # Build and compile model
    model = build_crnn(X_train.shape[1:], y_train.shape[1])
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate on the validation fold
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Fold {fold} - Validation Accuracy: {val_accuracy * 100:.2f}%")
    fold_accuracies.append(val_accuracy)
    
    # Generate ROC Curve for each class
    y_val_pred = model.predict(X_val)
    fpr = {}
    tpr = {}
    roc_auc = {}
    num_classes = y_val.shape[1]

    plt.figure(figsize=(10, 6))
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_val[:, i], y_val_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title(f'Fold {fold} - ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()
    
    # Plot Loss and Accuracy for the Fold
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Fold {fold} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Fold {fold} - Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
    fold += 1

# Cross-Validation Summary
mean_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)
print(f"\nCross-Validation Results:")
print(f"Mean Accuracy: {mean_accuracy * 100:.2f}%")
print(f"Standard Deviation: {std_accuracy * 100:.2f}%")
