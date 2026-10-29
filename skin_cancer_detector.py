import os
import ssl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# GPU driver ko bypass karne ke liye (Kaggle/CPU error fix)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ssl._create_default_https_context = ssl._create_unverified_context

# Skin Cancer key major categories
class_names = ['Benign', 'Malignant']

print("Loading dataset and creating 70/20/10 split...")
np.random.seed(42)
total_samples = 2000
images_placeholder = np.random.rand(total_samples, 64, 64, 3).astype(np.float32)
labels_placeholder = np.random.randint(0, 2, total_samples)

# Data Split (70% Train, 20% Test, 10% Validation)
X_train, X_temp, y_train, y_temp = train_test_split(images_placeholder, labels_placeholder, test_size=0.30, random_state=42, stratify=labels_placeholder)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=(1/3), random_state=42, stratify=y_temp)

# ==========================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("--- Performing EDA ---")
print(f"Training Set Shape: {X_train.shape}")
print(f"Testing Set Shape:  {X_test.shape}")

# Class distribution plot
plt.figure(figsize=(8, 4))
sns.countplot(x=[class_names[i] for i in y_train], order=class_names, palette="coolwarm")
plt.title("Class Distribution")
plt.close()

# ==========================================
# CNN MODEL IMPLEMENTATION & TRAINING
# ==========================================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

epochs = 5
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), batch_size=32)

# ==========================================
# EVALUATION & PREDICTIONS
# ==========================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=class_names))

# Save the model architecture layout
model.save('skin_cancer_detector_model.keras')
print("Execution completed successfully!")
