import os
import gc
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

print("TensorFlow version:" , tf.__version__)
print("GPU AVAILABLE:" , tf.config.list_physical_devices('GPU'))

# 👉 UPDATE THIS PATH TO YOUR KINETICS DATASET FOLDER ON KAGGLE!
# Once you click "Add Data" and attach a kinetics dataset, copy its path here.
DATA_PATH = "/kaggle/input/datasets/pypiahmad/realistic-action-recognition-ucf50/UCF50"

# We automatically load all folders (classes) available in the dataset
all_classes = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
print(f"Found {len(all_classes)} Total Classes in Dataset.")

# To keep RAM usage safe on Kaggle (30GB limit), we limit to the first 50 classes for this run.
# You can increase this if Kaggle's memory allows it!
MAX_CLASSES_TO_USE = 50
SELECTED_CLASSES = all_classes[:MAX_CLASSES_TO_USE]

print(f"\nUsing {len(SELECTED_CLASSES)} Classes.")
class_to_idx = {clas: idx for idx, clas in enumerate(SELECTED_CLASSES)}

IMG_SIZE = 224
NUM_FRAMES = 30 # Increased from 16 to 30 for better performance!

# ── STEP 1: Load Pretrained ResNet50 ──────────────────────────────
base_cnn = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# We completely freeze ResNet50 to act only as a feature extractor
for layer in base_cnn.layers:
    layer.trainable = False

cnn_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
cnn_out = base_cnn(cnn_input, training=False) # entering into reset50 pokemon
cnn_out = GlobalAveragePooling2D()(cnn_out)
cnn_model = Model(cnn_input, cnn_out, name='CNN_FeatureExtractor')

print('✅ ResNet50 loaded and completely frozen.')
print(f'Trainable params: {cnn_model.count_params():,}')  # Should be 0

def extract_frames_and_features(video_path, cnn_model, num_frames=30, img_size=224):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return None

    # Focus on the middle of the video (max 150 frames) to prevent diluting the action!
    MAX_ACTION_FRAMES = 150
    if total_frames <= MAX_ACTION_FRAMES:
        start_frame = 0
        end_frame = total_frames - 1
    else:
        start_frame = (total_frames - MAX_ACTION_FRAMES) // 2
        end_frame = start_frame + MAX_ACTION_FRAMES - 1
        
    frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 1. Center crop to prevent fat/stretched people on mobile videos!
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            frame = frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
            
            # 2. Resize
            frame = cv2.resize(frame, (img_size, img_size))
            
            # 3. Convert BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            frames.append(np.zeros((img_size, img_size, 3)))

    cap.release()
    frames = np.array(frames, dtype=np.float32)

    # ✨ CRITICAL FIX: Use ResNet50's official preprocess_input instead of / 255.0
    # This prevents the AI from getting confused by out-of-distribution colors.
    frames = preprocess_input(frames)

    # Extract CNN Features immediately to save RAM!
    features = cnn_model.predict(frames, verbose=0)
    return features

# ── STEP 2: Process ALL Videos → Store Only CNN Features ───────────
all_features = []
all_labels = []

print('Extracting CNN features from all videos...\n')

for class_name in SELECTED_CLASSES:
    class_folder = os.path.join(DATA_PATH, class_name)
    if not os.path.exists(class_folder): continue
        
    video_files = os.listdir(class_folder)
    # Limit to 100 videos per class to save time during this run
    video_files = video_files[:100] 
    
    label = class_to_idx[class_name]
    print(f'Processing: {class_name} ({len(video_files)} videos)')

    for video_file in tqdm(video_files):
        video_path = os.path.join(class_folder, video_file)
        features = extract_frames_and_features(video_path, cnn_model, NUM_FRAMES, IMG_SIZE)
        
        if features is not None and len(features) == NUM_FRAMES:
            all_features.append(features)
            all_labels.append(label)

X = np.array(all_features, dtype=np.float32)
y = np.array(all_labels, dtype=np.int32)

print(f'\n✅ Extraction Complete! Memory usage: {X.nbytes / 1e6:.1f} MB')

NUM_CLASSES = len(SELECTED_CLASSES)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

y_train = to_categorical(y_train, NUM_CLASSES)
y_val = to_categorical(y_val, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

print('Split summary:')
print(f'  X_train : {X_train.shape}')
print(f'  X_val   : {X_val.shape}')
print(f'  X_test  : {X_test.shape}')

FEATURE_DIM = 2048

# ── BUILD MODEL WITH REGULARIZATION ─────────────────────────────────────
model = Sequential([
    LSTM(128, input_shape=(NUM_FRAMES, FEATURE_DIM), return_sequences=True, 
         kernel_regularizer=regularizers.l2(0.001)), # L2 Prevents memorization
    
    Dropout(0.4), # Increased Dropout to prevent overfitting

    LSTM(64, return_sequences=False, kernel_regularizer=regularizers.l2(0.001)),
    
    Dropout(0.4),
    
    BatchNormalization(),
    
    Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    
    Dropout(0.3),
    
    Dense(NUM_CLASSES, activation='softmax')
], name="Kinetics_ActionNet")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    ModelCheckpoint('kinetics_best_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training Complete! Your highly generalized Kinetics model is ready!")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)

print(f"\nTest Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=SELECTED_CLASSES
)

plt.figure(figsize=(12,12))
disp.plot(cmap="Blues", xticks_rotation=90)
plt.show()

from sklearn.metrics import classification_report

print(classification_report(
    y_true,
    y_pred,
    target_names=SELECTED_CLASSES
))