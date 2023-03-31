import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


import numpy as np
import pandas as pd
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import  Flatten, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
# from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras import mixed_precision
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications.inception_v3 import InceptionV3


# read the CSV file
df_full = pd.read_csv('/mnt/data/MS39_Raw/CSO2/metadata/new_skanda_230320231356.csv', low_memory=False)

# Filter by unique Image ID 
df_byImageID= df_full.drop_duplicates(subset=['ImageID'])

# Drop unnecessary columns
df_byImageID.drop(["ImageDateTime", 
                   "QualityScoreFront", 
                   "VAunit", 
                   "VAvalue", 
                   "VAbaseValue", 
                   "QualityScoreBack", 
                   "Sex", 
                   "File",  
                   "VAmethod", 
                   "Comments", 
                   "TagID", 
                   "TasassignmentsID", 
                   "TagName", 
                   "FileType", 
                   "Stage", 
                   "TestID", 
                   "Surname", 
                   "FirstName",
                   "estimate_noise",
                   "estimate_signal2noise",
                   "estimate_contrast",
                   "estimate_uniformity",
                   "estimate_sharpness",
                   "avg_intensity",
                   "std_intensity",
                   "med_intensity",    
                   "DOB", 
                   "Date",
                   "ThinnestPointPachymetry", 
                   "Laterality",  
                   "TestDateTime", 
                   "UnknownID", 
                   "AcqCode", 
                   "PatientID", 
                   "ImageID" ], 
                   axis=1, inplace=True)

df_byImageID.dropna()

# Filter the DataFrame based on the 'Resolution' column and select the first 1500 rows
df_byImageID = df_byImageID[df_byImageID['Resolution'].str.contains("1024x1800")].sample(1000)

import tensorflow as tf
from tensorflow import keras

print(tf.__version__, ' ', tf.keras.__version__)

df = df_byImageID

# Load images from file paths
def load_images(df):
    images = []
    for file_path in df['FilePath']:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Assuming grayscale images
        img = cv2.resize(img, (128, 225))  # Resize the image (change dimensions as needed)
        img = np.stack((img,)*3, axis=-1)  # Stack the grayscale image to create a 3-channel image
        img = np.transpose(img, (1, 0, 2))  # Transpose the image to swap height and width
        images.append(img)
    return np.array(images)

# Load images into memory
images = load_images(df_byImageID)

# Separate features (images) and target (K2) values
X = images
y = df_byImageID['K2'].values

# Filter X and y simultaneously, removing rows with NaNs in y
X_filtered, y_filtered = zip(*[(x, target) for x, target in zip(X, y) if not np.isnan(target)])

# Convert the filtered X and y back to numpy arrays
X_filtered = np.array(X_filtered)
y_filtered = np.array(y_filtered)

# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Normalize the image data
train_images = train_images / 255.0
test_images = test_images / 255.0

#3.5

# Create a mask for non-NaN values in train_labels
non_nan_mask = ~np.isnan(train_labels)

# Filter train_images and train_labels using the mask
train_images_filtered = train_images[non_nan_mask]
train_labels_filtered = train_labels[non_nan_mask]

# Check if there are any NaN values in train_labels_filtered
print("Are there NaN values in train_labels_filtered?", np.isnan(train_labels_filtered).any())

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
tf.keras.mixed_precision.experimental.set_policy(policy)

inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(128, 225, 3))

for layer in inception.layers[:-10]:
    layer.trainable = False

for layer in inception.layers[-10:]:
    layer.trainable = True

model = Sequential([
    inception,
    Flatten(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.1)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.1)),
    BatchNormalization(),
    Dropout(0.6),
    Dense(1, kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1), dtype='float32')
])

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.96)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=60, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, verbose=1, min_lr=1e-6)


datagen.fit(train_images_filtered)

history = model.fit(datagen.flow(train_images_filtered, train_labels_filtered, batch_size=128, shuffle=True), 
                    steps_per_epoch=len(train_images_filtered) // 128, epochs=500, 
                    validation_data=(test_images, test_labels), callbacks=[early_stopping, reduce_lr])

loss, mae = model.evaluate(test_images, test_labels)
print('Loss:', loss)
print('Mean Absolute Error:', mae)