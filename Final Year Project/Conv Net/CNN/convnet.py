# # %%
# %matplotlib inline

# %% [markdown]
# ### Library Imports

# %%
import tensorflow as tf

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# %%
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
from keras.callbacks import EarlyStopping,  ReduceLROnPlateau
from keras.applications import InceptionV3
from keras.layers import LeakyReLU
from sklearn.model_selection import KFold




# %% [markdown]
# ###  I. Data Loading

# %%
# read the CSV file
df_full = pd.read_csv('/mnt/data/MS39_Raw/CSO2/metadata/new_skanda_230320231356.csv', low_memory=False)

# %%
# Filter by unique Image ID 
df_byImageID= df_full.drop_duplicates(subset=['ImageID'])

# %% [markdown]


# %% [markdown]
# ### Dropping unnecessary predictors

# %%
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


sample_incomplete_rows = df_byImageID[df_byImageID.isnull().any(axis=1)].head()
sample_incomplete_rows

# %%
df_byImageID.dropna()

# %%
# Filter the DataFrame based on the 'Resolution' column and select the first 1500 rows
df_byImageID = df_byImageID[df_byImageID['Resolution'].str.contains("1024x1800")].sample(1400, random_state=42)


df_byImageID.info()

# %% [markdown]
# ## Conv Net

# %%
import tensorflow as tf
from tensorflow import keras

# %%
print(tf.__version__, ' ', tf.keras.__version__)



def load_images(df):
    images = []
    for file_path in df['FilePath']:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Assuming grayscale images
        img = cv2.resize(img, (256, 450))  # Resize the image (change dimensions as needed)
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

# %% [markdown]
# ### Splitting the dataset and normalising the image data

# %%
# Split the dataset into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# Normalize the image data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Create a mask for non-NaN values in train_labels
non_nan_mask = ~np.isnan(train_labels)

# Filter train_images and train_labels using the mask
train_images_filtered = train_images[non_nan_mask]
train_labels_filtered = train_labels[non_nan_mask]

# Check if there are any NaN values in train_labels_filtered
print("Are there NaN values in train_labels_filtered?", np.isnan(train_labels_filtered).any())

# %% [markdown]
# 

# %% [markdown]
# ### Data augmentation: Use data augmentation

# %%
# datagen = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     # vertical_flip=True,
#     # brightness_range=(0.8, 1.2),
#     # channel_shift_range=0.2,
#     # fill_mode='nearest'
#     )

# %% [markdown]
# ### a. Inception V3

# %%
policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)
from keras.applications import InceptionV3


#Load InceptionV3 model pre-trained on ImageNet
inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(256, 450, 3))

total_layers = len(inception.layers)
print("Total number of layers in the InceptionV3 model:", total_layers)


# Unfreeze the last few layers of the InceptionV3 model
for layer in inception.layers[:-10]:
    layer.trainable = False

for layer in inception.layers[-10:]:
    layer.trainable = True

# Add new layers to the InceptionV3 model
model = Sequential([
    inception,
    Flatten(),
    Dense(64, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    LeakyReLU(alpha=0.1),
    # Dropout(0.7),
    Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
    LeakyReLU(alpha=0.1),
    # Dropout(0.6),
    Dense(1, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))  # Regression output layer
])

# Set the learning rate schedule
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0001,
    decay_steps=10000,
    decay_rate=0.96)
# Create the optimizer with the learning rate schedule
opt = keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile the model
model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mae'])

early_stopping = EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_images_filtered, train_labels_filtered, batch_size=256, shuffle=True, 
                    steps_per_epoch=len(train_images_filtered) // 256, epochs=500, validation_data=(test_images, test_labels),
                    callbacks=[early_stopping])


# Evaluate the model
loss, mae = model.evaluate(test_images, test_labels)
print('Loss:', loss)
print('Mean Absolute Error:', mae)

# %%
import matplotlib.pyplot as plt

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Adjust spacing between subplots
fig.subplots_adjust(wspace=0.2)

# Plot training and validation loss
axs[0].plot(history.history['loss'], label='Training loss', linestyle='-', linewidth=2, marker='.', markersize=1)
axs[0].plot(history.history['val_loss'], label='Validation loss', linestyle='-', linewidth=2, marker='.', markersize=1)
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].set_title('Training and Validation Loss')  
axs[0].legend()
axs[0].grid(True) 

# Plot training and validation MAE
axs[1].plot(history.history['mae'], label='Training MAE', linestyle='-', linewidth=2, marker=',', markersize=1)
axs[1].plot(history.history['val_mae'], label='Validation MAE', linestyle='-', linewidth=2, marker=',', markersize=1)
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('MAE')
axs[1].set_title('Training and Validation Mean Absolute Error (MAE)')  # Add title
axs[1].legend()
axs[1].grid(True)  # Add gridlines

plt.show()

# %%
y_pred = model.predict(test_images)


def plot_predictions(test_labels, y_pred):

    # Make predictions on the test set
    # Plot the predicted vs actual values
    plt.scatter(test_labels, y_pred, alpha=0.5)  # Add transparency to better visualize overlapping points
    plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], 'k--', lw=2)  # Add a diagonal line
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Predicted vs Actual Values')  # Add a title to the plot
    plt.grid(True)  # Add a grid to the plot
    plt.show()

    # Compute the correlation coefficient
    corr = np.corrcoef(test_labels, y_pred[:, 0])[0, 1]
    print('Correlation coefficient:', round(corr, 4))  # Round the correlation coefficient to 4 decimal places

    # Print the range of predicted values
    print('Range of predicted values:', round(y_pred.min(), 2), '-', round(y_pred.max(), 2))  # Round the range values to 2 decimal places

# Make predictions on the test set
y_pred = model.predict(test_images)

# Call the function to plot predictions and display metrics
plot_predictions(test_labels, y_pred)






