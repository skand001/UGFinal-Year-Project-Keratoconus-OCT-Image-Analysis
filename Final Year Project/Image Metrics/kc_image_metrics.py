import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import datetime
import multiprocessing
from tqdm import tqdm

# Define the input CSV file path for full, and the filtered data
filtered_byImages = pd.read_csv('/mnt/data/MS39_Raw/CSO2/metadata/filtered_byImages.csv', low_memory=False)

# Define the output CSV file path
output_csv = ('/mnt/data/MS39_Raw/CSO2/metadata/metrics_test.csv')

# Check if the output file already exists
if os.path.exists(output_csv):
    # If it exists, add a timestamp to the filename before the file extension
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_csv = output_csv.split('.csv')[0] + '_' + timestamp + '.csv'

# Read the list of image file paths from the CSV file
image_files = filtered_byImages['FilePath'].tolist()

# Create an empty dataframe to store the statistics
stats_df = pd.DataFrame(columns=['filename', 
                                 'estimate_noise', 
                                 'estimate_signal2noise', 
                                 'estimate_contrast',
                                 'estimate_uniformity',
                                 'estimate_sharpness', 
                                 'avg_intensity', 
                                 'med_intensity', 
                                 'std_intensity']
                                 )

# Defining Metrics Functions

# Estimating the noise level in an image using Laplacian variance.
def estimate_noise(img_arr):
    est_noise = cv2.Laplacian(img_arr, cv2.CV_64F).var()
    return est_noise

# Estimating the signal-to-noise ratio (SNR)
def estimate_signal2noise(I):
    est_sign2noise= 20 * np.log10(np.max(I) / estimate_noise(I))
    return est_sign2noise  

# Estimating the contrast of an image using its standard deviation.
def estimate_contrast(I):
    est_contrast= np.std(I)
    return est_contrast

# Estimating the uniformity of an image using the ratio of its standard deviation to its mean.
def estimate_uniformity(I):
    est_uniforimity= np.std(I) / np.mean(I)
    return  est_uniforimity

# Estimating the sharpness of an image using Sobel filtering.
def estimate_sharpness(I):
    sobel_x = cv2.Sobel(I, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(I, cv2.CV_64F, 0, 1)
    sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    est_sharpness= np.mean(sobel)
    return est_sharpness

# Estimating average pixel density
def estimate_avgPixelDensity(I):
    est_avgPixelDensity= cv2.mean(I)[0]
    return est_avgPixelDensity
 
# Estimating median pixel density
def estimate_medianPixelDensity(I):
    est_medianPixelDensity= np.median(I)
    return est_medianPixelDensity

# Estimate Standard Deviation
def estimate_std(I):
    est_std= np.std(I)
    return est_std

# Define the number of processes to use
num_processes = 8

# Split the list of image files into chunks
image_file_chunks = [image_files[i:i+num_processes] for i in range(0, len(image_files), num_processes)]

def process_chunk(image_files):
    dfs = []  # list to store dataframes
    for file_path in tqdm(image_files, desc='Processing', unit='file'):
        filename = os.path.basename(file_path)

        try:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        except:
            # If there is an error loading the image, 
            # add a row with NaN values to the output dataframe
            row = {'filename': filename, 
                   'estimate_noise': np.nan, 
                   'estimate_signal2noise': np.nan, 
                   'estimate_contrast': np.nan,
                   'estimate_uniformity': np.nan,
                   'estimate_sharpness': np.nan, 
                   'avg_intensity': np.nan, 
                   'med_intensity': np.nan, 
                   'std_intensity': np.nan
                   }
            
        else:
            # Calculate the required statistics
            estimate_noise_val = estimate_noise(image)
            estimate_signal2noise_val = estimate_signal2noise(image)
            estimate_contrast_val = estimate_contrast(image)
            estimate_uniformity_val = estimate_uniformity(image)
            estimate_sharpness_val = estimate_sharpness(image)
            avg_intensity_val = estimate_avgPixelDensity(image)
            med_intensity_val = estimate_medianPixelDensity(image)
            std_intensity_val = estimate_std(image)

            # Create a dictionary for the row of statistics
            row = {'filename': filename,
                   'estimate_noise':estimate_noise_val, 
                   'estimate_signal2noise':estimate_signal2noise_val, 
                   'estimate_contrast':estimate_contrast_val, 
                   'estimate_uniformity':estimate_uniformity_val, 
                   'estimate_sharpness': estimate_sharpness_val, 
                   'avg_intensity': avg_intensity_val, 
                   'med_intensity': med_intensity_val, 
                   'std_intensity': std_intensity_val
                   }

        # Append the row to the list of dataframes
        dfs.append(pd.DataFrame([row]))

    # Concatenate the list of dataframes into a single dataframe
    stats_df = pd.concat(dfs, ignore_index=True)
    return stats_df

# Create a pool of processes and apply the process_chunk function to each chunk of image files
with multiprocessing.Pool(processes=num_processes) as pool:
    results = pool.map(process_chunk, image_file_chunks)

# Concatenate the results into a single dataframe
stats_df = pd.concat(results, ignore_index=True)

# Set the index of the output dataframe to match the input filenames
stats_df.set_index('filename', inplace=True)

# Write the output dataframe to a CSV file
stats_df.to_csv(output_csv)
num_rows_added = len(stats_df) 
print('Data has been written to the output file. Number of rows added: {}'.format(num_rows_added))
