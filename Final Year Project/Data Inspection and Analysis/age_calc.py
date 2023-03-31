import pandas as pd
import concurrent.futures
from tqdm import tqdm
import os

def process_chunk(chunk):
    # convert date and DOB columns to datetime objects
    chunk['Date'] = pd.to_datetime(chunk['Date'], format='%d/%m/%Y')
    chunk['DOB'] = pd.to_datetime(chunk['DOB'], format='%d/%m/%Y', errors='coerce')

    # drop rows with invalid dates
    chunk.dropna(subset=['DOB'], inplace=True)

    # calculate age in years
    chunk['Age'] = (chunk['Date'] - chunk['DOB']).astype('timedelta64[Y]')

    # insert Age column after DOB column
    dob_col_idx = chunk.columns.get_loc('DOB')
    chunk.insert(dob_col_idx + 1, 'Age', chunk.pop('Age'))

    return chunk

# read the CSV file in chunks
input_file = '/mnt/data/MS39_Raw/CSO2/metadata/images_with_metrics_kc_image_stages_skanda.csv'
chunksize = 10000

# determine the number of chunks to process
total_rows = sum(1 for _ in open(input_file, 'r')) - 1  # subtract 1 to exclude header
total_chunks = (total_rows // chunksize) + (1 if total_rows % chunksize != 0 else 0)

chunks = pd.read_csv(input_file, low_memory=False, chunksize=chunksize)

# process chunks in parallel using multiprocessing
with concurrent.futures.ProcessPoolExecutor() as executor:
    processed_chunks = list(tqdm(executor.map(process_chunk, chunks), total=total_chunks, desc="Processing CSV Chunks"))

# concatenate processed chunks into a single DataFrame
df = pd.concat(processed_chunks)

# save the updated DataFrame to a new CSV file
df.to_csv('/mnt/data/MS39_Raw/CSO2/metadata/new_skanda_230320231356.csv', index=False)
print('new file saved')
