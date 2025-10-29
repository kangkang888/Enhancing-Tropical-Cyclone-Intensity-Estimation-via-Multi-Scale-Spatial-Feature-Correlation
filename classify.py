import h5py
import pandas as pd

# Define the function to classify wind speeds
def classify_wind_speed(vmax):
    if vmax <= 10.3:
        return 'NC'
    elif vmax <= 17.0:
        return 'TD'
    elif vmax <= 32.4:
        return 'TS'
    elif vmax <= 42.2:
        return 'H1'
    elif vmax <= 48.9:
        return 'H2'
    elif vmax <= 57.6:
        return 'H3'
    elif vmax <= 70.0:
        return 'H4'
    else:
        return 'H5'

# Path to the data file
data_path = "./data/TCIR-ALL_2017.h5"

# Load the data information
data_info = pd.read_hdf(data_path, key="info", mode='r')

# Load the data matrix
with h5py.File(data_path, 'r') as hf:
    data_matrix = hf['matrix'][:]

# Classify the wind speeds and count each category
data_info['category'] = data_info['Vmax'].apply(classify_wind_speed)
category_counts = data_info['category'].value_counts()

print(category_counts)
