import os
import csv
import h5py
import shutil
from collections import defaultdict

# Get Metadata
def get_meta(file_path):
    with h5py.File(file_path, 'r') as f:
        metadata = {key: value[()] for key, value in f['metadata'].items()}
    return metadata

file_path = "../xr_knee_final/images/5170192/3.h5"
metadata = get_meta(file_path)
print(metadata)
# print(metadata.get("SeriesDescription", "").decode("utf-8").lower())
# folder_data[file_path] = metadata
# print(metadata)
# print(type(metadata), 111)
if "sun" in metadata.get("SeriesDescription", "").decode("utf-8").lower() or "pat" in metadata.get("SeriesDescription", "").decode("utf-8").lower() \
    or "sun" in metadata.get("ViewPosition", "").decode("utf-8").lower() or "pat" in metadata.get("ViewPosition", "").decode("utf-8").lower() \
    or "sun" in metadata.get("AcquisitionDeviceProcessingDescription", "").decode("utf-8").lower() or "pat" in metadata.get("AcquisitionDeviceProcessingDescription", "").decode("utf-8").lower():
    view = "sunrise"
elif "ap" in metadata.get("SeriesDescription", "").decode("utf-8").lower() or "pa" in metadata.get("SeriesDescription", "").decode("utf-8").lower() \
    or "ap" in metadata.get("ViewPosition", "").decode("utf-8").lower() or "pa" in metadata.get("ViewPosition", "").decode("utf-8").lower() \
    or "ap" in metadata.get("AcquisitionDeviceProcessingDescription", "").decode("utf-8").lower() or "pa" in metadata.get("AcquisitionDeviceProcessingDescription", "").decode("utf-8").lower():
    view = "ap"
elif "lat" in metadata.get("SeriesDescription", "").decode("utf-8").lower() and "bilat" not in metadata.get("SeriesDescription", "").decode("utf-8").lower() \
    or "lat" in metadata.get("ViewPosition", "").decode("utf-8").lower() and "bilat" not in metadata.get("ViewPosition", "").decode("utf-8").lower() \
    or "lat" in metadata.get("AcquisitionDeviceProcessingDescription", "").decode("utf-8").lower() and "bilat" not in metadata.get("AcquisitionDeviceProcessingDescription", "").decode("utf-8").lower():
    view = "lat"
print(view)