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

def view_filter(base_folder, output_folder):
    view_categories = defaultdict(list)
    uncategorized_metadata = []
    i = 0
    for root, _, files in os.walk(base_folder):
        # print(root)
        # folder_data = {}
        # if i == 100:
        #     break
        for file in files:
            view = ""
            # if i == 100:
            #     break
            if file.endswith('.h5'):
                file_path = os.path.join(root, file)
                # print(root, file)
                
                # try:
                metadata = get_meta(file_path)
                # print(metadata)
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
                else:
                    uncategorized_metadata.append({"file": file_path, **metadata})
                print(view, metadata.get("SeriesDescription", "").decode("utf-8").lower())
                if view != "":
                    view_categories[view].append(file_path)
                # i+=1
                # if i == 100:
                #     break
                # except Exception as e:
                #     print(f"Error processing file {file_path}: {e}")

    if uncategorized_metadata:
        csv_path = os.path.join(output_folder, "uncategorized_metadata.csv")
        # Ensure output folder and subfolder for uncategorized metadata exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        with open(csv_path, mode="w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["file"] + list(uncategorized_metadata[0].keys()))
            writer.writeheader()
            writer.writerows(uncategorized_metadata)

    categorized_csv_path = os.path.join(output_folder, "categorized_views.csv")
    # Ensure output folder and subfolder for uncategorized metadata exist
    os.makedirs(os.path.dirname(categorized_csv_path), exist_ok=True)

    with open(categorized_csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["View Category", "File Path"])
        for view, files in view_categories.items():
            for path in files:
                writer.writerow([view, path])
    print(view_categories.items())
    for view, file_paths in view_categories.items():
        view_folder = os.path.join(output_folder, view)
        os.makedirs(view_folder, exist_ok=True)
        print(len(files))
        for file_path in file_paths:
            # try:
                if not os.path.exists(file_path):
                    print(f"文件不存在: {file_path}")
                    continue  # 跳过不存在的文件
                metadata = get_meta(file_path)
                accession_number = metadata.get("AccessionNumber").decode("utf-8")
                # Copy files to the view folder

                original_name = os.path.basename(file_path)
                unique_name = f"{accession_number}_{original_name}"
                print(1)
                target_path = os.path.join(view_folder, unique_name)

                # Move the file with the new name
                shutil.copy(file_path, target_path)
                print(2)
                # os.rename(file_path, os.path.join(view_folder, os.path.basename(file_path)))
            # except Exception as e:
            #     print(f"Error moving file {file_path}: {e}")

    print("Done!")


base_folder = "../../automated_resident/report_generation/datasets/xr_knee/v1.0.0-20241204"
output_folder = "../xr_knee_abc/classified"
view_filter(base_folder, output_folder)
