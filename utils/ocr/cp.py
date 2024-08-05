# # Open the input file in read mode
# with open("./none.txt", "r") as file:
#     lines = file.readlines()

# # Lists to hold paths based on their suffixes
# false_false_paths = []
# true_false_paths = []

# # Process each line
# for line in lines:
#     # Remove any leading or trailing whitespace
#     path = line.strip()
    
#     if path.endswith("FalseFalse"):
#         false_false_paths.append(path[:-10])
#     elif path.endswith("TrueFalse"):
#         true_false_paths.append(path[:-9])

# # Write the lists to separate text files
# with open("false_false_paths.txt", "w") as file:
#     for path in false_false_paths:
#         file.write(path + "\n")

# with open("true_false_paths.txt", "w") as file:
#     for path in true_false_paths:
#         file.write(path + "\n")

# print("Paths have been separated and written to 'false_false_paths.txt' and 'true_false_paths.txt'.")

import os
import shutil

# Define the input file name
input_file = "none.txt"

# Define the output directories
false_false_dir = "false_false_images"
true_false_dir = "true_false_images"

# Create the output directories if they don't exist
os.makedirs(false_false_dir, exist_ok=True)
os.makedirs(true_false_dir, exist_ok=True)

# Lists to hold paths based on their suffixes
false_false_paths = []
true_false_paths = []

# Read the input file and process each line
with open(input_file, "r") as file:
    lines = file.readlines()

for line in lines:
    # Remove any leading or trailing whitespace
    path = line.strip()
    
    if path.endswith("FalseFalse"):
        false_false_paths.append(path[:-10])
    elif path.endswith("TrueFalse"):
        true_false_paths.append(path[:-9])

# Write the lists to separate text files
with open("false_false_paths.txt", "w") as file:
    for path in false_false_paths:
        file.write(path + "\n")

with open("true_false_paths.txt", "w") as file:
    for path in true_false_paths:
        file.write(path + "\n")

# Move the images to the new directories
for path in false_false_paths:
    try:
        shutil.copy2(path, os.path.join(false_false_dir, os.path.basename(path)))
    except FileNotFoundError:
        print(f"File not found: {path}")

for path in true_false_paths:
    try:
        shutil.copy2(path, os.path.join(true_false_dir, os.path.basename(path)))
    except FileNotFoundError:
        print(f"File not found: {path}")

print("Paths have been separated, written to text files, and images have been moved to new directories.")
