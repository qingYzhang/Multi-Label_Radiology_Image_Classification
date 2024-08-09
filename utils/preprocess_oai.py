import os
import json

# Define the root directory and subdirectories for train, val, and test
root_dir = '../OAI'
subdirs = ['train', 'val', 'test']

# Initialize the output lists for merged train/val and test datasets
train_val_data = []
test_data = []

# Iterate through subdirectories (train, val, and test)
for subdir in subdirs:
    # Get the full path for the subdir
    subdir_path = os.path.join(root_dir, subdir)
    
    # Iterate through each class directory (0, 1, 2, etc.)
    for class_dir in os.listdir(subdir_path):
        class_path = os.path.join(subdir_path, class_dir)
        
        if os.path.isdir(class_path):
            # Create the target list based on the class directory name
            target = [0] * 5  # Assuming 5 classes
            target[int(class_dir)] = 1
            
            # Iterate through images in the class directory
            for image_file in os.listdir(class_path):
                # Construct the relative image path
                # img_path = os.path.join(f"../Dataset/Chest/all_images", image_file)
                img_path = os.path.join(subdir_path, class_dir, image_file)
                
                # Create the dictionary for the current image
                data_entry = {
                    "target": target,
                    "img_path": img_path
                }
                
                # Append to the corresponding output list
                if subdir == 'test':
                    test_data.append(data_entry)
                else:
                    train_val_data.append(data_entry)


print(train_val_data)
# Write the merged train/val and test datasets to JSON files
train_val_json_path = 'train_val_dataset.json'
test_json_path = 'test_dataset.json'

with open(train_val_json_path, 'w') as train_val_file:
    json.dump(train_val_data, train_val_file, indent=4)

with open(test_json_path, 'w') as test_file:
    json.dump(test_data, test_file, indent=4)

print(f"Train/Val JSON saved to {train_val_json_path}")
print(f"Test JSON saved to {test_json_path}")
