import scipy
import imageio
import argparse

import numpy as np
import pandas as pd


def get_masks_and_sizes_of_connected_components(img_mask):
    """
    Finds the connected components from the mask of the image
    """
    mask, num_labels = scipy.ndimage.label(img_mask)

    mask_pixels_dict = {}
    for i in range(num_labels + 1):
        this_mask = (mask == i)
        if img_mask[this_mask][0] != 0:
            # Exclude the 0-valued mask
            mask_pixels_dict[i] = np.sum(this_mask)

    return mask, mask_pixels_dict


def get_mask_of_largest_connected_component(img_mask):
    """
    Finds the largest connected component from the mask of the image
    """
    mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(img_mask)
    print(pd.Series(mask_pixels_dict))
    largest_mask_index = pd.Series(mask_pixels_dict).idxmax()
    largest_mask = mask == largest_mask_index
    return largest_mask


def get_edge_values(img, largest_mask, axis):
    """
    Finds the bounding box for the largest connected component
    """
    assert axis in ["x", "y"]
    has_value = np.any(largest_mask, axis=int(axis == "y"))
    edge_start = np.arange(img.shape[int(axis == "x")])[has_value][0]
    edge_end = np.arange(img.shape[int(axis == "x")])[has_value][-1] + 1
    return edge_start, edge_end


def mark_connected_component(image, start_point):
    """
    Finds the connected pixels of the center pixel
    """
    rows, cols = image.shape
    visited = np.zeros_like(image, dtype=bool)
    component = np.zeros_like(image, dtype=np.uint8)
    
    stack = [start_point]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while stack:
        x, y = stack.pop()
        if visited[x, y] or not image[x, y]:
            continue
        visited[x, y] = True
        component[x, y] = 255  # Mark the component in the new image
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and image[nx, ny]:
                stack.append((nx, ny))
    
    return component


def crop_img(img_path, target_path, threshold, iterations):
    """
    Performs erosion on the mask of the image, selects largest connected component,
    dialates the largest connected component

    input:
        - img_path: location of the image to be cropped

    output: 
        - img: cropped image
        - window_location: location of cropping window w.r.t. original dicom image so that segmentation
           map can be cropped in the same way for training.
        
    """
    img = np.array(imageio.imread(img_path))
    print(img)

    img_mask = img > threshold
    img_mask_clean = img_mask

    # Erosion in order to remove thin lines in the background
    img_mask = scipy.ndimage.morphology.binary_erosion(img_mask, iterations=iterations)

    # Select mask for largest connected component
    largest_mask = get_mask_of_largest_connected_component(img_mask)

    # Dilation to recover the original mask, excluding the thin lines
    largest_mask = scipy.ndimage.morphology.binary_dilation(largest_mask, iterations=iterations)

    # figure out where to crop
    y_edge_top, y_edge_bottom = get_edge_values(img, largest_mask, "y")
    x_edge_left, x_edge_right = get_edge_values(img, largest_mask, "x")

    # figure out the center of cropping and mark the connected componenet in the orginal mask
    x, y = (x_edge_left+x_edge_right)//2,(y_edge_top+y_edge_bottom)//2
    component = mark_connected_component(img_mask_clean, [x,y])

    # Generate the output png
    result = np.zeros_like(img)
    result[component == 255] = img[component == 255]
    imageio.imwrite(target_path, result, format='png')
    # return (y_edge_top, y_edge_bottom, x_edge_left, x_edge_right)
    return


def main():
    parser = argparse.ArgumentParser(description='Remove background of image and save cropped files')
    parser.add_argument('--img_path', default='./IM00567.png')
    parser.add_argument('--target_path', default='./cropped_image/test.png')
    # parser.add_argument('--data_path', default='./new/new.pkl')
    parser.add_argument('--threshold', default=71, type=int)
    parser.add_argument('--num-iterations', default=100, type=int)
    args = parser.parse_args()

    crop_img(
        img_path=args.img_path,
        target_path=args.target_path,
        # data_path=args.data_path,
        threshold=args.threshold,
        iterations=args.num_iterations
    )


if __name__ == "__main__":
    main()