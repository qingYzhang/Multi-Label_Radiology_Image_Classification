import os
import scipy
import argparse

import cv2 as cv
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from glob import glob

def get_masks_and_sizes_of_connected_components(img_mask):
    """
    Finds the connected components from the mask of the image
    """
    mask, num_labels = scipy.ndimage.label(img_mask)
    # print(mask, num_labels)
    mask_pixels_dict = {}
    for i in range(num_labels + 1):
        this_mask = (mask == i)
        print(img_mask[this_mask])
        # if len(img_mask[this_mask])==0
        if len(img_mask[this_mask]) != 0:
            if img_mask[this_mask][0] != 0:
                # Exclude the 0-valued mask
                mask_pixels_dict[i] = np.sum(this_mask)

    return mask, mask_pixels_dict

def get_mask_of_largest_connected_component(img_mask, component_rank=1):
    """
    Finds the largest connected component from the mask of the image.
    
    Parameters:
    img_mask (ndarray): The input image mask.
    component_rank (int): The rank of the component to find (1 for largest, 2 for second largest, etc.).
    
    Returns:
    ndarray: A mask with the largest connected component.
    """
    mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(img_mask)
    mask_pixels_series = pd.Series(mask_pixels_dict)
    
    # Get the index of the specified largest component
    sorted_indices = mask_pixels_series.sort_values(ascending=False).index
    # print(sorted_indices)
    if component_rank - 1 < len(sorted_indices):
        selected_mask_index = sorted_indices[component_rank - 1]
    else:
        raise ValueError("component_rank is out of bounds for the number of connected components")
    
    selected_mask = mask == selected_mask_index
    return selected_mask


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
        component[x, y] = 1  # Mark the component in the new image
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny] and image[nx, ny]:
                stack.append((nx, ny))
    
    return component


def include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size):
    """
    Includes buffer in all sides of the image in y-direction
    """
    if y_edge_top > 0:
        y_edge_top -= min(y_edge_top, buffer_size)
    if y_edge_bottom < img.shape[0]:
        y_edge_bottom += min(img.shape[0] - y_edge_bottom, buffer_size)
    return y_edge_top, y_edge_bottom


def include_buffer_x_axis(img, x_edge_left, x_edge_right, buffer_size):
    """
    Includes buffer in only one side of the image in x-direction
    """
    if x_edge_left > 0:
        x_edge_left -= min(x_edge_left, buffer_size)
    if x_edge_right < img.shape[1]:
        x_edge_right += min(img.shape[1] - x_edge_right, buffer_size)
    return x_edge_left, x_edge_right

def create_mask_and_crop(img, x_edge_left, x_edge_right, y_edge_top, y_edge_bottom):
    """
    Creates a mask the same size as the specified area and crops the original image to the specified area
    """
    mask_height = y_edge_bottom - y_edge_top
    mask_width = x_edge_right - x_edge_left
    mask = np.ones((mask_height, mask_width), dtype=img.dtype)
    
    cropped_img = img[y_edge_top:y_edge_bottom, x_edge_left:x_edge_right]
    
    return mask, cropped_img

def calculate_mask_percentage(mask):
    """
    calculate the masked pixel percentage in a mask
    """
    # Count the number of pixels in the mask with value 1
    mask_pixels = np.sum(mask == 1)
    
    # Calculate the total number of pixels in the image
    total_pixels = mask.size
    
    # Compute the percentage of mask coverage
    mask_percentage = (mask_pixels / total_pixels) * 100
    return mask_percentage

def msk_to_img(img, th_mask, buffer_size):
    # Get the rectangle of the component
    y_edge_top, y_edge_bottom = get_edge_values(img, th_mask, "y")
    x_edge_left, x_edge_right = get_edge_values(img, th_mask, "x")

    # include the buffer size
    # x_edge_left, x_edge_right = include_buffer_x_axis(img, x_edge_left, x_edge_right, buffer_size)
    # y_edge_top, y_edge_bottom = include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size)

    # Get the cropped img and mask
    mask, cropped_img = create_mask_and_crop(img, x_edge_left, x_edge_right, y_edge_top, y_edge_bottom)

    # Apply the mask to the cropped image
    result = cropped_img * mask
    return result, x_edge_left, x_edge_right, y_edge_top, y_edge_bottom   

def crop_img(img_path, target_path, iterations, gap, buffer_size):
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

    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img, (5, 5), 0)
    _, th_mask = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    
    output_path = 'binary_mask_org.png'
    imageio.imwrite(output_path, th_mask, format= "png")


    th_mask = scipy.ndimage.morphology.binary_erosion(th_mask, iterations=iterations)

    output_path = 'binary_mask_1.png'
    imageio.imwrite(output_path, th_mask.astype(np.uint8) * 255, format= "png")
    th_mask = scipy.ndimage.morphology.binary_dilation(th_mask, iterations=iterations+gap)

    output_path = 'binary_mask.png'
    imageio.imwrite(output_path, th_mask.astype(np.uint8) * 255, format= "png")


    # Get the largest connected componenet in the mask
    th_mask_1 = get_mask_of_largest_connected_component(th_mask, component_rank=1)
    try: 
        th_mask_2 = get_mask_of_largest_connected_component(th_mask, component_rank=2)
    except:
        mode = 1
    else:
        mode = 2
    # th_mask_3 = get_mask_of_largest_connected_component(th_mask, component_rank=3)
    # th_mask_4 = get_mask_of_largest_connected_component(th_mask, component_rank=4)


    # Get masked pixel percentage
    th_mask_1_percentage = calculate_mask_percentage(th_mask_1)
    if mode == 2:
        th_mask_2_percentage = calculate_mask_percentage(th_mask_2)
    # th_mask_3_percentage = calculate_mask_percentage(th_mask_3)
    # th_mask_4_percentage = calculate_mask_percentage(th_mask_4)

    # print(th_mask_1_percentage, th_mask_2_percentage)
    # print(th_mask_1_percentage/th_mask_2_percentage)
    result1, _, _, _, _ = msk_to_img(img, th_mask_1, buffer_size)

    if mode == 2 and 0.4 < th_mask_1_percentage/th_mask_2_percentage < 2.5:
            # print("two knee mode")
            result2, _, _, _, _ = msk_to_img(img, th_mask_2, buffer_size)
    else:
        mode = 1
    
    if mode == 2:
        base, ext = os.path.splitext(target_path)
        target_path_2 = f"{base}_2{ext}"
        imageio.imwrite(target_path, result1, format='png')
        imageio.imwrite(target_path_2, result2, format='png')
    else:
        imageio.imwrite(target_path, result1, format='png')


    # img_mask = img > threshold
    # img_mask_clean = img_mask 

    # # Select mask for largest connected component
    # largest_mask = get_mask_of_largest_connected_component(img_mask, component_rank=1)

    # # figure out where to crop
    # y_edge_top, y_edge_bottom = get_edge_values(img, largest_mask, "y")
    # x_edge_left, x_edge_right = get_edge_values(img, largest_mask, "x")

    # # figure out the center of cropping and mark the connected componenet in the orginal mask
    # x, y = (x_edge_left+x_edge_right)//2,(y_edge_top+y_edge_bottom)//2
    # component = mark_connected_component(img_mask_clean, [x,y])

    # # Generate the output png
    # result = np.zeros_like(img)
    # result[component == 1] = img[component == 1]
    # imageio.imwrite(target_path, result, format='png')
    # # return (y_edge_top, y_edge_bottom, x_edge_left, x_edge_right)
    return


def main():
    parser = argparse.ArgumentParser(description='Remove background of image and save cropped files')
    parser.add_argument('--img_path', default='./IM00013.png')
    parser.add_argument('--target_path', default='./cropped_image')
    # parser.add_argument('--data_path', default='./new/new.pkl')
    # parser.add_argument('--threshold', default=90, type=int)
    parser.add_argument('--num-iterations', default=5, type=int)
    parser.add_argument('--gap', default=10, type=int)
    parser.add_argument('--buffer_size', default=30, type=int)
    args = parser.parse_args()



    if os.path.isfile(args.img_path):  # Single image path
        img_paths = [args.img_path]
    elif os.path.isdir(args.img_path):  # Directory containing images
        img_paths = glob(os.path.join(args.img_path, '*.png'))  # Adjust file extension as needed
        print(img_paths)
    else:
        raise ValueError(f"Invalid path: {args.img_path}")

    os.makedirs(args.target_path, exist_ok=True)

    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        target_file = os.path.join(args.target_path, img_name)
        crop_img(
            img_path=img_path,
            target_path=target_file,
            # data_path=args.data_path,
            # threshold=args.threshold,
            iterations=args.num_iterations,
            gap=args.gap,
            buffer_size=args.buffer_size
        )

        # crop_img(
        #     img_path=args.img_path,
        #     target_path=args.target_path,
        #     # data_path=args.data_path,
        #     # threshold=args.threshold,
        #     iterations=args.num_iterations,
        #     buffer_size=args.buffer_size
        # )


if __name__ == "__main__":
    main()