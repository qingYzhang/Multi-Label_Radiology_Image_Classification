import os
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
# MMOCR
from mmocr.apis.inferencers import MMOCRInferencer
from mmocr.utils import poly2bbox
# SAM
from segment_anything import SamPredictor, sam_model_registry


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--inputs',
        type=str,
        default=
        '../our_knee/knee_png_sample_L/',
        help='Input image file or folder path.')
    parser.add_argument(
        '--outdir',
        type=str,
        default='./new/',
        help='Output directory of results.')
    # MMOCR parser
    parser.add_argument(
        '--det',
        type=str,
        default=
        'mmocr_dev/configs/textdet/dbnetpp/dbnetpp_swinv2_base_w16_in21k.py',
        help='Pretrained text detection algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--det-weights',
        type=str,
        # required=True,
        default='checkpoints/mmocr/db_swin_mix_pretrain.pth',
        help='Path to the custom checkpoint file of the selected det model.')
    parser.add_argument(
        '--rec',
        type=str,
        default='mmocr_dev/configs/textrecog/abinet/abinet_20e_st-an_mj.py',
        help='Pretrained text recognition algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--rec-weights',
        type=str,
        # required=True,
        default=
        'checkpoints/mmocr/abinet_20e_st-an_mj_20221005_012617-ead8c139.pth',
        help='Path to the custom checkpoint file of the selected recog model.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    # SAM Parser
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        # required=True,
        default='checkpoints/sam/sam_vit_h_4b8939.pth',
        help="path to checkpoint file")
    parser.add_argument(
        "--sam_type",
        type=str,
        default='vit_h',
        help="path to checkpoint file")
    parser.add_argument(
        "--show", action='store_true', help="whether to show the result")
    args = parser.parse_args()
    return args


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# # Function to merge predictions into the result dictionary
# def merge_predictions(all_results, new_predictions):
#     if 'rec_texts' in new_predictions:
#         all_results['rec_texts'].extend(new_predictions['rec_texts'])
#     if 'rec_scores' in new_predictions:
#         all_results['rec_scores'].extend(new_predictions['rec_scores'])
#     if 'det_polygons' in new_predictions:
#         all_results['det_polygons'].extend(new_predictions['det_polygons'])
#     if 'det_scores' in new_predictions:
#         all_results['det_scores'].extend(new_predictions['det_scores'])

if __name__ == '__main__':
    # Build MMOCR
    args = parse_args()
    mmocr_inferencer = MMOCRInferencer(
        args.det,
        args.det_weights,
        args.rec,
        args.rec_weights,
        device=args.device)
    # Build SAM
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    sam_predictor = SamPredictor(sam)
    # Run
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    ori_inputs = mmocr_inferencer._inputs_to_list(args.inputs)
    for i, ori_input in enumerate(ori_inputs):
        print(f'Processing {ori_input} [{i + 1}]/[{len(ori_inputs)}]')
        begin = time.time()
        img = cv2.imread(ori_input)
        
        # Initialize an empty dictionary to store all results
        all_results = {
            'rec_texts': [],
            'rec_scores': [],
            'det_polygons': [],
            'det_scores': []
        }

        # Set the rotation angle
        angle = 45

        for i in range(16):
            if i > 7:
                img = cv2.flip(img, 1)

            # Set the center point of the image
            center = (img.shape[1] // 2, img.shape[0] // 2)
            
            # Get the rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle * i, 1.0)
            
            # Rotate the image
            rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
            
            # MMOCR inference
            result = mmocr_inferencer(rotated_img, save_vis=True, out_dir=args.outdir)['predictions'][0]
            print(result)
            rec_texts = result['rec_texts']
            if "l" in rec_texts or "r" in rec_texts or "left" in rec_texts or "lt" in rec_texts or "rt" in rec_texts or "radiology" in rec_texts:
                img = rotated_img
                break
            # # Merge the new predictions with the previous results
            # merge_predictions(all_results, result)

        #TODO
        #LEFT LT RT study exam


        # Now `all_results` contains the merged predictions from all rotations
        # print(all_results)

        # # Set the rotation angle and center point
        # angle = 45  # Rotate by 45 degrees
        # center = (img.shape[1] // 2, img.shape[0] // 2)  # Center of the image

        # # Get the rotation matrix
        # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # # Rotate the image
        # rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))

        # MMOCR inference
        # result = mmocr_inferencer(
        #     img, save_vis=True, out_dir=args.outdir)['predictions'][0]
        # print(result)
        t = time.time() - begin
        print("{} image(s) take {:.2f}s".format(i, t))
        rec_texts = result['rec_texts']
        if rec_texts != [] and ("l" in rec_texts or "r" in rec_texts or "left" in rec_texts or "lt" in rec_texts or "rt" in rec_texts or "radiology" in rec_texts):
            if "l" in rec_texts or "left" in rec_texts or "lt" in rec_texts:\
                lr = "l"
            else:
                lr = "r"
            det_polygons = result['det_polygons']
            det_bboxes = torch.tensor(np.array([poly2bbox(poly) for poly in det_polygons]),
                                    device=sam_predictor.device)
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(
                det_bboxes, img.shape[:2])
            # SAM inference
            sam_predictor.set_image(img, image_format='BGR')
            masks, _, _ = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )
            # Show results
            plt.figure(figsize=(10, 10))
            # convert img to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            for mask, rec_text, polygon in zip(masks, rec_texts, det_polygons):
                show_mask(mask.cpu(), plt.gca(), random_color=True)
                polygon = np.array(polygon).reshape(-1, 2)
                # convert polygon to closed polygon
                polygon = np.concatenate([polygon, polygon[:1]], axis=0)
                plt.plot(polygon[:, 0], polygon[:, 1], color='r', linewidth=2)
                plt.text(polygon[0, 0], polygon[0, 1], rec_text, color='r')
            if args.show:
                plt.show()
            print(type(ori_input))
            parts = ori_input.split('/')
            dir = parts[-1]
            res = dir.split(".")[0]
            plt.savefig(os.path.join(args.outdir, f'{res}.png'))
            with open ("result.txt", 'a') as file:
                file.write(ori_input+" "+lr+"\n")
        else:
            # print("hahah")
            if not ("exam" in rec_texts or "study" in rec_texts):
                with open("none.txt", 'a') as file:
                    file.write(ori_input+str(rec_texts != [])+str(("l" in rec_texts or "r" in rec_texts or "left" in rec_texts or "lt" in rec_texts or "rt" in rec_texts or "radiology" in rec_texts))+"\n")
