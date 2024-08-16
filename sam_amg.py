import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import argparse
import json
import os
from typing import Any, Dict, List
import numpy as np
import shutil



SCANNET_COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                        6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                        13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}


parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as either PNGs or COCO-style RLEs. Requires open-cv, "
        "as well as pycocotools if saving in RLE format."
    )
)


parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM checkpoint to use for mask generation.",
)

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")

parser.add_argument(
    "--convert-to-rle",
    action="store_true",
    help=(
        "Save masks as COCO RLEs in a single json instead of as a folder of PNGs. "
        "Requires pycocotools."
    ),
)

amg_settings = parser.add_argument_group("AMG Settings")

amg_settings.add_argument(
    "--points-per-side",
    type=int,
    default=None,
    help="Generate masks by sampling a grid over the image with this many points to a side.",
)

amg_settings.add_argument(
    "--points-per-batch",
    type=int,
    default=None,
    help="How many input points to process simultaneously in one batch.",
)

amg_settings.add_argument(
    "--pred-iou-thresh",
    type=float,
    default=None,
    help="Exclude masks with a predicted score from the model that is lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-thresh",
    type=float,
    default=None,
    help="Exclude masks with a stability score lower than this threshold.",
)

amg_settings.add_argument(
    "--stability-score-offset",
    type=float,
    default=None,
    help="Larger values perturb the mask more when measuring stability score.",
)

amg_settings.add_argument(
    "--box-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding a duplicate mask.",
)

amg_settings.add_argument(
    "--crop-n-layers",
    type=int,
    default=None,
    help=(
        "If >0, mask generation is run on smaller crops of the image to generate more masks. "
        "The value sets how many different scales to crop at."
    ),
)

amg_settings.add_argument(
    "--crop-nms-thresh",
    type=float,
    default=None,
    help="The overlap threshold for excluding duplicate masks across different crops.",
)

amg_settings.add_argument(
    "--crop-overlap-ratio",
    type=int,
    default=None,
    help="Larger numbers mean image crops will overlap more.",
)

amg_settings.add_argument(
    "--crop-n-points-downscale-factor",
    type=int,
    default=None,
    help="The number of points-per-side in each layer of crop is reduced by this factor.",
)

amg_settings.add_argument(
    "--min-mask-region-area",
    type=int,
    default=None,
    help=(
        "Disconnected mask regions or holes with area smaller than this value "
        "in pixels are removed by postprocessing."
    ),
)



def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    anns = masks.copy()
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        # cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    # metadata_path = os.path.join(path, "metadata.csv")
    # with open(metadata_path, "w") as f:
    #     f.write("\n".join(metadata))
    

    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    

    img_color = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img_color[:,:,3] = 1.0
    
    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))
    part_count = 0

    color_arr = np.arange(20)
    np.random.shuffle(color_arr)

    for ann in sorted_anns:
        m = ann['segmentation']

        # remove small parts
        if ann['area'] >= 40:
            # color_mask = np.concatenate([np.random.random(3), [1.0]])
            cmap_ind = color_arr[part_count % 20]
            color_mask = np.concatenate([np.array(SCANNET_COLOR_MAP_20[cmap_ind]) / 255.0, [1.0]])
            img_color[m] = color_mask
            img[m] = part_count
            part_count += 1
    img[img < 0] = part_count  # set unmasked pixels to the largest index

    return img, img_color
    


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.cuda()
    output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)


    obj_lists = sorted(os.listdir("output/mvimgs"), key=str.lower)
    img_type_flag, img_ext = 'sync', '.png'
    
    img_path_dir = 'output/mvimgs'

    for obj_name in obj_lists:
        obj_path = os.path.join(img_path_dir, obj_name, '0' + img_ext)
        sam_views = []
        sam_views_color = []
        rgb_views = []

        print(f"Processing '{obj_name}' ...")
        image = cv2.imread(obj_path, -1)
        if image.shape[2] == 4:   # with alpha channel
            image = image / 255.0
            image = image[...,:3]*image[...,-1:] + (1.-image[...,-1:])
            image = (image * 255.0).astype(np.uint8)
        if image is None:
            print(f"Could not load '{obj_path}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_views.append(image)
        targets = np.split(image, 6, axis=1) if 'w3d' in img_type_flag else np.split(image, 16, axis=1)

        view_id = 0
        for t in targets:
            masks = generator.generate(t)

            save_base = os.path.join(img_path_dir, obj_name)    

            if output_mode == "binary_mask":
                os.makedirs(save_base, exist_ok=True)
                
                cur_sam_mask, cur_sam_mask_color = write_masks_to_folder(masks, save_base)
                sam_views.append(cur_sam_mask)
                sam_views_color.append(cur_sam_mask_color)
            else:
                save_file = save_base + ".json"
                with open(save_file, "w") as f:
                    json.dump(masks, f)
            view_id += 1
        
        sam_mask_full = np.concatenate(sam_views, axis=1)
        sam_mask_full_color = (np.concatenate(sam_views_color, axis=1) * 255).astype(np.uint8)
        rgb_views = np.concatenate(rgb_views, axis=1)
        np.save(os.path.join(save_base, f'mvout.npy'), sam_mask_full.astype(np.int16))
        cv2.imwrite(os.path.join(save_base, 'fusemask.png'), sam_mask_full_color[:,:,[2,1,0]])
        shutil.copyfile(os.path.join(img_path_dir, obj_name, '0' + img_ext), os.path.join(save_base, f'mvout.png'))
    print("Done!")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
