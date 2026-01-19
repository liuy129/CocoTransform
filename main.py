"""
transformCocoLabels.py

Utility for transforming COCO-format segmentation labels and images.
Supports mask decoding/encoding, cropping, padding, broadcasting, bounding box transformation,
and mask rotation. Can be used to preprocess COCO datasets for training or analysis.

Usage:
    python transformCocoLabels.py --input_coco <input.json> --output_coco <output.json> [options]

Options:
    --image_folder <folder>         Optional folder containing images.
    --src_num_sources <int>         Number of source splits per image (default: 2).
    --tgt_num_sources <int>         Number of target splits per image (default: 1).
    --remove_product_during_search  Remove product name suffix from filenames (default: False).
    --product_name <str>            Product name string to add/remove (default: "").
    --add_product_to_output_filename Add product name to output filenames (default: False).
    --expected_tgt_ext <str>        Output file extension (default: ".jpeg").
    --rotate_angle_cw <int>         Optional rotation angle in degrees (clockwise).
    --visualize                     Show mask/bbox visualizations (default: False).

Example:
    python transformCocoLabels.py --input_coco input.json --output_coco output.json --src_num_sources 2 --tgt_num_sources 1

Author: Yizhen Liu
License: MIT
"""

import argparse
import json
import copy
import numpy as np
from pycocotools import mask as maskUtils
import os
from pathlib import Path
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def decode_mask_once(ann, height, width):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rles = maskUtils.frPyObjects(seg, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(seg, dict) and "size" in seg and "counts" in seg:
        if isinstance(seg["counts"], bytes) or isinstance(seg["counts"], str):
            rle = seg
        elif isinstance(seg["counts"], list):
            rle = maskUtils.frPyObjects([seg], height, width)[0]
        else:
            raise TypeError(f"Unknown counts type: {type(seg['counts'])}")
    elif isinstance(seg, dict) and "counts" in seg:
        rle = seg
    else:
        raise ValueError("Unknown segmentation format: {}".format(type(seg)))
    return maskUtils.decode(rle)

def encode_mask(mask):
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

def crop_mask(mask, x0, x1):
    return mask[:, x0:x1]

def pad_mask(mask, x0, x1, total_width):
    H = mask.shape[0]
    out = np.zeros((H, total_width), dtype=mask.dtype)
    out[:, x0:x1] = mask
    return out

def broadcast_to_width(arr, target_width):
    H, W = arr.shape
    if W == target_width:
        return arr.copy()
    elif W > target_width:
        return arr[:, :target_width]
    else:
        out = np.zeros((H, target_width), dtype=arr.dtype)
        out[:, :W] = arr
        return out

def make_image_entry(image_id, file_name, width, height):
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height
    }

def reindex_annotation(ann, ann_id, image_id):
    ann["id"] = ann_id
    ann["image_id"] = image_id
    return ann

def remove_product_name(filename, product_name):
    base, ext = os.path.splitext(filename)
    if base.endswith(product_name):
        base = base[:-len(product_name)]
    return base + ext

def add_product_name(filename, product_name):
    base, ext = os.path.splitext(filename)
    if not base.endswith(product_name):
        base += product_name
    return base + ext

def visualize_masks(
    before_mask, 
    after_mask, 
    title_before="Before", 
    title_after="After", 
    bbox_before=None, 
    bbox_after=None
):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(before_mask, cmap='gray')
    axs[0].set_title(title_before)
    axs[0].axis('off')
    if bbox_before is not None:
        x, y, w, h = bbox_before
        rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        axs[0].add_patch(rect)
    axs[1].imshow(after_mask, cmap='gray')
    axs[1].set_title(title_after)
    axs[1].axis('off')
    if bbox_after is not None:
        if isinstance(bbox_after, list) and all(isinstance(b, (list, tuple)) and len(b) == 4 for b in bbox_after):
            for bbox in bbox_after:
                x, y, w, h = bbox
                rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                axs[1].add_patch(rect)
        else:
            x, y, w, h = bbox_after
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            axs[1].add_patch(rect)
    plt.tight_layout()
    plt.show()

def replace_file_extension(filename, new_ext):
    if not new_ext:
        return filename
    base, _ = os.path.splitext(filename)
    return base + new_ext

# ------------------------------------------------------------
# Transformation logic
# ------------------------------------------------------------

def map_mask_to_target_coords(
    mask, bbox, src_idx, src_width, tgt_idx, tgt_width, 
    src_num, tgt_num, total_height, mode
):
    def get_indices(idx, width, num):
        x0 = int(round(idx * width / num))
        x1 = int(round((idx + 1) * width / num))
        return x0, x1

    slices = []
    for idx in range(src_num):
        x0, x1 = get_indices(idx, src_width, src_num)
        slices.append(mask[:, x0:x1])
    merged_mask = slices[0].copy()
    for slice_mask in slices[1:]:
        merged_mask = np.bitwise_or(merged_mask, slice_mask)

    if merged_mask.shape[1] < tgt_width:
        new_mask = np.zeros((total_height, tgt_width), dtype=merged_mask.dtype)
        new_mask[:, :merged_mask.shape[1]] = merged_mask
    else:
        new_mask = merged_mask[:, :tgt_width]

    src_x0, src_x1 = get_indices(src_idx, src_width, src_num)
    tgt_x0, tgt_x1 = get_indices(tgt_idx, tgt_width, tgt_num)

    new_bboxes = []
    for idx in range(src_num):
        slice_x0, slice_x1 = get_indices(idx, src_width, src_num)
        inter_x0 = max(bbox[0], slice_x0)
        inter_x1 = min(bbox[0] + bbox[2], slice_x1)
        if inter_x1 > inter_x0:
            rel_x0 = inter_x0 - slice_x0
            rel_x1 = inter_x1 - slice_x0
            tgt_slice_x0, tgt_slice_x1 = get_indices(tgt_idx, tgt_width, tgt_num)
            mapped_x0 = tgt_slice_x0 + rel_x0
            mapped_x1 = tgt_slice_x0 + rel_x1
            new_bbox = [mapped_x0, bbox[1], mapped_x1 - mapped_x0, bbox[3]]
            new_bboxes.append(new_bbox)

    final_bbox = new_bboxes if new_bboxes else [[0, 0, 0, 0]]
    return new_mask, final_bbox

def rotate_mask_per_instance(mask, angle, visualize=False):
    import math
    if angle is None or angle == 0:
        return mask.copy()

    def get_rotated_shape(h, w, angle_deg):
        angle_rad = np.deg2rad(angle_deg % 360)
        abs_cos = abs(np.cos(angle_rad))
        abs_sin = abs(np.sin(angle_rad))
        new_w = int(np.round(h * abs_sin + w * abs_cos))
        new_h = int(np.round(h * abs_cos + w * abs_sin))
        return new_h, new_w

    unique_values = np.unique(mask)
    h, w = mask.shape
    rot_h, rot_w = get_rotated_shape(h, w, angle)
    canvas = np.zeros((rot_h, rot_w), dtype=np.uint8)

    for val in unique_values:
        if val == 0:
            continue
        instance_mask = (mask == val).astype(np.uint8) * 255
        if visualize:
            plt.figure()
            plt.title(f"instance mask for value {val}")
            plt.imshow(instance_mask, cmap='gray')
            plt.axis('off')
            plt.show()
        (h, w) = instance_mask.shape
        pad = max(h, w)
        padded = np.zeros((pad, pad), dtype=np.uint8)
        y_off = (pad - h) // 2
        x_off = (pad - w) // 2
        padded[y_off:y_off+h, x_off:x_off+w] = instance_mask
        center = (pad // 2, pad // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated_padded = cv2.warpAffine(
            padded, M, (pad, pad),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        mask_thresh = (rotated_padded > 0).astype(np.uint8) * val
        mask_thresh_cropped = (
            mask_thresh[x_off: mask_thresh.shape[1]-x_off-1, y_off: mask_thresh.shape[0]-y_off]
            if angle == 90 or angle == 270
            else mask_thresh[y_off: mask_thresh.shape[0]-y_off, x_off: mask_thresh.shape[1]-x_off-1]
        ) if mask_thresh.shape[0] > 0 and mask_thresh.shape[1] > 0 else mask_thresh
        if visualize:
            plt.figure()
            plt.title(f"rotated_instance for value {val}")
            plt.imshow(mask_thresh_cropped, cmap='gray')
            plt.axis('off')
            plt.show()
        ch, cw = mask_thresh_cropped.shape
        canvas[:ch, :cw] = np.where(mask_thresh_cropped == val, val, canvas[:ch, :cw])
    return canvas

def rotate_bbox(bbox, image_shape, angle):
    x, y, w, h = bbox
    H, W = image_shape[:2]
    angle = angle % 360
    corners = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ])
    cx, cy = W / 2, H / 2
    corners_centered = corners - np.array([cx, cy])
    if angle == 90:
        rot_mat = -np.array([[0, 1], [-1, 0]])
        new_W, new_H = H, W
    elif angle == 180:
        rot_mat = np.array([[-1, 0], [0, -1]])
        new_W, new_H = W, H
    elif angle == 270:
        rot_mat = -np.array([[0, -1], [1, 0]])
        new_W, new_H = H, W
    elif angle == 0:
        return [x, y, w, h]
    else:
        raise ValueError("Only 0, 90, 180, 270 degree rotations are supported.")
    rotated = np.dot(corners_centered, rot_mat.T)
    if angle in [90, 270]:
        ncx, ncy = new_W / 2, new_H / 2
    else:
        ncx, ncy = cx, cy
    rotated += np.array([ncx, ncy])
    x_min, y_min = rotated.min(axis=0)
    x_max, y_max = rotated.max(axis=0)
    new_bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
    return new_bbox

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def load_coco_json(input_coco):
    logger.info("Loading COCO file")
    with open(input_coco, "r") as f:
        coco = json.load(f)
    return coco["images"], coco["annotations"], coco["categories"]

def get_source_filenames(image_folder, exts, remove_product_during_search, product_name):
    source_filenames = [
        f.name for f in Path(image_folder).iterdir()
        if f.is_file() and f.suffix.lower() in exts
    ]
    source_filenames.sort()
    if remove_product_during_search:
        source_filenames = [
            remove_product_name(fn, product_name)
            for fn in source_filenames
        ]
    return source_filenames

def build_image_dim_map(images):
    return {os.path.splitext(img["file_name"])[0]: (img["width"], img["height"]) for img in images}

def build_output_images(
    source_filenames, image_dim_map, images, src_num_sources, tgt_num_sources,
    add_product_to_output_filename, product_name, expected_tgt_ext
):
    filename_to_image_id = {}
    images_out = {}
    for idx, filename in enumerate(source_filenames):
        img_dims = image_dim_map.get(os.path.splitext(filename)[0], (images[0]["width"], images[0]["height"]))
        img_width, img_height = img_dims
        width_out = (img_width // src_num_sources) * tgt_num_sources
        height_out = img_height
        out_image_id = idx + 1
        if add_product_to_output_filename:
            filename_out = add_product_name(filename, product_name)
        else:
            filename_out = filename
        filename_out = replace_file_extension(filename_out, expected_tgt_ext)
        if tgt_num_sources > 1:
            for tgt_idx in range(tgt_num_sources):
                base, ext = os.path.splitext(filename_out)
                indexed_filename = f"{base}_{tgt_idx}{ext}"
                image_id = out_image_id * tgt_num_sources + tgt_idx
                images_out[image_id] = make_image_entry(
                    image_id=image_id,
                    file_name=indexed_filename,
                    width=width_out,
                    height=height_out
                )
                filename_to_image_id[indexed_filename] = image_id
        else:
            images_out[out_image_id] = make_image_entry(
                image_id=out_image_id,
                file_name=filename_out,
                width=width_out,
                height=height_out
            )
            filename_to_image_id[filename_out] = out_image_id
    return images_out, filename_to_image_id

def determine_transform_mode(src_num_sources, tgt_num_sources):
    if src_num_sources == 1 and tgt_num_sources == 1:
        return "single2single"
    elif src_num_sources > 1 and tgt_num_sources == 1:
        return "multi2single"
    elif src_num_sources == 1 and tgt_num_sources > 1:
        return "single2multi"
    elif src_num_sources > 1 and tgt_num_sources > 1:
        return "multi2multi"
    else:
        raise ValueError("Invalid source/target num_sources combination.")

def process_annotations(
    annotations, image_meta, src_num_sources, tgt_num_sources, mode, filename_to_image_id,
    remove_product_during_search, product_name, add_product_to_output_filename, expected_tgt_ext,
    rotate_angle_cw, visualize
):
    annotations_out = []
    image_to_annotations = defaultdict(list)
    ann_id_counter = 1
    for ann in annotations:
        image_id = ann["image_id"]
        img = image_meta[image_id]
        H = img["height"]
        W = img["width"]
        tgt_width = (W // src_num_sources) * tgt_num_sources
        decoded_mask = decode_mask_once(ann, H, W)
        bbox = ann["bbox"]
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        x_center = bbox_x + bbox_w * 0.5
        if src_num_sources > 1:
            src_idx = int(x_center // W)
        else:
            src_idx = 0
        if tgt_num_sources > 1:
            tgt_indices = range(tgt_num_sources)
        else:
            tgt_indices = [0]
        for tgt_idx in tgt_indices:
            new_mask, new_bbox = map_mask_to_target_coords(
                decoded_mask.copy(), bbox.copy(), src_idx, W, tgt_idx, tgt_width,
                src_num_sources, tgt_num_sources, H, mode=mode
            )
            if tgt_num_sources > 1 and tgt_idx > 0:
                if 'new_bbox_first' in locals():
                    new_bbox = new_bbox_first
            elif tgt_num_sources > 1:
                new_bbox_first = new_bbox.copy()
            if new_mask.sum() == 0:
                continue
            rotated_bbox = None
            if rotate_angle_cw is not None:
                rotated = rotate_mask_per_instance(new_mask, rotate_angle_cw, visualize=visualize)
                is_list_of_bboxes = isinstance(new_bbox, list) and all(isinstance(b, (list, tuple)) for b in new_bbox)
                rotated_bbox = [rotate_bbox(b, new_mask.shape, rotate_angle_cw) for b in new_bbox] if is_list_of_bboxes else rotate_bbox(new_bbox, new_mask.shape, rotate_angle_cw)
                new_bbox = rotated_bbox
            else:
                rotated = new_mask
            rle = encode_mask(rotated)
            new_ann = ann.copy()
            new_ann["segmentation"] = rle
            new_ann["bbox"] = new_bbox
            new_ann["area"] = float(maskUtils.area(rle))
            new_ann["iscrowd"] = int(new_ann.get("iscrowd", 0))
            orig_filename = img["file_name"]
            if remove_product_during_search:
                orig_filename = remove_product_name(orig_filename, product_name)
            src_base = orig_filename
            if add_product_to_output_filename:
                out_filename = add_product_name(src_base, product_name)
            else:
                out_filename = src_base
            out_filename = replace_file_extension(out_filename, expected_tgt_ext)
            if tgt_num_sources > 1:
                base, ext = os.path.splitext(out_filename)
                out_filename = f"{base}_{tgt_idx}{ext}"
            out_image_id = filename_to_image_id.get(out_filename)
            if out_image_id is None:
                continue
            new_ann = reindex_annotation(new_ann, ann_id_counter, out_image_id)
            ann_id_counter += 1
            annotations_out.append(new_ann)
            image_to_annotations[out_image_id].append(new_ann["id"])
    return annotations_out, image_to_annotations

def save_coco_json(output_coco, images_out, annotations_out, categories, image_to_annotations):
    for img_id, img_entry in images_out.items():
        img_entry["annotation_ids"] = image_to_annotations.get(img_id, [])
    coco_out = {
        "images": list(images_out.values()),
        "annotations": annotations_out,
        "categories": categories
    }
    with open(output_coco, "w") as f:
        json.dump(coco_out, f)
    logger.info(
        "Saved %d annotations for %d images",
        len(annotations_out),
        len(images_out)
    )

def main(
    input_coco,
    output_coco,
    image_folder=None,
    src_num_sources=2,
    tgt_num_sources=1,
    remove_product_during_search=False,
    product_name="",
    add_product_to_output_filename=False,
    expected_tgt_ext = ".jpeg",
    rotate_angle_cw = None,
    visualize=False
):
    # Load COCO data
    images, annotations, categories = load_coco_json(input_coco)
    image_meta = {img["id"]: img for img in images}

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    if image_folder is not None:
        source_filenames = get_source_filenames(
            image_folder, exts, remove_product_during_search, product_name
        )
    else:
        # Use filenames from COCO images
        source_filenames = [img["file_name"] for img in images]
        if remove_product_during_search:
            source_filenames = [
                remove_product_name(fn, product_name)
                for fn in source_filenames
            ]

    # Build image dimension map
    image_dim_map = build_image_dim_map(images)

    # Build output images and filename-to-id mapping
    # this is aimed to deal with thescenario where the filename might be different on CVAT compared to what is stored in COCO
    images_out, filename_to_image_id = build_output_images(
        source_filenames, image_dim_map, images, src_num_sources, tgt_num_sources,
        add_product_to_output_filename, product_name, expected_tgt_ext
    )

    # Determine transformation mode
    mode = determine_transform_mode(src_num_sources, tgt_num_sources)
    logger.info(f"Transform mode: {mode}")

    # Process annotations
    annotations_out, image_to_annotations = process_annotations(
        annotations, image_meta, src_num_sources, tgt_num_sources, mode, filename_to_image_id,
        remove_product_during_search, product_name, add_product_to_output_filename, expected_tgt_ext,
        rotate_angle_cw, visualize
    )

    # Save output COCO JSON
    save_coco_json(output_coco, images_out, annotations_out, categories, image_to_annotations)



if __name__ == "__main__":
    main(
        input_coco=r"inputCocoPath",
        output_coco=r"OutputPath",
        image_folder=None,# image path for reference.
        src_num_sources=2,
        tgt_num_sources=1,
        remove_product_during_search=True,
        product_name="_appendix_before_file_extension",
        add_product_to_output_filename=True,
        expected_tgt_ext = ".jpeg",
        rotate_angle_cw = 0,
        visualize=False
    )
