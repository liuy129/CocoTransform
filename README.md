# CocoTransform

Utility for transforming COCO-format segmentation labels and images.
Supports mask decoding/encoding, cropping, padding, broadcasting, bounding box transformation,
and mask rotation. Can be used to preprocess COCO datasets for training or analysis.

Assume that source mask is a hybrid image that stitches multiple-coregistered-source-image side by side. 
This script is then to map a segmentation mask and bounding box from source coordinates to target coordinates.
This is useful when splitting or merging image regions (e.g., for multi-source/multi-target scenarios).

Example:
    # Suppose you have a mask of shape (100, 200) and want to merge 2 source splits into 1 target split.
    mask = np.ones((100, 200), dtype=np.uint8)
    bbox = [50, 10, 100, 80]
    new_mask, new_bbox = map_mask_to_target_coords(
        mask, bbox, src_idx=0, src_width=200, tgt_idx=0, tgt_width=200,
        src_num=2, tgt_num=1, total_height=100, mode="multi2single"
    )
    # new_mask will be the merged mask, new_bbox will be the mapped bounding box.


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
