# CocoTransform

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
