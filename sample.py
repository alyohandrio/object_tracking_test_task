"""Creates new config with sampled objects."""

import argparse
import json

import torch
from detection.boxes import Boxes


def main(config, out_path, num_objects):
    """Create new config for visualization with num_objects objects."""
    if "tracking" in config:
        del config["tracking"]

    boxes_data = torch.load(config["bboxes_path"])
    pred_boxes = Boxes(boxes_data["boxes"])
    all_ids = torch.unique(pred_boxes.get_ids())
    generator = torch.Generator()
    generator.manual_seed(42)
    ids = torch.randperm(all_ids.shape[0], generator=generator)[:num_objects]

    config["visualization"]["ids"] = ids.tolist()
    save_path = f"sample_{num_objects}_" + config["visualization"]["save_path"]
    config["visualization"]["save_path"] = save_path
    with open(out_path, "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample ids")
    parser.add_argument(
        "-i",
        "--in_config",
        default="config.json",
        type=str,
        help="in config file path",
    )
    parser.add_argument(
        "-o",
        "--out_config",
        default="sample_config.json",
        type=str,
        help="out config file path",
    )
    parser.add_argument(
        "-n",
        "--num_objects",
        type=int,
        help="num objects to sample",
    )
    args = parser.parse_args()

    with open(args.in_config, "r") as f:
        config = json.load(f)

    main(config, args.out_config, args.num_objects)

