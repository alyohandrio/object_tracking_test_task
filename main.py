import argparse
import json
import torch
from detection.tracker import Tracker

def main(config):
    """Run tracking and plot results according to config.
    
    If visualization section is not specified only tracking will be performed.
    If tracking section is not specified only visualization will be performed.
    """
    model_path = config["model_path"]
    model_url = config.get("model_url", None)
    classes = config.get("classes", None)
    video_path = config["video_path"]
    bboxes_path = config["bboxes_path"]
    device = config["device"]
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
    device = torch.device(device)
    
    tracker = Tracker(model_path, model_url=model_url, classes=classes)
    tracker.to(device)
    
    tracking_conf = config.get("tracking", None)
    if tracking_conf is not None:
        track_args = tracking_conf.get("track_args", {})
        tracker.run_detection(video_path, bboxes_path, track_args=track_args)
    
    visualization_conf = config.get("visualization", None)
    if visualization_conf is not None:
        plot_args = visualization_conf.get("plot_args", {})
        ids = visualization_conf.get("ids", None)
        out_path = visualization_conf["save_path"]
        tracker.add_boxes(video_path, bboxes_path, out_path, ids=ids, plot_args=plot_args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tracking programm")
    parser.add_argument(
        "-c",
        "--config",
        default="config.json",
        type=str,
        help="config file path",
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    main(config)
