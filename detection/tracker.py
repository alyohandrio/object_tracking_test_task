"""This file contains Tracker responsible for video processing."""

from __future__ import annotations

import torch
from detection.boxes import Boxes
from detection.utils import get_file
from detection.video import VideoReader, VideoWriter
from tqdm.auto import trange
from ultralytics import YOLO


class Tracker:
    """Class responsible for tracking objects and drawing bounding boxes."""

    def __init__(
        self, model_path: str, model_url: str = None, classes: list[int] = None
    ):
        """Download model if needed and create new Tracker.

        Args:
            model_path (str): either path to existing model or
              path where model downloaded from model_url will be saved.
            model_url (str | None): url to download model from.
            classes (list[int] | None): detect only these classes.
        """
        if model_url is not None:
            get_file(model_url, model_path)
        self.model = YOLO(model_path)
        self.classes = classes

    def to(self, device: torch.device):
        """Move model to device.

        Args:
            device (torch.device): target device
        """
        self.model = self.model.to(device)

    def run_detection(
        self,
        video_path: str,
        save_path: str,
        track_args: dict = {"verbose": False}
    ):
        """Run tracking with model, save resulting tensors to save_path.

        Args:
            video_path (str): path to video to process.
            save_path (str): path where resulting tensors will be saved.
            track_args (dict): additional args for model.
        """
        video = VideoReader(video_path)
        boxes = []
        starts = [0]
        total_num_bboxes = 0
        length = video.get_length()

        if not video.isOpened():
            print("Failed to read video")
            return

        for iter in trange(length, desc="Detecting"):
            success, frame = video.read()
            if success:
                results = self.model.track(
                    frame, persist=True, classes=self.classes, **track_args
                )
                boxes.append(results[0].boxes.data)
                total_num_bboxes += boxes[-1].shape[0]
                starts.append(total_num_bboxes)
            else:
                break

        boxes = torch.cat(boxes)
        unique, inv = torch.unique(
            boxes[:, 4],
            sorted=True,
            return_inverse=True
        )
        boxes[:, 4] = inv
        print(f"Found total of {unique.shape[0]} unique objects")
        torch.save({"boxes": boxes, "starts": torch.tensor(starts)}, save_path)
        video.release()

    def add_boxes(
        self,
        video_path: str,
        boxes_path: str,
        save_path: str,
        plot_args: dict = {},
        ids=None,
    ):
        """Add selected bounding boxes to video.

        Args:
            video_path (str): path to original video.
            boxes_path (str): path to bboxes tensors
              (created in run_detection).
            save_path (str): path to store resulting video.
            plot_args (dict): additional args for Boxes.plot function.
            ids (list[int] | None): list of object's ids to plot
              (by default all bboxes are drawn).
        """
        video = VideoReader(video_path)
        writer = VideoWriter(video, save_path)
        length = video.get_length()
        boxes_data = torch.load(boxes_path)
        pred_boxes = boxes_data["boxes"]
        starts = boxes_data["starts"]

        if not video.isOpened():
            print("Failed to read video")

        for iter in trange(length, desc="Visualizing"):
            success, frame = video.read()
            if success:
                cur_boxes = pred_boxes[starts[iter]:starts[iter + 1]]
                cur_boxes = Boxes(cur_boxes, ids)
                cur_boxes.plot(frame, self.model.names, **plot_args)
                writer.write(frame)
            else:
                break

        video.release()
        writer.release()

