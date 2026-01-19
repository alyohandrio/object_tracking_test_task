"""This file contains classes for working with bboxes."""

from __future__ import annotations

import cv2
import numpy as np
import torch
from ultralytics.utils.plotting import colors


class Box:
    """Stores data of single bounding box.

    Used to access properties of a bounding box and plot bbox on images.
    """

    def __init__(self, data: torch.Tensor):
        """Create new Box object from tensor of size [7].

        Args:
            data (torch.Tensor): tensor of size [7] representing a bbox.
        """
        data = data.detach().cpu()
        assert data.shape == torch.Size([7])
        self.xyxy = data[:4].int().tolist()
        self.id = data[4].int().item()
        self.conf = data[5].item()
        self.cls = data[6].int().item()

    def plot(
        self,
        img: np.ndarray,
        label: str = "",
        line_width: int = 1,
        color: tuple = (128, 128, 128),
        txt_color: tuple = (255, 255, 255),
    ):
        """Plot bounding box on single image.

        Args:
            img (np.ndarray): image to plot on.
            label (str): objects description to plot.
            line_width (int): controls the width of bbox's line and font size.
            color (tuple): controls line's color.
            txt_color (tuple): controls text's color.

        Returns:
            (np.ndarray): image with bounding box.
        """
        thickness = max(line_width - 1, 1)  # font thickness
        scale = line_width / 3  # font scale
        p1 = (int(self.xyxy[0]), int(self.xyxy[1]))
        cv2.rectangle(
            img,
            p1,
            (int(self.xyxy[2]), int(self.xyxy[3])),
            color,
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )
        if label:
            w, h = cv2.getTextSize(
                label,
                0,
                fontScale=scale,
                thickness=thickness
            )[0]  # text width, height
            h += 3  # add pixels to pad text
            outside = p1[1] >= h  # label fits outside box
            if (
                p1[0] > img.shape[1] - w
            ):  # shape is (h, w), check if label extend beyond right side
                p1 = img.shape[1] - w, p1[1]
            p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
            cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(
                img,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
                0,
                scale,
                txt_color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
            return img


class Boxes:
    """Stores data of several bounding boxes for the same frame.

    Used to access properties of bounding boxes and plot bboxes on images.
    """

    def __init__(self, data: torch.Tensor, ids: list[int] = None):
        """Create new Boxes object from tensor of size [N, 7].

        Args:
            data (torch.Tensor): tensor of size [N, 7], represents N bboxes.
            ids (list[int] | None): list of objects' ids to use.
        """
        assert data.shape[1] == 7
        self.data = data.detach().cpu()
        if ids is not None:
            matched = torch.isin(self.data[:, 4].int(), torch.tensor(ids))
            self.data = self.data[matched]

    def get_xyxys(self):
        """Return xyxy properties of bboxes."""
        return self.data[:, :4].int()

    def get_ids(self):
        """Return ids of bboxes' objects."""
        return self.data[:, 4].int()

    def get_confs(self):
        """Return confidences of bboxes."""
        return self.data[:, 5]

    def get_clss(self):
        """Return classes of bboxes' objects."""
        return self.data[:, 6].int()

    def __getitem__(self, idx: int):
        """Return single bbox by index.

        Args:
            idx (int): bboxes id.
        """
        return Box(self.data[idx])

    def __len__(self):
        """Return number of bboxes."""
        return self.data.shape[0]

    def plot(
        self,
        img: np.ndarray | torch.Tensor,
        names: dict[int, str],
        use_conf: bool = True,
        use_names: bool = True,
        use_labels: bool = True,
        line_width: float = 1,
        color_mode: str = "class",
        txt_color: tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Plot detection results on an input BGR image.

        Args:
            img (np.ndarray | torch.Tensor): Image to plot on.
            names (dict[int, str]): Classes ids to corresponding names
            use_conf (bool): Whether to plot detection confidence scores.
            use_names (bool): Whether to plot classes names.
            use_labels (bool): Whether to plot labels of bounding boxes.
            line_width (float | None): Line width of bounding boxes.
            color_mode (str): Color mode, e.g., 'instance' or 'class'.
            txt_color (tuple[int, int, int]): Text color in BGR format.

        Returns:
            (np.ndarray): Annotated image as a NumPy array (BGR).
        """
        assert color_mode in {
            "instance",
            "class",
        }, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if isinstance(img, torch.Tensor):
            img = (
                img.detach().permute(1, 2, 0).contiguous() * 255
            ).byte().cpu().numpy()

        # Plot Detect results
        for box in self:
            name = f"id:{box.id}" + (f" {names[box.cls]}" if use_names else "")
            label = None
            if use_labels:
                label = f"{name} {box.conf:.2f}" if use_conf else name
            box.plot(
                img,
                label,
                line_width,
                color=colors(
                    box.cls if color_mode == "class" else box.id,
                    True
                ),
                txt_color=txt_color,
            )
        return img

