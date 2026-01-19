from __future__ import annotations

import cv2

class VideoReader(cv2.VideoCapture):
    """Used to read video and get its properties
    """
    
    def __init__(self, video_path: str):
        """Create reader.

        Args:
            video_path (str): path to video to read.
        """
        super().__init__(video_path)
    
    def get_length(self):
        """Get number of frames in video.
        """
        return int(self.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self):
        """Get video's fps.
        """
        return int(self.get(cv2.CAP_PROP_FPS))
    
    def get_width(self):
        """Get single frame's width.
        """
        return int(self.get(3))
    
    def get_height(self):
        """Get single frame's height.
        """
        return int(self.get(4))

class VideoWriter(cv2.VideoWriter):
    """Used to write video.
    """
    
    def __init__(self, reader: VideoReader, save_path: str):
        """Creates new writer object with properties from reader object.
    
        Args:
            reader (VideoReader): original video's reader.
            save_path (str): path where resulting video will be stored.
        """
        fps = reader.get_fps()
        size = (reader.get_width(), reader.get_height())
        super().__init__(save_path, cv2.VideoWriter_fourcc('m','p','4','v'), fps, size)
