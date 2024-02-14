
import numpy as np


class FrameInfo:
    def __init__(self, frame_num=1, point=None) -> None:
        self.frame_num = frame_num
        self.point = np.array(point, dtype=np.float32)  # centre point
        self.frame = None
        self.tips_points = []
            
    def deep_copy_point(self):
        return np.copy(self.point)
    def __str__(self) -> str:
        return f"({self.point[0]},{self.point[1]})"