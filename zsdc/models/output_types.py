from dataclasses import dataclass, field

import numpy as np


@dataclass
class GeometryPrediction:
    depth: np.ndarray
    points: np.ndarray
    intrinsics: np.ndarray  # official normalized intrinsics
    pred_mask: np.ndarray
    depth_per_step: list[np.ndarray] | None = None
    points_per_step: list[np.ndarray] | None = None
    intrinsics_per_step: list[np.ndarray] | None = None
    normal: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        h, w = self.depth.shape
        if self.points.shape != (h, w, 3) or self.pred_mask.shape != (h, w):
            raise ValueError("Prediction shapes do not agree")
        if self.intrinsics.shape != (3, 3):
            raise ValueError("Expected normalized 3x3 intrinsics")
