from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class CompletionInput:
    image: np.ndarray  # HWC float32 RGB [0,1]
    sensor_depth: np.ndarray  # HW optical-axis Z, meters
    sensor_mask: np.ndarray
    K: np.ndarray | None = None  # pixel intrinsics, integer pixel centers
    sensor_confidence: np.ndarray | None = None
    sample_id: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        shape = self.sensor_depth.shape
        if len(shape) != 2 or self.image.shape != (*shape, 3) or self.sensor_mask.shape != shape:
            raise ValueError("RGB, sensor and mask must have aligned HWC/HW shapes")
        if not np.isfinite(self.image).all() or self.image.min() < 0 or self.image.max() > 1:
            raise ValueError("RGB must be finite in [0,1]")
        self.sensor_mask = (self.sensor_mask.astype(bool) & np.isfinite(self.sensor_depth)
                            & (self.sensor_depth > 0))
        if self.K is not None:
            validate_K(self.K)
        if self.sensor_confidence is not None:
            c = self.sensor_confidence
            if c.shape != shape or not np.isfinite(c).all() or np.any((c < 0) | (c > 1)):
                raise ValueError("Confidence must be finite HW in [0,1]")


def validate_K(K):
    if (np.shape(K) != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0
            or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1])):
        raise ValueError("Invalid pixel intrinsics")


@dataclass
class EvaluationTarget:
    gt_depth: np.ndarray
    gt_valid_mask: np.ndarray

    def __post_init__(self):
        if self.gt_depth.shape != self.gt_valid_mask.shape:
            raise ValueError("GT/mask shape mismatch")
        self.gt_valid_mask = (self.gt_valid_mask.astype(bool) & np.isfinite(self.gt_depth)
                              & (self.gt_depth > 0))


@dataclass
class CompletionResult:
    depth: np.ndarray
    metadata: dict = field(default_factory=dict)
    correction: np.ndarray | None = None
    weights: np.ndarray | None = None
    graph_degree: np.ndarray | None = None


class CompletionMethod(Protocol):
    def complete(self, sample: CompletionInput, prior) -> CompletionResult: ...
