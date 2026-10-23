import numpy as np

from .schema import validate_K


def normalized_to_pixel(K, shape):
    """utils3d uses UV=(pixel+0.5)/(width,height)."""
    h, w = shape
    T = np.array([[w, 0, -0.5], [0, h, -0.5], [0, 0, 1.]])
    return T @ K


def pixel_to_normalized(K, shape):
    h, w = shape
    T = np.array([[1/w, 0, 0.5/w], [0, 1/h, 0.5/h], [0, 0, 1.]])
    return T @ K


def resize_K(K, old_shape, new_shape):
    return normalized_to_pixel(pixel_to_normalized(K, old_shape), new_shape)


def rays(shape, K):
    validate_K(K)
    y, x = np.indices(shape)
    v = np.stack([x, y, np.ones(shape)], axis=-1)
    r = v @ np.linalg.inv(K).T
    return r / r[..., 2:3]


def unproject(depth, K):
    return rays(depth.shape, K) * depth[..., None]


def project(points, K):
    p = points @ K.T
    return p[..., :2] / p[..., 2:3]


def range_to_z(depth, K):
    return depth / np.linalg.norm(rays(depth.shape, K), axis=-1)


def horizontal_fov(K, shape):
    # MoGe supports centered focal/FOV, not arbitrary principal point/skew.
    return float(np.degrees(2 * np.arctan(shape[1] / (2 * K[0, 0]))))
