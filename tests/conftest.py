from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zsdc.scripts.synthetic_smoke import synthetic_scene


@pytest.fixture
def scene():
    return synthetic_scene()


@pytest.fixture
def triplets(tmp_path):
    """Actual uint16 PNG fixtures; train/val scenes are distinct."""
    lists = {}
    for split, scene in (("train", "scene_a"), ("val", "scene_b")):
        folder = tmp_path / "lingbot-depth_pub" / "RobbySimVal" / "val_view" / scene / "00"
        folder.mkdir(parents=True)
        rgb = folder/"0001_cam0_rgb.left.jpg"
        raw = folder/"0001_cam0_rawdepth.left.png"
        gt = folder/"0001_cam0_depth_left.png"
        cv2.imwrite(str(rgb), np.full((8, 12, 3), 64, np.uint8))
        value = np.full((8, 12), 2000, np.uint16)
        value[0, 0], value[0, 1] = 0, 65535
        cv2.imwrite(str(raw), value)
        cv2.imwrite(str(gt), value)
        np.savetxt(folder/"intrinsics.txt", [[20, 0, 5.5], [0, 20, 3.5], [0, 0, 1]])
        filename = tmp_path/(split+".txt")
        filename.write_text(" ".join(p.relative_to(tmp_path).as_posix() for p in (rgb, raw, gt))+"\n")
        lists[split] = filename
    return tmp_path, lists
