import hashlib
import json
import os
import platform
import subprocess
from importlib import metadata
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False), encoding="utf-8")


def source_identity():
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(ROOT), *args], stderr=subprocess.DEVNULL).decode().strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    # Content hashes also cover uncommitted implementation changes.
    source = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p)
              for folder in ("zsdc", "moge") for p in sorted((ROOT/folder).rglob("*.py"))}
    return {"git_sha": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "source_sha256": fingerprint(source), "moge_source_sha256": fingerprint({k:v for k,v in source.items() if k.startswith("moge/")})}


def environment():
    packages = {}
    for name in ("numpy", "scipy", "opencv-python-headless", "opencv-python", "torch", "torchvision", "triton", "flex-gemm", "utils3d_moge"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def peak_rss_bytes():
    if os.name == "posix":
        import resource
        # Linux ru_maxrss is KiB, macOS is bytes.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == "Darwin" else 1024))
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in
                        ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                         "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        ok = get_info(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    except (AttributeError, OSError):
        return None
