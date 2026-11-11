#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fire-VAD dataset segmentation tool.

Segmentation GUI for the fire-VAD dataset (RGB + IR videos stored as frame
sequences). The annotator marks the frame index where each new segment starts;
segments are consecutive intervals covering the whole video in which "the
overall situation does not change". There are no caption fields.

Output: one JSON file per (annotator, environment, scene) under
  annotations/<annotator>/environment_X/Y.json
with video_id, environment, annotator_id and the list of segments
(segment_id, start_frame, end_frame, temp_min_c, temp_max_c). The temperature
range of every segment is filled automatically. Per-frame fire labels
(annotations.csv) are NEVER loaded or shown to annotators.

Dependencies: Python 3.9+ with Tk 8.6+, numpy and pillow
(see requirements.txt: pip install -r requirements.txt).
Usage:
    python segment_gui.py               # normal GUI
    python segment_gui.py --selftest    # headless data/logic validation
    python segment_gui.py --guitest     # scripted end-to-end GUI test (dev)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import unicodedata
import zipfile
from datetime import datetime

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_NAME = "Fire-VAD Segmentation Tool"
FROZEN = getattr(sys, "frozen", False)      # True in the PyInstaller builds


def _app_dir():
    """Folder the annotator sees the program in (used to look for the videos)."""
    if not FROZEN:
        return os.path.dirname(os.path.abspath(__file__))
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    bundle = re.fullmatch(r"(.*)\.app/Contents/MacOS", exe_dir.replace(os.sep, "/"))
    return os.path.dirname(bundle.group(1)) if bundle else exe_dir


SCRIPT_DIR = _app_dir()
# The installed app folder can be read-only or randomly relocated (macOS App
# Translocation), so a frozen build keeps the annotations in the home folder.
PROJECT_DIR = (os.path.join(os.path.expanduser("~"), "FireVAD-Segmenter")
               if FROZEN else SCRIPT_DIR)       # overridden by --guitest

IR_DIR_NAME = "IR"
RGB_DIR_NAME = "RGB"

# Three IR layouts are supported:
#   IR_delta.npz  (Antunovic_VAD_rgb_ir_dataset) frame-to-frame differences of the
#                 float16 temperature matrices, split into byte planes "lo" and
#                 "hi" (format "ir-delta-shuffle-v1", see IR_COMPRESSION.md there),
#   IR.npz        key "frames": (N, H, W) float16 array of °C temperature matrices,
#   IR/           one .npy temperature matrix per frame (legacy).
IR_DELTA_NAME = "IR_delta.npz"
IR_DELTA_FORMAT = "ir-delta-shuffle-v1"
IR_DELTA_CHUNK = 32           # frames decoded at a time: keeps the peak memory low
IR_ARCHIVE_NAME = "IR.npz"
IR_ARCHIVE_KEY = "frames"

# 256-entry RGB colormap LUTs (generated from OpenCV's applyColorMap so the
# GUI does not need OpenCV as a dependency).
COLORMAP_B64 = {
    "HOT": "AAAAAgAABQAACAAACgAADAAADwAAEgAAFAAAFgAAGQAAGwAAHgAAIAAAIwAAJgAAKAAAKgAALQAAMAAAMgAANAAANwAAOQAAPAAAPgAAQQAARAAARgAASAAASwAATgAAUAAAUgAAVQAAWAAAWgAAXAAAXwAAYgAAZAAAZgAAaQAAbAAAbgAAcAAAcwAAdQAAeAAAegAAfQAAgAAAggAAhAAAhwAAigAAjAAAjgAAkQAAlAAAlgAAmAAAmwAAngAAoAAAogAApQAAqAAAqgAArAAArwAAsgAAtAAAtgAAuQAAvAAAvgAAwAAAwwAAxgAAyAAAygAAzQAA0AAA0gAA1AAA1wAA2gAA3AAA3wAA4QAA5AAA5gAA6AAA6wAA7gAA8AAA8wAA9QAA+AAA+gAA/AAA/QIA/gQA/gYA/wgA/woA/w0A/w8A/xIA/xQA/xYA/xkA/xwA/x4A/yAA/yMA/yYA/ygA/yoA/y0A/zAA/zIA/zQA/zcA/zoA/zwA/z4A/0EA/0QA/0YA/0gA/0sA/04A/1AA/1IA/1UA/1gA/1oA/1wA/18A/2IA/2QA/2YA/2kA/2wA/24A/3AA/3MA/3YA/3gA/3oA/30A/4AA/4IA/4QA/4cA/4oA/4wA/44A/5EA/5QA/5YA/5gA/5sA/54A/6AA/6IA/6UA/6gA/6oA/6wA/68A/7IA/7QA/7YA/7kA/7wA/74A/8AA/8MA/8YA/8gA/8oA/80A/9AA/9IA/9QA/9cA/9oA/9wA/94A/+EA/+QA/+YA/+gA/+sA/+4A//AA//IA//UA//gA//oA//wC//0F//4I//8L//8P//8U//8Z//8e//8j//8o//8t//8y//83//88//9B//9G//9L//9Q//9V//9a//9f//9k//9p//9u//9z//94//99//+C//+H//+M//+R//+W//+b//+g//+l//+q//+v//+0//+5//++///D///I///N///S///X///c///h///m///r///w///1///6////",
    "INFERNO": "AAAEAQAFAQEGAQEIAgEKAgIMAgIOAwIQBAMSBAMUBQQXBgQZBwUbCAUdCQYfCgciCwckDAgmDQgpDgkrEAktEQowEgoyFAs0FQs3Fgs5GAw8GQw+GwxBHAxDHgxFHwxIIQxKIwxMJAxPJgxRKAtTKQtVKwtXLQtZLwpbMQpcMgpeNApfNglhOAliOQljOwlkPQllPglmQApnQgpoRApoRQppRwtqSQtqSgxrTAxrTQ1sTw1sUQ5sUg5tVA9tVQ9tVxBuWRBuWhFuXBJuXRJuXxNuYRNuYhRuZBVuZRVuZxZuaRZuahdubBhubRhubxlucRluchpudBpudRtudxxteBxteh1tfB1tfR5tfx5sgB9sgiBshCBrhSFrhyFriCJqiiJqjCNpjSNpjyRpkCVokiVokyZnlSZnlydmmCdmmihlmylknSlknypjoCpjoitioyxhpSxgpi1gqC5fqS5eqy9erTBdrjBcsDFbsTJaszJatDNZtjRYtzVXuTVWujZVvDdUvThTvzlSwDpRwTpQwztPxDxOxj1Nxz5MyD9LykBKy0FJzEJIzkNHz0RG0EVF0kZE00dD1EhC1UpB10s/2Ew+2U092k4821A73VE63lI431M34FU24VY14lc041kz5Fox5Vww5l0v514u6GAt6WEr6mMq62Qp62Yo7Gcm7Wkl7mok72wj724h8G8g8XEf8XMd8nQc83Yb83gZ9HkY9XsX9X0V9n4U9oAT94IS94QQ+IUP+IcO+IkM+YsL+YwK+Y4J+pAI+pIH+pQH+5YG+5cG+5kG+5sG+50H/J8H/KEI/KMJ/KUK/KYM/KgN/KoP/KwR/K4S/LAU/LIW/LQY+7Ya+7gd+7of+7wh+74j+sAm+sIo+sQq+sYt+ccv+cky+cs1+M03+M8699E999NA9tVD9tdG9dlJ9dtM9N1P9N9T9OFW8+Na8+Vd8uZh8uhl8upp8ext8e1x8e918fF58vJ98vSC8/WG8/aK9PiO9fmS9vqW+Pua+fyd+v2h/P+k",
    "JET": "AACAAACEAACIAACMAACQAACUAACYAACcAACgAACkAACoAACsAACwAAC0AAC4AAC8AADAAADEAADIAADMAADQAADUAADYAADcAADgAADkAADoAADsAADwAAD0AAD4AAD8AAD/AAT/AAj/AAz/ABD/ABT/ABj/ABz/ACD/ACT/ACj/ACz/ADD/ADT/ADj/ADz/AED/AET/AEj/AEz/AFD/AFT/AFj/AFz/AGD/AGT/AGj/AGz/AHD/AHT/AHj/AHz/AID/AIT/AIj/AIz/AJD/AJT/AJj/AJz/AKD/AKT/AKj/AKz/ALD/ALT/ALj/ALz/AMD/AMT/AMj/AMz/AND/ANT/ANj/ANz/AOD/AOT/AOj/AOz/APD/APT/APj/APz/Av/+Bv/6Cv/2Dv/yEv/uFv/qGv/mHv/iIv/eJv/aKv/WLv/SMv/ONv/KOv/GPv/CQv++Rv+6Sv+2Tv+yUv+uVv+qWv+mXv+iYv+eZv+aav+Wbv+Scv+Odv+Kev+Gfv+Cgv9+hv96iv92jv9ykv9ulv9qmv9mnv9iov9epv9aqv9Wrv9Ssv9Otv9Kuv9Gvv9Cwv8+xv86yv82zv8y0v8u1v8q2v8m3v8i4v8e5v8a6v8W7v8S8v8O9v8K+v8G/v8B//wA//gA//QA//AA/+wA/+gA/+QA/+AA/9wA/9gA/9QA/9AA/8wA/8gA/8QA/8AA/7wA/7gA/7QA/7AA/6wA/6gA/6QA/6AA/5wA/5gA/5QA/5AA/4wA/4gA/4QA/4AA/3wA/3gA/3QA/3AA/2wA/2gA/2QA/2AA/1wA/1gA/1QA/1AA/0wA/0gA/0QA/0AA/zwA/zgA/zQA/zAA/ywA/ygA/yQA/yAA/xwA/xgA/xQA/xAA/wwA/wgA/wQA/wAA/AAA+AAA9AAA8AAA7AAA6AAA5AAA4AAA3AAA2AAA1AAA0AAAzAAAyAAAxAAAwAAAvAAAuAAAtAAAsAAArAAAqAAApAAAoAAAnAAAmAAAlAAAkAAAjAAAiAAAhAAAgAAA",
    "PLASMA": "DQiHEAeIEweJFgeKGQaMGwaNHQaOIAaPIgaQJAaRJgWRKAWSKgWTLAWULgWVLwWWMQWXMwWXNQSYNwSZOASaOgSaPASbPgScPwScQQSdQwOeRAOeRgOfSAOfSQOgSwOhTAKhTgKiUAKiUQKjUwKjVQKkVgGkWAGkWQGlWwGlXAGmXgGmYAGmYQCnYwCnZACnZgCnZwCoaQCoagCobACobgCobwCocQCocgGodAGodQGodwGoeAGoegKoewKofQOofgOogASogQSngwWnhAWnhgamhwemiAimigmliwqljQuljgykjw2kkQ6jkg+jlBCilRGhlhOhmBSgmRWfmhafnBeenRidnhmdoBqcoRuboh2aox6apR+ZpiCYpyGXqCKWqiOVqySUrCaUrSeTriiSsCmRsSqQsiuPsyyOtC6NtS+MtjCLtzGKuDKJujOIuzSIvDWHvTeGvjiFvzmEwDqDwTuCwjyBwz2AxD5/xUB+xkF9x0J8yEN7yUR6ykV6y0Z5zEd4zEl3zUp2zkt1z0x00E1z0U5y0k9x01Fx1FJw1VNv1VRu1lVt11Zs2Fdr2Vhq2lpq2ltp21xo3F1n3V5m3l9l3mFk32Jj4GNj4WRi4mVh4mZg42hf5Gle5Wpd5Wtd5mxc525b529a6HBZ6XFY6XJX6nRX63VW63ZV7HdU7XlT7XpS7ntR73xR735Q8H9P8IBO8YFN8YNM8oRL84VL84dK9IhJ9IlI9YtH9YxG9o1F9o9E95BE95FD95NC+JRB+JVA+Zc/+Zg++Zo++ps9+pw8+p47+586+6E5+6I4/KM4/KU3/KY2/Kg1/Kk0/asz/awz/a4y/a8x/bEw/bIv/bQv/bUu/rct/rgs/ros/rsr/r0q/r4q/sAp/cIp/cMo/cUn/cYn/cgn/com/csm/M0l/M4l/NAl/NIl+9Mk+9Uk+9ck+tgk+tok+dwk+d0l+N8l+OEl9+Il9+Ql9uYm9ugm9ekm9esn9O0n8+4n8/An8vIn8fQm8fUl8Pck8Pkh",
    "VIRIDIS": "RAFURAJWRQRXRQVZRgdaRghcRgpdRgteRw1gRw5hRxBjRxFkRxNlSBRnSBZoSBdpSBhqSBpsSBttSBxuSB1vSB9wSCBxSCFzSCN0SCR1SCV2SCZ3SCh4SCl5Ryp6Ryx6Ry17Ry58Ry99RjB+RjJ+RjN/RjSARTWBRTeBRTiCRDmDRDqDRDuEQz2EQz6FQj+FQkCGQkGGQUKHQUSHQEWIQEaIP0eIP0iJPkmJPkqJPkyKPU2KPU6KPE+KPFCLO1GLO1KLOlOLOlSMOVWMOVaMOFiMOFmMN1qMN1uNNlyNNl2NNV6NNV+NNGCNNGGNM2KNM2ONMmSOMmWOMWaOMWeOMWiOMGmOMGqOL2uOL2yOLm2OLm6OLm+OLXCOLXGOLHGOLHKOLHOOK3SOK3WOKnaOKneOKniOKXmOKXqOKXuOKHyOKH2OJ36OJ3+OJ4COJoGOJoKOJoKOJYOOJYSOJYWOJIaOJIeOI4iOI4mOI4qNIouNIoyNIo2NIY6NIY+NIZCNIZGMIJKMIJKMIJOMH5SMH5WLH5aLH5eLH5iLH5mKH5qKHpuKHpyJHp2JH56JH5+IH6CIH6GIH6GHH6KHIKOGIKSGIaWFIaaFIqeFIqiEI6mDJKqDJauCJayCJq2BJ62BKK6AKa9/KrB/LLF+LbJ9LrN8L7R8MbV7MrZ6NLZ5Nbd5N7h4OLl3Orp2O7t1Pbx0P7xzQL1yQr5xRL9wRsBvSMFuSsFtTMJsTsNrUMRqUsVpVMVoVsZnWMdlWshkXMhjXsliYMpgY8tfZcteZ8xcac1bbM1abs5YcM9Xc9BWddBUd9FTetFRfNJQf9NOgdNNhNRLhtVJidVIi9ZGjtZFkNdDk9dBldhAmNg+m9k8ndk7oNo5oto3pds2qNs0qtwyrdwwsN0vst0ttd4ruN4put4ovd8mwN8lwt8jxeAhyOAgyuEfzeEd0OEc0uIb1eIa2OIZ2uMZ3eMY3+MY4uQY5eQZ5+QZ6uUa7OUb7+Uc8eUd9OYe9uYg+OYh++cj/ecl"
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def load_colormaps():
    """Decode embedded colormap tables -> {name: (256, 3) uint8 RGB array}."""
    luts = {}
    for name, b64 in COLORMAP_B64.items():
        luts[name] = np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(256, 3)
    return luts


LUTS = load_colormaps()


def parse_frame_index(stem):
    """Extract the frame index from a frame filename stem.

    Handles both "12.npy" (test_VAD layout) and legacy
    "ir_frame_12_28.728231.npy" names (first number wins).
    """
    if re.fullmatch(r"\d+", stem):
        return int(stem)
    m = re.search(r"\d+", stem)
    return int(m.group()) if m else None


def list_frame_files(folder, extensions):
    """Return [(index, path), ...] sorted by frame index."""
    out = []
    if not os.path.isdir(folder):
        return out
    for name in os.listdir(folder):
        base, ext = os.path.splitext(name)
        if ext.lower() not in extensions:
            continue
        idx = parse_frame_index(base)
        if idx is None:
            continue
        out.append((idx, os.path.join(folder, name)))
    out.sort(key=lambda t: t[0])
    return out


def sanitize_annotator_dir(name):
    """Folder name for an annotator, e.g. "Ivana Šimić" -> "Ivana_Simic"."""
    # NFKD splits č/ć/š/ž into a base letter plus an accent; đ has no
    # decomposition, so it gets the usual Croatian transliteration.
    name = name.strip().replace("đ", "dj").replace("Đ", "Dj")
    name = "".join(c for c in unicodedata.normalize("NFKD", name)
                   if not unicodedata.combining(c))
    # Leading/trailing dots would allow "." / ".." and are dropped by Windows.
    d = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_.")
    return d or "annotator"


def save_json_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def reveal_in_file_manager(path):
    """Show a file in Explorer / Finder / the Linux file manager (best effort)."""
    try:
        if sys.platform == "win32":
            subprocess.Popen('explorer /select,"%s"' % os.path.normpath(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])
    except OSError:
        return False
    return True


def enable_windows_dpi_awareness():
    """Sharp text and video on scaled Windows displays (125 %, 150 %, ...)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.OleDLL("shcore").SetProcessDpiAwareness(1)   # same call as IDLE
    except (ImportError, AttributeError, OSError):
        pass


# --------------------------------------------------------------------------
# Dataset discovery
# --------------------------------------------------------------------------

def _read_npy_header(f):
    """(shape, dtype) from the header of an .npy stream; None for unknown versions."""
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, _, dtype = np.lib.format.read_array_header_2_0(f)
    else:
        return None
    return shape, dtype


def npz_stack_info(path, key=IR_ARCHIVE_KEY):
    """Return (shape, dtype) of the stacked array inside an .npz archive.

    Reads only the zip entry's npy header, so scene discovery stays fast
    without decompressing the (potentially large) frame data.
    Returns None when the archive is missing the expected key or unreadable.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            member = key + ".npy"
            if member not in zf.namelist():
                return None
            with zf.open(member) as f:
                return _read_npy_header(f)
    except Exception:
        return None


def delta_archive_info(path):
    """Shape (N, H, W) of an IR_delta.npz archive, or None when it is not one."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            if not {"format.npy", "lo.npy", "hi.npy"} <= names:
                return None
            with zf.open("lo.npy") as f:
                info = _read_npy_header(f)
        shape = info[0] if info else None
        return shape if shape and len(shape) == 3 else None
    except Exception:
        return None


def _read_exact(f, nbytes):
    data = f.read(nbytes)
    if len(data) != nbytes:
        raise ValueError("The IR archive is truncated or damaged.")
    return data


def decode_delta_archive(path, progress=None):
    """Decode an IR_delta.npz archive into a float16 (N, H, W) stack of °C.

    The byte planes are streamed in chunks of IR_DELTA_CHUNK frames, so the peak
    memory is the decoded stack plus a few megabytes (not the whole archive
    unpacked). The result is bit-identical to the original frames:
    d = hi << 8 | lo;  u = cumulative sum of d along the frame axis modulo 2**16;
    frames = u viewed as float16.  `progress(done, total)` is called per chunk.
    """
    with zipfile.ZipFile(path) as zf:
        with zf.open("format.npy") as f:
            fmt = str(np.load(io.BytesIO(f.read()), allow_pickle=False))
        if fmt != IR_DELTA_FORMAT:
            raise ValueError("Unsupported IR archive format %r." % fmt)
        with zf.open("lo.npy") as flo, zf.open("hi.npy") as fhi:
            lo_info, hi_info = _read_npy_header(flo), _read_npy_header(fhi)
            if (not lo_info or not hi_info or lo_info[0] != hi_info[0]
                    or len(lo_info[0]) != 3):
                raise ValueError("The IR archive has unexpected contents.")
            n, h, w = lo_info[0]
            out = np.empty((n, h, w), dtype=np.float16)
            prev = np.zeros((h, w), dtype=np.uint16)
            for start in range(0, n, IR_DELTA_CHUNK):
                k = min(IR_DELTA_CHUNK, n - start)
                lo = np.frombuffer(_read_exact(flo, k * h * w), dtype=np.uint8).reshape(k, h, w)
                hi = np.frombuffer(_read_exact(fhi, k * h * w), dtype=np.uint8).reshape(k, h, w)
                d = (hi.astype(np.uint16) << 8) | lo
                d[0] += prev                    # wraps modulo 2**16, as encoded
                u = np.cumsum(d, axis=0, dtype=np.uint16)
                out[start:start + k] = u.view(np.float16)
                prev = u[-1]
                if progress:
                    progress(start + k, n)
    return out


def make_ir_loader(scene):
    """Return (loader, n_frames) giving per-frame 2D temperature matrices.

    Legacy folder format: loader(i) loads the i-th IR/*.npy file from disk.
    Archive formats: loader(i) indexes the decoded stack of the scene. The stack
    is decoded exactly once and cached in the closure (a lock keeps the first
    load safe when the stats thread races the UI); loader.preload(progress)
    decodes it ahead of time so the UI can show a progress bar.
    Out-of-range indices are clamped (last frame is held), mirroring the
    legacy behaviour when RGB and IR counts differ.
    """
    fmt = scene.get("ir_format")
    if fmt in ("delta", "archive"):
        state = {"stack": None}
        lock = threading.Lock()

        def stack_of(progress=None):
            with lock:
                if state["stack"] is None:
                    if fmt == "delta":
                        state["stack"] = decode_delta_archive(scene["ir_archive"], progress)
                    else:
                        with np.load(scene["ir_archive"]) as z:
                            state["stack"] = z[IR_ARCHIVE_KEY]
                return state["stack"]

        def load(i):
            stack = stack_of()
            return stack[min(max(i, 0), stack.shape[0] - 1)]

        load.preload = lambda progress=None: stack_of(progress)
        if fmt == "delta":
            info = delta_archive_info(scene["ir_archive"])
            return load, (int(info[0]) if info else 0)
        info = npz_stack_info(scene["ir_archive"])
        return load, (int(info[0][0]) if info else 0)

    files = scene["ir_files"]

    def load(i):
        j = min(max(i, 0), len(files) - 1)
        return np.load(files[j][1])

    return load, len(files)


def discover_scenes(dataset_root):
    """Scan dataset_root/environment_N/<scene_id> and list scenes.

    Three IR layouts are supported (see IR_DELTA_NAME): IR_delta.npz, IR.npz and
    the legacy IR/ folder of .npy files. The first one present is used, in that order.
    """
    scenes = []
    if not dataset_root or not os.path.isdir(dataset_root):
        return scenes
    for entry in sorted(os.listdir(dataset_root)):
        m = re.fullmatch(r"environment_(\d+)", entry)
        if not m:
            continue
        env_dir = os.path.join(dataset_root, entry)
        if not os.path.isdir(env_dir):
            continue
        env_id = int(m.group(1))
        for sub in sorted(os.listdir(env_dir)):
            try:
                scene_id = int(sub)
            except ValueError:
                continue
            scene_dir = os.path.join(env_dir, sub)
            rgb_files = list_frame_files(os.path.join(scene_dir, RGB_DIR_NAME),
                                         {".jpg", ".jpeg", ".png"})
            if not rgb_files:
                continue
            delta = os.path.join(scene_dir, IR_DELTA_NAME)
            archive = os.path.join(scene_dir, IR_ARCHIVE_NAME)
            if os.path.isfile(delta):
                shape = delta_archive_info(delta)
                if shape is None:
                    continue  # unreadable or unknown archive: skip scene
                ir_format, ir_count, archive = "delta", int(shape[0]), delta
                ir_files = []
            elif os.path.isfile(archive):
                info = npz_stack_info(archive)
                if info is None or len(info[0]) < 2:
                    continue  # unreadable or non-stack archive: skip scene
                ir_format, ir_count = "archive", int(info[0][0])
                ir_files = []
            else:
                ir_files = list_frame_files(os.path.join(scene_dir, IR_DIR_NAME), {".npy"})
                if not ir_files:
                    continue
                ir_format, ir_count = "folder", len(ir_files)
            scenes.append({
                "environment": env_id,
                "scene": scene_id,
                "path": scene_dir,
                "ir_format": ir_format,
                "ir_archive": archive if ir_format in ("delta", "archive") else None,
                "ir_files": ir_files,
                "ir_count": ir_count,
                "rgb_files": rgb_files,
                "n_frames": max(ir_count, len(rgb_files)),
            })
    scenes.sort(key=lambda s: (s["environment"], s["scene"]))
    return scenes


# Dataset folders the tool looks for, in order of preference.
DATASET_DIR_NAMES = ("Antunovic_VAD", "test_VAD")
_ENV_DIR_RE = re.compile(r"environment_\d+")
_SEARCH_SKIP = {"annotations", "__pycache__", "node_modules"}

NO_DATASET_MESSAGE = (
    "No fire-VAD videos were found in that folder or its subfolders "
    "(looking for environment_1, environment_2, ... folders).\n\n"
    "If you downloaded a ZIP file, extract (unzip) it first and then "
    "select the extracted folder.")


def _dataset_rank(name):
    """0 for the preferred dataset folder names, then the next ones, then the rest."""
    low = name.lower()
    for rank, prefix in enumerate(DATASET_DIR_NAMES):
        if low.startswith(prefix.lower()):
            return rank
    return len(DATASET_DIR_NAMES)


def _is_dir(entry):
    try:
        return entry.is_dir()
    except OSError:
        return False


def looks_like_dataset_root(path):
    """True when path directly contains environment_<N> folders."""
    if not path or not os.path.isdir(path):
        return False
    try:
        with os.scandir(path) as it:
            return any(_ENV_DIR_RE.fullmatch(e.name) and _is_dir(e) for e in it)
    except OSError:
        return False


def find_dataset_root(path, max_depth=3, budget=None, exclude=()):
    """Return path itself or the shallowest folder below it that is a dataset root.

    Unzipping often adds wrapper folders (Windows "Extract All" turns
    a ZIP file into <name>/<name>/environment_1), so annotators may
    pick whichever folder they downloaded.  ``budget`` (a one-item list,
    shared between calls) caps the number of folders listed so that a large
    Downloads folder is searched quickly; ``exclude`` holds folders that were
    already searched and need not be entered again.
    """
    if not path or not os.path.isdir(path):
        return None
    budget = [2000] if budget is None else budget
    skip = {os.path.normcase(os.path.abspath(p)) for p in exclude}
    level = [os.path.abspath(path)]
    for depth in range(max_depth + 1):
        deeper = []
        for folder in level:
            if budget[0] <= 0:
                return None
            budget[0] -= 1
            try:
                with os.scandir(folder) as it:
                    # preferred dataset folders first, then alphabetical: deterministic
                    # when several datasets sit side by side.
                    subdirs = sorted((e for e in it if _is_dir(e)), key=lambda e: (
                        _dataset_rank(e.name), e.name.lower()))
            except OSError:
                continue
            if any(_ENV_DIR_RE.fullmatch(e.name) for e in subdirs):
                return folder
            if depth < max_depth:
                deeper.extend(
                    e.path for e in subdirs
                    if not e.name.startswith(".") and e.name not in _SEARCH_SKIP
                    and os.path.normcase(os.path.abspath(e.path)) not in skip)
        level = deeper
    return None


def _user_home():
    return os.path.abspath(os.path.expanduser("~"))


def _is_root_or_top_level(path):
    up = os.path.dirname(path)
    return up == path or os.path.dirname(up) == up


def _contains(parent, child):
    try:
        return os.path.commonpath([parent, child]) == parent
    except ValueError:          # different drives on Windows
        return False


def dataset_search_folders():
    """The tool folder plus up to two parents: where a fresh download usually is.

    The search never covers the home folder (or anything above it), a drive
    root or a top-level folder such as /Volumes: that would be slow and, on
    macOS, would trigger privacy prompts for Desktop, Documents and USB drives.
    """
    folders = [SCRIPT_DIR]
    home = os.path.normcase(_user_home())
    current = SCRIPT_DIR
    for _ in range(2):
        parent = os.path.dirname(current)
        if (parent == current or _is_root_or_top_level(parent)
                or _contains(os.path.normcase(parent), home)):
            break
        folders.append(parent)
        current = parent
    return folders


def locate_dataset_near_tool():
    budget = [4000]
    searched = []
    for folder in dataset_search_folders():
        found = find_dataset_root(folder, budget=budget, exclude=searched)
        if found:
            return found
        searched.append(folder)
    return None


def default_dataset_root():
    """A dataset found near the tool (independent of cwd), else <tool>/<first dataset name>."""
    return locate_dataset_near_tool() or os.path.join(SCRIPT_DIR, DATASET_DIR_NAMES[0])


# --------------------------------------------------------------------------
# Segment math
# --------------------------------------------------------------------------

def boundaries_to_segments(bounds, n_frames):
    """Boundary = first frame of a new segment. Returns [(start, end_inclusive)]."""
    bs = sorted(b for b in bounds if 0 < b < n_frames)
    starts = [0] + bs
    ends = [b - 1 for b in bs] + [n_frames - 1]
    return list(zip(starts, ends))


# A segment this short (or shorter) is almost certainly a mistake: the annotator
# is asked to delete one of its boundaries.
SHORT_SEGMENT_FRAMES = 40


def short_segment_indices(segs, limit=SHORT_SEGMENT_FRAMES):
    """Indices of the (start, end_inclusive) segments with at most `limit` frames."""
    return [i for i, (s, e) in enumerate(segs) if e - s + 1 <= limit]


def short_segment_warning(segs, limit=SHORT_SEGMENT_FRAMES):
    """Text telling the annotator which segments are too short, or "" when none is."""
    idx = short_segment_indices(segs, limit)
    if not idx:
        return ""
    names = ", ".join(str(i) for i in idx)
    return ("Segment%s %s %s %d frames or fewer. Delete a boundary of %s "
            "to merge %s with a neighbouring segment."
            % ("s" if len(idx) > 1 else "", names,
               "are" if len(idx) > 1 else "is", limit,
               "each of them" if len(idx) > 1 else "it",
               "them" if len(idx) > 1 else "it"))


def segments_to_boundaries(pairs):
    return [p[0] for p in pairs[1:]]


def add_boundary_value(bounds, frame, n_frames):
    """Insert a boundary, return (ok, message)."""
    if frame is None:
        return False, "No current frame."
    if frame <= 0:
        return False, "Frame 0 is already the start of the first segment."
    if frame >= n_frames:
        return False, "Frame index out of range."
    if frame in bounds:
        return False, "A boundary already exists at frame %d." % frame
    bounds.append(frame)
    bounds.sort()
    return True, "Boundary added at frame %d." % frame


def nudge_boundary_value(bounds, index, delta, n_frames):
    """Move a boundary by +/-1, return (ok, message, new_bounds)."""
    if index is None or not (0 <= index < len(bounds)):
        return False, "Select a boundary first.", bounds
    new = bounds[index] + delta
    if new <= 0:
        return False, "A boundary cannot be at frame 0.", bounds
    if new >= n_frames:
        return False, "Boundary out of range.", bounds
    others = [b for i, b in enumerate(bounds) if i != index]
    if new in others:
        return False, "Another boundary already exists at frame %d." % new, bounds
    bounds[index] = new
    bounds.sort()
    return True, "Boundary moved to frame %d." % new, bounds


# --------------------------------------------------------------------------
# Annotation payload / paths
# --------------------------------------------------------------------------

def build_annotation_payload(environment, video_id, annotator_id, segments):
    """Assemble the JSON payload: the segments of one video with their temperatures."""
    return {
        "video_id": video_id,
        "environment": environment,
        "annotator_id": annotator_id,
        "segments": [{
            "segment_id": i,
            "start_frame": seg["start_frame"],
            "end_frame": seg["end_frame"],
            "temp_min_c": seg.get("temp_min_c"),
            "temp_max_c": seg.get("temp_max_c"),
        } for i, seg in enumerate(segments)],
    }


class SceneAnnotation:
    """The segment boundaries of one video."""

    def __init__(self, n_frames):
        self.n_frames = n_frames
        self.boundaries = []

    def segments(self):
        return boundaries_to_segments(self.boundaries, self.n_frames)

    def index_of_frame(self, frame):
        for i, (s, e) in enumerate(self.segments()):
            if s <= frame <= e:
                return i
        return None

    def add_boundary(self, frame):
        return add_boundary_value(self.boundaries, frame, self.n_frames)

    def remove_boundary(self, start):
        """Remove a segment's start boundary, merging it into the previous one."""
        if start not in self.boundaries:
            return False, "The first segment has no start boundary to remove."
        self.boundaries.remove(start)
        return True, "Boundary at frame %d removed." % start

    def nudge_boundary(self, start, delta):
        """Move a segment's start boundary. Returns (ok, msg, new_start)."""
        if start not in self.boundaries:
            return False, "The first segment always starts at frame 0.", start
        ok, msg, self.boundaries = nudge_boundary_value(
            self.boundaries, self.boundaries.index(start), delta, self.n_frames)
        if not ok:
            return False, msg, start
        return True, msg, start + delta

    def to_segments(self, stats=None):
        out = []
        for s, e in self.segments():
            tmin, tmax = stats.segment_minmax(s, e) if stats else (None, None)
            out.append({"start_frame": s, "end_frame": e,
                        "temp_min_c": tmin, "temp_max_c": tmax})
        return out

    @classmethod
    def from_payload(cls, data, n_frames):
        """Rebuild from a saved payload; invalid or out-of-range starts are dropped."""
        model = cls(n_frames)
        segs = data.get("segments") if isinstance(data, dict) else None
        if not isinstance(segs, list):
            return model
        starts = set()
        for seg in segs:
            try:
                s = int(seg["start_frame"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0 < s < n_frames:
                starts.add(s)
        model.boundaries = sorted(starts)
        return model


def annotations_root():
    return os.path.join(PROJECT_DIR, "annotations")


def annotator_dir(annotator_id):
    return os.path.join(annotations_root(), sanitize_annotator_dir(annotator_id))


def scene_file_path(annotator_id, environment, scene):
    return os.path.join(annotator_dir(annotator_id),
                        "environment_%d" % environment, "%d.json" % scene)


def scene_status(annotator_id, environment, scene, n_frames):
    """Short progress text for the scene list."""
    data = load_json_safe(scene_file_path(annotator_id, environment, scene))
    if data:
        total = len(SceneAnnotation.from_payload(data, n_frames).segments())
        return "✓ saved (%d segment%s)" % (total, "" if total == 1 else "s")
    return "not annotated"


def config_path():
    return os.path.join(PROJECT_DIR, "annotator_config.json")


def load_config():
    return load_json_safe(config_path()) or {}


def save_config(cfg):
    save_json_atomic(config_path(), cfg)


# --------------------------------------------------------------------------
# IR frame rendering / stats
# --------------------------------------------------------------------------

def render_ir(frame2d, lo, hi, lut):
    """Map a 2D temperature matrix to an RGB uint8 image via a colormap LUT."""
    if frame2d.ndim == 3:
        frame2d = frame2d[:, :, 0]
    if hi <= lo:
        hi = lo + 1e-6
    norm = (frame2d.astype(np.float64) - float(lo)) / float(hi - lo)
    np.clip(norm, 0.0, 1.0, out=norm)
    idx = (norm * 255.0).astype(np.uint8)
    return lut[idx]


class FrameStats:
    """Per-frame IR min/max temperatures, computed in a background thread.

    A JSON cache is written next to the annotator's output (not inside the
    dataset folders) once all frames have been processed.
    """

    def __init__(self, n_frames, ir_loader, cache_path=None):
        self.n = n_frames
        self._loader = ir_loader
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.fmin = [None] * n_frames
        self.fmax = [None] * n_frames
        self.next_idx = 0
        self.complete = False

    def try_load_cache(self):
        if not self.cache_path or not os.path.exists(self.cache_path):
            return False
        data = load_json_safe(self.cache_path)
        if not data or data.get("n_frames") != self.n or not data.get("complete"):
            return False
        fmin, fmax = data.get("frame_min"), data.get("frame_max")
        if not isinstance(fmin, list) or not isinstance(fmax, list):
            return False
        if len(fmin) != self.n or len(fmax) != self.n:
            return False
        with self.lock:
            self.fmin = [float(v) for v in fmin]
            self.fmax = [float(v) for v in fmax]
            self.next_idx = self.n
            self.complete = True
        return True

    def start(self):
        t = threading.Thread(target=self._work, daemon=True)
        t.start()
        return t

    def _work(self):
        while True:
            with self.lock:
                if self.next_idx >= self.n:
                    break
                i = self.next_idx
                arr = self._loader(i)
                self.fmin[i] = float(arr.min())
                self.fmax[i] = float(arr.max())
                self.next_idx += 1
        with self.lock:
            self.complete = True
        self.save_cache()

    def ensure_range(self, start, end):
        with self.lock:
            for i in range(start, min(end, self.n - 1) + 1):
                if self.fmin[i] is None:
                    arr = self._loader(i)
                    self.fmin[i] = float(arr.min())
                    self.fmax[i] = float(arr.max())
            # advance next_idx past the contiguous computed prefix
            while self.next_idx < self.n and self.fmin[self.next_idx] is not None:
                self.next_idx += 1
            if self.next_idx >= self.n:
                self.complete = True

    def progress(self):
        with self.lock:
            return self.next_idx, self.n

    def video_range(self):
        with self.lock:
            vals = [v for v in self.fmin if v is not None]
            hi = [v for v in self.fmax if v is not None]
        if not vals or not hi:
            return None, None
        return min(vals), max(hi)

    def segment_minmax(self, start, end):
        """(min, max) over frames [start..end], rounded to 2 decimals."""
        if start > end:
            start, end = end, start
        self.ensure_range(start, end)
        with self.lock:
            vals = [v for v in self.fmin[start:end + 1] if v is not None]
            hi = [v for v in self.fmax[start:end + 1] if v is not None]
        if not vals or not hi:
            return None, None
        return round(min(vals), 2), round(max(hi), 2)

    def save_cache(self):
        if not self.cache_path:
            return
        with self.lock:
            if self.next_idx < self.n:
                return
            payload = {
                "n_frames": self.n,
                "complete": True,
                "frame_min": list(self.fmin),
                "frame_max": list(self.fmax),
            }
        save_json_atomic(self.cache_path, payload)


# --------------------------------------------------------------------------
# Guidance texts (from the annotation instructions document)
# --------------------------------------------------------------------------

GUIDANCE_BOUNDARIES = """\
Segments are consecutive intervals that together cover the whole video.
A new segment starts when something appears or disappears, or when a person
changes their action. Pause at the first frame of the new situation and press B.

A video in which nothing changes is a single segment: save it without
adding any boundary."""

KEYBOARD_HELP = """\
Keys
  Space         play / pause
  Left / Right  one frame back / forward
  Up / Down     10 frames back / forward
  Home / End    first / last frame
  B             a new segment starts at the current frame
  Delete        remove the start boundary of the selected segment
  Ctrl+S        save (macOS: Cmd+S)
  F1            this help"""


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class AnnotationApp(tk.Tk):

    def __init__(self):
        # A successful import or winfo_viewable() does not prove Tk can paint.
        # In particular, Apple's bundled Tk 8.5 shows blank windows on new macOS.
        if tk.TkVersion < 8.6:
            raise RuntimeError(
                "Tk %s is too old and can show a blank window. "
                "Run the tool with a Python that has Tk 8.6+ (install Python 3.13 "
                "from python.org, or run: brew install python-tk@3.13 on macOS)."
                % tk.TkVersion)
        super().__init__()
        self.title(APP_NAME)
        # 1.0 at 100 % display scaling; larger on scaled (DPI-aware) Windows.
        scale = max(1.0, self.winfo_fpixels("1i") / 96.0)
        self.geometry("%dx%d" % (780 * scale, 560 * scale))
        self.active_window = None

        self.cfg = load_config()
        self.annotator_id = (self.cfg.get("annotator_id") or "").strip()
        self.dataset_root = self.cfg.get("dataset_root")

        # Build the main window FIRST so there is always something visible on
        # screen, and only then ask for the annotator name (first run) on top
        # of it.  Hiding the root and popping a dialog parented to a hidden
        # window makes the dialog invisible/behind Terminal on macOS.
        self._build_ui()
        self._resolve_annotator()
        self._resolve_dataset_root()
        self._update_info()
        self.refresh_tree()
        self._force_to_front()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- setup helpers ----------------------------------------------------

    @staticmethod
    def _force_to_front_window(win, ms=300):
        """Keep a window above everything briefly (macOS focus stealing fix)."""
        try:
            win.attributes("-topmost", True)
            win.after(ms, lambda: win.attributes("-topmost", False))
        except Exception:
            pass

    def _force_to_front(self):
        self.update_idletasks()
        self.deiconify()
        self.lift()
        self._force_to_front_window(self)
        try:
            self.focus_force()
        except Exception:
            pass

    def _ask_name(self, initial=""):
        """Modal dialog asking for the annotator name. Returns name or None."""
        result = {"name": None, "viewable": None}
        dlg = tk.Toplevel(self)
        dlg.title("Annotator name")
        dlg.transient(self)
        dlg.resizable(False, False)
        var = tk.StringVar(value=initial)

        ttk.Label(dlg, justify="left", wraplength=400, text=(
            "Welcome to the Fire-VAD segmentation tool.\n\n"
            "Enter your full name — it identifies all your segmentation files:"))\
            .pack(anchor="w", padx=16, pady=(14, 8))
        entry = ttk.Entry(dlg, textvariable=var, width=42)
        entry.pack(padx=16, pady=(0, 10))
        entry.icursor("end")

        def ok(event=None):
            if not var.get().strip():
                try:
                    dlg.bell()
                except Exception:
                    pass
                return
            result["name"] = var.get().strip()
            dlg.destroy()

        def cancel(event=None):
            dlg.destroy()

        btns = ttk.Frame(dlg)
        btns.pack(pady=(0, 14))
        ttk.Button(btns, text="OK", command=ok).pack(side="left", padx=6)
        ttk.Button(btns, text="Cancel", command=cancel).pack(side="left", padx=6)
        dlg.bind("<Return>", ok)
        dlg.bind("<Escape>", cancel)
        dlg.protocol("WM_DELETE_WINDOW", cancel)

        # dev/test hook: auto-answer the dialog (used by --guitest)
        auto = os.environ.get("FIREVAD_GUITEST_NAME")
        if auto:
            var.set(auto)

            def _auto_ok():
                result["viewable"] = bool(dlg.winfo_ismapped() and dlg.winfo_viewable())
                ok()
            dlg.after(400, _auto_ok)

        dlg.update_idletasks()
        # center the dialog over the main window
        px = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
        py = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry("+%d+%d" % (max(px, 40), max(py, 40)))
        dlg.grab_set()
        dlg.lift()
        self._force_to_front_window(dlg)
        try:
            dlg.focus_force()
        except Exception:
            pass
        entry.focus_set()
        dlg.wait_window()
        self._name_dlg_viewable = result.get("viewable")
        return result["name"]

    def _resolve_annotator(self):
        name = (self.cfg.get("annotator_id") or "").strip()
        while not name:
            entered = self._ask_name()
            if entered is None:
                print("Annotator name is required. Exiting.")
                self.destroy()
                sys.exit(1)
            name = entered
        self.annotator_id = name
        if self.cfg.get("annotator_id") != name:
            self.cfg["annotator_id"] = name
            save_config(self.cfg)

    def _resolve_dataset_root(self):
        root = find_dataset_root(self.cfg.get("dataset_root")) or locate_dataset_near_tool()
        if not root:
            messagebox.showinfo(
                APP_NAME, "The video folder (Antunovic_VAD_rgb_ir_dataset) was not found next to the tool.\n\n"
                "In the next window, select the folder where you extracted "
                "the dataset.", parent=self)
        while not root:
            chosen = filedialog.askdirectory(
                title="Select the folder with the fire-VAD videos (environment_1, ...)", parent=self)
            if not chosen:
                break
            root = find_dataset_root(chosen)
            if not root:
                messagebox.showwarning(APP_NAME, NO_DATASET_MESSAGE, parent=self)
        self.dataset_root = root
        if root and self.cfg.get("dataset_root") != root:
            self.cfg["dataset_root"] = root
            save_config(self.cfg)

    # -- UI ----------------------------------------------------------------

    def _build_ui(self):
        head = ttk.Frame(self)
        head.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Label(head, text=APP_NAME, font=("Helvetica", 15, "bold")).pack(anchor="w")
        self.info_var = tk.StringVar()
        info = ttk.Label(head, textvariable=self.info_var, wraplength=740)
        info.pack(fill="x", pady=(3, 0))
        head.bind("<Configure>", lambda e: info.configure(wraplength=max(100, e.width)))

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=10)

        settings = ttk.Frame(body)
        settings.pack(fill="x", pady=(0, 5))
        ttk.Button(settings, text="Change annotator…", command=self.change_annotator)\
            .pack(side="right", padx=(6, 0))
        ttk.Button(settings, text="Change dataset folder…", command=self.change_dataset)\
            .pack(side="right")

        tree_wrap = ttk.Frame(body)
        tree_wrap.pack(fill="both", expand=True)
        cols = ("env", "scene", "frames", "status")
        self.tree = ttk.Treeview(tree_wrap, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("env", text="Environment")
        self.tree.heading("scene", text="Scene")
        self.tree.heading("frames", text="Frames (RGB / IR)")
        self.tree.heading("status", text="Your progress")
        self.tree.column("env", width=110, anchor="w")
        self.tree.column("scene", width=80, anchor="w")
        self.tree.column("frames", width=130, anchor="w")
        self.tree.column("status", width=260, anchor="w")
        vsb = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<Double-1>", lambda e: self.open_scene())
        self.tree.bind("<Return>", lambda e: self.open_scene())

        actions = ttk.Frame(body)
        actions.pack(fill="x", pady=6)
        ttk.Button(actions, text="Open scene", command=self.open_scene).pack(side="left")
        ttk.Button(actions, text="Refresh list", command=self.refresh_tree).pack(side="left", padx=6)
        ttk.Button(actions, text="Export all my annotations",
                   command=self.export_all).pack(side="left")
        ttk.Button(actions, text="Help", command=lambda: self.show_help()).pack(side="right")

        hint = ttk.Label(
            self,
            text="Select a scene and press Open. Pause at the frame where each new segment starts\n"
                 "and press B. Save when the whole video is divided into segments.")
        hint.pack(anchor="w", padx=10, pady=(0, 10))
        self._update_info()

    def _update_info(self):
        self.info_var.set("Annotator: %s   |   Dataset: %s"
                          % (self.annotator_id, self.dataset_root or "—"))

    # -- actions ------------------------------------------------------------

    def change_annotator(self):
        name = self._ask_name(initial=self.annotator_id)
        if not name:
            return
        self.annotator_id = name
        self.cfg["annotator_id"] = self.annotator_id
        save_config(self.cfg)
        self._update_info()
        self.refresh_tree()

    def change_dataset(self):
        chosen = filedialog.askdirectory(
            title="Select the folder with the fire-VAD videos (environment_1, ...)", parent=self)
        if not chosen:
            return
        root = find_dataset_root(chosen)
        if not root:
            if not messagebox.askyesno(APP_NAME, NO_DATASET_MESSAGE +
                                       "\n\nUse this folder anyway?", parent=self):
                return
            root = chosen
        self.dataset_root = root
        self.cfg["dataset_root"] = root
        save_config(self.cfg)
        self._update_info()
        self.refresh_tree()

    def refresh_tree(self):
        self.scenes = discover_scenes(self.dataset_root)
        self.tree.delete(*self.tree.get_children())
        for i, sc in enumerate(self.scenes):
            self.tree.insert("", "end", iid=str(i), values=(
                "environment_%d" % sc["environment"],
                "%d" % sc["scene"],
                "%d / %d" % (len(sc["rgb_files"]), sc["ir_count"]),
                scene_status(self.annotator_id, sc["environment"], sc["scene"],
                             sc["n_frames"])))

    def _annotate_open(self):
        w = self.active_window
        if w is None:
            return False
        try:
            return bool(w.winfo_exists())
        except Exception:
            return False

    def open_scene(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo(APP_NAME, "Select a scene first.", parent=self)
            return
        if self._annotate_open():
            messagebox.showinfo(APP_NAME, "Close the open segmentation window first.", parent=self)
            return
        scene = self.scenes[int(sel[0])]
        loader = make_ir_loader(scene)
        if not self._preload_ir(scene, loader[0]):
            return
        self.active_window = AnnotateWindow(self, scene, loader)

    def _preload_ir(self, scene, load):
        """Decode the IR stack of a scene behind a progress bar. False on failure."""
        preload = getattr(load, "preload", None)
        if preload is None:
            return True
        dlg = tk.Toplevel(self)
        dlg.title("Loading")
        dlg.transient(self)
        dlg.resizable(False, False)
        ttk.Label(dlg, text="Loading environment %d · scene %d …"
                  % (scene["environment"], scene["scene"])).pack(padx=20, pady=(16, 8))
        bar = ttk.Progressbar(dlg, length=320, maximum=max(1, scene["ir_count"]))
        bar.pack(padx=20, pady=(0, 16))
        dlg.protocol("WM_DELETE_WINDOW", lambda: None)
        dlg.update_idletasks()
        px = self.winfo_x() + (self.winfo_width() - dlg.winfo_width()) // 2
        py = self.winfo_y() + (self.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry("+%d+%d" % (max(px, 40), max(py, 40)))
        dlg.grab_set()      # nothing else can be clicked while the scene decodes

        def progress(done, total):
            bar["value"] = done
            dlg.update()

        try:
            dlg.update()
            preload(progress)
        except Exception as e:
            dlg.destroy()
            messagebox.showerror(APP_NAME, "Could not load the IR frames of this scene:\n%s" % e,
                                 parent=self)
            return False
        dlg.destroy()
        return True

    def export_all(self):
        out = {
            "annotator_id": self.annotator_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_root": self.dataset_root,
            "annotations": [],
        }
        base = annotator_dir(self.annotator_id)
        if os.path.isdir(base):
            for env_name in sorted(os.listdir(base)):
                env_dir = os.path.join(base, env_name)
                if not (re.fullmatch(r"environment_\d+", env_name) and os.path.isdir(env_dir)):
                    continue
                for fn in sorted(os.listdir(env_dir)):
                    if fn.endswith(".json"):
                        data = load_json_safe(os.path.join(env_dir, fn))
                        if data:
                            out["annotations"].append(data)
        out["annotations"].sort(key=lambda d: (d.get("environment", 0), d.get("video_id", 0)))
        path = os.path.join(base, "combined_export.json")
        save_json_atomic(path, out)
        if messagebox.askyesno(
                APP_NAME,
                "Exported %d annotated video(s) to:\n%s\n\n"
                "Send this file to the dataset owner.\n\nOpen its folder now?"
                % (len(out["annotations"]), path), parent=self):
            if not reveal_in_file_manager(path):
                messagebox.showinfo(
                    APP_NAME, "Could not open the folder. The file is here:\n%s" % path,
                    parent=self)

    def show_help(self, parent=None):
        top = tk.Toplevel(parent or self)
        top.title("Help — Fire-VAD segmentation")
        top.geometry("560x420")
        txt = tk.Text(top, wrap="word", padx=10, pady=8)
        sb = ttk.Scrollbar(top, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        txt.insert("1.0", GUIDANCE_BOUNDARIES + "\n\n" + KEYBOARD_HELP)
        txt.configure(state="disabled")

    def _on_close(self):
        if self._annotate_open():
            self.active_window._on_close()
        self.destroy()


# --------------------------------------------------------------------------
# Segmentation window (viewer + segment list)
# --------------------------------------------------------------------------

PANEL_W = 430                 # provisional width before the first fit
PANEL_MIN_H = 120
PANEL_ASPECT = 540 / 400      # IR frame width / height
TIMELINE_H = 34


class AnnotateWindow(tk.Toplevel):

    def __init__(self, app, scene, ir_loader=None):
        super().__init__(app)
        self.app = app
        self.scene = scene
        self.env = scene["environment"]
        self.vid = scene["scene"]
        self.n_frames = scene["n_frames"]

        self.ir_files = scene["ir_files"]
        self.rgb_files = scene["rgb_files"]
        self._ir_loader, self.n_ir = ir_loader or make_ir_loader(scene)
        self.n_rgb = len(self.rgb_files)
        if self.n_ir != self.n_rgb:
            self.after(300, lambda: messagebox.showwarning(
                APP_NAME,
                "RGB and IR frame counts differ (%d vs %d). The last available "
                "frame is held when one stream ends."
                % (self.n_rgb, self.n_ir),
                parent=self))

        self.frame_idx = 0
        self.playing = False
        self._play_after = None
        self._poll_after = None
        self._fit_after = None
        self.dirty = False
        self._cur_ir = None
        self._photos = {}
        self._panel_geom = {}
        self._ir_cache_key = None
        self._ir_cache_arr = None
        self._rgb_cache_key = None
        self._rgb_cache_arr = None
        self._resume_msg = None
        self._abs_touched = False
        self.stats_ready = False

        # state
        self.model = SceneAnnotation(self.n_frames)
        self.cur_seg_start = 0        # start frame of the selected segment
        self.frame_info_var = tk.StringVar(value="")
        self._ir_temp_var = tk.StringVar(value="IR: —")
        self._frame_temp_var = tk.StringVar(value="")
        self._stats_progress_var = tk.StringVar(value="IR stats: …")
        self.dirty_var = tk.StringVar(value="")

        self.title("%s — environment %d · scene %d" % (APP_NAME, self.env, self.vid))
        self.colormap_var = tk.StringVar(value="JET")
        self.fps_var = tk.IntVar(value=30)
        self.abs_lo_var = tk.DoubleVar(value=0.0)
        self.abs_hi_var = tk.DoubleVar(value=40.0)

        self._load_previous_work()
        self._build_ui()
        self._bind_keys()
        self._init_stats()
        self._render_panels()
        self._measure_layout()
        self._draw_timeline()
        self._update_status()

        self.grab_set()
        self.transient(app)
        self.focus_set()
        self.lift()
        self.app._force_to_front_window(self)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -- data access --------------------------------------------------------

    def _load_ir(self, idx):
        if self._ir_cache_key == idx and self._ir_cache_arr is not None:
            return self._ir_cache_arr
        arr = self._ir_loader(idx)
        self._ir_cache_key = idx
        self._ir_cache_arr = arr
        return arr

    def _load_rgb(self, idx):
        if self._rgb_cache_key == idx and self._rgb_cache_arr is not None:
            return self._rgb_cache_arr
        i = min(max(idx, 0), len(self.rgb_files) - 1)
        with Image.open(self.rgb_files[i][1]) as im:
            arr = np.array(im.convert("RGB"))
        self._rgb_cache_key = idx
        self._rgb_cache_arr = arr
        return arr

    def _init_stats(self):
        cache = os.path.join(annotator_dir(self.app.annotator_id), "cache",
                             "environment_%d" % self.env, "%d.json" % self.vid)
        self.stats = FrameStats(self.n_frames, self._ir_loader, cache_path=cache)
        if not self.stats.try_load_cache():
            self.stats.start()
        self._poll_stats()

    def _poll_stats(self):
        done, total = self.stats.progress()
        if not self.stats.complete:
            self._stats_progress_var.set("IR stats: %d/%d frames" % (done, total))
            self._poll_after = self.after(400, self._poll_stats)
        else:
            self._stats_ready()

    def _stats_ready(self):
        if self.stats_ready:
            return
        self.stats_ready = True
        self._stats_progress_var.set("IR stats: done")
        lo, hi = self.stats.video_range()
        if lo is not None and hi is not None and not self._abs_touched:
            self.abs_lo_var.set(round(0, 1))
            self.abs_hi_var.set(round(100, 1))
            self._render_panels()

    # -- previous work resume ------------------------------------------------

    def _load_previous_work(self):
        prev = load_json_safe(scene_file_path(self.app.annotator_id, self.env, self.vid))
        if prev:
            self.model = SceneAnnotation.from_payload(prev, self.n_frames)
            self._resume_msg = "Loaded your saved segments for this scene."

    # -- UI -------------------------------------------------------------------

    def _build_ui(self):
        width = min(self.winfo_screenwidth() - 20, 1900)
        height = min(self.winfo_screenheight() - 100, 1150)
        self.geometry("%dx%d+10+30" % (width, height))
        self.minsize(1000, 640)
        # The screen size includes the Windows taskbar, which would cover the
        # Save bar (especially at 125-150 % scaling); a maximized window never does.
        self._start_zoomed = sys.platform == "win32"
        if self._start_zoomed:
            self.state("zoomed")
        # Provisional; _fit_panels() sizes the videos to the space left over
        # once the rest of the window has been laid out.
        self.panel_w, self.panel_height = PANEL_W, PANEL_MIN_H

        viewer = ttk.Frame(self)
        viewer.pack(fill="x", padx=8, pady=(6, 0))

        panels = ttk.Frame(viewer)
        panels.pack()
        self._panels_frame = panels
        self._panel_labels = {}
        for key, title in (("rgb", "RGB"), ("ir_rel", "IR — relative (per-frame scale)"),
                           ("ir_abs", "IR — absolute (°C, fixed scale)")):
            box = ttk.LabelFrame(panels, text=title)
            box.pack(side="left", padx=(0, 6))
            lbl = tk.Label(box, bd=1, relief="solid", bg="black")
            lbl.pack(padx=2, pady=2)
            lbl.bind("<Button-1>", lambda e: self.focus_set())
            self._panel_labels[key] = lbl
            if key.startswith("ir"):
                lbl.bind("<Motion>", lambda e, k=key: self._on_ir_motion(e, k))
                lbl.bind("<Leave>", lambda e: self._ir_temp_var.set("IR: —"))

        self.timeline = tk.Canvas(viewer, height=TIMELINE_H, bg="white",
                                  highlightthickness=1, highlightbackground="#bbb")
        self.timeline.pack(fill="x", pady=(4, 0))
        self.timeline.bind("<Button-1>", self._timeline_seek)
        self.timeline.bind("<B1-Motion>", self._timeline_seek)
        self.timeline.bind("<Configure>", lambda e: self._draw_timeline())

        # controls
        ctrl = ttk.Frame(viewer)
        ctrl.pack(fill="x", pady=3)
        self.play_btn = ttk.Button(ctrl, text="▶ Play", width=8, command=self._toggle_play)
        self.play_btn.pack(side="left")
        ttk.Button(ctrl, text="◀ -1", width=6,
                   command=lambda: self._seek(self.frame_idx - 1)).pack(side="left", padx=2)
        ttk.Button(ctrl, text="+1 ▶", width=6,
                   command=lambda: self._seek(self.frame_idx + 1)).pack(side="left")
        ttk.Label(ctrl, text="FPS:").pack(side="left", padx=(12, 2))
        fps_lbl = ttk.Label(ctrl, text=str(self.fps_var.get()), width=3)
        ttk.Scale(ctrl, from_=1, to=30, orient="horizontal", length=80,
                  variable=self.fps_var,
                  command=lambda v: fps_lbl.configure(text="%d" % round(float(v)))
                  ).pack(side="left")
        fps_lbl.pack(side="left", padx=(2, 0))

        ttk.Label(ctrl, text="Colormap:").pack(side="left", padx=(16, 2))
        cb = ttk.Combobox(ctrl, textvariable=self.colormap_var, state="readonly",
                          width=8, values=sorted(LUTS.keys()))
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self._render_panels())

        ttk.Label(ctrl, text="Abs. range (°C):").pack(side="left", padx=(16, 2))
        lo_spin = ttk.Spinbox(ctrl, from_=-50.0, to=2000.0, increment=0.1, width=7,
                              textvariable=self.abs_lo_var, command=self._abs_changed)
        hi_spin = ttk.Spinbox(ctrl, from_=-50.0, to=2000.0, increment=0.1, width=7,
                              textvariable=self.abs_hi_var, command=self._abs_changed)
        lo_spin.pack(side="left")
        hi_spin.pack(side="left", padx=2)
        for w in (lo_spin, hi_spin):
            w.bind("<Return>", lambda e: self._abs_changed())
            w.bind("<FocusOut>", lambda e: self._abs_changed())
        ttk.Button(ctrl, text="Video range", command=self._reset_abs_range).pack(side="left", padx=4)
        ttk.Label(ctrl, textvariable=self._stats_progress_var).pack(side="left", padx=8)

        bottom = ttk.Frame(self)
        bottom.pack(fill="both", expand=True, padx=8, pady=4)

        self._build_segments_panel(bottom)
        self._refresh_segment_list()

        # save bar
        savebar = ttk.Frame(self)
        savebar.pack(fill="x", side="bottom", padx=8, pady=(0, 2))
        ttk.Button(savebar, text="Segment instructions…",
                   command=self._show_boundaries_help).pack(side="left")
        ttk.Label(savebar, textvariable=self.dirty_var, foreground="#c62828").pack(side="left", padx=8)
        ttk.Button(savebar, text="Save & close", command=self._save_and_close).pack(side="right", padx=6)
        self.save_btn = ttk.Button(savebar, text="Save  (Ctrl+S)", command=self.save)
        self.save_btn.pack(side="right")
        ttk.Button(savebar, text="Close", command=self._on_close).pack(side="right", padx=6)

        # status bar
        status = ttk.Frame(self)
        status.pack(fill="x", side="bottom", padx=8, pady=(0, 6))
        ttk.Label(status, textvariable=self.frame_info_var).pack(side="left")
        ttk.Label(status, textvariable=self._ir_temp_var).pack(side="right")
        ttk.Label(status, textvariable=self._frame_temp_var).pack(side="right", padx=12)
        # Pack expandable content last so it cannot push the footer offscreen.
        bottom.pack_forget()
        bottom.pack(fill="both", expand=True, padx=8, pady=4)

        if self._resume_msg:
            msg = self._resume_msg
            self.after(400, lambda: messagebox.showinfo(APP_NAME, msg, parent=self))
            self._resume_msg = None

    def _build_segments_panel(self, parent):
        box = ttk.LabelFrame(parent, text="Segments — select one to jump to its start")
        box.pack(fill="both", expand=True)
        # Always two lines high, so the layout does not jump when a warning appears.
        self.short_warning_var = tk.StringVar(value="")
        self.short_warning = tk.Label(box, textvariable=self.short_warning_var, height=2,
                                      fg="#c62828", justify="left", anchor="nw",
                                      font=("Helvetica", 11, "bold"))
        self.short_warning.pack(fill="x", padx=6, pady=(2, 0))
        box.bind("<Configure>", lambda e: self.short_warning.configure(
            wraplength=max(100, e.width - 16)))
        self.seg_list = tk.Listbox(box, height=7, activestyle="dotbox", exportselection=False,
                                  font=("Helvetica", 12))
        self.seg_list.pack(fill="both", expand=True, padx=4, pady=4)
        self.seg_list.bind("<<ListboxSelect>>", self._on_list_select)

        btns = ttk.Frame(box)
        btns.pack(fill="x", padx=4)
        ttk.Button(btns, text="New segment starts here (B)",
                   command=self.add_boundary).pack(side="left")
        self.del_btn = ttk.Button(btns, text="Remove start boundary (Del)",
                                  command=self.delete_boundary)
        self.del_btn.pack(side="left", padx=4)

        btns2 = ttk.Frame(box)
        btns2.pack(fill="x", padx=4, pady=4)
        ttk.Label(btns2, text="Move start:").pack(side="left")
        self.nudge_down = ttk.Button(btns2, text="◀ -1", width=5,
                                     command=lambda: self.nudge_boundary(-1))
        self.nudge_down.pack(side="left", padx=(4, 0))
        self.nudge_up = ttk.Button(btns2, text="+1 ▶", width=5,
                                   command=lambda: self.nudge_boundary(1))
        self.nudge_up.pack(side="left", padx=2)
        ttk.Button(btns2, text="Go to start",
                   command=lambda: self._seek(self.cur_seg_start)).pack(side="left", padx=(8, 0))

    # -- key bindings -----------------------------------------------------------

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self._nav_key(-1))
        self.bind("<Right>", lambda e: self._nav_key(1))
        self.bind("<Up>", lambda e: self._nav_key(-10))
        self.bind("<Down>", lambda e: self._nav_key(10))
        self.bind("<Home>", lambda e: self._seek(0))
        self.bind("<End>", lambda e: self._seek(self.n_frames - 1))
        self.bind("<space>", lambda e: self._space_key())
        self.bind("<Key-b>", lambda e: self._b_key())
        self.bind("<Key-B>", lambda e: self._b_key())
        self.bind("<Delete>", lambda e: self._delete_key())
        self.bind("<F1>", lambda e: self._show_key_help())
        self.bind("<Control-s>", lambda e: self.save())
        self.bind("<Meta-s>", lambda e: self.save())

    _TEXTY_CLASSES = {"Text", "Entry", "TEntry", "Spinbox", "TSpinbox",
                      "TCombobox", "Combobox"}
    _BUTTONY_CLASSES = {"Button", "TButton", "TCheckbutton", "Checkbutton",
                        "TRadiobutton", "Radiobutton", "Scale"}

    def _nav_allowed(self):
        """Hotkeys that move frames: blocked while typing in a text widget/listbox."""
        w = self.focus_get()
        if w is None:
            return True
        cls = w.winfo_class()
        return cls not in self._TEXTY_CLASSES and cls not in ("Listbox", "TListbox")

    def _space_allowed(self):
        w = self.focus_get()
        if not self._nav_allowed():
            return False
        return w is None or w.winfo_class() not in self._BUTTONY_CLASSES

    def _nav_key(self, delta):
        if self._nav_allowed():
            self._seek(self.frame_idx + delta)

    def _space_key(self):
        if self._space_allowed():
            self._toggle_play()

    def _b_key(self):
        if self._nav_allowed():
            self.add_boundary()

    def _delete_key(self):
        w = self.focus_get()
        if w is None:
            self.delete_boundary()
            return
        cls = w.winfo_class()
        if cls in ("Listbox", "TListbox"):
            self.delete_boundary()
        elif cls not in self._TEXTY_CLASSES:
            self.delete_boundary()

    # -- frame navigation -----------------------------------------------------

    def _seek(self, frame, pause=True):
        frame = int(frame)
        frame = max(0, min(self.n_frames - 1, frame))
        changed = frame != self.frame_idx
        self.frame_idx = frame
        if pause and self.playing:
            self._toggle_play()
        if changed or "rgb" not in self._photos:
            self._render_panels()
        self._draw_timeline()
        self._update_status()
        self._sync_to_playhead()

    def _toggle_play(self):
        self.playing = not self.playing
        self.play_btn.configure(text="⏸ Pause" if self.playing else "▶ Play")
        if not self.playing:
            self._sync_to_playhead()
        if self.playing:
            if self.frame_idx >= self.n_frames - 1:
                self.frame_idx = 0
            self._schedule_play()

    def _schedule_play(self):
        if not self.playing:
            return
        delay = max(10, int(1000.0 / max(1, int(self.fps_var.get()))))
        self._play_after = self.after(delay, self._play_tick)

    def _play_tick(self):
        if not self.playing:
            return
        nxt = self.frame_idx + 1
        if nxt > self.n_frames - 1:
            self.playing = False
            self.play_btn.configure(text="▶ Play")
            self._update_status()
            self._sync_to_playhead()
            return
        self.frame_idx = nxt
        self._render_panels()
        self._draw_timeline()
        self._update_status()
        self._schedule_play()

    # -- rendering ------------------------------------------------------------

    def _render_panels(self):
        try:
            ir = self._load_ir(self.frame_idx)
            rgb = self._load_rgb(self.frame_idx)
        except Exception as e:
            self.playing = False
            self.play_btn.configure(text="▶ Play")
            messagebox.showerror(APP_NAME, "Failed to load frame %d:\n%s"
                                 % (self.frame_idx, e), parent=self)
            return
        self._cur_ir = ir
        lut = LUTS[self.colormap_var.get()]
        fmin, fmax = float(ir.min()), float(ir.max())
        rel = render_ir(ir, fmin, fmax, lut)
        lo, hi = self.abs_lo_var.get(), self.abs_hi_var.get()
        ab = render_ir(ir, lo, hi, lut)
        self._set_panel("rgb", Image.fromarray(rgb))
        self._set_panel("ir_rel", Image.fromarray(rel))
        self._set_panel("ir_abs", Image.fromarray(ab))
        self._frame_temp_var.set("frame %d: min %.1f °C  max %.1f °C"
                                 % (self.frame_idx, fmin, fmax))

    def _fit_panels(self):
        """Give the three videos all the space the rest of the layout does not need."""
        win_w, win_h = self.winfo_width(), self.winfo_height()
        if win_w < 100 or win_h < 100:      # not mapped yet
            win_w, win_h = self._initial_size
        avail_h = win_h - self._fixed_h
        avail_w = (win_w - self._fixed_w) // 3
        h = max(PANEL_MIN_H, min(avail_h, int(avail_w / PANEL_ASPECT)))
        w = int(h * PANEL_ASPECT)
        if (w, h) != (self.panel_w, self.panel_height):
            self.panel_w, self.panel_height = w, h
            self._render_panels()

    def _measure_layout(self):
        """Space taken by everything except the three video images."""
        self.update_idletasks()
        self._initial_size = (self.winfo_reqwidth(), self.winfo_reqheight())
        geo = re.match(r"(\d+)x(\d+)", self.geometry())
        if geo and int(geo.group(1)) > 100:
            self._initial_size = (int(geo.group(1)), int(geo.group(2)))
        self._fixed_h = self.winfo_reqheight() - self.panel_height
        self._fixed_w = self._panels_frame.winfo_reqwidth() - 3 * self.panel_w + 16
        self._fit_panels()
        self._fit_after = None
        self.bind("<Configure>", self._on_configure)

    def _restore_start_size(self):
        """Return to the size the window opened with."""
        if self._start_zoomed:
            self.state("zoomed")
        else:
            self.geometry("%dx%d" % self._initial_size)

    def _on_configure(self, event):
        if event.widget is not self:
            return
        if self._fit_after is not None:
            self.after_cancel(self._fit_after)
        self._fit_after = self.after(120, self._fit_panels)

    def _set_panel(self, key, img):
        pw, ph = self.panel_w, self.panel_height
        scale = min(pw / img.width, ph / img.height)
        new_w = max(1, round(img.width * scale))
        new_h = max(1, round(img.height * scale))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        off_x = (pw - new_w) // 2
        off_y = (ph - new_h) // 2
        canvas = Image.new("RGB", (pw, ph), (0, 0, 0))
        canvas.paste(resized, (off_x, off_y))
        photo = ImageTk.PhotoImage(canvas)
        self._panel_labels[key].configure(image=photo)
        self._photos[key] = photo
        self._panel_geom[key] = (scale, off_x, off_y, img.width, img.height)

    def _on_ir_motion(self, event, key):
        if self._cur_ir is None or key not in self._panel_geom:
            return
        scale, off_x, off_y, src_w, src_h = self._panel_geom[key]
        sx = (event.x - off_x) / scale
        sy = (event.y - off_y) / scale
        if 0 <= sx < src_w and 0 <= sy < src_h:
            val = float(self._cur_ir[int(sy), int(sx)])
            self._ir_temp_var.set("IR @ (x=%d, y=%d): %.1f °C" % (int(sx), int(sy), val))
        else:
            self._ir_temp_var.set("IR: —")

    def _abs_changed(self):
        self._abs_touched = True
        self._render_panels()

    def _reset_abs_range(self):
        lo, hi = self.stats.video_range()
        if lo is None:
            messagebox.showinfo(APP_NAME, "IR statistics are still being computed.",
                                parent=self)
            return
        self.abs_lo_var.set(round(lo, 1))
        self.abs_hi_var.set(round(hi, 1))
        self._abs_touched = False
        self._render_panels()

    # -- timeline ----------------------------------------------------------------

    def _tl_geom(self):
        w = max(int(self.timeline.winfo_width()), 200)
        pad = 8
        usable = max(1, w - 2 * pad)
        return pad, usable

    def _draw_timeline(self):
        c = self.timeline
        c.delete("all")
        pad, usable = self._tl_geom()
        n = self.n_frames
        if n < 2:
            return

        def x(f):
            return pad + (f / (n - 1)) * usable

        segs = self._segments_view()
        colors = ("#e6e6e6", "#cfcfcf")
        for i, (s, e) in enumerate(segs):
            x0 = x(s)
            x1 = max(x(e), x(s) + 1)
            short = e - s + 1 <= SHORT_SEGMENT_FRAMES
            fill = "#90caf9" if s == self.cur_seg_start else colors[i % 2]
            if short and s != self.cur_seg_start:
                fill = "#ffab91"
            c.create_rectangle(x0, 3, x1, 19, fill=fill,
                               outline="#c62828" if short else "#999",
                               width=2 if short else 1)
            if x1 - x0 > 34:
                c.create_text((x0 + x1) / 2, 11, text="%d–%d" % (s, e),
                              font=("Helvetica", 7), fill="#333")
        for s, _ in segs[1:]:
            c.create_line(x(s), 0, x(s), 22, fill="#404040", width=2)
        px = x(self.frame_idx)
        c.create_line(px, 0, px, 23, fill="#d32f2f", width=2)
        c.create_text(px, 29, text=str(self.frame_idx),
                      font=("Helvetica", 8, "bold"), fill="#d32f2f")

    def _timeline_seek(self, event):
        pad, usable = self._tl_geom()
        if self.n_frames < 2:
            return
        frac = (event.x - pad) / usable
        frac = max(0.0, min(1.0, frac))
        self._seek(int(round(frac * (self.n_frames - 1))))

    # -- segments -------------------------------------------------------------------

    def _segments_view(self):
        return self.model.segments()

    def _current_segment_index(self):
        return self.model.index_of_frame(self.frame_idx)

    def _selected_index(self):
        for i, (s, _) in enumerate(self.model.segments()):
            if s == self.cur_seg_start:
                return i
        return 0

    def _update_status(self):
        seg = self._segments_view()
        idx = self._current_segment_index()
        if idx is not None and seg:
            seg_txt = "segment %d/%d (frames %d–%d)" % (idx + 1, len(seg), seg[idx][0], seg[idx][1])
        else:
            seg_txt = ""
        self.frame_info_var.set("Frame %d / %d   %s" % (self.frame_idx, self.n_frames - 1, seg_txt))

    def _segment_label(self, i, start, end):
        n = end - start + 1
        short = "   ⚠ too short: delete a boundary" if n <= SHORT_SEGMENT_FRAMES else ""
        return "Segment %d   frames %d–%d   (%d frame%s)%s" % (
            i, start, end, n, "" if n == 1 else "s", short)

    def _refresh_segment_list(self):
        segs = self.model.segments()
        lb = self.seg_list
        lb.delete(0, "end")
        for i, (s, e) in enumerate(segs):
            lb.insert("end", self._segment_label(i, s, e))
            if e - s + 1 <= SHORT_SEGMENT_FRAMES:
                lb.itemconfig(i, foreground="#c62828")
        self.short_warning_var.set(short_segment_warning(segs))
        sel = self._selected_index()
        lb.selection_set(sel)
        lb.see(sel)
        state = "disabled" if sel == 0 else "normal"
        for b in (self.del_btn, self.nudge_down, self.nudge_up):
            b.configure(state=state)
        self._draw_timeline()
        self._update_status()

    def _select_segment(self, idx, jump=False):
        segs = self.model.segments()
        idx = max(0, min(idx, len(segs) - 1))
        self.cur_seg_start = segs[idx][0]
        self._refresh_segment_list()
        if jump:
            self._seek(self.cur_seg_start)

    def _sync_to_playhead(self):
        """While paused, the selected segment follows the playhead."""
        if self.playing:
            return
        idx = self._current_segment_index()
        if idx is not None and self.model.segments()[idx][0] != self.cur_seg_start:
            self._select_segment(idx)

    def _on_list_select(self, event=None):
        sel = self.seg_list.curselection()
        if sel:
            self._select_segment(sel[0], jump=True)
            self.focus_set()    # keep B, Space and the arrows working after a click

    def add_boundary(self):
        ok, msg = self.model.add_boundary(self.frame_idx)
        if ok:
            self._mark_dirty(silent=True)
            self._select_segment(self.model.index_of_frame(self.frame_idx))
        self._flash(msg)

    def delete_boundary(self):
        segs = self.model.segments()
        i = self._selected_index()
        if i == 0:
            self._flash("Segment 0 always starts at frame 0; select a later segment.")
            return
        start = segs[i][0]
        ok, msg = self.model.remove_boundary(start)
        self.cur_seg_start = segs[i - 1][0]
        self._mark_dirty(silent=True)
        self._refresh_segment_list()
        self._flash(msg)

    def nudge_boundary(self, delta):
        ok, msg, new = self.model.nudge_boundary(self.cur_seg_start, delta)
        if ok:
            self.cur_seg_start = new
            self._mark_dirty(silent=True)
            self._refresh_segment_list()
            self._seek(new)
        self._flash(msg)

    def _flash(self, msg):
        self.frame_info_var.set(msg)
        if msg:
            self.after(2500, self._update_status)

    # -- help dialogs ---------------------------------------------------------------

    def _show_boundaries_help(self):
        top = tk.Toplevel(self)
        top.title("Segment instructions")
        top.geometry("560x420")
        txt = tk.Text(top, wrap="word", padx=10, pady=8)
        sb = ttk.Scrollbar(top, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        txt.insert("1.0", GUIDANCE_BOUNDARIES + "\n\n" + KEYBOARD_HELP)
        txt.configure(state="disabled")

    def _show_key_help(self):
        top = tk.Toplevel(self)
        top.title("Keyboard controls")
        txt = tk.Text(top, wrap="word", padx=10, pady=8, height=14)
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", KEYBOARD_HELP)
        txt.configure(state="disabled")

    # -- saving --------------------------------------------------------------------

    def _mark_dirty(self, silent=False):
        self.dirty = True
        self.dirty_var.set("● unsaved changes")
        if not silent:
            self._draw_timeline()
            self._update_status()

    def save(self):
        warning = short_segment_warning(self.model.segments())
        if warning and not messagebox.askyesno(
                APP_NAME, warning + "\n\nSave anyway? You can continue later.", parent=self):
            return False
        segs = self.model.to_segments(self.stats)
        payload = build_annotation_payload(
            self.env, self.vid, self.app.annotator_id, segs)
        path = scene_file_path(self.app.annotator_id, self.env, self.vid)
        save_json_atomic(path, payload)
        self.dirty = False
        self.dirty_var.set("")
        self._flash("Saved %d segment(s) to %s" % (len(payload["segments"]), path))
        self.app.refresh_tree()
        return True

    def _save_and_close(self):
        if self.save():
            self._force_close()

    # -- closing ---------------------------------------------------------------------

    def _on_close(self):
        if self.dirty:
            answer = messagebox.askyesnocancel(
                APP_NAME, "You have unsaved changes.\n\nSave before closing?", parent=self)
            if answer is None:
                return
            if answer:
                if not self.save():
                    return
        self._force_close()

    def _force_close(self):
        if self._play_after is not None:
            try:
                self.after_cancel(self._play_after)
            except Exception:
                pass
        for pending in (self._poll_after, self._fit_after):
            if pending is not None:
                try:
                    self.after_cancel(pending)
                except Exception:
                    pass
        self.playing = False
        # Drop the IR loader (and any cached frames) so a decompressed IR.npz
        # stack is released as soon as the window is gone.
        self._ir_loader = None
        self._ir_cache_key = None
        self._ir_cache_arr = None
        self._cur_ir = None
        try:
            self.grab_release()
        except Exception:
            pass
        self.destroy()
        self.app.active_window = None
        self.app.refresh_tree()


# --------------------------------------------------------------------------
# Selftest (headless)
# --------------------------------------------------------------------------

def run_selftest(dataset_root=None, full_scene_limit=1):
    if dataset_root:
        root = find_dataset_root(dataset_root) or dataset_root
    else:
        root = default_dataset_root()
    failures = []
    checks = []

    def check(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        if not cond:
            failures.append("%s %s" % (name, detail))

    scenes = discover_scenes(root)
    check("scene discovery", len(scenes) > 0, "found %d scenes" % len(scenes))

    check("annotator folder names",
          sanitize_annotator_dir("Ivana Šimić") == "Ivana_Simic"
          and sanitize_annotator_dir(" Đurđa Kovačević ") == "Djurdja_Kovacevic"
          and sanitize_annotator_dir("..") == "annotator",
          sanitize_annotator_dir("Ivana Šimić"))
    check("parse '0'", parse_frame_index("0") == 0)
    check("parse legacy", parse_frame_index("ir_frame_203_28.728231") == 203)
    check("parse junk", parse_frame_index("abc") is None)

    segs = boundaries_to_segments([34, 210], 632)
    check("segments", segs == [(0, 33), (34, 209), (210, 631)], str(segs))
    check("no boundaries", boundaries_to_segments([], 632) == [(0, 631)])
    check("bounds roundtrip", segments_to_boundaries(segs) == [34, 210])
    ok, msg = add_boundary_value([34], 34, 632)
    check("duplicate boundary rejected", not ok)
    ok, msg, b = nudge_boundary_value([34, 100], 0, -1, 632)
    check("nudge", ok and b == [33, 100], str(b))

    tmp = tempfile.mkdtemp(prefix="firevad_selftest_")
    try:
        # IR_delta.npz decoding: synthetic archive, built with the documented encoder.
        # 70 frames cross the chunk boundaries; large random jumps force modulo-2**16 wrap.
        rng = np.random.default_rng(1)
        synth = (rng.standard_normal((70, 6, 8)) * 40 + 25).astype(np.float16)
        u = synth.view(np.uint16)
        diff = u.copy()
        diff[1:] = u[1:] - u[:-1]
        synth_path = os.path.join(tmp, "synthetic_IR_delta.npz")
        np.savez_compressed(synth_path, format=np.array(IR_DELTA_FORMAT),
                            frame_indices=np.arange(70, dtype=np.int32),
                            lo=(diff & 0xFF).astype(np.uint8), hi=(diff >> 8).astype(np.uint8))
        seen = []
        decoded = decode_delta_archive(synth_path, lambda d, t: seen.append((d, t)))
        check("delta decode is bit-exact",
              decoded.shape == synth.shape and np.array_equal(decoded.view(np.uint16), u))
        check("delta decode progress", seen and seen[-1] == (70, 70) and len(seen) == 3, str(seen))
        check("delta archive header", delta_archive_info(synth_path) == (70, 6, 8)
              and delta_archive_info(os.path.join(tmp, "missing.npz")) is None)
        bad_path = os.path.join(tmp, "bad_format.npz")
        np.savez_compressed(bad_path, format=np.array("other"), lo=diff.astype(np.uint8),
                            hi=diff.astype(np.uint8))
        try:
            decode_delta_archive(bad_path)
            check("unknown archive format rejected", False)
        except ValueError:
            check("unknown archive format rejected", True)

        sampled, seen_env = set(), set()
        for si, sc in enumerate(scenes):         # loading is checked on one scene per environment
            if sc["environment"] not in seen_env:
                seen_env.add(sc["environment"])
                sampled.add(si)
        for si, sc in enumerate(scenes):
            env, vid, n = sc["environment"], sc["scene"], sc["n_frames"]
            loader, n_ir = make_ir_loader(sc)
            n_rgb = len(sc["rgb_files"])
            check("scene %d/%d files" % (env, vid), n_ir > 0 and n_rgb > 0,
                  "ir=%d rgb=%d format=%s" % (n_ir, n_rgb, sc["ir_format"]))
            if si not in sampled:
                continue
            try:
                ir = loader(0)
                if sc["ir_format"] == "delta":
                    # the three-line decode documented in the dataset's IR_COMPRESSION.md
                    with np.load(sc["ir_archive"]) as z:
                        dd = (z["hi"].astype(np.uint16) << 8) | z["lo"].astype(np.uint16)
                        ref = np.cumsum(dd, axis=0, dtype=np.uint16).view(np.float16)
                        ref_idx = z["frame_indices"]
                    last = loader(n_ir - 1)
                    check("scene %d/%d delta matches reference decode" % (env, vid),
                          np.array_equal(ref[0].view(np.uint16), ir.view(np.uint16))
                          and np.array_equal(ref[-1].view(np.uint16), last.view(np.uint16))
                          and len(ref_idx) == n_ir
                          and [i for i, _ in sc["rgb_files"]] == list(map(int, ref_idx)),
                          "frames=%d" % n_ir)
                    del ref, dd
                with Image.open(sc["rgb_files"][0][1]) as im:
                    rgb = np.array(im.convert("RGB"))
                check("scene %d/%d ir 2D" % (env, vid),
                      ir.ndim == 2 and np.issubdtype(ir.dtype, np.number),
                      "shape=%s dtype=%s" % (ir.shape, ir.dtype))
                check("scene %d/%d rgb 3ch" % (env, vid),
                      rgb.ndim == 3 and rgb.shape[2] == 3, str(rgb.shape))
                for cname in sorted(LUTS):
                    img = Image.fromarray(
                        render_ir(ir, float(ir.min()), float(ir.max()), LUTS[cname]))
                    img.save(os.path.join(tmp, "e%ds%d_%s.png" % (env, vid, cname)))
                check("scene %d/%d render" % (env, vid), True)
            except Exception as e:
                check("scene %d/%d load" % (env, vid), False, repr(e))

            if si == min(sampled):
                cache = os.path.join(tmp, "cache_e%d_s%d.json" % (env, vid))
                partial = {"n_frames": n, "complete": False,
                           "frame_min": [0.0] * n, "frame_max": [1.0] * n}
                save_json_atomic(cache, partial)
                stats2 = FrameStats(n, lambda i, ld=loader: ld(i),
                                    cache_path=cache)
                check("partial cache ignored", not stats2.try_load_cache())
                stats = FrameStats(n, lambda i, ld=loader: ld(i),
                                   cache_path=cache)
                stats.ensure_range(0, n - 1)
                stats.save_cache()
                stats3 = FrameStats(n, lambda i, ld=loader: ld(i),
                                   cache_path=cache)
                check("full cache accepted", stats3.try_load_cache())
                lo, hi = stats.video_range()
                check("video range", lo is not None and hi is not None and hi >= lo,
                      "%.2f..%.2f" % (lo, hi))
                smin, smax = stats.segment_minmax(0, min(9, n - 1))
                check("segment stats", smin is not None and smax >= smin,
                      "%.2f..%.2f" % (smin, smax))

        segs_payload = [
            {"start_frame": 0, "end_frame": 33, "temp_min_c": 12.0, "temp_max_c": 22.0},
            {"start_frame": 34, "end_frame": 631, "temp_min_c": 12.0, "temp_max_c": 300.0},
        ]
        payload = build_annotation_payload(1, 2, "Test Annotator", segs_payload)
        path = os.path.join(tmp, "payload.json")
        save_json_atomic(path, payload)
        back = load_json_safe(path)
        check("payload top keys",
              set(back.keys()) == {"video_id", "environment", "annotator_id", "segments"},
              str(sorted(back.keys())))
        expected_seg = {"segment_id", "start_frame", "end_frame", "temp_min_c", "temp_max_c"}
        check("segment keys (no captions)",
              all(set(sg.keys()) == expected_seg for sg in back["segments"]),
              str(sorted(back["segments"][0].keys())))
        check("video_id/env", back["video_id"] == 2 and back["environment"] == 1)
        check("segment ids", [sg["segment_id"] for sg in back["segments"]] == [0, 1])

        m = SceneAnnotation(632)
        check("single segment by default", m.segments() == [(0, 631)])
        m.add_boundary(100)
        m.add_boundary(50)
        check("model split", m.segments() == [(0, 49), (50, 99), (100, 631)], str(m.segments()))
        check("model index_of_frame",
              m.index_of_frame(0) == 0 and m.index_of_frame(99) == 1 and m.index_of_frame(631) == 2)
        check("model rejects frame 0 and duplicates",
              not m.add_boundary(0)[0] and not m.add_boundary(50)[0])
        ok, _, new = m.nudge_boundary(100, 1)
        check("model nudge", ok and new == 101 and m.boundaries == [50, 101], str(m.boundaries))
        check("model nudge collision", not m.nudge_boundary(50, 51)[0])
        lim = SHORT_SEGMENT_FRAMES
        edge = boundaries_to_segments([lim, lim + 1 + lim, 3 * lim], 5 * lim)
        check("short segments: <= limit flagged, limit+1 not",
              short_segment_indices(edge) == [0, 2] and edge[1] == (lim, 2 * lim)
              and short_segment_indices([(0, lim - 1), (lim, 2 * lim)]) == [0],
              str(short_segment_indices(edge)))
        check("short segment warning text",
              short_segment_warning([(0, 99)]) == ""
              and "Segment 1 is" in short_segment_warning([(0, 99), (100, 120), (121, 400)])
              and "Segments 0, 2 are" in short_segment_warning([(0, 9), (10, 99), (100, 120)]))
        check("model segment 0 fixed", not m.remove_boundary(0)[0]
              and not m.nudge_boundary(0, 1)[0])
        m.remove_boundary(50)
        check("model merge", m.segments() == [(0, 100), (101, 631)], str(m.segments()))
        back2 = SceneAnnotation.from_payload(
            build_annotation_payload(1, 2, "T", m.to_segments()), 632)
        check("model roundtrip", back2.boundaries == [101])
        messy = {"segments": [{"start_frame": 0, "end_frame": 29},
                              {"start_frame": 30, "end_frame": 700},
                              {"start_frame": 900, "end_frame": 950},
                              {"end_frame": 5}]}
        check("invalid starts dropped",
              SceneAnnotation.from_payload(messy, 632).boundaries == [30])
        with_captions = {"segments": [
            {"start_frame": 0, "end_frame": 29, "caption_rgb_initial": "a"},
            {"start_frame": 30, "end_frame": 631, "caption_rgb_dynamics": "b"}]}
        check("files with captions load as boundaries",
              SceneAnnotation.from_payload(with_captions, 632).boundaries == [30])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("Fire-VAD segmentation tool — selftest")
    for name, ok, detail in checks:
        print("  %-34s %s %s" % (name, "PASS" if ok else "FAIL", "" if ok else detail))
    print("%d/%d checks passed." % (len(checks) - len(failures), len(checks)))
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        return 1
    return 0


# --------------------------------------------------------------------------
# Scripted GUI test (dev only)
# --------------------------------------------------------------------------

_GUITEST_FAILED = False


def run_guitest(dataset_root=None):
    global PROJECT_DIR, _GUITEST_FAILED
    PROJECT_DIR = tempfile.mkdtemp(prefix="firevad_guitest_")
    _GUITEST_FAILED = False
    # No annotator_id in the config on purpose: this exercises the REAL
    # first-run path (name dialog). The env var auto-answers the dialog.
    os.environ["FIREVAD_GUITEST_NAME"] = "GUI Test"
    save_config({
        "dataset_root": dataset_root or default_dataset_root(),
    })
    real_info = messagebox.showinfo
    messagebox.showinfo = lambda *a, **k: None   # resume/export dialogs would block
    try:
        app = AnnotationApp()
        app.after(600, lambda: _guitest_guard(app, _guitest_step1))
        app.mainloop()
    finally:
        messagebox.showinfo = real_info
    os.environ.pop("FIREVAD_GUITEST_NAME", None)
    shutil.rmtree(PROJECT_DIR, ignore_errors=True)
    if _GUITEST_FAILED:
        return 1
    return 0


def _guitest_guard(app, fn):
    global _GUITEST_FAILED
    try:
        fn(app)
    except Exception:
        traceback.print_exc()
        print("GUI TEST FAILED")
        _GUITEST_FAILED = True
        try:
            app.destroy()
        except Exception:
            pass


def _guitest_step1(app):
    # first-run name dialog must have been visible and accepted
    assert app.annotator_id == "GUI Test", "first-run name dialog did not work"
    assert getattr(app, "_name_dlg_viewable", None) is True, "name dialog was not visible"
    assert bool(app.winfo_viewable()), "main window not visible"
    app.tree.selection_set(app.tree.get_children()[0])
    app.open_scene()
    app.after(2500, lambda: _guitest_guard(app, _guitest_step2))


def _guitest_step2(app):
    w = app.active_window
    assert w is not None, "segmentation window did not open"
    assert bool(w.winfo_viewable()), "segmentation window not visible"
    for widget in [w.save_btn, w.seg_list]:
        assert widget.winfo_viewable(), "the segment list or Save control is hidden"
        assert widget.winfo_rooty() + widget.winfo_height() <= \
            w.winfo_rooty() + w.winfo_height(), "the segment list or Save control is clipped"
    print("video panels: %dx%d px in a %dx%d window"
          % (w.panel_w, w.panel_height, w.winfo_width(), w.winfo_height()))
    # videos follow the window size, and the segment list/Save stay reachable
    big = w.panel_height
    if w._start_zoomed:
        assert w.state() == "zoomed", "segmentation window did not open maximized"
        w.state("normal")
    w.geometry("1000x640")
    w.update()
    w._fit_panels()
    w.update()
    assert w.panel_height < big, "videos did not shrink with the window"
    for widget in [w.save_btn, w.seg_list]:
        assert widget.winfo_rooty() + widget.winfo_height() <= \
            w.winfo_rooty() + w.winfo_height(), "control clipped in a small window"
    w._restore_start_size()
    w.update()
    w._fit_panels()
    assert w.panel_height == big, "videos did not grow back"
    assert not hasattr(w, "caption_texts"), "caption fields should not exist"
    assert w.seg_list.size() == 1 and w.cur_seg_start == 0, "new scene should have 1 segment"
    assert str(w.del_btn.cget("state")) == "disabled", "segment 0 start must not be removable"
    w._seek(50)
    w.add_boundary()
    assert w.cur_seg_start == 50 and w.seg_list.size() == 2, "new segment not selected after B"
    assert str(w.del_btn.cget("state")) == "normal", "later segment start should be removable"
    w._seek(120)
    w.add_boundary()
    assert w.seg_list.size() == 3 and w.cur_seg_start == 120, "expected 3 segments"
    assert w.dirty, "adding boundaries should mark the window unsaved"
    assert not w.short_warning_var.get(), "no short segment yet, but a warning is shown"
    # a segment of <= 40 frames is flagged everywhere and asked about when saving
    w._seek(120 + SHORT_SEGMENT_FRAMES - 1)
    w.add_boundary()
    assert w.model.segments()[2] == (120, 120 + SHORT_SEGMENT_FRAMES - 2), w.model.segments()
    assert "Segment 2 is" in w.short_warning_var.get() and "Delete a boundary" in \
        w.short_warning_var.get(), "short segment warning missing"
    assert "⚠" in w.seg_list.get(2) and "⚠" not in w.seg_list.get(1), "list does not flag it"
    prompts = []
    real_ask = messagebox.askyesno
    messagebox.askyesno = lambda *a, **k: prompts.append(a[1]) or False
    try:
        assert not w.save() and len(prompts) == 1 and "Segment 2" in prompts[0], \
            "saving with a short segment did not ask"
        assert load_json_safe(scene_file_path(app.annotator_id, w.env, w.vid)) is None, \
            "a declined save wrote a file"
    finally:
        messagebox.askyesno = real_ask
    w._seek(120 + SHORT_SEGMENT_FRAMES - 1)   # inside the last segment
    assert w.cur_seg_start == 120 + SHORT_SEGMENT_FRAMES - 1
    w.nudge_boundary(1)
    assert w.short_warning_var.get(), "40 frames is still too short"
    w.nudge_boundary(1)
    assert w.model.segments()[2][1] - w.model.segments()[2][0] + 1 == SHORT_SEGMENT_FRAMES + 1 \
        and not w.short_warning_var.get(), "a segment of 41 frames must not be flagged"
    w.nudge_boundary(-1)
    assert w.short_warning_var.get(), "warning did not come back"
    w._select_segment(3, jump=True)
    w.delete_boundary()
    assert not w.short_warning_var.get() and w.seg_list.size() == 3, "warning not cleared"
    # paused playhead in segment 0 -> the selection follows it
    w._seek(10)
    assert w.cur_seg_start == 0, "segment selection did not follow the playhead"
    # B, Space and the arrows keep working after a click in the segment list
    w.seg_list.selection_clear(0, "end")
    w.seg_list.selection_set(1)
    w._on_list_select()
    assert w.frame_idx == 50, "list selection did not jump to segment start"
    w.focus_get = lambda: w.seg_list
    try:
        assert not w._nav_allowed(), "a focused list should keep its own arrow keys"
    finally:
        del w.focus_get
    # move segment 1's start by +1
    w.nudge_boundary(1)
    assert w.cur_seg_start == 51 and w.frame_idx == 51, "nudge failed"
    assert w.model.boundaries == [51, 120], w.model.boundaries
    # remove the last segment's start boundary
    w._select_segment(2, jump=True)
    w.delete_boundary()
    assert w.seg_list.size() == 2 and w.cur_seg_start == 51, "remove boundary failed"
    w._seek(w.n_frames - 1)
    w._seek(w.n_frames + 100)
    assert w.frame_idx == w.n_frames - 1, "seek must clamp to the last frame"
    assert w.save(), "save failed"
    assert not w.dirty and not w.dirty_var.get(), "saving should clear the unsaved mark"
    data = load_json_safe(scene_file_path(app.annotator_id, w.env, w.vid))
    assert [(s["start_frame"], s["end_frame"]) for s in data["segments"]] == \
        [(0, 50), (51, w.n_frames - 1)], "saved segments wrong"
    assert all(not any("caption" in k for k in sg) for sg in data["segments"]), \
        "saved segments must not hold captions"
    assert data["segments"][0]["temp_min_c"] is not None, "temp stats missing"
    assert "saved" in app.tree.item(app.tree.get_children()[0])["values"][3], "status not saved"
    w._force_close()
    app.after(600, lambda: _guitest_guard(app, _guitest_step3))


def _guitest_step3(app):
    # reopen: saved work is restored
    app.tree.selection_set(app.tree.get_children()[0])
    app.open_scene()
    w = app.active_window
    assert w.model.boundaries == [51], "saved boundaries not restored"
    # closing with unsaved changes asks; answering "No" to saving discards nothing on disk
    w._seek(200)
    w.add_boundary()
    assert w.dirty
    asked = []
    real_ask = messagebox.askyesnocancel
    messagebox.askyesnocancel = lambda *a, **k: asked.append(a) or False
    try:
        w._on_close()
    finally:
        messagebox.askyesnocancel = real_ask
    assert asked, "closing with unsaved changes did not ask"
    assert load_json_safe(scene_file_path(app.annotator_id, w.env, w.vid))["segments"][-1][
        "start_frame"] == 51, "unsaved change reached the disk"
    app.after(400, lambda: _guitest_guard(app, _guitest_step4))


def _guitest_step4(app):
    asked = []
    real_ask = messagebox.askyesno
    # "No" to "Open its folder now?" so no file manager pops up during the test.
    messagebox.askyesno = lambda *a, **k: asked.append(a) or False
    try:
        app.export_all()
    finally:
        messagebox.askyesno = real_ask
    assert asked and "combined_export.json" in asked[0][1], "export did not offer the folder"
    combined = load_json_safe(
        os.path.join(annotator_dir(app.annotator_id), "combined_export.json"))
    assert combined and len(combined["annotations"]) == 1, "combined export wrong"
    assert combined["annotator_id"] == "GUI Test", "combined export annotator wrong"
    # change annotator through the dialog (auto-answered)
    os.environ["FIREVAD_GUITEST_NAME"] = "GUI Test Rename"
    app.change_annotator()
    assert app.annotator_id == "GUI Test Rename", "change annotator failed"
    assert getattr(app, "_name_dlg_viewable", None) is True, "rename dialog not visible"
    print("GUI TEST OK — first-run dialog, boundaries, resume, unsaved-changes prompt, "
          "export and rename validated.")
    app.after(300, app.destroy)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--selftest", action="store_true",
                        help="headless data/logic validation, then exit")
    parser.add_argument("--guitest", action="store_true",
                        help="scripted end-to-end GUI test (dev only)")
    parser.add_argument("--dataset-root", default=None,
                        help="path to the dataset root (folder with environment_* subfolders)")
    args = parser.parse_args()

    try:
        import numpy  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError as e:
        print("Missing dependency: %s\nInstall with:  pip install numpy pillow" % e)
        sys.exit(1)

    if args.dataset_root and not (args.selftest or args.guitest):
        cfg = load_config()
        cfg["dataset_root"] = os.path.abspath(args.dataset_root)
        save_config(cfg)

    if args.selftest:
        sys.exit(run_selftest(args.dataset_root))
    enable_windows_dpi_awareness()      # must precede the first Tk window
    if args.guitest:
        sys.exit(run_guitest(args.dataset_root))
    try:
        app = AnnotationApp()
        app.mainloop()
    except Exception:
        report_fatal_error()
        sys.exit(1)


def report_fatal_error():
    """Print the traceback; the windowed builds have no console, so show it in a dialog."""
    text = traceback.format_exc()
    if sys.stderr:
        sys.stderr.write(text)
    if not FROZEN:
        return
    log = os.path.join(PROJECT_DIR, "error.log")
    try:
        os.makedirs(PROJECT_DIR, exist_ok=True)
        with open(log, "w", encoding="utf-8") as f:
            f.write(text)
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            APP_NAME, "The tool stopped because of an error.\n\n"
            "Send this file to the coordinator:\n%s\n\n%s"
            % (log, text.strip().splitlines()[-1]))
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    main()
