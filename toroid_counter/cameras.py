"""Camera sources: USB / built-in cameras by name, or a network stream (phone app, RTSP, MJPEG)."""
from __future__ import annotations

import sys

import cv2

URL_HINT = "Network / phone camera (URL)..."


def list_cameras(max_probe: int = 6) -> list[tuple[int, str]]:
    """[(index, name)] of the cameras the computer can see.

    Windows: names come from DirectShow (pygrabber), in the same order OpenCV uses with CAP_DSHOW.
    Elsewhere, or if that fails, indexes are probed and named "Camera N".
    """
    if sys.platform == "win32":
        try:
            from pygrabber.dshow_graph import FilterGraph
            names = FilterGraph().get_input_devices()
            if names:
                return list(enumerate(names))
        except Exception:
            pass
    found = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(i)
        ok = cap.isOpened()
        cap.release()
        if ok:
            found.append((i, f"Camera {i}"))
    return found


def label(index: int, name: str) -> str:
    return f"{index}: {name}"


def parse_label(text: str):
    """Combobox text -> int index, URL string, or None."""
    text = (text or "").strip()
    if not text or text == URL_HINT:
        return None
    if "://" in text:
        return text
    head = text.split(":", 1)[0].strip()
    return int(head) if head.isdigit() else None


def open_source(src, width: int = 3840, height: int = 2160):
    """Open a camera index or a stream URL. Returns an opened cv2.VideoCapture or raises."""
    if isinstance(src, str):
        # phone IP-camera apps: http://<phone-ip>:8080/video (MJPEG) or rtsp://...
        cap = cv2.VideoCapture(src)
    else:
        cap = cv2.VideoCapture(src, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(src)
        # ask for the highest resolution the camera offers; it falls back to what it supports
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera {src!r}")
    return cap
