# Copyright (c) 2026 Zetta Contributors
"""Episode-owned RGB storage and immutable public snapshots."""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np
from PIL import Image

CAMERAS = ("front_rgb", "left_rgb", "right_rgb")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ImageStore:
    def __init__(self, root: Path, writer=None):
        self.root = root
        self.writer = writer
        self.registered: set[str] = set()

    def publish(self, images):
        references, owned = {}, {}
        for camera in CAMERAS:
            pixels = np.array(images[camera], copy=True, order="C")
            if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError("camera must be uint8 RGB")
            stream = io.BytesIO()
            # Real-time mode uses inexpensive lossless PNG packaging. File SHA
            # and the public byte contract stay known before publication.
            Image.fromarray(pixels).save(stream, format="PNG", **(
                {"compress_level": 0} if self.writer is not None else {}))
            payload = stream.getvalue()
            sha = hashlib.sha256(payload).hexdigest()
            content_id = "rgb-" + sha
            if content_id not in self.registered:
                path = self.root / (content_id + ".png")
                if self.writer is None:
                    atomic_write(path, payload)
                else:
                    self.writer.submit(atomic_write, path, payload)
                self.registered.add(content_id)
            references[camera] = {
                "content_id": content_id,
                "sha256": sha,
                "width": pixels.shape[1],
                "height": pixels.shape[0],
                "encoding": "png",
            }
            pixels.flags.writeable = False
            owned[camera] = pixels
        return references, owned
