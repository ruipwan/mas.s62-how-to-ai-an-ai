import hashlib
import json
import os
from pathlib import Path
import time
import tempfile
import threading


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Independent experiment/report processes can publish at the same time.
    # Give each writer its own temporary file before the atomic replacement.
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            tmp = Path(stream.name)
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Events:
    def __init__(self, directory):
        self.path = Path(directory) / "events.jsonl"
        self.rows = []
        self.start = time.monotonic()
        self.round = -1
        self.lock = threading.RLock()

    def __call__(self, event):
        with self.lock:
            self.append(event)

    def append(self, event):
        row = {"seq": len(self.rows), "elapsed": round(time.monotonic() - self.start, 3),
               "round": self.round, **event}
        self.rows.append(row)
        with self.path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()

    def emit(self, kind, **values):
        self({"event": kind, **values})
