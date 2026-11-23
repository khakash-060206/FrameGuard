"""Conservative box association and bounded, timestamped sliding windows."""

from collections import deque
from dataclasses import dataclass, field
import math

SAMPLE_FPS = 8
WINDOW = 16
STRIDE = 4
MAX_GAP = 0.25


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class Track:
    box: tuple
    timestamp: float
    items: deque = field(default_factory=lambda: deque(maxlen=WINDOW))
    count: int = 0

    def append(self, timestamp, value):
        if self.items and not 0 < timestamp - self.items[-1][0] <= MAX_GAP:
            self.items.clear()
            self.count = 0
        self.items.append((timestamp, value))
        self.count += 1
        if len(self.items) == WINDOW and (self.count - WINDOW) % STRIDE == 0:
            return list(self.items)
        return None


class Tracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.last_time = None

    def update(self, boxes, timestamp, reset=False):
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Timestamp must be finite and nonnegative")
        if any(len(b) != 4 or not all(math.isfinite(x) for x in b) or b[2] <= b[0] or b[3] <= b[1] for b in boxes):
            raise ValueError("Invalid face bounding box")
        if reset or (self.last_time is not None and not 0 < timestamp - self.last_time <= MAX_GAP):
            self.tracks.clear()
        self.last_time = timestamp
        candidates = [[key for key, track in self.tracks.items() if iou(box, track.box) >= 0.3] for box in boxes]
        uses = {}
        for matches in candidates:
            for key in matches:
                uses[key] = uses.get(key, 0) + 1
        updated, assigned = {}, []
        for box, matches in zip(boxes, candidates):
            # shortcut: IoU is not identity recognition; reset ambiguous matches, upgrade after real-video validation.
            if len(matches) == 1 and uses[matches[0]] == 1:
                key = matches[0]
                track = self.tracks[key]
            else:
                key = self.next_id
                self.next_id += 1
                track = Track(tuple(box), timestamp)
            track.box, track.timestamp = tuple(box), timestamp
            updated[key] = track
            assigned.append(key)
        self.tracks = updated
        return assigned


def window_starts(timestamps):
    """Only continuous 16-frame windows; a gap restarts the stride schedule."""
    segment_start = 0
    for end, timestamp in enumerate(timestamps):
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid timestamp")
        if end and not 0 < timestamp - timestamps[end - 1] <= MAX_GAP:
            segment_start = end
        length = end - segment_start + 1
        if length >= WINDOW and (length - WINDOW) % STRIDE == 0:
            yield end - WINDOW + 1

