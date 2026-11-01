"""Verify real checkpoint, streamed CPU inference and notebook-equivalent final labels."""
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import threading
import urllib.request
import urllib.error

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from inference import CHECKPOINT, EXPECTED_SHA256, SOURCE, CLASS_NAMES, MEAN, STD, Segmenter
from server import create_server

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "validation"
OUTPUT.mkdir(exist_ok=True)


@torch.inference_mode()
def notebook_prediction(model, image):
    """Reference preprocessing/merge from pspnet-modal.ipynb, independent of progress hooks."""
    width, height = image.size
    scale = 512 / max(width, height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    a = np.asarray(image.resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.
    x = torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1).copy())[None]
    x = F.pad(x, (0, max(0, 473 - w), 0, max(0, 473 - h)))
    hp, wp = x.shape[-2:]
    scores, count = torch.zeros(150, hp, wp), torch.zeros(1, hp, wp)
    def positions(n):
        if n <= 473:
            return [0]
        return sorted(set(list(range(0, n - 473 + 1, 316)) + [n - 473]))
    for top in positions(hp):
        for left in positions(wp):
            scores[:, top:top + 473, left:left + 473] += model(x[:, :, top:top + 473, left:left + 473]).softmax(1)[0]
            count[:, top:top + 473, left:left + 473] += 1
    probabilities = (scores / count)[:, :h, :w]
    best = torch.full((height, width), -float("inf"))
    labels = torch.zeros((height, width), dtype=torch.long)
    for begin in range(0, 150, 10):
        values, ids = F.interpolate(probabilities[None, begin:begin + 10], (height, width),
                                    mode="bilinear", align_corners=True)[0].max(0)
        mask = values > best
        labels[mask] = ids[mask] + begin
        best[mask] = values[mask]
    return labels.numpy().astype(np.uint8)


def main():
    with CHECKPOINT.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    assert digest == EXPECTED_SHA256, digest
    official_names = (SOURCE / "data/ade20k/ade20k_names.txt").read_text().splitlines()
    assert CLASS_NAMES == official_names, "Class names/order differs from upstream ADE20K"
    sample = SOURCE / "figure/demo/ADE_val_00001515.jpg"
    raw = sample.read_bytes()
    segmenter = Segmenter()
    server = create_server("127.0.0.1", 0, segmenter)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        status = json.load(urllib.request.urlopen(url + "/api/status"))
        assert status["checkpoint"] and not status["loaded"]
        for page in ("/", "/app.js", "/style.css"):
            assert urllib.request.urlopen(url + page).status == 200
        request = urllib.request.Request(url + "/api/predict", data=b"not an image", method="POST")
        try:
            urllib.request.urlopen(request)
            raise AssertionError("Invalid image was accepted")
        except urllib.error.HTTPError as error:
            assert error.code == 400
        events = []
        request = urllib.request.Request(url + "/api/predict", data=raw,
                                         headers={"X-Base-Size": "512"}, method="POST")
        with urllib.request.urlopen(request, timeout=600) as response:
            for line in response:
                event = json.loads(line)
                assert event["type"] != "error", event
                events.append(event)
                print(event["type"], event.get("stage", ""), flush=True)
        assert events[-1]["type"] == "complete"
        final = events[-1]
        assert len([e for e in events if e["type"] == "activation"]) == final["tiles"] * 6
        assert len([e for e in events if e["type"] == "tile_done"]) == final["tiles"]
        progress = [e["progress"] for e in events if "progress" in e]
        assert progress == sorted(progress)
        decoded = base64.b64decode(final["labels"].split(",", 1)[1])
        labels = np.asarray(Image.open(BytesIO(decoded)))
        assert labels.min() >= 0 and labels.max() < 150
        for key in ("overlay", "mask", "labels"):
            (OUTPUT / f"example_{key}.png").write_bytes(base64.b64decode(final[key].split(",", 1)[1]))
        with Image.open(sample) as image:
            reference = notebook_prediction(segmenter.model, image.convert("RGB"))
        assert np.array_equal(labels, reference), f"Differing pixels: {np.sum(labels != reference)}"
        assert not any(module._forward_hooks for module in segmenter.model.modules())
        report = {"checkpoint_sha256": digest, "device": "cpu", "model": "b7", "classes": 150,
                  "streamed_events": len(events), "tiles": final["tiles"],
                  "inference_seconds_with_first_load": final["seconds"],
                  "pixel_identical_to_notebook_reference": True,
                  "sample": "hszhao/semseg figure/demo/ADE_val_00001515.jpg",
                  "checks": ["checkpoint hash", "strict model loading", "official class order",
                             "HTTP assets/status", "invalid image rejection", "real layer progress",
                             "monotonic progress", "output IDs", "reference pixel equality", "hook cleanup"]}
        (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
