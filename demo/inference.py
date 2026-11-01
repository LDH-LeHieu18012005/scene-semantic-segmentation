"""CPU-only PSPNet b7 inference, with real intermediate activation/progress frames."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "checkpoints" / "b7_seed17_last.pt"
SOURCE = ROOT / "demo" / "vendor" / "semseg"
EXPECTED_SHA256 = "b7f060e54ec271ea41049daa18f82d607b2581b93d642884971d192650ac89cd"
MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)

# Use the author's class list rather than maintaining a second label mapping.
CLASS_NAMES = (SOURCE / "data" / "ade20k" / "ade20k_names.txt").read_text(encoding="utf-8").splitlines()
assert len(CLASS_NAMES) == 150


def palette():
    """Deterministic VOC-style colors; class IDs stay distinct, including class 0."""
    colors = np.zeros((150, 3), dtype=np.uint8)
    for index in range(150):
        value = index + 1
        for bit in range(8):
            for channel in range(3):
                colors[index, channel] |= ((value >> channel) & 1) << (7 - bit)
            value >>= 3
    return colors


COLORS = palette()


def encoded_image(image, fmt="JPEG"):
    buffer = BytesIO()
    image.save(buffer, format=fmt, **({"quality": 86} if fmt == "JPEG" else {}))
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def starts(length, crop=473):
    if length <= crop:
        return [0]
    return sorted(set(range(0, length - crop + 1, 316)) | {length - crop})


class Segmenter:
    def __init__(self, checkpoint=CHECKPOINT, threads=4):
        self.checkpoint = Path(checkpoint)
        self.model = None
        torch.set_num_threads(max(1, threads))

    def load(self, emit):
        if self.model is not None:
            return
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Chưa có checkpoint: {self.checkpoint}")
        if not (SOURCE / "model" / "pspnet.py").is_file():
            raise FileNotFoundError("Thiếu mã kiến trúc tại demo/vendor/semseg.")
        emit({"type": "stage", "progress": 2, "stage": "Đang nạp model vào RAM", "detail": "Chỉ cần nạp ở lần dự đoán đầu tiên."})
        sys.path.insert(0, str(SOURCE))
        from model.pspnet import PSPNet

        class DemoPSPNet(PSPNet):
            # Match the notebook's ResearchPSPNet, rather than upstream argmax output.
            def forward(self, x):
                size = x.shape[-2:]
                x = self.layer2(self.layer1(self.layer0(x)))
                x = self.layer4(self.layer3(x))
                return F.interpolate(self.cls(self.ppm(x)), size=size,
                                     mode="bilinear", align_corners=True)

        cp = torch.load(self.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if cp.get("variant") != "b7" or cp.get("bins") != [1, 2, 3, 7]:
            raise ValueError("Checkpoint phải là b7 gốc với bins [1,2,3,7].")
        if cp.get("step") != 10000:
            raise ValueError("Checkpoint chưa đủ 10.000 updates như kết quả đã báo cáo.")
        model = DemoPSPNet(layers=101, bins=(1, 2, 3, 7), classes=150,
                          zoom_factor=8, pretrained=False)
        model.load_state_dict(cp["model"], strict=True)
        self.model = model.cpu().eval()
        del cp

    @torch.inference_mode()
    def predict(self, image, emit, base_size=512):
        started = time.perf_counter()
        self.load(emit)
        original_size = image.size
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        emit({"type": "input", "image": encoded_image(image), "width": image.width,
              "height": image.height, "resized": original_size != image.size})
        scale = base_size / max(image.size)
        w, h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
        resized = image.resize((w, h), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.
        x = torch.from_numpy(((array - MEAN) / STD).transpose(2, 0, 1).copy())[None]
        x = F.pad(x, (0, max(0, 473 - w), 0, max(0, 473 - h)))
        hp, wp = x.shape[-2:]
        scores = torch.zeros(150, hp, wp)
        count = torch.zeros(1, hp, wp)
        tiles = [(top, left) for top in starts(hp) for left in starts(wp)]
        canvas = Image.new("RGB", (wp, hp), tuple(int(v) for v in MEAN * 255))
        canvas.paste(resized)
        stage_names = [("layer0", "Stem · đặc trưng ban đầu"),
                       ("layer1", "ResNet · tầng 1"), ("layer2", "ResNet · tầng 2"),
                       ("layer3", "ResNet · tầng 3"), ("layer4", "ResNet · tầng 4"),
                       ("ppm", "PPM · ngữ cảnh đa tỉ lệ")]
        try:
            for index, (top, left) in enumerate(tiles):
                tile_started = time.perf_counter()
                patch_image = canvas.crop((left, top, left + 473, top + 473))
                box = [left / wp, top / hp, (left + 473) / wp, (top + 473) / hp]
                emit({"type": "tile_start", "tile": index + 1, "total": len(tiles),
                      "box": box, "canvas": encoded_image(canvas),
                      "progress": 5 + 88 * index / len(tiles),
                      "stage": f"Đang xử lý cửa sổ {index + 1}/{len(tiles)}"})
                handles = []
                def hook_for(stage_index, title):
                    def hook(module, inputs, output):
                        activation = output[0].float().abs().mean(0)
                        activation = F.interpolate(activation[None, None], (240, 240),
                                                   mode="bilinear", align_corners=False)[0, 0].numpy()
                        lo, hi = np.percentile(activation, [2, 98])
                        normalized = np.clip((activation - lo) / max(float(hi - lo), 1e-8), 0, 1)
                        heat = np.stack([255 * normalized,
                                         210 * np.sqrt(normalized), 255 * (1 - normalized)], -1).astype(np.uint8)
                        view = Image.blend(patch_image.resize((240, 240)), Image.fromarray(heat), .57)
                        emit({"type": "activation", "stage": title,
                              "shape": list(output.shape), "activation": encoded_image(view),
                              "progress": 5 + 88 * (index + (stage_index + 1) / 7) / len(tiles),
                              "detail": "Bản đồ cường độ kích hoạt thực; chưa phải nhãn phân đoạn."})
                    return hook
                for j, (name, title) in enumerate(stage_names):
                    handles.append(getattr(self.model, name).register_forward_hook(hook_for(j, title)))
                try:
                    logits = self.model(x[:, :, top:top + 473, left:left + 473])
                    probability = logits.float().softmax(1)[0]
                finally:
                    for handle in handles:
                        handle.remove()
                scores[:, top:top + 473, left:left + 473] += probability
                count[:, top:top + 473, left:left + 473] += 1
                covered = count[0].numpy() > 0
                labels = scores.argmax(0).numpy()
                color = Image.fromarray(COLORS[labels])
                overlay = np.array(Image.blend(canvas, color, .48))
                overlay[~covered] = np.asarray(canvas)[~covered]
                emit({"type": "tile_done", "preview": encoded_image(Image.fromarray(overlay)),
                      "tile": index + 1, "total": len(tiles), "box": box,
                      "tile_seconds": round(time.perf_counter() - tile_started, 2),
                      "progress": 5 + 88 * (index + 1) / len(tiles),
                      "stage": f"Đã ghép {index + 1}/{len(tiles)} cửa sổ"})
                del logits, probability
            emit({"type": "stage", "progress": 95, "stage": "Khôi phục kích thước và xuất kết quả"})
            # Match notebook: average probabilities first, resize them, then argmax.
            probabilities = (scores / count.clamp_min(1))[:, :h, :w]
            best = torch.full((image.height, image.width), -float("inf"))
            prediction = torch.zeros((image.height, image.width), dtype=torch.uint8)
            for begin in range(0, 150, 10):
                values, ids = F.interpolate(probabilities[None, begin:begin + 10],
                                            (image.height, image.width), mode="bilinear",
                                            align_corners=True)[0].max(0)
                better = values > best
                prediction[better] = (ids[better] + begin).to(torch.uint8)
                best[better] = values[better]
            labels = prediction.numpy()
            mask = Image.fromarray(COLORS[labels])
            overlay = Image.blend(image, mask, .48)
            counts = np.bincount(labels.ravel(), minlength=150)
            legend = [{"id": int(k), "name": CLASS_NAMES[k],
                       "color": "#" + "".join(f"{v:02x}" for v in COLORS[k]),
                       "percent": round(100 * int(counts[k]) / labels.size, 2)}
                      for k in np.argsort(counts)[::-1] if counts[k] > 0]
            emit({"type": "complete", "progress": 100, "stage": "Hoàn thành",
                  "seconds": round(time.perf_counter() - started, 2),
                  "overlay": encoded_image(overlay, "PNG"), "mask": encoded_image(mask, "PNG"),
                  "labels": encoded_image(Image.fromarray(labels), "PNG"), "legend": legend,
                  "tiles": len(tiles), "base_size": base_size})
        finally:
            # Keep weights resident for the next image; per-image tensors leave scope.
            pass
