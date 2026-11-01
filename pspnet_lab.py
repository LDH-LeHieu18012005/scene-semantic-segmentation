"""Controlled PSPNet PPM study. Generated notebook embeds this file verbatim.

Architecture is imported from pinned hszhao/semseg (MIT), not torchvision.
Run `python pspnet_lab.py --help` for the notebook's subprocess interface.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import contextlib
import copy
import csv
import gc
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
import torch
from torch import nn
import torch.nn.functional as F

COMMIT = "4f274c3f276778228bc14a4565822d46359f0cc8"
ADE_URL = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
BACKBONE_DRIVE_ID = "1V-sfLnqSuwTPgNMrqUp4HDZpqtoBH4xm"
# variant -> (PPM bins, extension): None, "link" (LinkedPPM) or "aspp" (DeepLabv3-style atrous branches)
VARIANTS = {"b6": ([1, 2, 3, 6], None), "b7": ([1, 2, 3, 7], None),
            "b6_link": ([1, 2, 3, 6], "link"), "b7_link": ([1, 2, 3, 7], "link"),
            "b6_aspp": ([1, 2, 3, 6], "aspp"), "b7_aspp": ([1, 2, 3, 7], "aspp")}
ASPP_RATES = (12, 24, 36)  # DeepLabv3 atrous rates for output stride 8
ASPP_CHANNELS = 256  # DeepLabv3 branch width
MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)
PREFLIGHT_UPDATES = 3  # time the last one; the first includes cold start/cuDNN autotuning


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_csv(path, row):
    path = Path(path)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def persist_volume():
    """Flush Volume v2 in notebooks/subprocesses without an implicit SDK reload."""
    mount = os.environ.get("PSPNET_MODAL_MOUNT")
    if mount:
        subprocess.run(["sync", "-f", mount], check=True, timeout=300)


def download_file(url, destination, attempts=3):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(attempts):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                mode = "ab" if append else "wb"
                received = offset if append else 0
                length = response.headers.get("Content-Length")
                expected = received + int(length) if length is not None else None
                last_report = received // (256 * 2**20)
                with partial.open(mode) as target:
                    while chunk := response.read(2**20):
                        target.write(chunk)
                        received += len(chunk)
                        report = received // (256 * 2**20)
                        if report > last_report:
                            print(f"Downloading {destination.name}: {received / 2**20:.0f} MiB", flush=True)
                            last_report = report
            if expected is not None and received != expected:
                raise OSError(f"Incomplete download: {received}/{expected} bytes")
            os.replace(partial, destination)
            return
        except (OSError, TimeoutError, http.client.HTTPException) as error:
            if getattr(error, "code", None) == 416 and partial.exists():
                # A prior process can finish the payload and stop before rename.
                with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
                    if int(response.headers.get("Content-Length", -1)) == partial.stat().st_size:
                        os.replace(partial, destination)
                        return
            if attempt + 1 == attempts:
                raise RuntimeError(f"Không tải được {url}. Tải thủ công vào {destination} rồi chạy lại.") from error
            print(f"Download retry {attempt + 1}: {error}", flush=True)


def resources(cfg, download_dataset=True):
    """Get real ADE20K and the author's matching ImageNet backbone weights."""
    if cfg["profile"] == "smoke":
        return
    weights = Path(cfg["initial_weights"])
    if not weights.is_file():
        if cfg["init_kind"] != "backbone" or weights.name != "resnet101_v2.pth":
            raise FileNotFoundError(f"Upload checkpoint vào {weights} trước khi chạy.")
        weights.parent.mkdir(parents=True, exist_ok=True)
        import gdown
        partial = weights.with_suffix(".pth.download")
        print("Tải resnet101_v2.pth từ Google Drive chính thức của hszhao...", flush=True)
        try:
            result = gdown.download(id=BACKBONE_DRIVE_ID, output=str(partial), quiet=False, resume=True)
            if not result or not partial.exists():
                raise RuntimeError("Google Drive returned no weights")
            # Reject HTML/error pages before treating the download as a checkpoint.
            if partial.stat().st_size < 1024 * 1024:
                raise ValueError("Downloaded file is too small to be ResNet-101")
            os.replace(partial, weights)
        except Exception as error:
            raise RuntimeError(f"Drive có thể giới hạn lượt tải. Tải file ID {BACKBONE_DRIVE_ID} thủ công vào {weights} rồi chạy lại.") from error
    # Validate BEFORE downloading/extracting the much larger dataset.
    setup_source(cfg["source_dir"])
    probe = make_model(cfg, "b6")
    if cfg["init_kind"] == "backbone":
        load_backbone(probe, cfg)
    else:
        probe.load_state_dict(load_external_state(weights), strict=True)
    del probe
    gc.collect()
    print("Pretrained weights tương thích.", flush=True)
    root = Path(cfg["data_root"])
    complete = all(len(list((root / "images" / split).glob("*.jpg"))) == count
                   and len(list((root / "annotations" / split).glob("*.png"))) == count
                   for split, count in [("training", 20210), ("validation", 2000)])
    if not complete:
        if not download_dataset:
            raise FileNotFoundError(f"ADE20K chưa đầy đủ tại {root}; bật download hoặc upload dataset.")
        archive = root.parent / "ADEChallengeData2016.zip"
        if not archive.is_file() or not zipfile.is_zipfile(archive):
            download_file(ADE_URL, archive)
        print("Giải nén ADE20K...", flush=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                target = (root.parent / member.filename).resolve()
                if not target.is_relative_to(root.parent.resolve()):
                    raise ValueError("Unsafe archive path")
            z.extractall(root.parent)
    persist_volume()
    print("Tài nguyên ADE20K sẵn sàng:", root, flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Bilinear/adaptive pooling backward can be nondeterministic on CUDA.
    # Warn rather than silently claim bitwise reproducibility.
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment():
    return {"python": sys.version, "torch": str(torch.__version__), "numpy": np.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count(), "source_commit": COMMIT}


def setup_source(root):
    root = Path(root).resolve()
    if not root.exists():
        subprocess.run(["git", "clone", "https://github.com/hszhao/semseg.git", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "checkout", "--detach", COMMIT], check=True, timeout=120)
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    assert actual == COMMIT
    changes = subprocess.check_output(["git", "-C", str(root), "diff", "HEAD", "--", "model"], text=True)
    if changes:
        raise RuntimeError("Mã model upstream đã thay đổi. Dùng một thư mục source sạch.")


def import_source(cfg):
    source = str(Path(cfg["source_dir"]).resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    from model.pspnet import PSPNet
    from model.resnet import resnet101
    return PSPNet, resnet101


class LinkedPPM(nn.Module):
    """Six directed adjacent edges, applied simultaneously on low-res features."""
    def __init__(self, ppm):
        super().__init__()
        self.features = ppm.features
        self.edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
        self.edge_weights = nn.Parameter(torch.zeros(len(self.edges)))

    def forward(self, x):
        z = [branch(x) for branch in self.features]
        mixed = list(z)
        for k, (src, dst) in enumerate(self.edges):
            size = z[dst].shape[-2:]
            if z[src].shape[-1] > size[-1]:
                aligned = F.adaptive_avg_pool2d(z[src], size)
            else:
                aligned = F.interpolate(z[src], size=size, mode="bilinear", align_corners=True)
            mixed[dst] = mixed[dst] + self.edge_weights[k].tanh() * aligned
        return torch.cat([x] + [F.interpolate(v, size=x.shape[-2:], mode="bilinear",
                                              align_corners=True) for v in mixed], dim=1)


class AtrousBranches(nn.Module):
    """DeepLabv3-style atrous branches on the same OS8 backbone features as the PPM.

    3x3 convolutions at rates 12/24/36, 256 channels each, concatenated after the pyramid
    features. Image-level pooling is already provided by the PPM 1x1 bin.
    """
    def __init__(self, in_dim=2048, channels=ASPP_CHANNELS, rates=ASPP_RATES):
        super().__init__()
        self.branches = nn.ModuleList(nn.Sequential(
            nn.Conv2d(in_dim, channels, kernel_size=3, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True)) for rate in rates)

    def forward(self, x):
        return [branch(x) for branch in self.branches]


def make_model(cfg, variant):
    PSPNet, _ = import_source(cfg)

    class ResearchPSPNet(PSPNet):
        # Keep upstream modules and state_dict names. Return logits for our trainer.
        def forward(self, x):
            size = x.shape[-2:]
            x = self.layer2(self.layer1(self.layer0(x)))
            aux_features = self.layer3(x)
            x = self.layer4(aux_features)
            features = self.ppm(x)
            if hasattr(self, "atrous"):
                features = torch.cat([features] + self.atrous(x), dim=1)
            logits = F.interpolate(self.cls(features), size=size,
                                   mode="bilinear", align_corners=True)
            if self.training:
                aux = F.interpolate(self.aux(aux_features), size=size,
                                    mode="bilinear", align_corners=True)
                return logits, aux
            return logits

    bins, extension = VARIANTS[variant]
    model = ResearchPSPNet(layers=101, bins=tuple(bins), classes=cfg["classes"],
                           zoom_factor=8, pretrained=False)
    if extension == "link":
        model.ppm = LinkedPPM(model.ppm)
    elif extension == "aspp":
        model.atrous = AtrousBranches()
        first = model.cls[0]
        # Classifier sees the extra atrous channels; apply_initial zeroes their weights.
        model.cls[0] = nn.Conv2d(first.in_channels + len(ASPP_RATES) * ASPP_CHANNELS, first.out_channels,
                                 kernel_size=3, padding=1, bias=False)
    return model


def load_external_state(path):
    # Only tensor/state-dict checkpoints from a trusted source; no unsafe fallback.
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in raw:
        raw = raw["state_dict"]
    elif "model" in raw:
        raw = raw["model"]
    return {k.removeprefix("module."): v for k, v in raw.items()}


def load_backbone(model, cfg):
    _, resnet101 = import_source(cfg)
    backbone = resnet101(pretrained=False)
    weights = load_external_state(cfg["initial_weights"])
    # Classifier fc is unused by segmentation; enforce every actual backbone tensor.
    weights = {k: v for k, v in weights.items() if not k.startswith("fc.")}
    result = backbone.load_state_dict(weights, strict=False)
    assert set(result.missing_keys) == {"fc.weight", "fc.bias"} and not result.unexpected_keys, result
    for name, idx in [("conv1", 0), ("bn1", 1), ("conv2", 3), ("bn2", 4),
                      ("conv3", 6), ("bn3", 7)]:
        model.layer0[idx].load_state_dict(getattr(backbone, name).state_dict(), strict=True)
    for name in ("layer1", "layer2", "layer3", "layer4"):
        getattr(model, name).load_state_dict(getattr(backbone, name).state_dict(), strict=True)


def create_synthetic(root, classes):
    """Synthetic files only exercise the pipeline; never research evidence."""
    root = Path(root)
    for split, count in [("training", 8), ("validation", 3)]:
        for category in ("images", "annotations"):
            (root / category / split).mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(2026 if split == "training" else 2027)
        for i in range(count):
            a = rng.integers(0, 256, (81, 97, 3), dtype=np.uint8)
            yy, xx = np.mgrid[:81, :97]
            mask = ((xx // 17 + yy // 21 + i) % classes + 1).astype(np.uint8)
            mask[:2] = 0
            Image.fromarray(a).save(root / "images" / split / f"demo_{i:03d}.jpg")
            Image.fromarray(mask).save(root / "annotations" / split / f"demo_{i:03d}.png")


def data_pairs(root, split):
    root = Path(root)
    pairs = [(p, root / "annotations" / split / (p.stem + ".png"))
             for p in sorted((root / "images" / split).glob("*.jpg"))]
    if not pairs or any(not y.exists() for _, y in pairs):
        raise FileNotFoundError(f"Thiếu ảnh/nhãn tại {root}, split={split}")
    return pairs


def read_mask(path, classes):
    raw = np.asarray(Image.open(path), dtype=np.int64)
    if raw.ndim != 2 or raw.min() < 0 or raw.max() > classes:
        raise ValueError(f"Nhãn phải là ADE gốc 0..{classes}: {path}")
    # ADE raw 0 = void; raw 1..150 -> train IDs 0..149.
    return np.where(raw == 0, 255, raw - 1).astype(np.uint8)


def tensor_image(image):
    a = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(((a - MEAN) / STD).transpose(2, 0, 1).copy())


def augment_pair(image_path, mask_path, cfg, seed, epoch, index):
    # Augmentation independent of model RNG, worker scheduling and variant.
    token = f"{seed}:{epoch}:{index}".encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(token).digest()[:8], "little"))
    image = Image.open(image_path).convert("RGB")
    mask = Image.fromarray(read_mask(mask_path, cfg["classes"]))
    scale = rng.uniform(*cfg["scale_range"])
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    image = image.resize(size, Image.Resampling.BILINEAR)
    mask = mask.resize(size, Image.Resampling.NEAREST)
    angle = rng.uniform(-cfg["rotation"], cfg["rotation"])
    fill = tuple(int(round(v * 255)) for v in MEAN)
    image = image.rotate(angle, Image.Resampling.BILINEAR, fillcolor=fill)
    mask = mask.rotate(angle, Image.Resampling.NEAREST, fillcolor=255)
    if rng.random() < .5:
        image, mask = ImageOps.mirror(image), ImageOps.mirror(mask)
    if rng.random() < .5:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(.1, 1.5)))
    crop = cfg["crop"]
    border = (0, 0, max(0, crop - image.width), max(0, crop - image.height))
    image, mask = ImageOps.expand(image, border, fill), ImageOps.expand(mask, border, 255)
    left, top = rng.randint(0, image.width - crop), rng.randint(0, image.height - crop)
    box = (left, top, left + crop, top + crop)
    return tensor_image(image.crop(box)), torch.from_numpy(np.array(mask.crop(box), dtype=np.int64))


def batch_position(micro_index, n, batch, seed):
    batches = n // batch
    if batches == 0:
        raise ValueError("Dataset nhỏ hơn batch thực.")
    epoch, within = divmod(micro_index, batches)
    # Cheap for ADE20K; deterministic random access also makes resume exact in ordering.
    perm = np.random.default_rng(seed + epoch).permutation(n)
    return epoch, perm[within * batch:(within + 1) * batch]


def get_batch(pairs, cfg, seed, micro_index):
    epoch, ids = batch_position(micro_index, len(pairs), cfg["batch_size"], seed)
    samples = [augment_pair(*pairs[int(i)], cfg, seed, epoch, int(i)) for i in ids]
    return torch.stack([v[0] for v in samples]), torch.stack([v[1] for v in samples])


class BatchPrefetcher:
    """Build upcoming micro-batches in threads while the GPU trains (PIL releases the GIL).

    Returns exactly get_batch(pairs, cfg, seed, i) for i = first, first+1, ...,
    so data order, augmentation and resume position match the serial loader.
    """
    def __init__(self, pairs, cfg, seed, first, depth=8):
        self.args = (pairs, cfg, seed)
        self.pool = ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1))
        self.queue = deque()
        self.next_index = first
        for _ in range(depth):
            self._submit()

    def _submit(self):
        self.queue.append((self.next_index, self.pool.submit(get_batch, *self.args, self.next_index)))
        self.next_index += 1

    def get(self, index):
        expected, future = self.queue.popleft()
        assert expected == index, "Prefetch order mismatch"
        self._submit()
        return future.result()

    def close(self):
        for _, future in self.queue:
            future.cancel()
        self.pool.shutdown(wait=True)


def prepare(cfg):
    validate_config(cfg)
    setup_source(cfg["source_dir"])
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if cfg["profile"] == "smoke":
        create_synthetic(cfg["data_root"], cfg["classes"])
    manifests = {}
    for split in ("training", "validation"):
        rows = []
        pairs = data_pairs(cfg["data_root"], split)
        if cfg["profile"] == "research":
            expected = 20210 if split == "training" else 2000
            if len(pairs) != expected:
                raise ValueError(f"Research ADE20K cần {expected} ảnh {split}, nhận {len(pairs)}.")
        for index, (image, mask) in enumerate(pairs):
            with Image.open(image) as im:
                size = im.size
            label = read_mask(mask, cfg["classes"])
            if label.shape != (size[1], size[0]):
                raise ValueError(f"Image/mask size mismatch: {image}")
            rows.append({"image": str(image.relative_to(cfg["data_root"])),
                         "mask": str(mask.relative_to(cfg["data_root"])),
                         "image_sha256": sha256(image), "mask_sha256": sha256(mask)})
            if (index + 1) % 1000 == 0:
                print("Hash/validate", split, index + 1, "/", len(pairs), flush=True)
        manifests[split] = rows
        print(split, len(rows), "ảnh đã kiểm tra", flush=True)
    # The official ADE20K split contains a few byte-identical train/val images.
    # Keep the official split (paper protocol, identical for every variant) and record them.
    train_by_hash = {}
    for r in manifests["training"]:
        train_by_hash.setdefault(r["image_sha256"], []).append(r["image"])
    duplicates = [{"validation": r["image"], "training": train_by_hash[r["image_sha256"]],
                   "image_sha256": r["image_sha256"]}
                  for r in manifests["validation"] if r["image_sha256"] in train_by_hash]
    if duplicates:
        print(f"CẢNH BÁO: {len(duplicates)} ảnh validation trùng byte với ảnh train trong split chính thức; "
              "giữ nguyên split và ghi danh sách vào train_val_duplicates.json.", flush=True)
    manifest_bytes = json.dumps(manifests, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    env = environment()
    env["source_files"] = {name: sha256(Path(cfg["source_dir"]) / "model" / name)
                           for name in ("pspnet.py", "resnet.py")}
    fingerprint = {"config": cfg, "data_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                   "code_sha256": sha256(__file__), "source_files": env["source_files"],
                   "initial_weights_sha256": sha256(cfg["initial_weights"]) if cfg["initial_weights"] else None,
                   "train_val_duplicates": len(duplicates)}
    lock = output / "experiment_lock.json"
    if lock.exists() and json.loads(lock.read_text(encoding="utf-8")) != fingerprint:
        raise ValueError("Cấu hình/dữ liệu/mã đã đổi. Dùng output_dir mới, không ghi đè thí nghiệm.")
    json_write(output / "data_manifest.json", manifests)
    json_write(output / "train_val_duplicates.json", duplicates)
    json_write(output / "environment.json", env)
    json_write(lock, fingerprint)
    print("Đã khóa thí nghiệm:", lock)


def validate_config(cfg):
    assert cfg["profile"] in ("smoke", "pilot", "research")
    assert cfg["init_kind"] in ("random_smoke", "backbone", "psp_checkpoint")
    assert cfg["batch_size"] >= 2, "Nhánh 1x1 cần batch thực >=2 khi train BatchNorm."
    assert cfg["accumulation"] >= 1 and cfg["max_steps"] >= 1
    assert cfg["crop"] >= 57 and (cfg["crop"] - 1) % 8 == 0
    assert cfg["eval_crop"] >= 57 and (cfg["eval_crop"] - 1) % 8 == 0
    assert cfg["benchmark_size"] >= 57 and (cfg["benchmark_size"] - 1) % 8 == 0
    assert len(cfg["seeds"]) >= 1 and len(set(cfg["seeds"])) == len(cfg["seeds"])
    assert cfg["variants"][:2] == ["b6", "b7"]
    if cfg["profile"] != "smoke":
        assert cfg["classes"] == 150
        assert cfg["init_kind"] != "random_smoke", "Research/pilot yêu cầu pretrained weights."
        assert Path(cfg["initial_weights"]).is_file(), "Đặt initial_weights tới checkpoint thật."
    else:
        assert cfg["init_kind"] == "random_smoke"
    assert cfg["precision"] in ("fp32", "amp_fp16", "amp_bf16")
    assert cfg.get("eval_every", 0) >= 0 and cfg.get("eval_subset", 500) >= 1
    if cfg["profile"] == "research":
        assert cfg["val_limit"] == 0


def check_lock(cfg):
    lock_path = Path(cfg["output_dir"]) / "experiment_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["config"] == cfg, "Cấu hình khác experiment_lock; hãy dùng thư mục mới."
    assert lock["code_sha256"] == sha256(__file__), "Mã đã đổi sau khi khóa."
    assert lock["data_sha256"] == sha256(lock_path.parent / "data_manifest.json"), "Manifest dữ liệu đã đổi sau khi khóa."
    for name, digest in lock["source_files"].items():
        assert sha256(Path(cfg["source_dir"]) / "model" / name) == digest
    return sha256(lock_path)


def initial_path(cfg, seed):
    return Path(cfg["output_dir"]) / "initial" / f"seed_{seed}.pt"


def initialize(cfg, seed):
    lock_hash = check_lock(cfg)
    path = initial_path(cfg, seed)
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["lock_sha256"] == lock_hash
        return path
    seed_all(seed)
    model = make_model(cfg, "b6")
    if cfg["init_kind"] == "backbone":
        load_backbone(model, cfg)
    elif cfg["init_kind"] == "psp_checkpoint":
        model.load_state_dict(load_external_state(cfg["initial_weights"]), strict=True)
    atomic_save({"model": model.state_dict(), "seed": seed,
                 "lock_sha256": lock_hash, "bins": [1, 2, 3, 6]}, path)
    return path


def apply_initial(model, state, variant):
    extension = VARIANTS[variant][1]
    if extension == "link":
        result = model.load_state_dict(state, strict=False)
        assert result.missing_keys == ["ppm.edge_weights"] and not result.unexpected_keys
        assert torch.count_nonzero(model.ppm.edge_weights) == 0
    elif extension == "aspp":
        # Shared PSPNet weights as in b6/b7; classifier weights for the atrous channels
        # start at zero, so the network initially computes exactly the original PSPNet.
        state = dict(state)
        base = state.pop("cls.0.weight")
        weight = torch.zeros_like(model.cls[0].weight)
        weight[:, :base.shape[1]] = base
        state["cls.0.weight"] = weight
        result = model.load_state_dict(state, strict=False)
        assert result.missing_keys and all(k.startswith("atrous.") for k in result.missing_keys)
        assert not result.unexpected_keys
    else:
        model.load_state_dict(state, strict=True)


def run_dir(cfg, variant, seed):
    return Path(cfg["output_dir"]) / f"{variant}_seed{seed}"


def device_for(cfg):
    if cfg["profile"] != "smoke" and not torch.cuda.is_available():
        raise RuntimeError("Huấn luyện ADE20K cần CUDA GPU. Chọn GPU trong Compute profile của Modal.")
    if cfg["precision"] == "amp_bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU không hỗ trợ BF16. Dùng L4/A100 hoặc đổi cả hai bản sang fp32 trong output mới.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_context(cfg, device):
    dtype = torch.bfloat16 if cfg["precision"] == "amp_bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype,
                          enabled=device.type == "cuda" and cfg["precision"] != "fp32")


def capture_rng():
    return {"cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(saved):
    torch.set_rng_state(saved["cpu"])
    if torch.cuda.is_available() and saved["cuda"]:
        torch.cuda.set_rng_state_all(saved["cuda"])


def set_train_backend(cfg):
    """Opt-in fast cuDNN for training only, same policy for every variant.

    Training is then not bitwise reproducible; evaluation/benchmark processes keep
    the deterministic settings from seed_all().
    """
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def preflight(cfg, variants=None):
    """A few real training updates per variant, discarded; research checkpoints stay untouched."""
    check_lock(cfg)
    device = device_for(cfg)
    pairs = data_pairs(cfg["data_root"], "training")
    seed = cfg["seeds"][0]
    path = initialize(cfg, seed)
    rows = []
    for variant in variants or cfg["variants"]:
        seed_all(seed)
        model = make_model(cfg, variant)
        cp = torch.load(path, map_location="cpu", weights_only=True)
        apply_initial(model, cp["model"], variant)
        del cp
        model.to(device).train()
        groups = [{"params": [p for n, p in model.named_parameters() if n.startswith("layer")], "lr": cfg["lr"]},
                  {"params": [p for n, p in model.named_parameters() if not n.startswith("layer")],
                   "lr": cfg["lr"] * cfg["head_lr_multiplier"]}]
        optimizer = torch.optim.SGD(groups, momentum=.9, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["precision"] == "amp_fp16")
        seed_all(seed + 100000)
        set_train_backend(cfg)
        # Build batches first: training overlaps data loading with GPU work via the prefetcher.
        batches = [get_batch(pairs, cfg, seed, i) for i in range(PREFLIGHT_UPDATES * cfg["accumulation"])]
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        losses = []
        x = y = main = aux = loss = logits = None
        try:
            for update in range(PREFLIGHT_UPDATES):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                for micro in range(cfg["accumulation"]):
                    x, y = batches[update * cfg["accumulation"] + micro]
                    assert torch.any(y != 255), "Batch toàn ignore label"
                    x, y = x.to(device), y.to(device)
                    with autocast_context(cfg, device):
                        main, aux = model(x)
                        loss = F.cross_entropy(main, y, ignore_index=255) + .4 * F.cross_entropy(aux, y, ignore_index=255)
                    assert torch.isfinite(loss), "Loss không hữu hạn"
                    if update == 0:
                        losses.append(float(loss.detach()))
                    scaler.scale(loss / cfg["accumulation"]).backward()
                scaler.unscale_(optimizer)
                assert model.layer0[0].weight.grad is not None
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), "Gradient overflow; dùng BF16 hoặc FP32 cho cả hai bản"
                scaler.step(optimizer)
                scaler.update()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                step_seconds = time.perf_counter() - started
            batches = None
            train_peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
            # Check inference separately, like evaluate/benchmark. Keeping the
            # SGD momentum/gradients here can cause an artificial preflight OOM.
            optimizer.zero_grad(set_to_none=True)
            optimizer = groups = None
            main = aux = loss = y = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            model.eval()
            with torch.inference_mode():
                logits = model(x[:1])  # evaluation path is FP32
                assert logits.shape == (1, cfg["classes"], cfg["crop"], cfg["crop"])
                assert torch.isfinite(logits).all()
            inference_peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None
            rows.append({"variant": variant, "loss": float(np.mean(losses)),
                         "params": sum(p.numel() for p in model.parameters()),
                         "step_seconds": step_seconds,
                         "step_seconds_scope": f"GPU update {PREFLIGHT_UPDATES} of {PREFLIGHT_UPDATES}; excludes data loading and periodic eval",
                         "estimated_training_hours": step_seconds * cfg["max_steps"] / 3600,
                         "train_peak_allocated_mib": train_peak,
                         "inference_peak_allocated_mib": inference_peak,
                         "initial_sha256": sha256(path)})
            print("PASS preflight:", rows[-1], flush=True)
        except torch.OutOfMemoryError as error:
            raise RuntimeError("GPU không đủ VRAM cho cấu hình này. Chọn GPU lớn hơn hoặc đổi crop/batch cho cả hai bản và output mới.") from error
        finally:
            del model, optimizer, scaler, groups, x, y, main, aux, loss, logits, batches
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if len(rows) == 2:  # a compared pair: 6x6 vs 7x7 with the same extension
        assert rows[0]["params"] == rows[1]["params"]
        assert rows[0]["initial_sha256"] == rows[1]["initial_sha256"]
    json_write(Path(cfg["output_dir"]) / "preflight.json",
               {"environment": environment(), "data": "synthetic" if cfg["profile"] == "smoke" else "ADE20K",
                "steps_discarded": PREFLIGHT_UPDATES, "rows": rows})
    persist_volume()


def train(cfg, variant, seed, chunk_steps=None, checkpoint_callback=None):
    lock_hash = check_lock(cfg)
    device = device_for(cfg)
    runtime_path = Path(cfg["output_dir"]) / "training_environment.json"
    current_env = environment()
    if runtime_path.exists():
        if json.loads(runtime_path.read_text(encoding="utf-8")) != current_env:
            raise ValueError("Mọi run trong thí nghiệm phải dùng cùng GPU/phần mềm; dùng output_dir mới nếu đổi.")
    else:
        json_write(runtime_path, current_env)
    path = initialize(cfg, seed)
    init_hash = sha256(path)
    directory = run_dir(cfg, variant, seed)
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = make_model(cfg, variant)
    initial = torch.load(path, map_location="cpu", weights_only=True)
    apply_initial(model, initial["model"], variant)
    del initial
    model.to(device).train()
    backbone = [p for n, p in model.named_parameters() if n.startswith("layer")]
    heads = [p for n, p in model.named_parameters() if not n.startswith("layer")]
    assert all(p.requires_grad for p in backbone)
    optimizer = torch.optim.SGD([{"params": backbone, "lr_scale": 1.0},
                                 {"params": heads, "lr_scale": cfg["head_lr_multiplier"]}],
                                lr=cfg["lr"], momentum=.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["precision"] == "amp_fp16")
    start, seen_micro, skipped, elapsed = 0, 0, 0, 0.0
    checkpoint_path = directory / "last.pt"
    if checkpoint_path.exists():
        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert cp["lock_sha256"] == lock_hash and cp["initial_sha256"] == init_hash
        assert cp["variant"] == variant and cp["seed"] == seed
        assert cp["environment"] == environment(), "Resume cần cùng môi trường; không âm thầm đổi GPU/phần mềm."
        model.load_state_dict(cp["model"], strict=True)
        optimizer.load_state_dict(cp["optimizer"])
        scaler.load_state_dict(cp["scaler"])
        start, seen_micro, skipped = cp["step"], cp["seen_micro"], cp["skipped_updates"]
        elapsed = cp["training_seconds"]
        restore_rng(cp["rng"])
        del cp
        print(f"Resume {variant}/{seed}: {start}/{cfg['max_steps']} bước", flush=True)
    else:
        # Reset after construction/loading so paired runs share dropout RNG start.
        seed_all(seed + 100000)
    if start == cfg["max_steps"]:
        print("Đã hoàn thành; dùng checkpoint hiện có.")
        return
    pairs = data_pairs(cfg["data_root"], "training")
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    eval_every = cfg.get("eval_every", 0)
    eval_pairs = validation_subset(cfg) if eval_every else []
    # Trim logs beyond the last atomic checkpoint after an interrupted session.
    history_path = directory / "train.csv"
    val_path = directory / "val_history.csv"
    for log_path in (history_path, val_path):
        if log_path.exists():
            old = pd.read_csv(log_path)
            old[old.step <= start].to_csv(log_path, index=False)
    set_train_backend(cfg)
    started = time.perf_counter()
    eval_seconds = 0.0  # excluded from training_seconds
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    end_step = min(cfg["max_steps"], start + chunk_steps) if chunk_steps else cfg["max_steps"]
    prefetch = BatchPrefetcher(pairs, cfg, seed, seen_micro)
    params = list(model.parameters())
    window = []
    for step in range(start, end_step):
        step_started = time.perf_counter()
        lr = cfg["lr"] * (1 - step / cfg["max_steps"]) ** .9
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_scale"]
        optimizer.zero_grad(set_to_none=True)
        # loss, main loss, aux loss, correct pixels, valid pixels; read back once per step.
        totals = torch.zeros(5, device=device)
        data_wait = 0.0
        for micro in range(cfg["accumulation"]):
            waited = time.perf_counter()
            x, y = prefetch.get(seen_micro)
            data_wait += time.perf_counter() - waited
            seen_micro += 1
            if not torch.any(y != 255):  # checked on CPU, no GPU sync
                raise RuntimeError("Batch toàn ignore label; điều chỉnh dữ liệu/augmentation cho cả hai bản.")
            x, y = x.to(device), y.to(device)
            with autocast_context(cfg, device):
                main, aux = model(x)
                main_loss, aux_loss = criterion(main, y), criterion(aux, y)
                loss = main_loss + .4 * aux_loss
            scaler.scale(loss / cfg["accumulation"]).backward()
            with torch.no_grad():
                valid = y != 255
                totals += torch.stack([loss.detach().float(), main_loss.detach().float(),
                                       aux_loss.detach().float(),
                                       ((main.argmax(1) == y) & valid).sum().float(), valid.sum().float()])
            x = y = main = aux = loss = main_loss = aux_loss = valid = None
        if step == start:
            assert backbone[0].grad is not None, "Backbone không nhận gradient."
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))  # measure only, no clipping
        loss_sum, main_sum, aux_sum, correct, valid_pixels, grad_value = (
            torch.cat([totals, grad_norm.reshape(1).float()]).tolist())
        if not (math.isfinite(loss_sum) and math.isfinite(grad_value)):
            raise FloatingPointError("Loss/gradient không hữu hạn; không cập nhật trọng số.")
        prior_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < prior_scale:
            skipped += 1
            raise FloatingPointError("AMP bỏ qua update do overflow. Chạy lại CẢ HAI bằng fp32 trong output_dir mới.")
        step_seconds = time.perf_counter() - step_started
        n = cfg["accumulation"]
        row = {"step": step + 1, "time_utc": utc_now(), "loss": loss_sum / n, "main_loss": main_sum / n,
               "aux_loss": aux_sum / n, "train_pixel_acc": correct / max(valid_pixels, 1.0), "lr": lr,
               "grad_norm": grad_value, "step_seconds": step_seconds, "data_wait_seconds": data_wait,
               "images_per_second": cfg["batch_size"] * n / step_seconds,
               "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None}
        append_csv(history_path, row)
        window.append(row)
        if (step + 1) % cfg["log_every"] == 0 or step == start:
            mean = lambda key: float(np.mean([r[key] for r in window]))
            eta = mean("step_seconds") * (cfg["max_steps"] - step - 1) / 3600
            memory = f" | VRAM peak {row['gpu_peak_mib'] / 1024:.1f} GiB" if row["gpu_peak_mib"] else ""
            print(f"{variant}/{seed} step {step + 1}/{cfg['max_steps']} | loss {mean('loss'):.3f} "
                  f"(main {mean('main_loss'):.3f}, aux {mean('aux_loss'):.3f}) | train acc {mean('train_pixel_acc'):.3f} "
                  f"| lr {lr:.5f} | grad {mean('grad_norm'):.2f} | {mean('step_seconds'):.2f}s/step "
                  f"(chờ dữ liệu {mean('data_wait_seconds'):.2f}s){memory} | ETA train {eta:.1f}h", flush=True)
            window = []
        if eval_every and ((step + 1) % eval_every == 0 or step + 1 == cfg["max_steps"]):
            # Monitoring only: the final checkpoint is always the one reported, never the best one.
            eval_started = time.perf_counter()
            model.eval()
            score = evaluate_pairs(model, eval_pairs, cfg, device)
            model.train()
            seconds = time.perf_counter() - eval_started
            eval_seconds += seconds
            append_csv(val_path, {"step": step + 1, "time_utc": utc_now(), "val_subset_miou": score["miou"],
                                  "val_subset_pixel_acc": score["pixel_accuracy"],
                                  "images": len(eval_pairs), "eval_seconds": seconds})
            print(f"{variant}/{seed} VAL step {step + 1}: mIoU {100 * score['miou']:.2f} | "
                  f"pixel acc {100 * score['pixel_accuracy']:.2f} ({len(eval_pairs)} ảnh, {seconds:.0f}s)", flush=True)
        if (step + 1) % cfg["save_every"] == 0 or step + 1 == end_step:
            if device.type == "cuda":
                torch.cuda.synchronize()
            atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "scaler": scaler.state_dict(), "rng": capture_rng(),
                         "step": step + 1, "seen_micro": seen_micro, "skipped_updates": skipped,
                         "training_seconds": elapsed + time.perf_counter() - started - eval_seconds,
                         "variant": variant, "seed": seed, "bins": VARIANTS[variant][0],
                         "initial_sha256": init_hash, "lock_sha256": lock_hash,
                         "environment": environment(), "config": cfg,
                         "train_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None},
                        checkpoint_path)
            if checkpoint_callback:
                checkpoint_callback()
            else:
                persist_volume()
    prefetch.close()
    print(f"Saved {variant}/{seed} at {end_step}/{cfg['max_steps']} steps", flush=True)


def load_trained(cfg, variant, seed, device):
    check_lock(cfg)
    cp = torch.load(run_dir(cfg, variant, seed) / "last.pt", map_location="cpu", weights_only=True)
    assert cp["step"] == cfg["max_steps"], "Chưa đủ ngân sách huấn luyện."
    assert cp["variant"] == variant and cp["bins"] == VARIANTS[variant][0]
    assert cp["lock_sha256"] == sha256(Path(cfg["output_dir"]) / "experiment_lock.json")
    model = make_model(cfg, variant)
    model.load_state_dict(cp["model"], strict=True)
    metadata = {k: cp[k] for k in ("initial_sha256", "step", "training_seconds", "train_peak_allocated_mib")}
    del cp
    return model.to(device).eval(), metadata


def confusion_update(matrix, prediction, target, classes):
    valid = target != 255
    labels = classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    matrix += np.bincount(labels, minlength=classes * classes).reshape(classes, classes)


def metrics_from_confusion(cm):
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(1)
    union = support + cm.sum(0) - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    return {"miou": float(np.nanmean(iou)), "pixel_accuracy": float(tp.sum() / max(cm.sum(), 1)),
            "iou": iou, "support": support, "present_classes": int((union > 0).sum())}


def window_starts(length, crop, stride):
    if length <= crop:
        return [0]
    return sorted(set(list(range(0, length - crop + 1, stride)) + [length - crop]))


@torch.inference_mode()
def predict_image(model, image, cfg, device):
    """Single scale, no flip; sliding windows; resize probabilities to original H,W."""
    width, height = image.size
    scale = cfg["eval_base_size"] / max(width, height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    x = tensor_image(image.resize((w, h), Image.Resampling.BILINEAR)).unsqueeze(0)
    crop = cfg["eval_crop"]
    x = F.pad(x, (0, max(0, crop - w), 0, max(0, crop - h)), value=0)
    hp, wp = x.shape[-2:]
    # Accumulate on CPU to avoid full-resolution 150-channel GPU tensors.
    scores = torch.zeros(cfg["classes"], hp, wp)
    count = torch.zeros(1, hp, wp)
    stride = max(1, math.ceil(crop * 2 / 3))
    for top in window_starts(hp, crop, stride):
        for left in window_starts(wp, crop, stride):
            patch = x[:, :, top:top + crop, left:left + crop].to(device)
            # Accuracy evaluation always FP32, irrespective of training AMP.
            probability = model(patch).float().softmax(1)[0].cpu()
            scores[:, top:top + crop, left:left + crop] += probability
            count[:, top:top + crop, left:left + crop] += 1
    assert torch.all(count > 0)
    scores = (scores / count)[:, :h, :w]
    # Chunk channels to limit CPU RAM at original resolution.
    best = torch.full((height, width), -float("inf"))
    prediction = torch.zeros((height, width), dtype=torch.long)
    for begin in range(0, cfg["classes"], 10):
        resized = F.interpolate(scores[None, begin:begin + 10], size=(height, width),
                                mode="bilinear", align_corners=True)[0]
        values, ids = resized.max(0)
        update = values > best
        prediction[update] = ids[update] + begin
        best[update] = values[update]
    return prediction.numpy().astype(np.uint8)


def validation_subset(cfg):
    """Fixed random validation subset, identical for every variant; monitoring during training only."""
    pairs = data_pairs(cfg["data_root"], "validation")
    count = min(cfg.get("eval_subset", 500), len(pairs))
    chosen = np.sort(np.random.default_rng(20260).choice(len(pairs), size=count, replace=False))
    return [pairs[int(i)] for i in chosen]


def evaluate_pairs(model, pairs, cfg, device):
    total = np.zeros((cfg["classes"], cfg["classes"]), dtype=np.int64)
    for image_path, mask_path in pairs:
        with Image.open(image_path) as image:
            prediction = predict_image(model, image.convert("RGB"), cfg, device)
        confusion_update(total, prediction, read_mask(mask_path, cfg["classes"]), cfg["classes"])
    return metrics_from_confusion(total)


def evaluate(cfg, variant, seed):
    device = device_for(cfg)
    model, meta = load_trained(cfg, variant, seed, device)
    directory = run_dir(cfg, variant, seed)
    (directory / "predictions").mkdir(exist_ok=True)
    pairs = data_pairs(cfg["data_root"], "validation")
    if cfg["val_limit"]:
        pairs = pairs[:cfg["val_limit"]]
    total = np.zeros((cfg["classes"], cfg["classes"]), dtype=np.int64)
    image_rows, times = [], []
    per_image = {"intersection": [], "gt": [], "pred": []}  # for the paired bootstrap in summarize()
    for index, (image_path, mask_path) in enumerate(pairs):
        image = Image.open(image_path).convert("RGB")
        target = read_mask(mask_path, cfg["classes"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = predict_image(model, image, cfg, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - started) * 1000)
        cm = np.zeros_like(total)
        confusion_update(cm, prediction, target, cfg["classes"])
        total += cm
        per_image["intersection"].append(np.diag(cm))
        per_image["gt"].append(cm.sum(1))
        per_image["pred"].append(cm.sum(0))
        score = metrics_from_confusion(cm)
        image_rows.append({"image": image_path.name, "image_miou_present": score["miou"],
                           "image_pixel_accuracy": score["pixel_accuracy"], "pipeline_ms": times[-1]})
        Image.fromarray(prediction).save(directory / "predictions" / (image_path.stem + ".png"))
        if index % 100 == 0:
            print("eval", variant, seed, index + 1, "/", len(pairs), flush=True)
    score = metrics_from_confusion(total)
    np.save(directory / "confusion.npy", total)
    pd.DataFrame({"class_id": range(cfg["classes"]), "iou": score["iou"],
                  "gt_pixels": score["support"]}).to_csv(directory / "per_class.csv", index=False)
    pd.DataFrame(image_rows).to_csv(directory / "per_image.csv", index=False)
    np.savez_compressed(directory / "per_image_stats.npz", images=np.array([p.name for p, _ in pairs]),
                        **{k: np.stack(v) for k, v in per_image.items()})
    result = {"variant": variant, "seed": seed, "profile": cfg["profile"],
              "miou": score["miou"], "pixel_accuracy": score["pixel_accuracy"],
              "present_classes": score["present_classes"], "images": len(pairs),
              "pipeline_ms_median": float(np.median(times)),
              "pipeline_ms_p95": float(np.percentile(times, 95)),
              "pipeline_scope": "resize+normalize+H2D+windows+softmax+CPU merge+resize+argmax; excludes disk IO; FP32",
              "checkpoint_sha256": sha256(directory / "last.pt"), **meta}
    json_write(directory / "metrics.json", result)
    print(result, flush=True)


@torch.inference_mode()
def benchmark(cfg, variant, seed, repeat):
    # CLI invocation = fresh process for each model/seed/repeat.
    device = device_for(cfg)
    model, _ = load_trained(cfg, variant, seed, device)
    size = cfg["benchmark_size"]
    x = torch.randn(1, 3, size, size, device=device)
    counts = {"conv_linear_macs": 0}
    handles = []
    def hook(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            counts["conv_linear_macs"] += output.numel() * (module.in_channels // module.groups) * math.prod(module.kernel_size)
        elif isinstance(module, nn.Linear):
            counts["conv_linear_macs"] += output.numel() * module.in_features
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    model(x)
    for handle in handles:
        handle.remove()
    torch.use_deterministic_algorithms(False)
    # Benchmark fixed policy shared across variants, independent of training.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for _ in range(cfg["benchmark_warmup"]):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(cfg["benchmark_repeats"]):
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = model(x)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            start = time.perf_counter()
            result = model(x)
            samples.append((time.perf_counter() - start) * 1000)
        del result
    all_params = sum(p.numel() for p in model.parameters())
    aux_params = sum(p.numel() for p in model.aux.parameters())
    row = {"variant": variant, "seed": seed, "repeat": repeat, "profile": cfg["profile"],
           "params_training": all_params, "params_inference_graph": all_params - aux_params,
           "gmacs_conv_linear": counts["conv_linear_macs"] / 1e9,
           "gflops_conv_linear": 2 * counts["conv_linear_macs"] / 1e9,
           "mac_scope": "Conv2d/Linear only; excludes pooling/BN/ReLU/interpolation/add; aux not executed",
           "latency_ms_median": float(np.median(samples)), "latency_ms_p95": float(np.percentile(samples, 95)),
           "allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
           "reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else None,
           "vram_scope": "resident full model (including aux weights), input, forward workspace/output; no optimizer",
           "input_shape": [1, 3, size, size], "precision": "fp32", "environment": environment()}
    # Broader operator estimate with explicit unsupported operators. fvcore calls a MAC one FLOP.
    try:
        from fvcore.nn import FlopCountAnalysis
        # JIT tracing fails on inference-mode tensors with in-place ReLU; trace under no_grad.
        with torch.inference_mode(False), torch.no_grad():
            analysis = FlopCountAnalysis(model, torch.randn(1, 3, size, size, device=device))
            analysis.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
            row["fvcore_operation_units"] = float(analysis.total())
            row["fvcore_by_operator"] = {k: float(v) for k, v in analysis.by_operator().items()}
            row["fvcore_unsupported"] = {k: int(v) for k, v in analysis.unsupported_ops().items()}
        row["fvcore_convention"] = "FMA=1; mixed operator units, not directly 2x total FLOPs"
    except Exception as error:
        row["fvcore_note"] = (f"fvcore failed ({type(error).__name__}: {error}); "
                              "conv/linear counts remain valid partial counts")
    directory = run_dir(cfg, variant, seed)
    np.save(directory / f"latency_repeat{repeat}.npy", np.asarray(samples))
    json_write(directory / f"benchmark_repeat{repeat}.json", row)
    print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)


def summarize(cfg):
    check_lock(cfg)
    root = Path(cfg["output_dir"])
    rows, class_rows = [], []
    benchmark_environments = []
    for seed in cfg["seeds"]:
        for variant in cfg["variants"]:
            directory = run_dir(cfg, variant, seed)
            if not (directory / "metrics.json").exists():
                print("Chưa có kết quả:", directory.name)
                continue
            row = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            benchmarks = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("benchmark_repeat*.json"))]
            if benchmarks:
                benchmark_environments.extend(b["environment"] for b in benchmarks)
                for key in ("params_training", "params_inference_graph", "gmacs_conv_linear", "gflops_conv_linear",
                            "latency_ms_median", "latency_ms_p95", "allocated_mib", "reserved_mib"):
                    values = [b[key] for b in benchmarks if b[key] is not None]
                    row[key] = float(np.median(values)) if values else np.nan
                row["benchmark_sessions"] = len(benchmarks)
            row["training_hours"] = row["training_seconds"] / 3600
            if (directory / "train.csv").exists():
                tail = pd.read_csv(directory / "train.csv").tail(1000)
                for key in ("loss", "main_loss", "train_pixel_acc"):
                    if key in tail:
                        row[f"{key}_last1000"] = float(tail[key].mean())
            if (directory / "val_history.csv").exists():
                history = pd.read_csv(directory / "val_history.csv")
                row["val_subset_miou_final"] = float(history.val_subset_miou.iloc[-1])
            rows.append(row)
            classes = pd.read_csv(directory / "per_class.csv")
            classes["variant"], classes["seed"] = variant, seed
            class_rows.append(classes)
    if not rows:
        print("Chưa có số liệu; không tạo kết quả giả.")
        return
    if benchmark_environments:
        assert all(e == benchmark_environments[0] for e in benchmark_environments), "Benchmark hardware/software differs between runs."
    raw = pd.DataFrame(rows)
    raw.to_csv(root / "results_per_seed.csv", index=False)
    pd.concat(class_rows).to_csv(root / "all_class_iou.csv", index=False)
    numeric = [c for c in ["miou", "params_training", "params_inference_graph", "gmacs_conv_linear",
                           "gflops_conv_linear", "latency_ms_median", "latency_ms_p95", "allocated_mib"] if c in raw]
    summary = (raw.groupby("variant")[numeric].first() if len(cfg["seeds"]) == 1
               else raw.groupby("variant")[numeric].agg(["mean", "std", "count"]))
    summary.to_csv(root / "ablation_summary.csv")
    print(summary.to_string())
    paired = []
    for base, test in [("b6", "b7"), ("b6", "b6_link"), ("b7", "b7_link"), ("b6_link", "b7_link")]:
        a = raw[raw.variant == base].set_index("seed")
        b = raw[raw.variant == test].set_index("seed")
        common = a.index.intersection(b.index)
        if not len(common):
            continue
        assert (a.loc[common, "initial_sha256"] == b.loc[common, "initial_sha256"]).all()
        delta = 100 * (b.loc[common, "miou"] - a.loc[common, "miou"])
        if len(common) == 1:
            paired.append({"comparison": f"{test} - {base}", "seed": int(common[0]),
                           "delta_miou_pp": float(delta.iloc[0])})
        else:
            paired.append({"comparison": f"{test} - {base}", "paired_seeds": len(common),
                           "delta_miou_pp_mean": delta.mean(), "delta_miou_pp_sd": delta.std(ddof=1)})
    pd.DataFrame(paired).to_csv(root / "paired_differences.csv", index=False)
    print(pd.DataFrame(paired).to_string(index=False))
    boot = []
    for base, test in [("b6", "b7"), ("b6", "b6_link"), ("b7", "b7_link"), ("b6_link", "b7_link")]:
        for seed in cfg["seeds"]:
            files = [run_dir(cfg, v, seed) / "per_image_stats.npz" for v in (base, test)]
            if all(f.exists() for f in files):
                result = paired_bootstrap(*files)
                for variant, key in ((base, "base_miou"), (test, "test_miou")):
                    reported = raw[(raw.variant == variant) & (raw.seed == seed)].miou.iloc[0]
                    assert abs(result[key] - reported) < 1e-9, "per_image_stats không khớp metrics.json"
                boot.append({"comparison": f"{test} - {base}", "seed": seed, **result})
    if boot:
        pd.DataFrame(boot).to_csv(root / "paired_bootstrap.csv", index=False)
        print(pd.DataFrame(boot).drop(columns=["scope"]).to_string(index=False))


def miou_from_stats(intersection, gt, pred):
    union = gt + pred - intersection
    iou = np.divide(intersection, union, out=np.full(intersection.shape, np.nan), where=union > 0)
    return np.nanmean(iou, axis=-1)


def paired_bootstrap(base_file, test_file, draws=2000):
    """Uncertainty of the dataset-level mIoU gap from the finite validation set.

    Images are resampled with replacement, identically for both variants (paired).
    This does NOT capture training randomness (seed variance).
    """
    a, b = dict(np.load(base_file)), dict(np.load(test_file))
    assert np.array_equal(a["images"], b["images"]), "Hai bản phải đánh giá cùng danh sách ảnh."
    n = len(a["images"])
    weights = np.random.default_rng(2026).multinomial(n, np.full(n, 1 / n), size=draws).astype(np.float64)

    def miou(stats, w):
        return miou_from_stats(*(w @ stats[k].astype(np.float64) for k in ("intersection", "gt", "pred")))

    full = np.ones((1, n))
    base_miou, test_miou = float(miou(a, full)[0]), float(miou(b, full)[0])
    deltas = 100 * (miou(b, weights) - miou(a, weights))
    return {"images": n, "draws": draws, "base_miou": base_miou, "test_miou": test_miou,
            "delta_miou_pp": 100 * (test_miou - base_miou),
            "ci95_low_pp": float(np.percentile(deltas, 2.5)), "ci95_high_pp": float(np.percentile(deltas, 97.5)),
            "prob_test_better": float(np.mean(deltas > 0)),
            "scope": "paired bootstrap over validation images; excludes training/seed variance"}


def selftest(cfg):
    seed_all(7)
    cm = np.zeros((2, 2), np.int64)
    confusion_update(cm, np.array([0, 1, 0, 1]), np.array([0, 1, 1, 255]), 2)
    assert np.array_equal(cm, [[1, 0], [1, 1]])
    assert metrics_from_confusion(cm)["miou"] == .5
    assert abs(miou_from_stats(np.diag(cm), cm.sum(1), cm.sum(0)) - .5) < 1e-12
    assert window_starts(700, 473, 316) == [0, 227]
    a = make_model(cfg, "b6").eval()
    b = make_model(cfg, "b7").eval()
    b.load_state_dict(a.state_dict(), strict=True)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    diffs = []
    for (na, ma), (nb, mb) in zip(a.named_modules(), b.named_modules()):
        assert na == nb and type(ma).__name__ == type(mb).__name__
        if isinstance(ma, nn.AdaptiveAvgPool2d) and ma.output_size != mb.output_size:
            diffs.append(na)
    assert diffs == ["ppm.features.3.0"]
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    del b
    # Check output stride and main/aux forward without large activations.
    x = torch.randn(2, 3, 65, 65)
    with torch.no_grad():
        y = a.layer4(a.layer3(a.layer2(a.layer1(a.layer0(x)))))
        assert y.shape == (2, 2048, 9, 9)
        assert a(x).shape == (2, cfg["classes"], 65, 65)
        original = a.ppm(y)
        linked = LinkedPPM(a.ppm).eval()
        assert torch.equal(original, linked(y)), "Zero gates must preserve original function."
    linked(y.detach()).square().mean().backward()
    assert linked.edge_weights.grad is not None
    assert torch.isfinite(linked.edge_weights.grad).all()
    assert torch.count_nonzero(linked.edge_weights.grad) > 0, "Gates must be learnable at zero initialization."
    # DeepLab-style atrous branches: zero classifier weights for the new channels preserve the function.
    c = make_model(cfg, "b7_aspp").eval()
    state = make_model(cfg, "b7").state_dict()
    apply_initial(c, state, "b7_aspp")
    reference = make_model(cfg, "b7").eval()
    reference.load_state_dict(state, strict=True)
    assert [branch[0].dilation for branch in c.atrous.branches] == [(r, r) for r in ASPP_RATES]
    with torch.no_grad():
        assert torch.allclose(reference(x), c(x), rtol=1e-4, atol=1e-5), "Zero atrous weights must preserve PSPNet."
    c.train()
    c_main, c_aux = c(x)
    (F.cross_entropy(c_main, torch.zeros((2, 65, 65), dtype=torch.long)) + .4 * F.cross_entropy(
        c_aux, torch.zeros((2, 65, 65), dtype=torch.long))).backward()
    assert c.cls[0].weight.grad[:, 4096:].abs().sum() > 0, "Atrous channels must be learnable."
    del c, reference, state
    a.zero_grad(set_to_none=True)
    a.train()
    main, aux = a(x)
    target = torch.zeros((2, 65, 65), dtype=torch.long)
    (F.cross_entropy(main, target) + .4 * F.cross_entropy(aux, target)).backward()
    assert a.layer0[0].weight.grad is not None
    assert torch.isfinite(a.layer0[0].weight.grad).all()
    print("PASS: metrics, window coverage, identical initial weights, only pooling differs, OS8, zero gates, "
          "zero-init atrous branches, backbone gradient.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["resources", "prepare", "selftest", "preflight", "train", "evaluate", "benchmark", "summarize"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=list(VARIANTS), default="b6")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--chunk-steps", type=int, default=None)
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=None,
                        help="preflight only: variants to check (default: cfg variants)")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    # Also applies to evaluation/benchmark subprocesses: fixed input RNG, TF32 off.
    seed_all(0)
    if args.command == "resources":
        resources(cfg, download_dataset=args.download_dataset)
    elif args.command == "preflight":
        preflight(cfg, args.variants)
    elif args.command in ("prepare", "selftest", "summarize"):
        globals()[args.command](cfg)
    elif args.command == "benchmark":
        benchmark(cfg, args.variant, args.seed, args.repeat)
    elif args.command == "train":
        if args.chunk_steps is not None and args.chunk_steps < 1:
            parser.error("--chunk-steps must be >=1")
        train(cfg, args.variant, args.seed, chunk_steps=args.chunk_steps)
    else:
        globals()[args.command](cfg, args.variant, args.seed)


if __name__ == "__main__":
    main()
