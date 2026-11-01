"""Modal GPU runner for the PSPNet study.

Quick test: modal run modal_app.py --action smoke --experiment smoke_modal_v1
Training chunk: modal run --detach modal_app.py --action train --config configs/modal_research.json --variant b6 --seed 17

Each training call advances at most --chunk-steps updates, preserving the global LR
schedule and total budget. Re-run the SAME command until the checkpoint is complete.
"""
from pathlib import Path, PurePosixPath
import json
import re

import modal

VOLUME_NAME = "pspnet-study-ade20k-v2"
SOURCE_COMMIT = "4f274c3f276778228bc14a4565822d46359f0cc8"
GPU = "L4"  # A fixed GPU type for both variants; change before starting a new study.
LOCAL_ROOT = Path(__file__).resolve().parent
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.5.1", "numpy==1.26.4", "pandas==2.2.3", "Pillow==11.1.0",
                 "matplotlib==3.9.4", "fvcore==0.1.5.post20221221", "gdown==5.2.0")
    .run_commands("git clone https://github.com/hszhao/semseg.git /opt/semseg",
                  f"git -C /opt/semseg checkout --detach {SOURCE_COMMIT}")
    .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONUNBUFFERED": "1",
          "PSPNET_MODAL_MOUNT": "/mnt/pspnet-study-ade20k-v2"})
    .add_local_file(LOCAL_ROOT / "pspnet_lab.py", "/root/pspnet_lab.py", copy=True)
)
app = modal.App("pspnet-ppm-study", image=image)


def validate_remote_config(cfg):
    """All persistent outputs must land on the attached Volume."""
    keys = ["output_dir", "data_root"]
    if cfg["profile"] != "smoke":
        keys.append("initial_weights")
    for key in keys:
        path = PurePosixPath(cfg[key])
        if (not path.is_absolute() or ".." in path.parts
                or not path.is_relative_to(PurePosixPath("/mnt/pspnet-study-ade20k-v2"))):
            raise ValueError(f"{key} must be below /mnt/pspnet-study-ade20k-v2")
    if cfg["source_dir"] != "/opt/semseg":
        raise ValueError("source_dir must be /opt/semseg for Modal Functions")


@app.function(volumes={"/mnt/pspnet-study-ade20k-v2": volume}, cpu=4, memory=16384,
              timeout=24 * 3600, max_containers=1)
def prepare_cpu(cfg, download_dataset=True):
    """Prepare/hash data without keeping a GPU allocated."""
    import subprocess
    import sys
    validate_remote_config(cfg)
    volume.reload()
    import pspnet_lab as lab
    lab.resources(cfg, download_dataset=download_dataset)
    lab.prepare(cfg)
    root = Path(cfg["output_dir"])
    lab.json_write(root / "modal_config.json", cfg)
    (root / "requirements-resolved.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    lab.persist_volume()
    return {"output_dir": str(root), "prepared": True}


@app.function(gpu=GPU, volumes={"/mnt/pspnet-study-ade20k-v2": volume}, cpu=4, memory=16384,
              timeout=24 * 3600, retries=0, max_containers=1)
def gpu_worker(command, cfg, variant="b6", seed=17, repeat=0, chunk_steps=1000):
    """One serialized task. Every core task executes in a fresh subprocess."""
    import os
    import subprocess
    import sys
    import torch
    import pspnet_lab as lab
    validate_remote_config(cfg)
    volume.reload()
    if variant not in cfg["variants"] or seed not in cfg["seeds"]:
        raise ValueError("variant/seed is absent from the locked protocol")
    if command not in ("selftest", "preflight", "train", "evaluate", "benchmark", "summarize"):
        raise ValueError(command)
    lab.check_lock(cfg)
    config_path = Path(cfg["output_dir"]) / "modal_config.json"
    lab.json_write(config_path, cfg)
    argv = [sys.executable, "-u", "/root/pspnet_lab.py", command, "--config", str(config_path),
            "--variant", variant, "--seed", str(seed), "--repeat", str(repeat)]
    if command == "train":
        if chunk_steps < 1:
            raise ValueError("chunk_steps must be >=1")
        argv += ["--chunk-steps", str(chunk_steps)]
    log_dir = Path(cfg["output_dir"]) / "console_logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"modal_{command}_{variant}_{seed}_{repeat}.log"
    try:
        with log_path.open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", env=dict(os.environ))
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if proc.wait():
                    raise RuntimeError(f"Task failed; inspect {log_path}")
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
    finally:
        lab.persist_volume()
    result = {"command": command, "variant": variant, "seed": seed, "output_dir": cfg["output_dir"]}
    if command == "train":
        cp = torch.load(lab.run_dir(cfg, variant, seed) / "last.pt", map_location="cpu", weights_only=True)
        result.update(step=cp["step"], max_steps=cfg["max_steps"], complete=cp["step"] == cfg["max_steps"])
    return result


def smoke_config(experiment):
    return {"profile": "smoke", "source_dir": "/opt/semseg",
            "data_root": "/mnt/pspnet-study-ade20k-v2/synthetic_smoke",
            "output_dir": f"/mnt/pspnet-study-ade20k-v2/results/{experiment}",
            "classes": 3, "init_kind": "random_smoke", "initial_weights": "",
            "variants": ["b6", "b7"], "seeds": [17], "crop": 65,
            "batch_size": 2, "accumulation": 1, "max_steps": 2,
            "lr": .01, "head_lr_multiplier": 10.0, "precision": "fp32",
            "scale_range": [1.0, 1.0], "rotation": 0.0, "save_every": 1, "log_every": 1,
            "eval_crop": 65, "eval_base_size": 81, "val_limit": 0,
            "benchmark_size": 65, "benchmark_warmup": 2, "benchmark_repeats": 5,
            "benchmark_sessions": 1}


@app.local_entrypoint()
def main(action: str = "prepare", config: str = "configs/modal_research.json", experiment: str = "smoke_modal_v1",
         variant: str = "b6", seed: int = 17, repeat: int = 0,
         chunk_steps: int = 1000, download_dataset: bool = True):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment):
        raise ValueError("experiment: use letters, digits, underscore or hyphen")
    cfg = smoke_config(experiment) if action == "smoke" else json.loads(Path(config).read_text(encoding="utf-8"))
    allowed = {"smoke", "prepare", "selftest", "preflight", "train", "evaluate", "benchmark", "summarize"}
    if action not in allowed:
        raise ValueError(f"action must be one of {sorted(allowed)}")
    if action == "prepare":
        print(prepare_cpu.remote(cfg, download_dataset=download_dataset))
    elif action == "smoke":
        if cfg["profile"] != "smoke":
            raise ValueError("action smoke requires profile=smoke")
        print(prepare_cpu.remote(cfg))
        print(gpu_worker.remote("selftest", cfg, seed=cfg["seeds"][0]))
        for variant in cfg["variants"]:
            for command in ("train", "evaluate", "benchmark"):
                print(gpu_worker.remote(command, cfg, variant, cfg["seeds"][0], chunk_steps=chunk_steps))
        print(gpu_worker.remote("summarize", cfg, seed=cfg["seeds"][0]))
    else:
        # --detach keeps this submitted call alive if the CLI disconnects.
        print(gpu_worker.spawn(action, cfg, variant, seed, repeat, chunk_steps).get())
