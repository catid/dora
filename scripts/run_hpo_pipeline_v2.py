#!/usr/bin/env python3
"""Run the 12-trial HPO quota for restart and compression pipeline cells."""
from __future__ import annotations
import argparse, json, os, subprocess, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rankstudy.registry import build_registry


def jobs(root, data_root, manifest_root, updates, seed, rank, target_rank):
    lrs = (1e-6, 1e-5, 1e-4, 1e-3)
    scales = (0.25, 1.0, 4.0)
    out = []
    for spec in build_registry():
        if spec.group not in {"restart", "restart_control", "compress"}:
            continue
        for lr in lrs:
            for scale in scales:
                name = f"hpo_{spec.id}_lr{lr:g}_s{scale:g}"
                out.append({"method": spec.id, "lr": lr, "scale": scale, "seed": seed,
                            "rank": rank, "target_rank": target_rank, "updates": updates,
                            "data_root": str(data_root), "manifest": str(manifest_root),
                            "out": str(Path(root) / name)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/var/tmp/dora-adaptive-rank/data_v2"))
    p.add_argument("--manifest-root", type=Path, default=Path("artifacts/manifests_v2"))
    p.add_argument("--updates", type=int, default=128)
    p.add_argument("--seed", type=int, default=1001)
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--python", default=".venv-study/bin/python")
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    gpu_ids = [x.strip() for x in a.gpus.split(",") if x.strip()]
    free, active, queue = list(gpu_ids), {}, []
    for job in jobs(a.root, a.data_root, a.manifest_root, a.updates, a.seed, 16, 8):
        if a.resume and (Path(job["out"]) / "metadata.json").exists():
            continue
        queue.append(job)
    i = 0
    while i < len(queue) or active:
        while i < len(queue) and free:
            job = queue[i]; i += 1
            out = Path(job["out"]); out.mkdir(parents=True, exist_ok=True)
            gpu = free.pop(0)
            cmd = [a.python, "scripts/run_pipeline_v2.py"]
            for key in ("method", "scale", "lr", "seed", "updates", "data_root",
                        "manifest", "out", "rank", "target_rank"):
                cmd += [f"--{key.replace('_', '-')}", str(job[key])]
            cmd += ["--microbatch", str(a.microbatch)]
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": "src",
                        "HF_HOME": os.environ.get("HF_HOME", "/var/tmp/dora-adaptive-rank/hf"),
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
            log = (out / "stdout.log").open("w")
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[proc.pid] = (proc, log, job, gpu)
            print(json.dumps({"event": "launch", "pid": proc.pid, "gpu": gpu,
                              "method": job["method"], "lr": job["lr"],
                              "scale": job["scale"]}), flush=True)
        time.sleep(1)
        for pid, (proc, log, job, gpu) in list(active.items()):
            rc = proc.poll()
            if rc is None:
                continue
            log.close(); del active[pid]; free.append(gpu); free.sort(key=gpu_ids.index)
            (Path(job["out"]) / "launcher_status.json").write_text(json.dumps({
                "status": "complete" if rc == 0 else "failed", "returncode": rc,
                "gpu": gpu, "stage": "v2_pipeline_hpo"}, indent=2) + "\n")
            print(json.dumps({"event": "finish", "pid": pid, "gpu": gpu,
                              "status": "complete" if rc == 0 else "failed",
                              "method": job["method"]}), flush=True)
    print(json.dumps({"jobs": len(queue), "root": str(a.root)}))


if __name__ == "__main__":
    main()
