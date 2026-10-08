"""Post-hoc CLIP diagnostic excluding an unlearned subject identifier.

Generated images and all original metrics stay unchanged. The frozen external
CLIP encoder has not learned that ``sks`` names this corgi, so this diagnostic
scores the class/background/style text after replacing ``sks dog`` with
``a dog``. It is not used for learning-rate or checkpoint selection.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HOME", "/var/tmp/dora-bench/huggingface")

from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor


MODEL = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def score_run(directory, model, processor):
    run = json.loads((directory / "results.json").read_text())
    seed = run["provenance"]["controls"]["seed"]
    results = []
    for row in run["results"]:
        images, prompts = [], []
        for sample in row["samples"]:
            if digest(sample["path"]) != sample["sha256"]:
                raise RuntimeError(f"Generated sample changed: {sample['path']}")
            images.append(Image.open(sample["path"]).convert("RGB"))
            prompts.append(sample["prompt"].replace("sks dog", "a dog"))
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        embeddings = model(**inputs)
        similarities = (embeddings.image_embeds * embeddings.text_embeds).sum(dim=-1).tolist()
        details = [{"generation_prompt": sample["prompt"], "scoring_prompt": prompt,
                    "cosine": value, "image": sample["path"], "image_sha256": sample["sha256"]}
                   for sample, prompt, value in zip(row["samples"], prompts, similarities)]
        results.append({"method": row["method"], "seed": seed,
                        "class_only_clip_mean": sum(similarities) / len(similarities),
                        "original_identifier_clip_mean": row["image_metrics"]["clip_text_image_cosine_mean"],
                        "per_prompt": details})
    metadata = {"diagnostic": "class-only CLIP text-image cosine", "post_hoc": True,
                "used_for_model_selection": False, "prompt_rule": "replace 'sks dog' with 'a dog'",
                "generated_images_changed": False, "original_metrics_changed": False,
                "model": MODEL, "revision": REVISION, "device": "cpu", "dtype": "float32",
                "torch": torch.__version__, "source_sha256": digest(__file__),
                "caveat": "An external embedding proxy; removing the identifier does not turn it into a human quality score.",
                "results": results}
    output = directory / "class_clip_metrics.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    source_dir = directory / "executed_sources"
    source_dir.mkdir(exist_ok=True)
    (source_dir / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"directory": str(directory), "seed": seed,
                      "scores": {row["method"]: row["class_only_clip_mean"] for row in results}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--wait", action="store_true", help="Wait for still-running experiments to finish")
    args = parser.parse_args()
    torch.set_num_threads(8)
    model = CLIPModel.from_pretrained(MODEL, revision=REVISION).eval()
    processor = CLIPProcessor.from_pretrained(MODEL, revision=REVISION)
    for directory in args.run_dirs:
        while not (directory / "results.json").exists():
            if not args.wait:
                raise FileNotFoundError(directory / "results.json")
            time.sleep(5)
        score_run(directory, model, processor)


if __name__ == "__main__":
    main()
