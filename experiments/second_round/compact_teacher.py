"""Package teacher checkpoints with captured deterministic problem regeneration."""

import argparse
import json
from pathlib import Path
import shutil

import torch

from experiments.second_round.verify_teacher_compact import file_hash, tensor_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/var/tmp/dora-bench/second_round/teacher"))
    parser.add_argument("--output", type=Path, default=Path("/var/tmp/dora-bench/round2/teacher_compact"))
    args = parser.parse_args()
    source, output = args.source, args.output
    output.mkdir(parents=True, exist_ok=True)
    recorded = json.loads((source / "results.json").read_text())
    for name in ("provenance.json", "stratified_summary.json", "audit.json", "rank1_capacity_check.json"):
        shutil.copy2(source / name, output / name)
    shutil.copy2(source / "results.json", output / "recorded_results.json")
    sources = output / "executed_sources"
    sources.mkdir(exist_ok=True)
    for name, expected in recorded["provenance"]["source_sha256"].items():
        assert file_hash(source / "executed_sources" / name) == expected
        shutil.copy2(source / "executed_sources" / name, sources / name)
    shutil.copy2(Path(__file__).with_name("verify_teacher_compact.py"), output / "verify.py")
    shutil.copy2(__file__, output / "build_source.py")
    shutil.copy2(Path(__file__).with_name("audit_teacher.py"), output / "full_audit_source.py")
    cells = {}
    for run in recorded["runs"]:
        cell, method, seed = run["cell_id"], run["method"], run["seed"]
        if cell not in cells:
            data = torch.load(source / cell / "problem.pt", weights_only=True, map_location="cpu")
            teacher = run["teacher"]
            cells[cell] = {
                "generator_arguments": [teacher["adapter_rank"], teacher["family"], teacher["teacher_direction_rank"], teacher["coordinates"]],
                "original_problem_container_sha256": teacher["problem_sha256"],
                "problem_tensor_fingerprints": {name: tensor_fingerprint(value) for name, value in data.items()},
            }
        checkpoint = source / cell / method / f"seed{seed}.pt"
        assert file_hash(checkpoint) == run["checkpoint"]["sha256"]
        target = output / "checkpoints" / cell / method / checkpoint.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, target)
        selection = output / "selections" / cell / f"{method}.json"
        if not selection.exists():
            selection.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / cell / method / "selection.json", selection)
    (output / "README.txt").write_text(
        "Portable selected-checkpoint teacher evidence\n\n"
        "Run: python verify.py --output reconstructed.json\n"
        "Requires PyTorch and NumPy; recorded PyTorch is in provenance.json. CPU only.\n"
        "The verifier executes make_problem from the captured teacher.py source,\n"
        "requires exact hashes for every regenerated tensor, and independently\n"
        "reconstructs dense weights and test metrics from all selected adapters.\n"
        "A library/RNG/math change that alters a tensor fails explicitly.\n"
        "No original artifact paths, local repository imports, CUDA, network, or\n"
        "downloaded checkpoints are required. Absolute paths retained inside\n"
        "recorded metadata are provenance only and are never opened.\n\n"
        "The full local audit attestation is audit.json, with its source retained.\n"
        "Validation selections/curves are verified as recorded; this compact\n"
        "verifier does not retrain adapters or independently rerun those curves.\n"
        "Original full data and checkpoints were preserved locally unchanged.\n"
    )
    paths = sorted(path for path in output.rglob("*") if path.is_file() and path.name not in ("manifest.json", "reconstructed.json", "portable_verification.json"))
    manifest = {"format": "dora-teacher-compact-v1", "final_run_count": len(recorded["runs"]),
                "original_results_sha256": file_hash(source / "results.json"),
                "original_full_audit_sha256": file_hash(source / "audit.json"),
                "files_sha256": {str(path.relative_to(output)): file_hash(path) for path in paths},
                "cells": cells}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(output), "cells": len(cells), "runs": len(recorded["runs"]),
                      "bytes": sum(path.stat().st_size for path in output.rglob("*") if path.is_file())}))


if __name__ == "__main__":
    main()
