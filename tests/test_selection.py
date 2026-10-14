import json

from scripts.run_screen_native import select


def _run(root, name, method, lr, scale, seed, nll):
    d = root / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"method": method, "lr": lr, "scale": scale, "seed": seed}))
    (d / "metadata.json").write_text(json.dumps({"status": "complete", "seed": seed, "dev_tune_endpoint_nll": nll}))


def test_screen_selection_uses_mean_over_tuning_seeds(tmp_path):
    # The first configuration has the best individual seed but the second has
    # the better paired mean and must be frozen for screening.
    _run(tmp_path, "L_FIXED_c1_seed1002", "L_FIXED", 1e-4, 1.0, 1002, 0.10)
    _run(tmp_path, "L_FIXED_c1_seed1003", "L_FIXED", 1e-4, 1.0, 1003, 0.40)
    _run(tmp_path, "L_FIXED_c2_seed1002", "L_FIXED", 1e-3, 1.0, 1002, 0.20)
    _run(tmp_path, "L_FIXED_c2_seed1003", "L_FIXED", 1e-3, 1.0, 1003, 0.20)
    chosen = select(tmp_path)
    assert chosen["L_FIXED"]["lr"] == 1e-3
