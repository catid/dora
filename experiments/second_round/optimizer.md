# Second-round optimizer integration

`experiments.second_round_adapters.parameter_groups()` returns adapter parameters
only, with `param_names` metadata. Re-enable task heads after adapter injection
and append their parameters explicitly. With the pinned PyTorch version, every
additional optimizer group must also carry `param_names` when the adapter groups
have that field. Names and parameters must appear in the same order.

```python
model.classifier.requires_grad_(True)
groups = parameter_groups(model, lr=1e-4, magnitude_lr_multiplier=0.1)
head = list(model.classifier.named_parameters(prefix="classifier"))
groups.append({
    "params": [parameter for _, parameter in head],
    "param_names": [name for name, _ in head],
    "group_name": "task_head",
    "lr": 1e-3,
    "weight_decay": 0.01,
})
optimizer = torch.optim.AdamW(groups)
```

The magnitude multiplier applies only to `dora_nora_mlr`. Magnitude and gain
parameters have zero weight decay; adapter factors use the requested decay.
The task-head decay is set separately in its group.

Run the math and optimizer integration checks with:

```bash
/var/tmp/dora-bench/venv/bin/python -m unittest experiments.second_round.test_adapters -v
```
