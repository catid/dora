"""Train-only moments and gradients for initialization, independent of task code.

The caller supplies batches and the existing task loss. This collector never
steps an optimizer. Both source-specific and pooled statistics are retained.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from experiments.init_common import sha, tensor_digest, write_once


class CalibrationCollector(AbstractContextManager):
    def __init__(self, model, targets, *, retain_batch_grams=True):
        self.model = model
        self.modules = {name: module for name, module in model.named_modules()
                        if isinstance(module, torch.nn.Linear)
                        and (targets(name, module) if callable(targets)
                             else any(name == suffix or name.endswith('.'+suffix) for suffix in targets))}
        assert self.modules
        self.retain_batch_grams = retain_batch_grams
        self.stats, self.handles, self.examples = {}, [], []
        self.mask, self.total_loss_weight, self.batches = None, 0., 0

    def __enter__(self):
        self.started = time.perf_counter()
        self.training = {module: module.training for module in self.model.modules()}
        self.requires_grad = {p: p.requires_grad for p in self.model.parameters()}
        assert all(p.grad is None for p in self.requires_grad)
        self.original_digest = tensor_digest(self.model.named_parameters())
        self.random_state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.cpu_rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        self.model.eval().requires_grad_(False)
        grouped = {}
        for name, module in self.modules.items():
            grouped.setdefault(name.rsplit('.', 1)[-1], []).append(float(module.weight.detach().float().std()))
        self.group_std = {key: round(statistics.mean(values), 3) for key,values in grouped.items()}
        for name, module in self.modules.items():
            module.weight.requires_grad_(True)
            self.stats[name] = {'input_count': 0, 'output_count': 0, 'output_grad_count': 0,
                                'forward_count': 0, 'backward_count': 0, 'input_gram_batches': []}
            self.handles.append(module.register_forward_hook(self._hook(name)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        return self

    @staticmethod
    def _accumulate(record, key, value):
        value = value.detach()
        if key in record:
            record[key].add_(value)
        else:
            record[key] = value.clone()

    @staticmethod
    def _rows(value, mask):
        rows = value.detach().reshape(-1, value.shape[-1]).float()
        if mask is not None:
            selected = mask.reshape(-1)
            assert selected.numel() == rows.shape[0], (value.shape, mask.shape)
            rows = rows[selected.to(device=rows.device, dtype=torch.bool)]
        assert rows.shape[0] and torch.isfinite(rows).all()
        return rows

    def _hook(self, name):
        def hook(module, args, output):
            record = self.stats[name]
            mask = self.mask.detach().clone() if self.mask is not None else None
            x, y = self._rows(args[0], mask), self._rows(output, mask)
            all_x, all_y = self._rows(args[0], None), self._rows(output, None)
            # Explicitly disable ambient BF16 autocast for covariance products.
            with torch.autocast(x.device.type, enabled=False):
                self._accumulate(record, 'input_sum', x.sum(0))
                self._accumulate(record, 'input_second_sum', x.T @ x)
                self._accumulate(record, 'output_sum', y.sum(0))
                self._accumulate(record, 'output_second_sum', y.T @ y)
                raw_gram = all_x.T @ all_x
                self._accumulate(record, 'filet_input_batch_cov_sum', raw_gram / len(all_x))
                input_scale = all_x.max().abs()
                assert torch.isfinite(input_scale) and input_scale > 0, f'CorDA scale is zero: {name}'
                normalized_x = all_x / input_scale
                self._accumulate(record, 'corda_gram_sum', normalized_x.T @ normalized_x)
                centered_x = all_x - all_x.mean(0)
                self._accumulate(record, 'tlora_gram_sum', centered_x.T @ centered_x)
                # Match Astra's unusual abs(max(y)), not max(abs(y)).
                scale = all_y.max().abs()
                assert torch.isfinite(scale) and scale > 0, f'Astra scale is zero: {name}'
                normalized = all_y / scale
                self._accumulate(record, 'astra_gram_sum', normalized.T @ normalized)
            if self.retain_batch_grams:
                record['input_gram_batches'].append(raw_gram.detach().cpu())
            record['input_count'] += len(x)
            record['output_count'] += len(y)
            record['forward_count'] += 1
            def backward(gradient):
                dy, all_dy = self._rows(gradient, mask), self._rows(gradient, None)
                with torch.autocast(dy.device.type, enabled=False):
                    self._accumulate(record, 'output_grad_second_sum', dy.T @ dy)
                    self._accumulate(record, 'output_grad_diagonal_sum', dy.square().sum(0))
                    self._accumulate(record, 'filet_output_grad_batch_cov_sum', all_dy.T @ all_dy / len(all_dy))
                record['output_grad_count'] += len(dy)
                record['backward_count'] += 1
            output.register_hook(backward)
        return hook

    def begin_batch(self, *, example_ids, attention_mask=None):
        self.model.zero_grad(set_to_none=True)
        self.mask = attention_mask
        self.examples.extend(example_ids)
        self.batches += 1

    def set_mask(self, attention_mask):
        """Retrieval has separate query/document forwards within one loss."""
        self.mask = attention_mask

    def backward(self, loss, *, loss_weight=1.):
        assert math.isfinite(loss_weight) and loss_weight > 0
        assert torch.isfinite(loss)
        loss.backward()
        for name, module in self.modules.items():
            gradient = module.weight.grad
            assert gradient is not None and torch.isfinite(gradient).all(), name
            self._accumulate(self.stats[name], 'weight_grad_sum', gradient.float() * loss_weight)
        self.total_loss_weight += loss_weight
        self.model.zero_grad(set_to_none=True)

    def save(self, root, metadata):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        assert self.batches and self.total_loss_weight > 0
        assert len(self.examples) == len(set(self.examples)), 'Calibration examples must be unique'
        assert self.original_digest == tensor_digest(self.model.named_parameters())
        files, hashes = {}, {}
        for index, (name, record) in enumerate(self.stats.items()):
            assert record['forward_count'] == record['backward_count']
            n, ny, ng = record['input_count'], record['output_count'], record['output_grad_count']
            assert min(n, ny, ng, record['forward_count']) > 0, name
            mean_x = record['input_sum'] / n
            mean_y = record['output_sum'] / ny
            cx = record['input_second_sum'] / n
            cy = record['output_second_sum'] / ny
            values = {
                'input_count': n, 'output_count': ny, 'output_grad_count': ng,
                'input_mean': mean_x, 'output_mean': mean_y,
                'input_cov': cx, 'input_centered_cov': cx - mean_x[:, None] * mean_x[None, :],
                'output_cov': cy, 'output_centered_cov': cy - mean_y[:, None] * mean_y[None, :],
                'weight_grad': record['weight_grad_sum'] / self.total_loss_weight,
                'output_grad_cov': record['output_grad_second_sum'] / ng,
                'output_fisher_diag': record['output_grad_diagonal_sum'] / ng,
                'filet_input_cov': record['filet_input_batch_cov_sum'] / record['forward_count'],
                'filet_output_grad_cov': record['filet_output_grad_batch_cov_sum'] / record['backward_count'],
                'astra_output_cov': record['astra_gram_sum'] / record['forward_count'],
                'corda_input_cov': record['corda_gram_sum'] / record['forward_count'],
                'tlora_input_cov': record['tlora_gram_sum'] / record['forward_count'],
                'input_gram_batches': record['input_gram_batches'],
                'population_size': len(self.examples), 'batch_count': self.batches,
                'forward_count': record['forward_count'], 'backward_count': record['backward_count'],
            }
            if 'training_population_size' in metadata:
                values['training_population_size'] = int(metadata['training_population_size'])
            values = {key: value.detach().cpu() if torch.is_tensor(value) else value for key,value in values.items()}
            for key,value in values.items():
                if torch.is_tensor(value):
                    assert torch.isfinite(value).all(), (name, key)
            path = root / f'module_{index:03d}.pt'
            assert not path.exists(), f'Calibration is immutable: {path}'
            torch.save(values, path)
            files[name], hashes[name] = str(path.resolve()), sha(path)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        result = {'schema': 'dora-init-calibration-v1', 'module_files': files, 'module_sha256': hashes,
                  'metadata': {**metadata, 'example_ids': self.examples, 'population_size': len(self.examples),
                               'batch_count': self.batches, 'total_loss_weight': self.total_loss_weight,
                               'model_mode': 'eval', 'parameters_unchanged_sha256': self.original_digest,
                               'inilora_group_std': self.group_std,
                               'source_sha256': sha(__file__), 'statistics_dtype': 'float32',
                               'output_grad_scaling': 'Raw gradients from caller loss; no batch/token rescaling',
                               'mask_policy': 'Pooled moments use supplied nonpadding mask; Astra/CorDA/TLoRA/FILet/CNT source-specific moments use unmasked forwards',
                               'seconds': time.perf_counter() - self.started,
                               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}}
        write_once(root / 'manifest.json', result)
        return result

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.model.zero_grad(set_to_none=True)
        for parameter, flag in self.requires_grad.items():
            parameter.requires_grad_(flag)
        for module, training in self.training.items():
            module.training = training
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng:
            torch.cuda.set_rng_state_all(self.cuda_rng)
        assert self.original_digest == tensor_digest(self.model.named_parameters())
        return False
