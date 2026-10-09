"""Deterministic portable evidence bundle; preserves every included file hash."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile


METHODS = ('lora', 'dora', 'nora', 'dora_nora', 'dora_nora_mlr', 'dora_nora_gain')
TEXT_SUFFIXES = {'.json', '.jsonl', '.log', '.py', '.md', '.tsv', '.npz', '.txt'}
PART_BYTES = 4 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def split_archive(path):
    """Write deterministic fixed-size publication parts in archive byte order."""
    parts = []
    with path.open('rb') as stream:
        while content := stream.read(PART_BYTES):
            name = f'{path.name}.part-{len(parts):03d}'
            (path.parent / name).write_bytes(content)
            parts.append({'path': name, 'bytes': len(content),
                          'sha256': hashlib.sha256(content).hexdigest()})
    return parts


def assemble_parts(report, output=None):
    """Verify each published part and atomically reconstruct the complete archive.

    Uses only the standard library and never reads an existing complete archive.
    Part order and names come from the checked manifest, not a shell wildcard.
    """
    report = Path(report)
    manifest = json.loads((report / 'raw_manifest.json').read_text())
    name = manifest['archive_path']
    if name != 'raw_artifacts.tar.gz' or manifest['archive_part_bytes'] != PART_BYTES:
        raise ValueError('Unsupported archive name or part size')
    parts = manifest['archive_parts']
    size = manifest['archive_bytes']
    if size <= 0 or len(parts) != (size + PART_BYTES - 1) // PART_BYTES:
        raise ValueError('Archive part count does not match total byte count')
    for index, row in enumerate(parts):
        if row['path'] != f'{name}.part-{index:03d}':
            raise ValueError('Archive part names must be consecutive and in byte order')
        if row['bytes'] != min(PART_BYTES, size - index * PART_BYTES):
            raise ValueError(f'Incorrect expected byte count for {row["path"]}')
    destination = Path(output) if output is not None else report / name
    protected = {report / 'raw_manifest.json', report / 'summary.json', *(report / row['path'] for row in parts)}
    if destination.resolve() in {path.resolve() for path in protected}:
        raise ValueError('Assembly output would overwrite a manifest or input part')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f'.{destination.name}.', suffix='.tmp',
                                         dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            complete_digest = hashlib.sha256()
            complete_bytes = 0
            for row in parts:
                path = report / row['path']
                digest = hashlib.sha256()
                count = 0
                with path.open('rb') as source:
                    while content := source.read(1 << 20):
                        digest.update(content)
                        complete_digest.update(content)
                        count += len(content)
                        complete_bytes += len(content)
                        stream.write(content)
                if count != row['bytes'] or digest.hexdigest() != row['sha256']:
                    raise ValueError(f'Archive part failed byte/hash verification: {row["path"]}')
            if complete_bytes != size or complete_digest.hexdigest() != manifest['archive_sha256']:
                raise ValueError('Assembled archive failed complete byte/hash verification')
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {'path': str(destination), 'sha256': manifest['archive_sha256'], 'bytes': size,
            'parts_verified': parts, 'part_bytes': PART_BYTES}


def selected_roots(raw):
    roots = [('retrieval', raw / 'retrieval')]
    roots += [(f'vision/rank{rank}_run', raw / 'vision' / f'rank{rank}_run') for rank in (2, 8)]
    roots += [(f'vision/{name}', raw / 'vision' / name) for name in ('numerical_control', 'numerical_control_full')]
    roots += [(f'cogs/{method}', raw / 'cogs' / method) for method in METHODS]
    roots += [('teacher_compact', raw / 'teacher_compact'), ('cogs/analysis_source', raw / 'cogs' / 'analysis_source')]
    return roots


def bundle(raw, output, source_root):
    files = {}
    for prefix, directory in selected_roots(raw):
        if not directory.exists():
            continue
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            if prefix.startswith('cogs/') and prefix != 'cogs/lora' and 'data' in path.relative_to(directory).parts:
                continue  # Identical pinned TSV caches are retained once under cogs/lora/data.
            allowed = path.suffix in TEXT_SUFFIXES or (prefix == 'teacher_compact' and path.suffix in {'.pt', '.npy'})
            if not allowed:
                continue
            if path.stat().st_size >= 95 * 1024 * 1024:
                raise ValueError(f'Included evidence file unexpectedly exceeds95MiB: {path}')
            files[str(Path(prefix) / path.relative_to(directory))] = path
    for name in ('cross_shard_baseline_audit.json', 'preparation_audit.json'):
        audit = raw / 'vision' / name
        if audit.exists():
            files[f'vision/{name}'] = audit
    diagnostic = raw / 'matched_magnitude_validation.json'
    if diagnostic.exists():
        files['matched_magnitude_validation.json'] = diagnostic
    for name in ('analysis.json', 'split_manifest.json'):
        path = raw / 'cogs' / name
        if path.exists():
            files[f'cogs/{name}'] = path
    for path in sorted(source_root.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in TEXT_SUFFIXES:
            files[str(Path('report_source') / path.relative_to(source_root))] = path
    for name in ('retrieval.log', 'retrieval_control.log', 'retrieval_analysis.log', 'cogs_gpu2_queue.log', *(f'cogs_{method}.log' for method in METHODS)):
        path = raw / name
        if path.exists():
            files[f'logs/{name}'] = path
    for rank in (2, 8):
        path = raw / 'vision' / f'rank{rank}_run.log'
        if path.exists():
            files[f'logs/vision_rank{rank}.log'] = path
    # Read each included file once: live progress logs can grow while a preview
    # is built, and manifest hashes must describe the exact archived bytes.
    contents = {name: path.read_bytes() for name, path in sorted(files.items())}
    manifest = {'format_version': 1, 'files': [
        {'archive_path': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
        for name, content in contents.items()],
        'exclusions': 'Pretrained weights; downstream adapter/head checkpoints; image caches; failed/obsolete pilots and smoke attempts; duplicate COGS TSV caches (one pinned copy retained under cogs/lora/data). Compact teacher checkpoints are retained separately inside teacher_compact when available.',
        'audit_scope': 'Included hashes and saved-prediction metrics can be checked without omitted downstream checkpoints. Full local checkpoint audits remain captured as attestations; portable reports count omitted payloads explicitly.'}
    payload = (json.dumps(manifest, indent=2) + '\n').encode()
    archive_path = output / 'raw_artifacts.tar.gz'
    with archive_path.open('wb') as destination:
        with gzip.GzipFile(filename='', mode='wb', fileobj=destination, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for name, content in contents.items():
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(content), 0o644, 0
                    archive.addfile(info, io.BytesIO(content))
                info = tarfile.TarInfo('evidence_manifest.json')
                info.size, info.mode, info.mtime = len(payload), 0o644, 0
                archive.addfile(info, io.BytesIO(payload))
    if archive_path.stat().st_size >= 95 * 1024 * 1024:
        raise ValueError('Archive exceeds95MiB; split compact teacher evidence before publishing')
    parts = split_archive(archive_path)
    manifest.update(archive_path=archive_path.name, archive_sha256=sha256(archive_path),
                    archive_bytes=archive_path.stat().st_size, archive_part_bytes=PART_BYTES, archive_parts=parts)
    (output / 'raw_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'path': archive_path.name, 'sha256': manifest['archive_sha256'], 'bytes': manifest['archive_bytes'],
            'file_count': len(files), 'part_bytes': PART_BYTES, 'parts': parts,
            'publication': 'Download the listed parts and verify/reassemble them with archive.py before extraction; the complete archive is retained locally.'}


def verify_extracted(raw):
    path = raw / 'evidence_manifest.json'
    if not path.exists():
        return {'included_file_hashes_verified': 0, 'scope': 'Live artifact tree; archive manifest not present.'}
    manifest = json.loads(path.read_text())
    for row in manifest['files']:
        path = raw / row['archive_path']
        assert path.is_file() and path.stat().st_size == row['bytes']
        assert sha256(path) == row['sha256'], path
    return {'included_file_hashes_verified': len(manifest['files']), 'scope': 'Every file listed in the extracted evidence manifest verified before scoring.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verify and reassemble published evidence archive parts using only the standard library.')
    parser.add_argument('--assemble-parts', action='store_true', required=True)
    parser.add_argument('--report-dir', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(assemble_parts(args.report_dir, args.output), indent=2))
