"""Rebuild a report offline from a read-only extracted archive and attest outputs.

The audit is written outside the archive only after all comparisons pass, avoiding
a circular archive hash. Original report files and test_validation.json are read
only. The extracted evidence and rebuild log remain available for inspection.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile

from experiments.second_round.archive import assemble_parts


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def verify(report, work_parent, audit_path, allow_partial=False, force_parts=False):
    report = report.resolve()
    summary = json.loads((report / 'summary.json').read_text())
    assert allow_partial or summary['complete'], 'Final report is incomplete'
    manifest = json.loads((report / 'raw_manifest.json').read_text())
    assert manifest['archive_parts'] == summary['raw_archive']['parts']
    assert manifest['archive_part_bytes'] == summary['raw_archive']['part_bytes']
    assert manifest['archive_sha256'] == summary['raw_archive']['sha256']
    assert manifest['archive_bytes'] == summary['raw_archive']['bytes']
    work_parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='report_offline_rebuild_', dir=work_parent.resolve()))
    archive = report / 'raw_artifacts.tar.gz'
    original_full_archive_present = archive.is_file()
    parts_only = force_parts or not original_full_archive_present
    assembly = None
    if parts_only:
        assembly = assemble_parts(report, work / 'raw_artifacts.tar.gz')
        archive = Path(assembly['path'])
    archive_hash = sha256(archive)
    assert archive_hash == manifest['archive_sha256']
    assert archive.stat().st_size == manifest['archive_bytes']
    raw, rebuilt = work / 'raw', work / 'rebuilt'
    raw.mkdir()
    with tarfile.open(archive) as source:
        source.extractall(raw, filter='data')
    for path in raw.rglob('*'):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((item for item in raw.rglob('*') if item.is_dir()), reverse=True):
        path.chmod(0o555)
    raw.chmod(0o555)
    isolated = {'PYTHONPATH': str(raw / 'report_source'), 'CUDA_VISIBLE_DEVICES': '',
                'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                'HF_HOME': str(work / 'absent_model_cache'), 'PYTHONDONTWRITEBYTECODE': '1'}
    command = [sys.executable, '-m', 'experiments.second_round.report',
               '--raw-root', str(raw), '--output-dir', str(rebuilt), '--skip-bundle']
    if summary['preview']:
        command.append('--allow-partial')
    with (work / 'rebuild.log').open('w') as log:
        completed = subprocess.run(command, cwd=work, env={**os.environ, **isolated},
                                   stdout=log, stderr=subprocess.STDOUT)
    assert completed.returncode == 0, f'Rebuild failed; inspect {work / "rebuild.log"}'
    rebuilt_summary = json.loads((rebuilt / 'summary.json').read_text())
    assert rebuilt_summary['complete'] == summary['complete']
    assert rebuilt_summary['data_complete'] == summary['data_complete']
    assert rebuilt_summary['missing'] == summary['missing']
    compared = []
    names = {path.name for path in report.iterdir() if path.suffix in {'.png', '.svg', '.csv'}}
    # Author-written README material is outside the report generator's outputs.
    names.update(path.name for path in rebuilt.glob('*.md'))
    names.update(('teacher_recomputed.json', 'matched_magnitude_validation.json'))
    for name in sorted(names):
        expected, actual = report / name, rebuilt / name
        assert expected.read_bytes() == actual.read_bytes(), f'Rebuild output differs: {name}'
        compared.append({'path': name, 'sha256': sha256(expected), 'bytes': expected.stat().st_size})
    source_hashes = {}
    for path in sorted((report / 'report_source').rglob('*.py')):
        name = path.relative_to(report)
        digest = sha256(path)
        assert digest == sha256(raw / name) == sha256(rebuilt / name), f'Source differs: {name}'
        source_hashes[str(name)] = digest
    # Full task audit objects distinguish locally present from omitted checkpoints.
    # Scores, uncertainty and initialization controls must reproduce exactly.
    structured_checks = {}
    for name in summary['task_order']:
        expected = json.loads((report / 'tasks' / f'{name}.json').read_text())
        actual = json.loads((rebuilt / 'tasks' / f'{name}.json').read_text())
        fields = ['ranks', 'paired_uncertainty', 'provenance']
        fields += [key for key in ('initial_numerical_controls', 'initial_numerical_control') if key in expected]
        for field in fields:
            assert expected[field] == actual[field], f'Task reconstruction differs: {name}.{field}'
        structured_checks[name] = fields
    included = json.loads((raw / 'evidence_manifest.json').read_text())['files']
    for row in included:
        path = raw / row['archive_path']
        assert path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'], row['archive_path']
    assert rebuilt_summary['evidence_validation']['included_file_hashes_verified'] == len(included)
    if assembly is not None:
        for row in assembly['parts_verified']:
            path = report / row['path']
            assert path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'], row['path']
    result = {'format_version': 1, 'passed': True, 'preview': summary['preview'],
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'archive': {'path': archive.name, 'sha256': archive_hash, 'bytes': archive.stat().st_size},
              'archive_input_mode': 'parts_only' if parts_only else 'complete_archive',
              'original_full_archive_present': original_full_archive_present,
              'original_full_archive_read': not parts_only,
              'parts_verified_before_and_after': assembly['parts_verified'] if assembly else [],
              'input_files_verified_before_and_after': len(included), 'read_only_evidence': True,
              'byte_identical_outputs': compared, 'structured_task_fields_equal': structured_checks,
              'source_sha256': source_hashes, 'verifier_sha256': sha256(Path(__file__)),
              'environment': {'python': sys.version, 'python_executable': sys.executable,
                              'platform': platform.platform(), 'report_versions': rebuilt_summary['versions'],
                              'variables': isolated, 'cwd': str(work)},
              'command': command, 'work_directory': str(work),
              'scope': 'Rebuilt from archived sources outside the repository, with GPUs disabled, offline model-library settings and a nonexistent model cache. Every included evidence file remained hash-identical. Compared report outputs, source files, task scores, uncertainty and numerical controls. Recomputed saved predictions/logits and compact teacher checkpoints; did not rerun downstream model inference or training.'}
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--work-parent', type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument('--audit-output', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--force-parts', action='store_true', help='Reassemble only from published parts, without reading a locally retained full archive')
    args = parser.parse_args()
    audit = args.audit_output or args.report_dir / 'offline_rebuild_audit.json'
    result = verify(args.report_dir, args.work_parent, audit, args.allow_partial, args.force_parts)
    print(json.dumps({'passed': result['passed'], 'archive_sha256': result['archive']['sha256'],
                      'compared_outputs': len(result['byte_identical_outputs']),
                      'archive_input_mode': result['archive_input_mode'],
                      'audit': str(audit), 'work_directory': result['work_directory']}, indent=2))


if __name__ == '__main__':
    main()
