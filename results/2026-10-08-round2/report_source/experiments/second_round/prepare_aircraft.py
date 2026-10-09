"""Prepare the checksum-pinned official FGVC-Aircraft archive for vision.py.

Existing provenance is preserved. A cached archive is checksum verified, and
an existing extracted tree is compared byte-for-byte with that archive.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
from urllib.request import Request, urlopen


URL = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
SIZE = 2753340328
SHA256 = "e4e323d410e29f0370c81eabdcbb0e2b813acea1de22891b70b58ff41bfc9834"
ARCHIVE_NAME = "fgvc-aircraft-2013b.tar.gz"
DIRECTORY_NAME = "fgvc-aircraft-2013b"


def digest_stream(handle):
    digest = hashlib.sha256()
    while chunk := handle.read(1 << 20):
        digest.update(chunk)
    return digest.hexdigest()


def file_digest(path):
    with path.open("rb") as handle:
        return digest_stream(handle)


def download(cache, workers):
    archive = cache / ARCHIVE_NAME
    if archive.exists():
        return archive
    width = (SIZE + workers - 1) // workers

    def fetch(index):
        first, last = index * width, min(SIZE, (index + 1) * width) - 1
        part = cache / f"{ARCHIVE_NAME}.part{index}"
        expected = last - first + 1
        if part.exists() and part.stat().st_size == expected:
            return index, part
        for attempt in range(3):
            try:
                request = Request(URL, headers={"Range": f"bytes={first}-{last}", "User-Agent": "DoRA-reproducible-benchmark/2"})
                with urlopen(request, timeout=120) as response:
                    if response.status == 206:
                        if response.headers.get("Content-Range") != f"bytes {first}-{last}/{SIZE}":
                            raise RuntimeError("Unexpected Content-Range; archive was not accepted")
                    elif not (workers == 1 and response.status == 200):
                        raise RuntimeError("Server did not honor byte ranges; retry with --workers 1")
                    with part.open("wb") as handle:
                        shutil.copyfileobj(response, handle, length=1 << 20)
                if part.stat().st_size != expected:
                    raise RuntimeError(f"Incomplete archive part {index}")
                print(json.dumps({"event": "download_part", "index": index, "bytes": expected}), flush=True)
                return index, part
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pieces = dict(future.result() for future in as_completed([pool.submit(fetch, index) for index in range(workers)]))
    temporary = cache / f"{ARCHIVE_NAME}.incomplete"
    with temporary.open("wb") as output:
        for index in range(workers):
            with pieces[index].open("rb") as source:
                shutil.copyfileobj(source, output, length=1 << 20)
    if temporary.stat().st_size != SIZE or file_digest(temporary) != SHA256:
        raise RuntimeError("Archive checksum mismatch; incomplete archive and parts retained for diagnosis")
    temporary.replace(archive)
    for path in pieces.values():
        path.unlink()
    return archive


def validate_annotation_counts(directory):
    data = directory / "data"
    for split, expected in (("train", 3334), ("val", 3333), ("test", 3333)):
        rows = (data / f"images_variant_{split}.txt").read_text().splitlines()
        assert len(rows) == expected, (split, len(rows))
        assert all((data / "images" / f"{row.split(' ', 1)[0]}.jpg").is_file() for row in rows)
    assert len((data / "variants.txt").read_text().splitlines()) == 100


def check_members(handle):
    members = handle.getmembers()
    for member in members:
        path = Path(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != DIRECTORY_NAME:
            raise RuntimeError(f"Unsafe archive path: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(f"Unsupported archive member: {member.name}")
    return members


def verify_existing_tree(archive, cache):
    verified = 0
    with tarfile.open(archive, "r:gz") as handle:
        for member in check_members(handle):
            path = cache / member.name
            if path.is_symlink() or not path.resolve().is_relative_to(cache.resolve()):
                raise RuntimeError(f"Unsafe extracted path: {path}")
            if member.isdir():
                assert path.is_dir(), path
            else:
                assert path.is_file() and path.stat().st_size == member.size, path
                with handle.extractfile(member) as source, path.open("rb") as existing:
                    assert digest_stream(source) == digest_stream(existing), path
                verified += 1
    return verified


def prepare(cache, workers=8):
    started = time.perf_counter()
    cache.mkdir(parents=True, exist_ok=True)
    archive = download(cache, workers)
    if archive.stat().st_size != SIZE or file_digest(archive) != SHA256:
        raise RuntimeError("Cached archive checksum mismatch; cache was left unchanged")
    directory = cache / DIRECTORY_NAME
    if directory.exists():
        verified = verify_existing_tree(archive, cache)
    else:
        with tempfile.TemporaryDirectory(prefix="aircraft_extract_", dir=cache) as stage:
            with tarfile.open(archive, "r:gz") as handle:
                members = check_members(handle)
                handle.extractall(stage, members=members, filter="data")
                verified = sum(member.isfile() for member in members)
            extracted = Path(stage) / DIRECTORY_NAME
            validate_annotation_counts(extracted)
            extracted.rename(directory)
    validate_annotation_counts(directory)
    provenance_path = cache / "download_provenance.json"
    pinned = {"url": URL, "bytes": SIZE, "sha256": SHA256}
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text())
        assert all(existing[key] == value for key, value in pinned.items())
    else:
        provenance_path.write_text(json.dumps({**pinned, "download_extract_seconds": time.perf_counter() - started}, indent=2) + "\n")
    result = {"status": "ready", "cache": str(cache), "verified_archive_sha256": SHA256,
              "verified_extracted_files": verified, "seconds": time.perf_counter() - started,
              "provenance": str(provenance_path)}
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/var/tmp/dora-bench/cache/aircraft"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be between 1 and 32")
    prepare(args.cache, args.workers)
