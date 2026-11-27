"""Shared execution identity and immutable stage checks."""

import hashlib
import json
import os
from pathlib import Path

from . import core
from .training import _utc

PROJECT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def source_hashes():
    return {p.name: core.file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def bind_contract(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise ValueError(f'Stage contract changed: {path}')
    core.dump(path, value)


def verify_stage(folder, required=()):
    folder = Path(folder).resolve()
    manifest = folder / 'COMMITTED.json'
    if not manifest.is_file():
        raise ValueError(f'Missing committed stage: {folder}')
    entries = read_json(manifest)
    if not set(required).issubset(entries):
        raise ValueError(f'Incomplete committed stage: {folder}')
    for name, sha in entries.items():
        p = (folder / name).resolve()
        if not p.is_relative_to(folder) or not p.is_file() or core.file_hash(p) != sha:
            raise ValueError(f'Committed integrity failure: {p}')
    return dict(path=str(folder), files=entries, committed_sha256=core.file_hash(manifest))


def status(path, state, **kwargs):
    core.dump(path, dict(state=state, pid=os.getpid() if state == 'running' else None,
                         updated_at=_utc(), **kwargs))
