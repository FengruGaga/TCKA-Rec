"""Stable split fingerprints used by training protocols."""

import hashlib
import json

import numpy as np

from .training import _rowids


def _array_hash(array):
    array = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    for offset in range(0, len(array), 65536):
        digest.update(np.ascontiguousarray(array[offset:offset + 65536]).tobytes())
    return digest.hexdigest()


def _object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def split_identity(spec, split, rows):
    signature = spec.get('data_signature', {})
    admitted = signature.get('splits', {}).get(split)
    if admitted is None:
        return dict(n=len(rows), names=list(rows.names), rowid_sha256=_array_hash(_rowids(rows)),
                    features_sha256=_array_hash(rows.x), old_labels_sha256=_array_hash(rows.y[:, :2]))
    schema = dict(features_dtype=str(rows.x.dtype), rowid_dtype=str(_rowids(rows).dtype),
                  labels_dtype=str(rows.y.dtype))
    if (admitted.get('rows') != len(rows) or signature.get('features') != list(rows.names)
            or signature.get('tasks') != spec['tasks']
            or any(admitted.get(k) != v for k, v in schema.items())
            or rows.x.shape != (len(rows), len(rows.names))
            or rows.y.shape != (len(rows), len(spec['tasks']))):
        raise ValueError(f'Data signature schema mismatch: {split}')
    old_labels = {name: admitted.get('labels', {}).get(name, {}).get('sha256')
                  for name in spec['initial']}
    if (not admitted.get('features_sha256') or not admitted.get('rowid_sha256')
            or not all(old_labels.values())):
        raise ValueError(f'Data signature fingerprints missing: {split}')
    return dict(n=len(rows), names=list(rows.names), **schema,
                rowid_sha256=admitted['rowid_sha256'],
                features_sha256=admitted['features_sha256'],
                old_labels_sha256=_object_hash(old_labels),
                old_labels_order=list(spec['initial']),
                fingerprint_source='dataset-signature-admission')
