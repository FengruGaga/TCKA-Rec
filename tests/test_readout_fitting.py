"""Small real-tensor contracts for frozen-source readout stages."""

import importlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tcka_rec import core
import numpy as np
import torch


class ReadoutFittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        generator = torch.Generator().manual_seed(17)
        self.caches, sets = {}, {}
        for i, split in enumerate(('train', 'val', 'test')):
            x = torch.randn(12, 3, generator=generator) + (20 if split != 'train' else 0)
            y = torch.tensor([[j % 2, (j // 2) % 2] for j in range(12)]).float()
            self.caches[split] = dict(x=x, g=x[:, :2].clone(), s0=x[:, :2].clone() / 2,
                                      s1=x[:, :2].clone() / 3, y=y, old=.1 + .8 * y)

            class Rows(SimpleNamespace):
                def __len__(self):
                    return len(self.y)

            sets[split] = Rows(y=y.numpy(), x=x.numpy(), row_ids=np.arange(i * 12, (i + 1) * 12))
        self.spec = dict(dataset='toy', sets=sets, tasks=['a', 'b'], initial=['a', 'b'],
                         widths=[1, 2], xdim=3, rdim=2, lr=.005, batch=4)
        self.cfg = dict(epochs=2, patience=2)

    def fitting(self):
        self.assertIsNotNone(importlib.util.find_spec('tcka_rec.readout_fitting'),
                             'Readout fitting implementation must exist')
        return importlib.import_module('tcka_rec.readout_fitting')

    def test_readout_epoch0_frozen_sources_raw_rows_and_recovery(self):
        fitting = self.fitting()
        from tcka_rec.expert import UniversalExpert
        from tcka_rec.backbone import MPTRec
        base = MPTRec(2, {}, 1, 3, [2], [2], device=core.DEVICE).to(core.DEVICE)
        u = UniversalExpert([1, 2], 2, seed=20).to(core.DEVICE)
        original = (core.model_hash(base), core.model_hash(u))
        folder = self.root / 'readout'
        result = fitting.fit_readout(base, u, self.spec, self.caches, 0, folder, self.cfg, 19)
        self.assertEqual(result['selected_epoch'], 0)  # original AUC already 1
        self.assertEqual(original, (core.model_hash(base), core.model_hash(u)))
        with np.load(folder / 'predictions.npz') as raw:
            for split in ('val', 'test'):
                np.testing.assert_array_equal(raw[split + '_p'], raw[split + '_base_p'])
                np.testing.assert_array_equal(raw[split + '_rowid'], self.spec['sets'][split].row_ids)
        (folder / 'COMMITTED.json').unlink()
        self.assertEqual(result, fitting.fit_readout(base, u, self.spec, self.caches, 0, folder, self.cfg, 19))
        self.assertTrue((folder / 'COMMITTED.json').is_file())
        with (folder / 'model.pt').open('ab') as stream:
            stream.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'integrity|hash'):
            fitting.fit_readout(base, u, self.spec, self.caches, 0, folder, self.cfg, 19)

    def test_pending_readout_rejects_raw_corruption_before_commit(self):
        fitting = self.fitting()
        from tcka_rec.expert import UniversalExpert
        from tcka_rec.backbone import MPTRec
        base = MPTRec(2, {}, 1, 3, [2], [2], device=core.DEVICE).to(core.DEVICE)
        u = UniversalExpert([1, 2], 2, seed=20).to(core.DEVICE)
        folder = self.root / 'head'
        fitting.fit_readout(base, u, self.spec, self.caches, 1, folder, self.cfg, 19)
        (folder / 'COMMITTED.json').unlink()
        with np.load(folder / 'predictions.npz') as raw:
            arrays = {key: raw[key] for key in raw.files}
        arrays['val_rowid'][0] += 100
        np.savez_compressed(folder / 'predictions.npz', **arrays)
        with self.assertRaisesRegex(ValueError, 'integrity|hash|rowid'):
            fitting.fit_readout(base, u, self.spec, self.caches, 1, folder, self.cfg, 19)
        self.assertFalse((folder / 'COMMITTED.json').exists())


if __name__ == '__main__':
    unittest.main()
