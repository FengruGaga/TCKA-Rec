"""Behavioral contracts for the conditional readout head."""

import importlib
import importlib.util
import unittest

import torch
from torch.nn import functional as F

from tcka_rec.backbone import MPTRec


class ReadoutModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("tcka_rec.readout_models"),
                             "Readout model implementation is missing")
        return importlib.import_module("tcka_rec.readout_models")

    def fixture(self):
        generator = torch.Generator().manual_seed(23)
        batch = {k: torch.randn(9, width, generator=generator)
                 for k, width in [('x', 7), ('g', 8), ('s0', 8), ('s1', 8)]}
        return batch

    def test_conditional_readout_counts_zero_endpoints_and_only_head_gradients(self):
        mod = self.module()
        for rdim, count in [(128, 1541), (64, 773)]:
            head = mod.ConditionalReadout(rdim)
            self.assertEqual(sum(p.numel() for p in head.parameters()), count)
            p = torch.tensor([0., 1., 1e-8, .3, .99999999], requires_grad=True)
            h = torch.randn(5, rdim, requires_grad=True)
            u = torch.randn(5, rdim, requires_grad=True)
            self.assertTrue(torch.equal(head.delta(h, u), torch.zeros(5)))
            self.assertTrue(torch.equal(head(h, u, p), p))
            head(h, u, p).sum().backward()
            self.assertIsNone(h.grad)
            self.assertIsNone(u.grad)
            self.assertIsNone(p.grad)
            self.assertTrue(all(parameter.grad is not None for parameter in head.parameters()))
            with torch.no_grad():
                head.out.weight.fill_(.25)
                head.out.bias.fill_(.1)
            expected_delta = head.out(torch.sigmoid(head.A(torch.cat([
                F.layer_norm(h, (rdim,)), F.layer_norm(u, (rdim,))], 1))) * head.C(
                    F.layer_norm(u, (rdim,)))).reshape(-1)
            self.assertTrue(torch.allclose(head.delta(h, u), expected_delta))

    def test_old_task_hidden_is_exact_original_fusion_including_one_row(self):
        mod = self.module()
        batch = self.fixture()
        base = MPTRec(2, {}, 4, 7, [12, 8], [6, 4])
        for n in [9, 1]:
            rows = {k: v[:n] for k, v in batch.items()}
            for position in [0, 1]:
                env = base.env_embedding_network.weight[position]
                experts = torch.stack([rows[f's{position}'] * env, rows['g']], 2)
                expected = (experts @ base.gate_networks[position](rows['x']).unsqueeze(2)).squeeze(-1)
                actual = mod.old_task_hidden(base, rows, position)
                self.assertEqual(actual.shape, (n, 8))
                self.assertTrue(torch.equal(actual, expected))
                self.assertFalse(actual.requires_grad)


if __name__ == '__main__':
    unittest.main()
