"""KuaiRand training diagnostics and immutable cache checks."""

from pathlib import Path

import torch

from . import core
from .pipeline_utils import read_json


class DiagnosticTrainManager(core.PairedTrainManager):
    """Add end-of-epoch measurements without changing the inherited optimizer."""

    def evaluation_two_task(self, loader):
        scores = super().evaluation_two_task(loader)
        if loader is self.val_loader:
            with torch.no_grad():
                emb = self.model.reg_embedding * self.model.embedding_network.get_l2_reg()
                total = self.model.get_l2_reg()
                squares = sum(p.detach().square().sum() for p in self.model.embedding_network.parameters())
                count = sum(p.numel() for p in self.model.embedding_network.parameters())
            self.epoch_records[-1].update(
                train_fused_bce_0=self.fused_loss_0_list[-1],
                train_fused_bce_1=self.fused_loss_1_list[-1],
                train_general_bce_0=self.uni_loss_0_list[-1],
                train_general_bce_1=self.uni_loss_1_list[-1],
                train_environment_nll=self.env_loss_list[-1],
                end_epoch_embedding_l2=float(emb), end_epoch_dnn_l2=float(total - emb),
                end_epoch_embedding_rms=float((squares / count).sqrt()))
        return scores


def _snapshot(path):
    entries = read_json(Path(path) / "COMMITTED.json")
    return dict(manifest_sha256=core.file_hash(Path(path) / "COMMITTED.json"),
                files={n: dict(sha256=h, bytes=(Path(path) / n).stat().st_size,
                              mtime_ns=(Path(path) / n).stat().st_mtime_ns)
                       for n, h in entries.items()})


def _check_cache(folder, receipt):
    path = Path(folder) / "feature-cache"
    if core.file_hash(path / "COMMITTED.json") != receipt["manifest_sha256"]:
        raise ValueError("Representation cache manifest changed")
    for name, proof in receipt["files"].items():
        stat = (path / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (proof["bytes"], proof["mtime_ns"]):
            raise ValueError("Immutable representation cache file changed")
