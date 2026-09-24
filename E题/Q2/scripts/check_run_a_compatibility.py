"""One-time architecture and forward/loss equivalence check against commit 5790875."""

import subprocess

import torch
from torch.nn import functional as F

from diagnostics.oracle_text import OracleTextBaseline


source = subprocess.check_output(
    ['git', 'show', '5790875:E题/Q2/diagnostics/oracle_text.py'], text=True)
namespace = {'__name__': 'oracle_5790875'}
exec(compile(source, 'oracle_5790875.py', 'exec'), namespace)
OldOracleTextBaseline = namespace['OracleTextBaseline']
cfg = {'d_model': 128, 'dropout': 0.15, 'use_position': False,
       'heads': 4, 'temporal_layers': 1}
torch.manual_seed(42)
old = OldOracleTextBaseline(cfg).eval()
torch.manual_seed(42)
new = OracleTextBaseline(cfg).eval()
new.load_state_dict(old.state_dict(), strict=True)
batch = {
    'text_teacher': torch.randn(2, 50, 768),
    'audio': torch.randn(2, 50, 74),
    'vision': torch.randn(2, 50, 35),
    'text_bert': torch.ones(2, 3, 50, dtype=torch.long),
}
with torch.no_grad():
    old_logits, old_reg = old(batch)
    new_logits, new_reg = new(batch)
    assert old_logits.shape == new_logits.shape == (2, 3)
    assert old_reg.shape == new_reg.shape == (2,)
    assert torch.equal(old_logits, new_logits)
    assert torch.equal(old_reg, new_reg)
    labels = torch.tensor([0, 2])
    targets = torch.tensor([-1.0, 1.0])
    weights = torch.tensor([1.1, 1.4, 0.5])
    old_loss = F.cross_entropy(old_logits, labels, weight=weights) + F.smooth_l1_loss(
        old_reg, targets, beta=0.5)
    new_loss = F.cross_entropy(new_logits, labels, weight=weights) + F.smooth_l1_loss(
        new_reg, targets, beta=0.5)
    assert torch.equal(old_loss, new_loss)
print('Run A state dict, output shapes, forward values, and loss: identical')
