import torch
from torch.nn import functional as F


def masked_smooth_l1(pred, target, mask, beta=1.0):
    if not bool(mask.any()):
        return pred.sum() * 0
    return F.smooth_l1_loss(pred[mask], target[mask], beta=beta)


def compute_loss(masked, clean, batch, valid, original_missing, artificial_missing,
                 class_weights, cfg, variant):
    lc = cfg['loss']
    cls = F.cross_entropy(masked['cls_logits'], batch['y_cls'], weight=class_weights)
    reg = F.smooth_l1_loss(masked['reg_pred'], batch['y_reg'], beta=0.5)
    bridge_observed = valid & ~original_missing['T'] & ~artificial_missing['T']
    bridge = masked_smooth_l1(masked['teacher_recon'], batch['text_teacher'], bridge_observed)
    rec = cls.new_zeros(())
    if variant >= 3:
        for m in 'TAV':
            target = clean['proj'][m].detach() if clean is not None else masked['proj'][m].detach()
            rec = rec + masked_smooth_l1(masked['recon'][m], target, artificial_missing[m])
        rec = rec / 3
    agreement = F.smooth_l1_loss(masked['cls_prob'][:, 2] - masked['cls_prob'][:, 0], masked['reg_pred']/3)
    total = lc['cls']*cls + lc['reg']*reg + lc['text_bridge']*bridge + lc['cls_reg_agreement']*agreement
    if variant >= 3:
        total = total + lc['reconstruction']*rec
    consistency = cls.new_zeros(())
    if variant >= 5 and clean is not None:
        teacher_prob = clean['cls_prob'].detach()
        consistency = F.kl_div(F.log_softmax(masked['cls_logits'], -1), teacher_prob, reduction='batchmean')
        consistency = consistency + F.smooth_l1_loss(masked['reg_pred'], clean['reg_pred'].detach())
        total = total + lc['consistency']*consistency
        total = total + 0.30 * (F.cross_entropy(clean['cls_logits'], batch['y_cls'], weight=class_weights)
                                + F.smooth_l1_loss(clean['reg_pred'], batch['y_reg'], beta=0.5))
    return total, {'cls': cls.item(), 'reg': reg.item(), 'bridge': bridge.item(),
                   'reconstruction': rec.item(), 'consistency': consistency.item(), 'agreement': agreement.item()}
