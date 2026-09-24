"""Neutral-aware two-stage sentiment classification and regression."""

import torch
from torch import nn
from torch.nn import functional as F


class HierarchicalSentimentHead(nn.Module):
    def __init__(self, d_model, dropout=0.15):
        super().__init__()
        self.neutral = nn.Linear(d_model, 1)
        self.polarity = nn.Linear(d_model, 1)
        self.regression = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(d_model // 2, 1))

    def forward(self, z):
        neutral_logit = self.neutral(z).squeeze(-1)
        polarity_logit = self.polarity(z).squeeze(-1)
        p_neutral = torch.sigmoid(neutral_logit)
        p_positive_given_non_neutral = torch.sigmoid(polarity_logit)
        probabilities = torch.stack(((1 - p_neutral) * (1 - p_positive_given_non_neutral),
                                     p_neutral, (1 - p_neutral) * p_positive_given_non_neutral), dim=-1)
        regression = 3 * torch.tanh(self.regression(z).squeeze(-1))
        return {'neutral_logit': neutral_logit, 'polarity_logit': polarity_logit,
                'cls_prob': probabilities, 'reg_pred': regression}


def hierarchical_classification_loss(outputs, y_cls, neutral_pos_weight=None,
                                     lambda_neutral=1.0, lambda_polarity=1.0):
    neutral_target = (y_cls == 1).float()
    neutral_loss = F.binary_cross_entropy_with_logits(
        outputs['neutral_logit'], neutral_target, pos_weight=neutral_pos_weight)
    non_neutral = y_cls != 1
    if non_neutral.any():
        polarity_target = (y_cls[non_neutral] == 2).float()
        polarity_loss = F.binary_cross_entropy_with_logits(
            outputs['polarity_logit'][non_neutral], polarity_target)
    else:
        polarity_loss = outputs['polarity_logit'].sum() * 0
    return lambda_neutral * neutral_loss + lambda_polarity * polarity_loss
