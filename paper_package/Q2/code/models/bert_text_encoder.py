"""Pretrained BERT text features with explicit token validation."""

import torch
from transformers import BertModel


class BertTextEncoder(BertModel):
    """A BertModel with the same state-dict keys as prior BERT checkpoints."""

    def configure_trainability(self, mode='frozen', unfreeze_last_n=0):
        if mode not in {'frozen', 'partial'}:
            raise ValueError(f'Unknown BERT mode: {mode}')
        if mode == 'frozen' and unfreeze_last_n != 0:
            raise ValueError('Frozen BERT requires unfreeze_last_n=0')
        if mode == 'partial' and unfreeze_last_n != 4:
            raise ValueError('Only the previously tested last-four-layer mode is supported')
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if mode == 'partial':
            for layer in self.encoder.layer[-4:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)

    @staticmethod
    def checked_tokens(text_bert, vocab_size):
        if text_bert.ndim != 3 or text_bert.shape[1] != 3:
            raise ValueError(f'text_bert must have shape [B,3,T], got {tuple(text_bert.shape)}')
        if text_bert.is_floating_point():
            if not torch.isfinite(text_bert).all():
                raise ValueError('text_bert contains NaN or Inf')
            rounded = text_bert.round()
            if not torch.allclose(text_bert, rounded, atol=1e-4, rtol=0):
                raise ValueError('text_bert contains fractional token values')
            text_bert = rounded.long()
        else:
            text_bert = text_bert.long()
        ids, attention, token_type = text_bert.unbind(dim=1)
        if ids.min() < 0 or ids.max() >= vocab_size:
            raise ValueError('BERT input_ids outside vocabulary')
        if not torch.isin(attention, torch.tensor([0, 1], device=attention.device)).all():
            raise ValueError('BERT attention_mask must be binary')
        if not torch.isin(token_type, torch.tensor([0, 1], device=token_type.device)).all():
            raise ValueError('BERT token_type_ids must be binary')
        return ids, attention, token_type

    def forward(self, text_bert, **kwargs):
        ids, attention, token_type = self.checked_tokens(text_bert, self.config.vocab_size)
        return super().forward(input_ids=ids, attention_mask=attention,
                               token_type_ids=token_type, **kwargs).last_hidden_state
