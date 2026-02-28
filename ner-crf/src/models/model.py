from __future__ import annotations

from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

from src.models.crf import CRF


class BertCrfForNer(nn.Module):
    """
    BERT + Linear + CRF for NER.

    Inputs:
      - input_ids: (B, T)
      - attention_mask: (B, T)
      - valid_mask: (B, T)  (1 for valid positions to tag; 0 otherwise)
      - label_ids: (B, T)   (only required for training)

    Outputs:
      - if label_ids is provided: loss (scalar)
      - else: pred_paths: List[List[int]] (decoded tag ids for valid positions)
    """

    def __init__(
        self,
        pretrained_name: str,
        num_tags: int,
        start_tag_id: int,
        stop_tag_id: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.pretrained_name = pretrained_name

        # Use HF config to get hidden_size robustly
        hf_config = AutoConfig.from_pretrained(pretrained_name)
        self.bert = AutoModel.from_pretrained(pretrained_name, config=hf_config)

        hidden_size = hf_config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_tags)

        self.crf = CRF(num_tags=num_tags, start_tag_id=start_tag_id, stop_tag_id=stop_tag_id)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        label_ids: Optional[torch.Tensor] = None,
    ):
        """
        Train:
          loss = model(input_ids, attention_mask, valid_mask, label_ids)

        Inference:
          pred_paths = model(input_ids, attention_mask, valid_mask)
        """
        # BERT forward
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, T, H)

        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)  # (B, T, num_tags)

        if label_ids is not None:
            # CRF NLL
            loss = self.crf(emissions=emissions, tags=label_ids, mask=valid_mask)
            return loss
        else:
            # Viterbi decode
            pred_paths = self.crf.decode(emissions=emissions, mask=valid_mask)
            return pred_paths

    @torch.no_grad()
    def predict_batch(
        self,
        batch: Dict[str, Any],
        device: Optional[torch.device] = None
    ) -> List[List[int]]:
        """
        Convenience helper: move batch tensors to device and decode.
        """
        if device is None:
            device = next(self.parameters()).device

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        valid_mask = batch["valid_mask"].to(device)

        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            valid_mask=valid_mask,
            label_ids=None
        )