from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer

from .label_map import LabelMap  # 使用相对导入


class DataCollatorForBertCrf:
    """
    Collator for BERT+CRF NER.
    Pads:
      - input_ids with tokenizer.pad_token_id
      - attention_mask with 0
      - label_ids with label_map.PAD_ID (usually O=0)
      - valid_mask with 0
    """

    def __init__(self, pretrained_name: str, label_map: LabelMap):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            # BERT usually has [PAD], but just in case
            raise ValueError("Tokenizer has no pad_token_id. Check pretrained model/tokenizer.")
        self.label_map = label_map

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # features: list of dataset items (dict)
        batch_size = len(features)
        max_len = max(len(f["input_ids"]) for f in features)

        input_ids = torch.full((batch_size, max_len), fill_value=self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        label_ids = torch.full((batch_size, max_len), fill_value=self.label_map.PAD_ID, dtype=torch.long)
        valid_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        ids = []
        tokens = []
        labels = []

        for i, f in enumerate(features):
            seq_len = len(f["input_ids"])

            input_ids[i, :seq_len] = torch.tensor(f["input_ids"], dtype=torch.long)
            attention_mask[i, :seq_len] = torch.tensor(f["attention_mask"], dtype=torch.long)
            label_ids[i, :seq_len] = torch.tensor(f["label_ids"], dtype=torch.long)
            valid_mask[i, :seq_len] = torch.tensor(f["valid_mask"], dtype=torch.long)

            ids.append(f.get("id", str(i)))
            tokens.append(f.get("tokens"))
            labels.append(f.get("labels"))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids,
            "valid_mask": valid_mask,
            "ids": ids,         # debug用
            "tokens": tokens,   # debug用（可后续删）
            "labels": labels    # debug用（可后续删）
        }