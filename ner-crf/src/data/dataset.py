from typing import Dict, List, Any, Optional
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .label_map import LabelMap  # 使用相对导入


class CluenerBertDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict],
        pretrained_name: str,
        label_map: LabelMap,
        max_length: int = 256
    ):
        self.examples = examples
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
        self.max_length = max_length
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        tokens: List[str] = ex["tokens"]
        labels: List[str] = ex["labels"]

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors=None
        )

        word_ids: List[Optional[int]] = enc.word_ids()

        # ====== 关键：对齐 label_ids + valid_mask ======
        aligned_label_ids: List[int] = []
        valid_mask: List[int] = []

        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                # [CLS], [SEP] 等
                aligned_label_ids.append(self.label_map.PAD_ID)
                valid_mask.append(0)
            else:
                # 一个原始 token 的第一个 subword 才参与 CRF
                if word_id != prev_word_id:
                    # word_id 对应 tokens/labels 的下标
                    aligned_label_ids.append(self.label_map.label2id[labels[word_id]])
                    valid_mask.append(1)
                else:
                    # 同一个 word 的后续 subword：不参与 CRF
                    aligned_label_ids.append(self.label_map.PAD_ID)
                    valid_mask.append(0)

            prev_word_id = word_id

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label_ids": aligned_label_ids,
            "valid_mask": valid_mask,
            "tokens": tokens,
            "labels": labels,
            "id": ex.get("id", str(idx)),
        }