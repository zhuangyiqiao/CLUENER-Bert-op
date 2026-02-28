from dataclasses import dataclass
from typing import Dict, List


@dataclass
class LabelMap:
    label2id: Dict[str, int]
    id2label: Dict[int, str]

    pad_label: str = "O"
    start_tag: str = "<START>"
    stop_tag: str = "<STOP>"

    @property
    def num_labels(self) -> int:
        return len(self.label2id)

    @property
    def PAD_ID(self) -> int:
        # 你现在 pad_label 设为 O，所以 PAD_ID 就是 O 的 id
        return self.label2id[self.pad_label]

    @property
    def O_ID(self) -> int:
        return self.label2id["O"]

    @property
    def START_ID(self) -> int:
        return self.label2id[self.start_tag]

    @property
    def STOP_ID(self) -> int:
        return self.label2id[self.stop_tag]

    def encode(self, labels: List[str]) -> List[int]:
        """['O','B-address',...] -> [0,1,...]"""
        return [self.label2id[x] for x in labels]

    def decode(self, ids: List[int]) -> List[str]:
        """[0,1,...] -> ['O','B-address',...]"""
        return [self.id2label[i] for i in ids]


def build_label_map_from_config(config: dict) -> LabelMap:
    """
    config 里要求：
      - config['labels']['label2id']
      - config['data']['pad_label'] (可选，默认 'O')
    """
    label2id = config["labels"]["label2id"]
    id2label = {v: k for k, v in label2id.items()}
    pad_label = config.get("data", {}).get("pad_label", "O")

    return LabelMap(
        label2id=label2id,
        id2label=id2label,
        pad_label=pad_label
    )