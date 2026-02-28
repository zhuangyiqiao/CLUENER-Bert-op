import json  # JSON 解析模块
from pathlib import Path  # 文件路径处理
from typing import Dict, List, Optional  # 类型注解


class CluenerProcessor:
    """
    CLUENER Processor (JSONL):
    每行一个json:
      - text: str
      - label: dict(可选)  {entity_type: {entity_text: [[start,end], ...], ...}, ...}
    输出统一格式:
      {"id": "...", "tokens": [...], "labels": [...], "raw_text": "..."}
    """  # 处理 CLUENER 数据集的类，输入 JSONL 格式

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)  # 将传入路径转换为 Path 对象

    def get_train_examples(self) -> List[Dict]:
        return self._create_examples(self.data_dir / "train.json", mode="train")  # 读取训练集

    def get_dev_examples(self) -> List[Dict]:
        return self._create_examples(self.data_dir / "dev.json", mode="dev")  # 读取开发集

    def get_test_examples(self) -> List[Dict]:
        return self._create_examples(self.data_dir / "test.json", mode="test")  # 读取测试集

    def _create_examples(self, input_path: Path, mode: str) -> List[Dict]:
        examples: List[Dict] = []  # 存放所有示例的列表
        input_path = Path(input_path)  # 确保输入路径是 Path 对象

        with open(input_path, "r", encoding="utf-8") as f:  # 逐行读取文件
            for idx, line in enumerate(f):
                line = line.strip()  # 去除行尾空白
                if not line:
                    continue  # 空行跳过

                obj = json.loads(line)  # 解析 JSON
                text: str = obj["text"]  # 获取文本字段
                label_entities: Optional[Dict] = obj.get("label", None)  # 获取标签字典，如果有

                tokens = list(text)  # 中文按字切分（与官方一致）
                labels = ["O"] * len(tokens)  # 初始化所有标签为 'O'

                # 如果有标注，做 span -> BIOES
                if label_entities is not None:
                    for ent_type, ent_dict in label_entities.items():
                        # ent_dict: {entity_text: [[start,end], ...], ...}
                        for ent_text, spans in ent_dict.items():
                            for start, end in spans:
                                # 校验span是否和原文一致（官方也做了这个assert）
                                assert "".join(tokens[start:end + 1]) == ent_text

                                if start == end:
                                    labels[start] = f"S-{ent_type}"  # 单字实体标记为 S-
                                else:
                                    labels[start] = f"B-{ent_type}"  # 开始标记 B-
                                    for i in range(start + 1, end + 1):
                                        labels[i] = f"I-{ent_type}"  # 中间标记 I-

                examples.append({
                    "id": f"{mode}_{idx}",  # 示例 ID
                    "tokens": tokens,  # 分词结果
                    "labels": labels if mode != "test" else labels,  # test 没有标签时仍保留全 O
                    "raw_text": text  # 原始文本
                })

        return examples  # 返回构建好的示例列表