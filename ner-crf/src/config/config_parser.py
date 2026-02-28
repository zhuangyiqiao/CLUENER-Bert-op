import json
import torch
from pathlib import Path


def load_config(config_path: str):
    """
    读取 JSON 配置文件
    返回处理后的 config dict
    """

    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # ==========================
    # 1️⃣ 处理路径
    # ==========================

    config["data_dir"] = Path(config["data_dir"])
    config["train_path"] = Path(config["train_path"])
    config["dev_path"] = Path(config["dev_path"])
    config["test_path"] = Path(config["test_path"])
    config["output_dir"] = Path(config["output_dir"])

    # 创建输出目录（如果不存在）
    config["output_dir"].mkdir(parents=True, exist_ok=True)

    # ==========================
    # 2️⃣ 自动生成 id2label
    # ==========================

    label2id = config["labels"]["label2id"]
    id2label = {v: k for k, v in label2id.items()}

    config["labels"]["id2label"] = id2label
    config["num_labels"] = len(label2id)

    # ==========================
    # 3️⃣ 自动选择 device
    # ==========================

    device_setting = config["train"]["device"]

    if device_setting == "auto":
        config["device"] = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        config["device"] = torch.device(device_setting)

    return config