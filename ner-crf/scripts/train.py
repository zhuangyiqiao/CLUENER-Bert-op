import os
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AdamW

from src.config.config_parser import load_config
from src.data.processor import CluenerProcessor
from src.data.label_map import build_label_map_from_config
from src.data.dataset import CluenerBertDataset
from src.data.collate import DataCollatorForBertCrf


from src.models.model import BertCrfForNer
from src.training.lr_scheduler import build_scheduler
from src.training.trainer import Trainer

from src.utils.logger import init_logger
from src.utils.seed import set_seed

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
def resolve_path(root: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp).resolve()



def main():
    # 1) load config
    cfg = load_config("configs/bert_crf.json")

    # 2) set seed (minimal)
    seed = cfg["model"].get("seed", 42)
    set_seed(seed,deterministic=True)

    # 3) paths
    data_dir = Path(cfg["data_dir"])
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)


    log_path = Path(output_dir) / "train.log"
    logger = init_logger(log_file=log_path, log_file_level=logging.INFO)
    logger.info(f"Using device: {cfg['train'].get('device', 'auto')}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Pretrained: {cfg['model']['pretrained_name']}")
    logger.info(f"Seed: {seed}")
    

    # 4) build label map
    label_map = build_label_map_from_config(cfg)
    num_tags = label_map.num_labels
    start_id = label_map.START_ID
    stop_id = label_map.STOP_ID

    # 5) load data (JSONL)
    processor = CluenerProcessor(data_dir)
    train_examples = processor.get_train_examples()
    dev_examples = processor.get_dev_examples()

    # 6) datasets
    pretrained_name = cfg["model"]["pretrained_name"]
    max_length = cfg["data"]["max_length"]

    train_ds = CluenerBertDataset(
        examples=train_examples,
        pretrained_name=pretrained_name,
        label_map=label_map,
        max_length=max_length
    )
    dev_ds = CluenerBertDataset(
        examples=dev_examples,
        pretrained_name=pretrained_name,
        label_map=label_map,
        max_length=max_length
    )

    # 7) dataloaders
    train_bs = cfg["train"]["train_batch_size"]
    eval_bs = cfg["train"]["eval_batch_size"]

    collator = DataCollatorForBertCrf(pretrained_name=pretrained_name, label_map=label_map)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_bs,
        shuffle=True,
        collate_fn=collator,
        num_workers=0
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=eval_bs,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )

    # 8) model
    dropout = cfg["model"].get("dropout", 0.1)
    model = BertCrfForNer(
        pretrained_name=pretrained_name,
        num_tags=num_tags,
        start_tag_id=start_id,
        stop_tag_id=stop_id,
        dropout=dropout,
    )

    # 9) optimizer
    lr = float(cfg["train"]["learning_rate"])
    wd = float(cfg["train"]["weight_decay"])

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 10) scheduler (needs total training steps)
    num_epochs = int(cfg["train"]["num_epochs"])
    grad_acc_steps = int(cfg["train"].get("gradient_accumulation_steps", 1))

    total_update_steps = (len(train_loader) * num_epochs) // max(1, grad_acc_steps)
    warmup_ratio = float(cfg["train"].get("warmup_ratio", 0.1))
    scheduler = build_scheduler(
        optimizer=optimizer,
        num_training_steps=total_update_steps,
        warmup_ratio=warmup_ratio
    )

    # 11) device
    device = resolve_device(cfg["train"].get("device", "auto"))

    # 12) trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_map=label_map,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        output_dir=output_dir,
        log_every_steps=int(cfg["train"].get("log_every_steps", 50)),
        max_grad_norm=float(cfg["train"].get("max_grad_norm", 1.0)),
        gradient_accumulation_steps=grad_acc_steps,
    )

    # 13) train
    trainer.train(num_epochs=num_epochs)

    # 14) (optional) save config for reproducibility
    with open(Path(output_dir) / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("[DONE] training finished.")
    logger.info("[DONE]Training finished.")

if __name__ == "__main__":
    main()

