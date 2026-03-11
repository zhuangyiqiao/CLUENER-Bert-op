from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import os
import json
import torch
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainState:
    global_step: int = 0
    best_f1: float = 0.0
    best_epoch: int = 0


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        label_map,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        device: torch.device,
        output_dir: str,
        log_every_steps: int = 50,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.label_map = label_map
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.device = device

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.log_every_steps = log_every_steps
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)

        self.state = TrainState()
        self.epoch_metrics: List[Dict[str, Any]] = []
        self.best_report: Optional[str] = None

        self.model.to(self.device)

    def train(self, num_epochs: int) -> Dict[str, Any]:
        last_train_loss = None
        last_dev_f1 = None
        last_dev_report = None

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            dev_f1, dev_report = self.evaluate()

            last_train_loss = train_loss
            last_dev_f1 = dev_f1
            last_dev_report = dev_report

            epoch_record = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "dev_f1": float(dev_f1),
                "global_step": int(self.state.global_step),
            }
            self.epoch_metrics.append(epoch_record)

            print(f"[Epoch {epoch}] train_loss={train_loss:.4f} dev_f1={dev_f1:.4f}")
            logger.info(f"[Epoch {epoch}] train_loss={train_loss:.4f} dev_f1={dev_f1:.4f}")

            # 每轮结束后判断 best
            if dev_f1 > self.state.best_f1:
                self.state.best_f1 = dev_f1
                self.state.best_epoch = epoch
                self.best_report = dev_report
                self.save_checkpoint("best.pt")
                print(f"[BEST] new best_f1={dev_f1:.4f} saved to best.pt")
                logger.info(f"[BEST] new best_f1={dev_f1:.4f} saved to best.pt")

            # 始终保存最后一轮状态，覆盖写，不再存 epoch_1/2/3...
            self.save_checkpoint("last.pt")

        # 训练结束后保存每轮指标记录
        self.save_epoch_metrics()

        return {
            "best_f1": self.state.best_f1,
            "best_epoch": self.state.best_epoch,
            "last_train_loss": last_train_loss,
            "last_dev_f1": last_dev_f1,
            "last_dev_report": last_dev_report,
            "epoch_metrics": self.epoch_metrics,
        }

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_steps = 0

        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader, start=1):
            batch = self._move_batch_to_device(batch)

            loss = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                valid_mask=batch["valid_mask"],
                label_ids=batch["label_ids"],
            )

            # 梯度累积
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if step % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.state.global_step += 1

                if self.log_every_steps > 0 and self.state.global_step % self.log_every_steps == 0:
                    print(f"[Train] epoch={epoch} global_step={self.state.global_step} loss={loss.item():.4f}")
                    logger.info(f"[Train] epoch={epoch} global_step={self.state.global_step} loss={loss.item():.4f}")

            total_loss += loss.item()
            num_steps += 1

        return total_loss / max(1, num_steps)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, str]:
        self.model.eval()

        all_true: List[List[str]] = []
        all_pred: List[List[str]] = []

        for batch in self.dev_loader:
            batch = self._move_batch_to_device(batch)

            pred_paths = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                valid_mask=batch["valid_mask"],
                label_ids=None,
            )

            true_paths = self._extract_gold_sequences(batch["label_ids"], batch["valid_mask"])

            # 转成字符串标签序列
            for pred_ids, true_ids in zip(pred_paths, true_paths):
                pred_labels = self.label_map.decode(pred_ids)
                true_labels = self.label_map.decode(true_ids)

                # 注意：seqeval 通常希望没有 START/STOP（我们这里也不会有）
                all_pred.append(pred_labels)
                all_true.append(true_labels)

        dev_f1 = f1_score(all_true, all_pred)
        report = classification_report(all_true, all_pred, digits=4)
        return dev_f1, report

    def _extract_gold_sequences(self, label_ids: torch.Tensor, valid_mask: torch.Tensor) -> List[List[int]]:
        """
        从 (B,T) 的 label_ids 里抽取 valid_mask==1 的位置，得到每条样本的 gold 序列（list of ids）。
        """
        label_ids = label_ids.detach().cpu()
        valid_mask = valid_mask.detach().cpu()

        B, T = label_ids.shape
        sequences: List[List[int]] = []
        for b in range(B):
            ids = []
            for t in range(T):
                if int(valid_mask[b, t].item()) == 1:
                    ids.append(int(label_ids[b, t].item()))
            sequences.append(ids)
        return sequences

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # ids/tokens/labels 这些 debug 字段不 move
        moved = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    def save_epoch_metrics(self) -> None:
        metrics_path = os.path.join(self.output_dir, "epoch_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(self.epoch_metrics, f, ensure_ascii=False, indent=2)

        md_path = os.path.join(self.output_dir, "epoch_metrics.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Epoch Metrics\n\n")
            for item in self.epoch_metrics:
                f.write(
                    f"- epoch {item['epoch']}: "
                    f"train_loss={item['train_loss']:.4f}, "
                    f"dev_f1={item['dev_f1']:.4f}, "
                    f"global_step={item['global_step']}\n"
                )

        if self.best_report is not None:
            report_path = os.path.join(self.output_dir, "best_dev_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(self.best_report)

    def save_checkpoint(self, filename: str) -> None:
        path = os.path.join(self.output_dir, filename)
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": None if self.scheduler is None else self.scheduler.state_dict(),
            "global_step": self.state.global_step,
            "best_f1": self.state.best_f1,
        }
        torch.save(payload, path)