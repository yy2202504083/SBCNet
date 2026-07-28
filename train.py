# train.py
# Concise ESCNet training script. Evaluation is implemented in evaluate.py.

import argparse
import inspect
import math
import os
import re

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import Config, load_config
from dataset import MyData
from evaluate import ModelEvaluator, merge_yaml_into_config, split_model_outputs
from loss import EdgeDiceLoss, StructureLoss
from models.ronghe4_Copy10 import ESCNet
from utils import AverageMeter, Logger, set_seed


def set_config(config, key, value):
    try:
        setattr(config, key, value)
    except Exception:
        object.__setattr__(config, key, value)


class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if torch.is_floating_point(value)
        }

    @torch.no_grad()
    def update(self, model):
        for key, value in model.state_dict().items():
            if not torch.is_floating_point(value):
                continue
            if key not in self.shadow:
                self.shadow[key] = value.detach().clone()
            else:
                self.shadow[key].mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )

    def state_dict(self, model):
        return {
            key: self.shadow.get(key, value).detach().clone()
            for key, value in model.state_dict().items()
        }


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(f"cuda:{int(config.device_ids[0])}")
        torch.cuda.set_device(self.device)

        self.save_dir = os.path.join(config.save_model_dir, config.name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = Logger(config, os.path.join(self.save_dir, "log.txt"))

        self.accumulation_steps = max(
            1, int(getattr(config, "accumulation_steps", 1))
        )
        self.deep_weights = list(
            getattr(config, "deep_supervision_weights", [0.04, 0.12, 0.40, 1.45])
        )
        self.edge_weight = float(getattr(config, "edge_loss_weight", 0.012))
        self.max_grad_norm = float(getattr(config, "max_grad_norm", 1.0))
        self.logit_clip = float(getattr(config, "loss_logit_clip", 20.0))

        self.use_amp, self.amp_dtype, use_scaler = self._amp_policy()
        self.scaler = GradScaler(enabled=use_scaler)

        self.model = ESCNet(config, pretrained=True).to(self.device)
        self.start_epoch = self._load_resume()
        self.forward_accepts_epoch = self._accepts_epoch()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )
        self.structure_loss = StructureLoss().to(self.device)
        self.edge_loss = EdgeDiceLoss().to(self.device)
        self.loss_meter = AverageMeter()
        self.train_loader = self._build_loader()

        self.use_ema = bool(getattr(config, "use_ema", False))
        self.ema_start = int(getattr(config, "ema_start_epoch", 0))
        self.ema = None
        self.ema_decay = float(getattr(config, "ema_decay", 0.9995))

        self.evaluator = ModelEvaluator(
            config=config,
            model=self.model,
            device=self.device,
            rank=0,
            logger=self.logger,
            amp_enabled=self.use_amp,
            amp_dtype=self.amp_dtype,
            logit_clip_value=self.logit_clip,
        )

    def log(self, message):
        self.logger.info(message)

    def _amp_policy(self):
        mode = str(getattr(self.config, "amp_dtype", "bf16")).lower()
        if mode in {"fp16", "float16"}:
            return True, torch.float16, True
        if mode in {"bf16", "bfloat16"} and torch.cuda.is_bf16_supported():
            return True, torch.bfloat16, False
        return False, torch.float32, False

    def _build_loader(self):
        dataset = MyData(
            self.config,
            dataset_dir=self.config.train_dir,
            image_size=self.config.img_size,
            is_train=True,
        )
        workers = min(
            int(getattr(self.config, "num_workers", 4)),
            int(self.config.batch_size),
        )
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=workers > 0,
        )

    def _accepts_epoch(self):
        try:
            params = inspect.signature(self.model.forward).parameters
            return "epoch" in params or any(
                item.kind == inspect.Parameter.VAR_KEYWORD
                for item in params.values()
            )
        except (TypeError, ValueError):
            return False

    def _forward(self, images, epoch):
        if self.forward_accepts_epoch:
            return self.model(images, epoch=epoch)
        return self.model(images)

    @staticmethod
    def _checkpoint_state(checkpoint):
        if isinstance(checkpoint, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if isinstance(checkpoint.get(key), dict):
                    return checkpoint[key]
        return checkpoint

    def _load_resume(self):
        path = str(getattr(self.config, "resume", "")).strip()
        if not path:
            return 1
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(self._checkpoint_state(checkpoint), strict=False)

        if isinstance(checkpoint, dict) and "epoch" in checkpoint:
            epoch = int(checkpoint["epoch"])
        else:
            match = re.search(r"epoch[_-]?(\d+)", os.path.basename(path), re.I)
            epoch = int(match.group(1)) if match else 0
        return epoch + 1

    def _adjust_lr(self, epoch):
        warmup = int(getattr(self.config, "warmup_epochs", 5))
        total_epochs = int(self.config.epochs)
        min_lr = float(getattr(self.config, "min_lr", self.config.lr / 10))

        if warmup > 0 and epoch <= warmup:
            lr = float(self.config.lr) * epoch / warmup
        else:
            progress = (epoch - warmup) / max(1, total_epochs - warmup)
            progress = min(max(progress, 0.0), 1.0)
            lr = min_lr + 0.5 * (float(self.config.lr) - min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )

        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _resize_logits(self, prediction, target):
        prediction = F.interpolate(
            prediction.float(),
            size=target.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        if self.logit_clip > 0:
            prediction = prediction.clamp(-self.logit_clip, self.logit_clip)
        return prediction

    def _compute_loss(self, outputs, masks, edges):
        edge_pred, mask_preds = split_model_outputs(outputs)
        if len(mask_preds) != len(self.deep_weights):
            raise ValueError(
                "deep_supervision_weights must match the mask output count."
            )

        mask_loss = sum(
            float(weight)
            * self.structure_loss(self._resize_logits(pred, masks), masks)
            for pred, weight in zip(mask_preds, self.deep_weights)
        )

        edge_loss = torch.zeros((), device=self.device)
        if edge_pred is not None and self.edge_weight > 0:
            edge_loss = self.edge_loss(
                self._resize_logits(edge_pred, edges), edges
            )

        total_loss = mask_loss + self.edge_weight * edge_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite loss detected.")
        return total_loss, mask_loss, edge_loss

    def _update_ema(self, epoch):
        if not self.use_ema or epoch < self.ema_start:
            return
        if self.ema is None:
            self.ema = ModelEMA(self.model, self.ema_decay)
        else:
            self.ema.update(self.model)

    def _model_state(self):
        if self.ema is not None:
            return self.ema.state_dict(self.model)
        return self.model.state_dict()

    def _save(self, epoch):
        if epoch % int(self.config.save_step) != 0:
            return
        path = os.path.join(self.save_dir, f"epoch_{epoch}.pth")
        torch.save(self._model_state(), path)
        self.log(f"Saved checkpoint: {path}")

    def train_epoch(self, epoch):
        self.model.train()
        self.loss_meter.reset()
        self._adjust_lr(epoch)
        self.optimizer.zero_grad(set_to_none=True)

        total_steps = len(self.train_loader)
        for step, (images, masks, edges) in enumerate(self.train_loader, 1):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).float().clamp_(0, 1)
            edges = edges.to(self.device, non_blocking=True).float().clamp_(0, 1)

            with autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                outputs = self._forward(images, epoch)

            with autocast(device_type="cuda", enabled=False):
                total_loss, mask_loss, edge_loss = self._compute_loss(
                    outputs, masks, edges
                )
                backward_loss = total_loss / self.accumulation_steps

            if self.scaler.is_enabled():
                self.scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            should_step = (
                step % self.accumulation_steps == 0 or step == total_steps
            )
            if should_step:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._update_ema(epoch)

            self.loss_meter.update(total_loss.item(), images.size(0))
            if step % 20 == 0 or step == total_steps:
                self.log(
                    f"Epoch[{epoch}/{self.config.epochs}] "
                    f"Step[{step}/{total_steps}] "
                    f"Mask:{mask_loss.item():.4f} "
                    f"Edge:{edge_loss.item():.4f} "
                    f"Total:{total_loss.item():.4f}"
                )

        self.log(
            f"Epoch[{epoch}/{self.config.epochs}] "
            f"Average Loss:{self.loss_meter.avg:.4f}"
        )

    def train(self):
        for epoch in range(self.start_epoch, int(self.config.epochs) + 1):
            self.train_epoch(epoch)
            self._save(epoch)

            if self.evaluator.is_scheduled(epoch):
                ema_state = self.ema.state_dict(self.model) if self.ema else None
                self.evaluator.evaluate_and_save_best(
                    epoch, state_dict_override=ema_state
                )


def parse_args():
    parser = argparse.ArgumentParser(description="ESCNet training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default="")
    parser.add_argument("--gpu", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = merge_yaml_into_config(load_config(args.config), args.config)
    if args.resume:
        set_config(config, "resume", args.resume)
    if args.gpu is not None:
        set_config(config, "device_ids", [args.gpu])

    set_seed(config.rand_seed)
    Trainer(config).train()


if __name__ == "__main__":
    main()
