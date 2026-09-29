from __future__ import annotations

import torch
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, scheduler=None, device="cpu", amp=True, grad_accum=1, clip_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.amp = amp
        self.grad_accum = grad_accum
        self.clip_grad_norm = clip_grad_norm
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    def train_epoch(self, loader, criterion):
        self.model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(loader, desc="Training")):
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)
            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type, enabled=self.amp and self.device.type == "cuda"):
                logits = self.model(images)
                loss = criterion(logits, targets)

            loss = loss / self.grad_accum
            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0 or (step + 1) == len(loader):
                if self.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum * images.size(0)

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / len(loader.dataset)

    def evaluate(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        preds = []
        targets = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                target = batch["target"].to(self.device)
                logits = self.model(images)
                loss = criterion(logits, target)
                total_loss += loss.item() * images.size(0)
                preds.append(logits.cpu())
                targets.append(target.cpu())

        preds = torch.cat(preds) if preds else torch.tensor([])
        targets = torch.cat(targets) if targets else torch.tensor([])
        return total_loss / max(len(loader.dataset), 1), preds, targets
