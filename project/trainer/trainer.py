import torch
import numpy as np

from base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, loss, metrics, config, optimizer, train_loader,
                 test_loader=None, lr_scheduler=None):
        super().__init__(model, loss, metrics, config)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler

    def _eval_metrics(self, output, target):
        metric_values = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            metric_values[i] += metric(output, target)
        return metric_values

    def _train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_index, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            # question: where do we denormalize the steering angles?
            total_metrics += self._eval_metrics(output, target)

        if self.lr_scheduler:
            self.lr_scheduler.step()

        log = {
            'loss': total_loss / len(self.train_loader),
            'metrics': (total_metrics / len(self.train_loader)).tolist()
        }

        if self.test_loader:
            val_log = self._test_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _test_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_metrics = np.zeros(len(self.metrics))
            for batch_index, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                total_loss += loss.item()
                total_metrics += self._eval_metrics(output, target)
        return {
            'val_loss': total_loss / len(self.test_loader),
            'val_metrics': (total_metrics / len(self.test_loader)).tolist()
        }
