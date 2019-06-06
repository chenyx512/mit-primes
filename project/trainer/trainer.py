import torch
import numpy as np

from base.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Args:
        model: the model, which should be on device already
        loss: the loss function
        metrics: list of metric functions
        config: ConfigParser object, see BaseTrainer for needed arguments
    """

    def __init__(self, model, loss, metrics, config, optimizer, train_loader,
                 test_loader=None, scheduler=None):
        super().__init__(model, loss, metrics, optimizer, scheduler, config)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def _eval_metrics(self, output, target):
        metric_values = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            metric_values[i] += metric(output, target)
        return metric_values

    def _train_epoch(self, epoch):
        self.model.train()

        total_output = torch.zeros(0).to(self.device)
        total_target = torch.zeros(0).to(self.device)
        for batch_index, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            total_output = torch.cat([total_output, output])
            total_target = torch.cat([total_target, target])
            # question: potential speed problem?

        # denormalize
        total_loss = self.loss(total_output, total_target).item()
        total_output *= self.train_loader.dataset.steering_angle_factor
        total_target *= self.train_loader.dataset.steering_angle_factor
        total_metrics = self._eval_metrics(total_output, total_target)

        if self.scheduler:
            self.scheduler.step()

        log = {
            'loss': total_loss,
            'metrics': total_metrics.tolist()
        }

        if self.test_loader:
            val_log = self._test_epoch(epoch)
            log = {**log, **val_log}

        return log

    def _test_epoch(self, epoch):
        self.model.eval()

        total_output = torch.zeros(0).to(self.device)
        total_target = torch.zeros(0).to(self.device)
        with torch.no_grad():
            for batch_index, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                total_output = torch.cat([total_output, output])
                total_target = torch.cat([total_target, target])

        total_loss = self.loss(total_output, total_target).item()
        total_output *= self.train_loader.dataset.steering_angle_factor
        total_target *= self.train_loader.dataset.steering_angle_factor
        total_metrics = self._eval_metrics(total_output, total_target)

        return {
            'val_loss': total_loss,
            'val_metrics': total_metrics,
        }
