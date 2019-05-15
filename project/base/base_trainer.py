from abc import abstractmethod

import torch


class BaseTrainer:
    def __init__(self, model, loss, metrics, config):
        self.model = model
        self.loss = loss
        self.metrics = metrics

        self.config = config
        trainer_cfg = config['trainer']
        self.logger = config.get_logger('trainer', trainer_cfg['verbosity'])
        self.epochs = trainer_cfg['epochs']

        self.device = torch.device('cuda')

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(1, self.epochs):
            result = self._train_epoch(epoch)

            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            for key, value in log.items():
                self.logger.info(f'{key:15s}: {value}')
