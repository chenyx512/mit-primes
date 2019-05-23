from abc import abstractmethod
from time import time

from numpy import inf
import torch


class BaseTrainer:
    """ This class is in charge of logging, saving the best performing model,
    saving periodically, early stopping, loading and resuming training
    # TODO: tensorboard

    Args:
        model: the model, which should be on device already
        loss: the loss function
        metrics: list of metric functions
        optimizer:
        scheduler: this can be None
        config: a ConfigParser object

    Arguments in config['trainer"]:
        epochs: the total number of epoch to train

        resume (optional): The path to checkpoint file.

        save_period (optional): the period to save the model. (default: inf)

        monitor (optional): it should be of the form "{mode} {metric}",
        the mode should be in {'min', 'max'}, and metric should be in
        config['metric']. If this is specified, the model that
        maximize/minimize the metric will be saved.

        early_stop (optional): if monitor is specified, the trainer will
        early stop if the monitoring metric is not improved for this number
        of epochs. (default: inf)

    All trainer classes that extend this class should implement the
    _train_epoch method. It may return a dictionary result that will be
    logged. Specifically, metrics need to be a list of floats in result with
    key 'metrics', in the order specified in config['metric'], and if there
    are separate validation metrics, they need to be a list of floats in
    result with key 'val_metrics', with the same order.
    """

    def __init__(self, model, loss, metrics, optimizer, scheduler, config):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.config = config
        trainer_cfg = config['trainer']
        self.logger = config.get_logger('trainer', trainer_cfg['verbosity'])
        self.epochs = trainer_cfg['epochs']
        self.start_epoch = 1

        self.save_period = trainer_cfg.get('save_period', inf)
        self.monitor = trainer_cfg.get('monitor', 'off')
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']
            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = trainer_cfg.get('early_stop', inf)

        self.device = torch.device('cuda')

        if config.resume is not None:
            self._load_checkpoint()

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            start_time = time()
            result = self._train_epoch(epoch)

            log = {'epoch': epoch, 'time': time() - start_time}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i]
                                for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            self.logger.info('---------------------------------')
            for key, value in log.items():
                self.logger.info(f'{key:15s}: {value}')

            # monitoring the best model and early stopping
            is_best = False
            not_improved_cnt = 0
            if self.mnt_mode != 'off':
                is_improved = (self.mnt_mode == 'min' and
                               log[self.mnt_metric] <= self.mnt_best) or \
                              (self.mnt_mode == 'max' and
                               log[self.mnt_metric] >= self.mnt_best)
                if is_improved:
                    not_improved_cnt = 0
                    self.mnt_best = log[self.mnt_metric]
                    self._save_checkpoint(epoch, 'best.pth')
                else:
                    not_improved_cnt += 1

                if not_improved_cnt > self.early_stop:
                    self.logger.info(f'monitoring metric not improving for \
                    {self.early_stop} epochs. EARLY STOP')
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, f'epoch{epoch}.pth')

    def _save_checkpoint(self, epoch, filename):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config
        }
        if self.scheduler:
            state['scheduler_state'] = self.scheduler.state_dict()

        path = str(self.config.save_dir / filename)
        torch.save(state, path)
        self.logger.info(f'model saved as {filename} ')

    def _load_checkpoint(self):
        self.logger.info(f'loading checkpoint {self.config.resume}')

        state = torch.load(self.config.resume)
        self.model.load_state_dict(state['model_state'])
        self.start_epoch = state['epoch'] + 1

        # load optimizer iff its type isn't changed
        if state['config']['optimizer']['type'] != \
                self.config['optimizer']['type']:
            self.logger.warning('optimizer not loaded, type not match')
        else:
            self.optimizer.load_state_dict(state['optimizer_state'])

        # load scheduler iff same scheduler exists in both config
        try:
            assert state['config']['scheduler']['type'] == \
                self.config['scheduler']['type']
            self.scheduler.load_state_dict(state['scheduler_state'])
        except KeyError or AssertionError:
            self.logger.warning('scheduler not loaded')

        self.logger.info(f'state loaded, start at epoch-{self.start_epoch}')
