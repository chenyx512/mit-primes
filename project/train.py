import argparse
import logging
import collections

import torch.nn.functional as F
import torch.optim

import data_loader.data_loaders as data_loader_module
import data_loader.dataset as dataset_module
import model.metric as metric_module
from parse_config import ConfigParser
from trainer.trainer import Trainer
from model.model import Model



def main(config):
    logger = config.get_logger('trainer')

    dataset = config.initialize('dataset', dataset_module)
    train_loader = config.initialize('data_loader', data_loader_module, dataset)
    test_loader = train_loader.get_test_loader()

    model = Model()
    logger.info(model)

    metrics = [getattr(metric_module, mtr) for mtr in config['metrics']]
    loss = getattr(F, config['loss'])

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    trainer = Trainer(model, loss, metrics, config, optimizer, train_loader,
                      test_loader=test_loader)
    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default='config.yaml', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-lr', '--learning_rate'], type=float, target=('optimizer', 'args', 'lr')),
        CustomArgs(['-bs', '--batch_size'], type=int, target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)
    main(config)
