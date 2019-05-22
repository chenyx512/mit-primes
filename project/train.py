import argparse
import logging
import collections

import data_loader.data_loaders as data_loader_module
import data_loader.event_fram_dataset as dataset_module
from parse_config import ConfigParser


def main(config):
    dataset = config.initialize('dataset', dataset_module)
    train_loader = config.initialize('data_loader', data_loader_module, dataset)
    test_loader = train_loader.get_test_loader()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-c', '--config', default='config.yaml', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['-lr', '--learning_rate'], type=float,
                   target=('optimizer', 'args', 'lr')),
        CustomArgs(['-bs', '--batch_size'], type=int,
                   target=('data_loader', 'args', 'batch_size'))
    ]
    config = ConfigParser(args, options)
    main(config)
