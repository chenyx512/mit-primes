from datetime import datetime
from functools import reduce
from operator import getitem
from pathlib import Path
import logging
import logging.config

from utils.utils import read_yaml, write_yaml


class ConfigParser:
    """ This class parse the yaml config file and command line flags

    Args:
        args: the ArgumentParser object, arguments should include either
            '--config' or '--resume', and the value should be the path to the
            config or checkpoint file
        options: This list of namedtuples with attributes 'flags' 'type'
            'target' that overrides the config file. 'flags' is a list of flags
            that will be added to argument parser. 'type' specifies the type of
            the value. 'target' is a tuple of keys that specifies the location
            of the variable to be changed.
    """
    def __init__(self, args, options=None):
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        if args.resume:
            self.resume = Path(args.resume)
            self.config_path = self.resume.parent.parent / 'config.yaml'
        else:
            assert args.config is not None, 'config not specified'
            self.resume = None
            self.config_path = Path(args.config)

        config = read_yaml(self.config_path)
        self.__config = _update_config(config, options, args)

        save_dir = Path(self.config['save_dir'])
        exper_name = self.config['name']
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
        self.__log_dir = save_dir / (exper_name + ':' + timestamp)
        self.__save_dir = self.log_dir / 'model'

        self.save_dir.mkdir(parents=True, exist_ok=True)

        write_yaml(self.config, self.log_dir / 'config.yaml')
        # make sure that logger writes into the log_dir
        for _, handler in config['logger']['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(self.log_dir / handler['filename'])
        logging.config.dictConfig(config['logger'])
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def initialize(self, name, module, *args):
        """ Initialize a instance according to config file
        :param name: the key name of the class in config
        :param module: the module in which the class is to be initialized
        :return: A initialized instance of class config[name]['type'] in
            module, with args config[name]['args']
        """
        module_config = self[name]
        return getattr(module, module_config['type'])\
            (*args, **module_config['args'])

    def get_logger(self, name, verbosity=2):
        """
        :param name: the name of the logger
        :param verbosity: 0=warning, 1=info, 2=debug, (default: 2)
        :return: a logger with specified name and verbosity
        """
        assert verbosity in self.log_levels, 'verbosity not integer from 0 to 2'
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def __getitem__(self, name):
        return self.config[name]

    @property
    def config(self):
        return self.__config

    @property
    def log_dir(self):
        return self.__log_dir

    @property
    def save_dir(self):
        return self.__save_dir


def _update_config(config, options, args):
    """Override the config file with the options entered through terminal"""
    for option in options:
        value = getattr(args, _get_opt_name(option))
        if value is not None:
            _set_by_path(config, option.target, value)
    return config


def _get_opt_name(option):
    for flag in option.flags:
        if flag.startswith('--'):
            return flag[2:]
    raise AttributeError('the option has no verbose flag')


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
