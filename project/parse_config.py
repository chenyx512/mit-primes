from datetime import datetime
from functools import reduce
from operator import getitem
from pathlib import Path
import logging, logging.config

from utils.utils import read_yaml, write_yaml


class ConfigParser:

    def __init__(self, args, options=None):
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        if args.resume:
            self.resume = Path(args.resume)
            self.config_path = self.resume.parent / 'config.yaml'
        else:
            assert args.config is not None, 'config not specified'
            self.resume = None
            self.config_path = Path(args.config)

        config = read_yaml(self.config_path)
        self.__config = _update_config(config, options, args)

        save_dir = Path(self.config['save_dir'])
        exper_name = self.config['name']
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
        self.__save_dir = save_dir / 'models' / (exper_name + timestamp)
        self.__log_dir = save_dir / 'log' / (exper_name + timestamp)

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        write_yaml(self.config, self.save_dir / 'config.yaml')
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
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_config = self[name]
        return getattr(module, module_config['type'])(*args, **module_config['args'])

    def get_logger(self, name, verbosity=2):
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
