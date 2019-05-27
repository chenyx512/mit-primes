import yaml


def read_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def write_yaml(data, path):
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)
