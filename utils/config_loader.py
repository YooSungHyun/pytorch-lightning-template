import json


class ConfigLoader:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = ConfigLoader(**v)  # use recursively
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def config_to_dict(config: ConfigLoader):
    result = dict()
    for k, v in config.items():
        if type(v) == ConfigLoader:
            result[k] = config_to_dict(v)
        else:
            result[k] = v
    return result


def load_config(config_filename: str):
    with open(config_filename, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = ConfigLoader(**config)
    return hparams
