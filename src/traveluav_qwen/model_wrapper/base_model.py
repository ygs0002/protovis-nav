import torch


class BaseModelWrapper:
    def __init__(self):
        self.model: torch.Module = None

    def prepare_inputs(self, episodes):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def run(self, *args, **kwds):
        raise NotImplementedError

