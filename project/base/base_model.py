import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """Overriding __str__ to output #trainable_params"""

    @abstractmethod
    def forward(self, *input):
        raise NotImplementedError

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nTrainable parameters: {params}'