
from torch import nn
import torch.nn.functional as F

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Mish: Mish Activation Function
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

class Mish(nn.Module):
    def __init__(self, inplace=False):
        super(Mish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.mish(x, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str