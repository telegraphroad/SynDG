#from . import bsds300
from .bsds300 import BSDS300
from .cifar10 import CIFAR10
from .gas import GAS
from .hepmass import HEPMASS
from .miniboone import MINIBOONE
from .mnist import MNIST
from .power import POWER

data_mapping = {'BSDS300': BSDS300,
                'GAS': GAS,
                'MINIBOONE': MINIBOONE,
                'POWER': POWER,
                'HEPMASS': HEPMASS
                }
