from .backbone import *
from .nodecls_gnns import *
from .graphcls_gnns import *
from .pretrain_gnns import *

def model_entry(config):
    return globals()[config['type']](**config['kwargs'])